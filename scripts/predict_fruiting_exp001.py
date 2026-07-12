#!/usr/bin/env python3
"""EXP-001 overfit sanity 예측 스크립트.

finetune_agri_fruiting_overfit + overfit-epoch=19.ckpt 로
val 40건 기준 버킷 Acc 62.5% / MAE Count 0.625 를 재현합니다.

Usage:
    python scripts/predict_fruiting_exp001.py
    python scripts/predict_fruiting_exp001.py --split val --show-errors

샘플별 상세 CSV (4열):
    outputs/berry2vec_fruiting_predictions/EXP-001/samples/<split>/seq_<id>.csv
    - 1열 input_token / 2열 saliency / 3열 output_class / 4열 probability
    - 예측 클래스는 output_class에 * 표시
    - 메타정보: seq_<id>_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import InputXGradient
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_new.datamodule import collate_encoded_documents
from src.transformer.agri_fruiting_model import (
    FRUITING_NUM_CLASSES,
    fruiting_bucket_to_count,
    fruiting_count_to_bucket,
)

# EXP-001 고정값 (docs/experiments/fruiting/EXP-001_overfit_sanity_62p5.md)
EXP_ID = "EXP-001"
DEFAULT_EXPERIMENT = "finetune_agri_fruiting_overfit"
DEFAULT_DATAMODULE = "agri_fruiting_exp001_predict"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints/agri/fruiting/l2v/overfit/overfit-epoch=19.ckpt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "berry2vec_fruiting_predictions" / EXP_ID
BUCKET_LABELS = ("≤3", "4", "5", "6", "7", "≥8")
EXPECTED_VAL_ACC = 0.625
EXPECTED_VAL_MAE = 0.625


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-001 overfit sanity — 버킷 Acc 62.5% 재현 예측"
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="Hydra experiment config (default: overfit sanity)",
    )
    parser.add_argument(
        "--datamodule",
        default=DEFAULT_DATAMODULE,
        help="Datamodule config (default: agri_growth_set vocab for EXP-001)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="EXP-001 best checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="평가 split (overfit 설정에서는 train/val/test 모두 40건)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="오답 샘플만 콘솔에 출력",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="기대 메트릭과의 허용 오차",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="샘플별 상세 CSV 저장 디렉터리 (기본: <output-dir>/samples/<split>)",
    )
    parser.add_argument(
        "--no-saliency",
        action="store_true",
        help="saliency 계산 생략 (토큰·확률만 저장)",
    )
    return parser.parse_args()


def _config_dir() -> str:
    return str((PROJECT_ROOT / "conf").resolve())


def load_checkpoint_hparams(checkpoint: Path) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    hp = ckpt["hyper_parameters"]
    if hasattr(hp, "items"):
        return dict(hp)
    return hp


def load_experiment_config(experiment: str, datamodule: str, checkpoint: Path):
    hp = load_checkpoint_hparams(checkpoint)
    vocab_size = int(hp["vocab_size"])
    overrides = [
        f"experiment={experiment}",
        f"datamodule={datamodule}",
        f"model.hparams.vocab_size={vocab_size}",
        "model.hparams.pretrained_model_path=none",
    ]
    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def load_model_and_datamodule(
    experiment: str,
    datamodule: str,
    checkpoint: Path,
    device: str,
):
    cfg = load_experiment_config(experiment, datamodule, checkpoint)
    dm = instantiate(cfg.datamodule, _convert_="all")
    dm.setup()

    model = instantiate(cfg.model, _convert_="all")
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model, dm, cfg


def bucket_label(bucket: int) -> str:
    return BUCKET_LABELS[int(bucket)]


def _extract_logits(model, outputs: torch.Tensor) -> torch.Tensor:
    if outputs.dim() == 3:
        return outputs[:, 0, :]
    return outputs


def _logits_to_probs(model, logits: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "sigsoftmax"):
        return model.sigsoftmax(logits)
    return F.softmax(logits, dim=-1)


def build_saliency_fn(model):
    def forward_for_attr(embeddings, meta):
        padding_mask = meta["padding_mask"].long()
        hidden = model.transformer.forward_finetuning_with_embeddings(
            embeddings, padding_mask
        )
        if getattr(model.hparams, "pooled", False):
            output = model.decoder(hidden, mask=padding_mask)
        else:
            output = model.decoder(hidden)
        logits = _extract_logits(model, output)
        return _logits_to_probs(model, logits)

    return InputXGradient(forward_for_attr)


def decode_input_tokens(
    token_ids: Sequence[int],
    abspos: Sequence[int],
    ages: Sequence[float],
    mask: Sequence[int],
    index2token: Dict[int, str],
) -> List[str]:
    tokens: List[str] = []
    for token_id, day, age, is_valid in zip(token_ids, abspos, ages, mask):
        if not is_valid:
            continue
        tokens.append(index2token.get(int(token_id), f"<UNK_{int(token_id)}>"))
    return tokens


def _summarize_token_saliency(attr: torch.Tensor, mask: torch.Tensor) -> List[float]:
    attr = attr.sum(dim=-1)
    attr = attr[mask]
    norm = torch.norm(attr)
    if norm > 0:
        attr = attr / norm
    return attr.detach().cpu().tolist()


def save_sample_detail_csv(
    path: Path,
    *,
    input_tokens: Sequence[str],
    saliency_scores: Sequence[float],
    class_probs: Dict[str, float],
    pred_class: str,
) -> None:
    """샘플 단위 4열 CSV 저장.

    1열 input_token | 2열 saliency | 3열 output_class | 4열 probability
    - 입력 토큰 행: 1~2열 채움
    - 출력 클래스 행(6개): 3~4열 채움
    """
    rows: List[Dict[str, Any]] = []
    for token, score in zip(input_tokens, saliency_scores):
        rows.append(
            {
                "input_token": token,
                "saliency": score,
                "output_class": "",
                "probability": "",
            }
        )
    for class_label in BUCKET_LABELS:
        marker = "*" if class_label == pred_class else ""
        rows.append(
            {
                "input_token": "",
                "saliency": "",
                "output_class": f"{class_label}{marker}",
                "probability": class_probs[class_label],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_sample_meta_json(path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def predict_split(
    model,
    dm,
    split: str,
    device: str,
    *,
    index2token: Dict[int, str],
    samples_dir: Path | None = None,
    compute_saliency: bool = True,
) -> pd.DataFrame:
    if split == "train":
        dataset = dm.train
    elif split == "val":
        dataset = dm.val
    elif split == "test":
        dataset = dm.test
    else:
        raise ValueError(f"Unknown split: {split}")

    loader = DataLoader(
        dataset,
        batch_size=dm.batch_size,
        shuffle=False,
        collate_fn=collate_encoded_documents,
        num_workers=0,
    )

    rows: List[Dict] = []
    attr_fn = build_saliency_fn(model) if compute_saliency else None

    for batch in tqdm(loader, desc=f"predict:{split}", leave=False):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        outputs = model(batch)
        logits = _extract_logits(model, outputs)
        probs = _logits_to_probs(model, logits)
        pred_buckets = torch.argmax(probs, dim=-1)
        pred_counts = fruiting_bucket_to_count(pred_buckets)

        with torch.no_grad():
            targets = batch["target"].detach().cpu().numpy().reshape(-1)
            target_buckets = fruiting_count_to_bucket(
                torch.tensor(targets, dtype=torch.float32)
            ).numpy()
            sequence_ids = batch["sequence_id"].detach().cpu().numpy().reshape(-1)
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()
            prob_rows = probs.detach().cpu().numpy()
            pred_buckets_cpu = pred_buckets.detach()

        saliency_tensor = None
        if attr_fn is not None:
            embeddings, _ = model.transformer.get_sequence_embedding(
                batch["input_ids"].long()
            )
            embeddings = embeddings.detach().requires_grad_(True)
            padding_mask = batch["padding_mask"].long()
            saliency_tensor = attr_fn.attribute(
                embeddings,
                target=pred_buckets_cpu.to(device),
                additional_forward_args={"padding_mask": padding_mask},
            )

        token_rows = batch["input_ids"][:, 0].detach().cpu().numpy()
        abspos_rows = batch["input_ids"][:, 1].detach().cpu().numpy()
        age_rows = batch["input_ids"][:, 2].detach().cpu().numpy()
        mask_rows = batch["padding_mask"].detach().cpu().numpy()

        for idx in range(len(sequence_ids)):
            mask = mask_rows[idx].astype(bool)
            input_tokens = decode_input_tokens(
                token_ids=token_rows[idx],
                abspos=abspos_rows[idx],
                ages=age_rows[idx],
                mask=mask_rows[idx],
                index2token=index2token,
            )
            if saliency_tensor is not None:
                saliency_scores = _summarize_token_saliency(saliency_tensor[idx], mask)
            else:
                saliency_scores = [0.0] * len(input_tokens)

            target = float(targets[idx])
            target_bucket = int(target_buckets[idx])
            pred_bucket = int(pred_buckets[idx].item())
            prediction = float(pred_counts[idx].item())
            sequence_id = int(sequence_ids[idx])
            pred_class_label = bucket_label(pred_bucket)
            class_probs = {
                bucket_label(class_idx): float(prob_rows[idx, class_idx])
                for class_idx in range(FRUITING_NUM_CLASSES)
            }

            if samples_dir is not None:
                sample_csv = samples_dir / f"seq_{sequence_id}.csv"
                save_sample_detail_csv(
                    sample_csv,
                    input_tokens=input_tokens,
                    saliency_scores=saliency_scores,
                    class_probs=class_probs,
                    pred_class=pred_class_label,
                )
                save_sample_meta_json(
                    samples_dir / f"seq_{sequence_id}_meta.json",
                    {
                        "exp_id": EXP_ID,
                        "split": split,
                        "sequence_id": sequence_id,
                        "target": target,
                        "target_bucket": target_bucket,
                        "target_bucket_label": bucket_label(target_bucket),
                        "pred_bucket": pred_bucket,
                        "pred_bucket_label": pred_class_label,
                        "prediction": prediction,
                        "confidence": float(confidences[idx]),
                        "bucket_correct": pred_bucket == target_bucket,
                        "abs_error": abs(target - prediction),
                        "n_input_tokens": len(input_tokens),
                        "detail_csv": sample_csv.name,
                    },
                )

            rows.append(
                {
                    "exp_id": EXP_ID,
                    "split": split,
                    "sequence_id": sequence_id,
                    "target": target,
                    "target_bucket": target_bucket,
                    "target_bucket_label": bucket_label(target_bucket),
                    "pred_bucket": pred_bucket,
                    "pred_bucket_label": pred_class_label,
                    "prediction": prediction,
                    "confidence": float(confidences[idx]),
                    "bucket_correct": pred_bucket == target_bucket,
                    "abs_error": abs(target - prediction),
                    "n_input_tokens": len(input_tokens),
                    "detail_csv": (
                        f"samples/{split}/seq_{sequence_id}.csv"
                        if samples_dir is not None
                        else ""
                    ),
                }
            )

    return pd.DataFrame(rows)


def bucket_distribution(df: pd.DataFrame, column: str) -> Dict[int, int]:
    counts = Counter(df[column].astype(int).tolist())
    return {bucket: counts.get(bucket, 0) for bucket in range(FRUITING_NUM_CLASSES)}


def print_bucket_table(df: pd.DataFrame) -> None:
    true_dist = bucket_distribution(df, "target_bucket")
    pred_dist = bucket_distribution(df, "pred_bucket")

    print("\n버킷 분포 (실제 vs 예측)")
    print(f"{'버킷':<8} {'라벨':<6} {'실제':>6} {'예측':>6}")
    print("-" * 30)
    for bucket in range(FRUITING_NUM_CLASSES):
        print(
            f"{bucket:<8} {BUCKET_LABELS[bucket]:<6} "
            f"{true_dist[bucket]:>6} {pred_dist[bucket]:>6}"
        )


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "n_samples": int(len(df)),
        "bucket_acc": float(df["bucket_correct"].mean()),
        "mae_count": float(df["abs_error"].mean()),
        "n_correct": int(df["bucket_correct"].sum()),
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"체크포인트가 없습니다: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, dm, cfg = load_model_and_datamodule(
        experiment=args.experiment,
        datamodule=args.datamodule,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    samples_dir = args.samples_dir
    if samples_dir is None:
        samples_dir = args.output_dir / "samples" / args.split

    index2token = dm.vocabulary.index2token
    results_df = predict_split(
        model,
        dm,
        args.split,
        args.device,
        index2token=index2token,
        samples_dir=samples_dir,
        compute_saliency=not args.no_saliency,
    )
    metrics = compute_metrics(results_df)

    predictions_path = args.output_dir / f"predictions_{args.split}.csv"
    results_df.to_csv(predictions_path, index=False)
    results_df.to_csv(args.output_dir / "predictions_all.csv", index=False)

    summary = {
        "exp_id": EXP_ID,
        "experiment": args.experiment,
        "datamodule": args.datamodule,
        "vocab_size": int(cfg.model.hparams.vocab_size),
        "vocab_name": str(cfg.datamodule.vocabulary.name),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": args.device,
        "pooled": bool(cfg.model.hparams.pooled),
        "loss_type": str(cfg.model.hparams.loss_type),
        "expected": {"bucket_acc": EXPECTED_VAL_ACC, "mae_count": EXPECTED_VAL_MAE},
        "metrics": metrics,
        "samples_dir": str(samples_dir),
        "n_sample_files": int(len(results_df)),
    }
    metrics_path = args.output_dir / "metrics_summary.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    n = metrics["n_samples"]
    acc = metrics["bucket_acc"]
    mae = metrics["mae_count"]
    n_correct = metrics["n_correct"]

    print(f"\n=== {EXP_ID} 예측 결과 ({args.split}, n={n}) ===")
    print(f"버킷 Acc : {acc:.1%} ({n_correct}/{n})")
    print(f"MAE Count: {mae:.3f}")
    print_bucket_table(results_df)

    acc_ok = abs(acc - EXPECTED_VAL_ACC) <= args.tolerance
    mae_ok = abs(mae - EXPECTED_VAL_MAE) <= args.tolerance
    if args.split == "val":
        status = "OK" if acc_ok and mae_ok else "MISMATCH"
        print(
            f"\n기대값 대비: Acc {'✓' if acc_ok else '✗'} "
            f"({EXPECTED_VAL_ACC:.1%}), MAE {'✓' if mae_ok else '✗'} "
            f"({EXPECTED_VAL_MAE:.3f}) → {status}"
        )

    if args.show_errors:
        errors = results_df[~results_df["bucket_correct"]]
        if errors.empty:
            print("\n오답 없음.")
        else:
            cols = [
                "sequence_id",
                "target",
                "target_bucket_label",
                "prediction",
                "pred_bucket_label",
                "confidence",
            ]
            print(f"\n오답 샘플 ({len(errors)}건):")
            print(errors[cols].to_string(index=False))

    print(f"\n저장: {predictions_path}")
    print(f"저장: {metrics_path}")
    print(f"샘플 상세: {samples_dir}/seq_<id>.csv ({len(results_df)}건)")


if __name__ == "__main__":
    main()
