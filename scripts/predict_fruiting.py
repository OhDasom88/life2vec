#!/usr/bin/env python3
"""Berry2vec finetuned model inference for fruiting-count prediction.

Loads the trained Transformer_AgriFruiting checkpoint, runs predictions on
train/val/test splits, and exports inputs, outputs (with confidence), and
token-level saliency scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import InputXGradient
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.scripts.paths import FINETUNE_CKPT, VOCAB_PATH
from src.data_new.datamodule import collate_encoded_documents


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "berry2vec_fruiting_predictions"
DEFAULT_EXPERIMENT = "finetune_agri_fruiting"
FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NotoSansCJKkr-Regular.otf"
SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
SPLIT_LABELS = {"train": "학습", "val": "검증", "test": "테스트"}
SALIENCY_SKIP_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def configure_korean_matplotlib() -> str:
    """Register bundled Korean font for matplotlib plots."""
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"한글 폰트 파일이 없습니다: {FONT_PATH}\n"
            "scripts/fonts/NotoSansCJKkr-Regular.otf 를 확인해 주세요."
        )
    fm.fontManager.addfont(str(FONT_PATH))
    font_name = fm.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=font_name)
    return font_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습된 berry2vec 모델로 착과수 예측 및 saliency 분석"
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="Hydra experiment config name",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=FINETUNE_CKPT,
        help="Finetuned berry2vec checkpoint path",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--show-top-k",
        type=int,
        default=10,
        help="콘솔에 출력할 샘플 수",
    )
    parser.add_argument(
        "--top-token-k",
        type=int,
        default=8,
        help="샘플별 saliency 상위 토큰 수",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="시각화 이미지 생성을 건너뜁니다.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="저장된 predictions/saliency 결과만으로 시각화를 생성합니다.",
    )
    parser.add_argument(
        "--saliency-plot-n",
        type=int,
        default=6,
        help="saliency 예시 그래프에 표시할 고오차 샘플 수",
    )
    return parser.parse_args()


def _config_dir() -> str:
    return str((PROJECT_ROOT / "conf").resolve())


def load_experiment_config(experiment: str) -> DictConfig:
    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        return compose(config_name="config", overrides=[f"experiment={experiment}"])


def load_vocab() -> pd.DataFrame:
    return pd.read_csv(VOCAB_PATH, sep="\t").set_index("ID")


def load_model_and_datamodule(
    experiment: str,
    checkpoint: Path,
    device: str,
):
    cfg = load_experiment_config(experiment)
    dm = instantiate(cfg.datamodule, _convert_="all")
    dm.setup()

    model = instantiate(cfg.model, _convert_="all")
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)
    return model, dm, cfg


def get_split_dataset(dm, split: str):
    if split == "train":
        return dm.train
    if split == "val":
        return dm.val
    if split == "test":
        return dm.test
    raise ValueError(f"Unknown split: {split}")


def make_dataloader(dm, dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_encoded_documents,
        num_workers=0,
    )


def decode_input_tokens(
    token_ids: Sequence[int],
    abspos: Sequence[int],
    ages: Sequence[float],
    mask: Sequence[int],
    index2token: Dict[int, str],
) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for pos, (token_id, day, age, is_valid) in enumerate(
        zip(token_ids, abspos, ages, mask)
    ):
        if not is_valid:
            continue
        tokens.append(
            {
                "position": pos,
                "token_id": int(token_id),
                "token": index2token.get(int(token_id), f"<UNK_{int(token_id)}>"),
                "abspos": int(day),
                "age": float(age),
            }
        )
    return tokens


def summarize_tokens(tokens: Sequence[Dict[str, Any]], max_tokens: int = 12) -> str:
    visible = tokens[:max_tokens]
    rendered = [
        f"{item['token']}(d={item['abspos']},a={item['age']:.0f})" for item in visible
    ]
    suffix = " ..." if len(tokens) > max_tokens else ""
    return " | ".join(rendered) + suffix


def top_tokens_by_saliency(
    tokens: Sequence[Dict[str, Any]],
    saliency: Sequence[float],
    top_k: int,
) -> str:
    pairs = list(zip(tokens, saliency))
    pairs.sort(key=lambda item: item[1], reverse=True)
    rendered = [
        f"{token['token']}:{score:.4f}" for token, score in pairs[:top_k]
    ]
    return " | ".join(rendered)


def compute_confidence(abs_errors: np.ndarray, calibration_mae: float) -> np.ndarray:
    scale = max(calibration_mae, 1e-6)
    return np.exp(-abs_errors / scale)


def _summarize_token_saliency(attr: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    attr = attr.sum(dim=-1)
    attr = attr[mask]
    norm = torch.norm(attr)
    if norm > 0:
        attr = attr / norm
    return attr.detach().cpu().numpy()


def build_saliency_fn(model):
    def forward_for_attr(embeddings, meta):
        padding_mask = meta["padding_mask"].long()
        hidden = model.transformer.forward_finetuning_with_embeddings(
            embeddings, padding_mask
        )
        output = model.decoder(hidden)
        if output.ndim == 3:
            output = output[:, 0, 0]
        return output.unsqueeze(-1)

    return InputXGradient(forward_for_attr)


@torch.no_grad()
def predict_batch(model, batch: Dict[str, torch.Tensor]) -> np.ndarray:
    outputs = model(batch)
    if outputs.ndim == 3:
        return outputs[:, 0, 0].detach().cpu().numpy()
    return outputs.reshape(outputs.shape[0], -1)[:, 0].detach().cpu().numpy()


def run_split(
    *,
    model,
    dm,
    split: str,
    device: str,
    index2token: Dict[int, str],
    attr_fn: InputXGradient,
    top_token_k: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset = get_split_dataset(dm, split)
    loader = make_dataloader(dm, dataset, batch_size=dm.batch_size)

    sample_rows: List[Dict[str, Any]] = []
    saliency_rows: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc=f"predict:{split}", leave=False):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        predictions = predict_batch(model, batch)
        targets = batch["target"].detach().cpu().numpy().reshape(-1)
        sequence_ids = batch["sequence_id"].detach().cpu().numpy().reshape(-1)

        embeddings, _ = model.transformer.get_sequence_embedding(
            batch["input_ids"].long()
        )
        embeddings = embeddings.detach().requires_grad_(True)
        padding_mask = batch["padding_mask"].long()
        saliency_tensor = attr_fn.attribute(
            embeddings,
            target=0,
            additional_forward_args={"padding_mask": padding_mask},
        )

        token_rows = batch["input_ids"][:, 0].detach().cpu().numpy()
        abspos_rows = batch["input_ids"][:, 1].detach().cpu().numpy()
        age_rows = batch["input_ids"][:, 2].detach().cpu().numpy()
        mask_rows = batch["padding_mask"].detach().cpu().numpy()

        for idx in range(len(sequence_ids)):
            mask = mask_rows[idx].astype(bool)
            tokens = decode_input_tokens(
                token_ids=token_rows[idx],
                abspos=abspos_rows[idx],
                ages=age_rows[idx],
                mask=mask_rows[idx],
                index2token=index2token,
            )
            token_saliency = _summarize_token_saliency(saliency_tensor[idx], mask)

            target = float(targets[idx])
            prediction = float(predictions[idx])
            abs_error = abs(target - prediction)

            sample_rows.append(
                {
                    "split": split,
                    "sequence_id": int(sequence_ids[idx]),
                    "target": target,
                    "prediction": prediction,
                    "abs_error": abs_error,
                    "sequence_length": int(mask.sum()),
                    "input_summary": summarize_tokens(tokens),
                    "top_tokens": top_tokens_by_saliency(
                        tokens, token_saliency.tolist(), top_token_k
                    ),
                }
            )
            saliency_rows.append(
                {
                    "split": split,
                    "sequence_id": int(sequence_ids[idx]),
                    "target": target,
                    "prediction": prediction,
                    "tokens": [
                        {
                            **token,
                            "saliency": float(score),
                        }
                        for token, score in zip(tokens, token_saliency.tolist())
                    ],
                }
            )

    return sample_rows, saliency_rows


def compute_split_metrics(df: pd.DataFrame) -> Dict[str, float]:
    errors = df["abs_error"].to_numpy()
    targets = df["target"].to_numpy()
    preds = df["prediction"].to_numpy()
    mae = float(np.mean(errors))
    rmse = float(math.sqrt(np.mean((targets - preds) ** 2)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n_samples": int(len(df))}


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_saved_results(output_dir: Path) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    predictions_path = output_dir / "predictions_all.csv"
    saliency_path = output_dir / "saliency_all.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"예측 결과 파일이 없습니다: {predictions_path}")
    if not saliency_path.exists():
        raise FileNotFoundError(f"saliency 결과 파일이 없습니다: {saliency_path}")

    results_df = pd.read_csv(predictions_path)
    saliency_rows: List[Dict[str, Any]] = []
    with saliency_path.open(encoding="utf-8") as handle:
        for line in handle:
            saliency_rows.append(json.loads(line))
    return results_df, saliency_rows


def _figures_dir(output_dir: Path) -> Path:
    path = output_dir / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_pred_vs_true(
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for split, group in results_df.groupby("split"):
        ax.scatter(
            group["target"],
            group["prediction"],
            label=SPLIT_LABELS.get(split, split),
            color=SPLIT_COLORS.get(split, "#666666"),
            s=70,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.9,
        )

    values = np.concatenate(
        [results_df["target"].to_numpy(), results_df["prediction"].to_numpy()]
    )
    lims = [float(values.min()) - 0.5, float(values.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("실제 착과수")
    ax.set_ylabel("예측 착과수")
    overall_mae = float(results_df["abs_error"].mean())
    ax.set_title(f"berry2vec 예측 vs 실제 (전체 MAE={overall_mae:.3f})")
    ax.legend(title="데이터 분할")
    ax.grid(alpha=0.2)

    out_path = figures_dir / "pred_vs_true.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return out_path


def plot_metrics_by_split(
    split_metrics: Dict[str, Dict[str, float]],
    figures_dir: Path,
) -> Path:
    metrics_df = pd.DataFrame(split_metrics).T.reset_index().rename(columns={"index": "split"})
    metrics_df["split_label"] = metrics_df["split"].map(lambda s: SPLIT_LABELS.get(s, s))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        ["mae", "rmse", "r2"],
        ["MAE", "RMSE", "R²"],
    ):
        colors = [SPLIT_COLORS.get(split, "#666666") for split in metrics_df["split"]]
        ax.bar(metrics_df["split_label"], metrics_df[metric], color=colors)
        ax.set_title(title)
        ax.set_xlabel("데이터 분할")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("데이터 분할별 예측 성능")
    fig.tight_layout()

    out_path = figures_dir / "metrics_by_split.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return out_path


def plot_error_and_confidence(
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, group in results_df.groupby("split"):
        sns.histplot(
            group["abs_error"],
            label=SPLIT_LABELS.get(split, split),
            color=SPLIT_COLORS.get(split, "#666666"),
            kde=True,
            stat="density",
            element="step",
            fill=False,
            ax=ax,
        )
    ax.set_xlabel("절대 오차")
    ax.set_ylabel("밀도")
    ax.set_title("데이터 분할별 절대 오차 분포")
    ax.legend()
    error_path = figures_dir / "error_distribution.png"
    fig.savefig(error_path, bbox_inches="tight", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for split, group in results_df.groupby("split"):
        ax.scatter(
            group["abs_error"],
            group["confidence"],
            label=SPLIT_LABELS.get(split, split),
            color=SPLIT_COLORS.get(split, "#666666"),
            s=60,
            alpha=0.85,
        )
    ax.set_xlabel("절대 오차")
    ax.set_ylabel("신뢰도")
    ax.set_title("신뢰도 vs 절대 오차")
    ax.legend(title="데이터 분할")
    ax.grid(alpha=0.2)
    confidence_path = figures_dir / "confidence_vs_error.png"
    fig.savefig(confidence_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return error_path, confidence_path


def _filter_saliency_tokens(tokens: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        token
        for token in tokens
        if token["token"] not in SALIENCY_SKIP_TOKENS
    ]


def plot_saliency_examples(
    results_df: pd.DataFrame,
    saliency_rows: Sequence[Dict[str, Any]],
    figures_dir: Path,
    top_n: int,
) -> Path | None:
    if not saliency_rows:
        return None

    saliency_by_id = {
        (row["split"], row["sequence_id"]): row for row in saliency_rows
    }
    ordered = results_df.sort_values("abs_error", ascending=False).head(top_n)
    if ordered.empty:
        return None

    n_cols = min(3, len(ordered))
    n_rows = math.ceil(len(ordered) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, ordered.iterrows()):
        key = (row["split"], int(row["sequence_id"]))
        saliency_row = saliency_by_id.get(key)
        if saliency_row is None:
            ax.axis("off")
            continue

        tokens = _filter_saliency_tokens(saliency_row["tokens"])
        if not tokens:
            ax.axis("off")
            continue

        top_tokens = sorted(tokens, key=lambda item: item["saliency"], reverse=True)[:15]
        labels = [item["token"] for item in top_tokens]
        scores = [item["saliency"] for item in top_tokens]
        y = np.arange(len(top_tokens))
        colors = plt.cm.Reds(
            np.array(scores) / (max(scores) + 1e-8)
        )
        ax.barh(y, scores, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(
            f"{SPLIT_LABELS.get(row['split'], row['split'])} "
            f"id={int(row['sequence_id'])} "
            f"실제={row['target']:.1f} 예측={row['prediction']:.1f}",
            fontsize=9,
        )

    for ax in axes[len(ordered) :]:
        ax.axis("off")

    fig.suptitle("고오차 샘플 saliency (상위 토큰, CLS/SEP 제외)")
    fig.tight_layout()
    out_path = figures_dir / "saliency_top_errors.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return out_path


def plot_prediction_table(
    results_df: pd.DataFrame,
    figures_dir: Path,
    top_k: int = 12,
) -> Path:
    preview = results_df.sort_values(["split", "sequence_id"]).head(top_k).copy()
    preview["prediction"] = preview["prediction"].map(lambda x: f"{x:.2f}")
    preview["confidence"] = preview["confidence"].map(lambda x: f"{x:.3f}")
    preview["abs_error"] = preview["abs_error"].map(lambda x: f"{x:.3f}")

    table_df = preview[
        ["split", "sequence_id", "target", "prediction", "confidence", "abs_error"]
    ].copy()
    table_df["split"] = table_df["split"].map(lambda s: SPLIT_LABELS.get(s, s))
    table_df = table_df.rename(
        columns={
            "split": "분할",
            "sequence_id": "개체ID",
            "target": "실제값",
            "prediction": "예측값",
            "confidence": "신뢰도",
            "abs_error": "절대오차",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(table_df) + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title("예측 결과 미리보기", pad=12)

    out_path = figures_dir / "prediction_preview_table.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return out_path


def generate_visualizations(
    *,
    results_df: pd.DataFrame,
    saliency_rows: Sequence[Dict[str, Any]],
    split_metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
    saliency_plot_n: int,
    show_top_k: int,
) -> List[Path]:
    configure_korean_matplotlib()
    figures_dir = _figures_dir(output_dir)
    generated = [
        plot_pred_vs_true(results_df, figures_dir),
        plot_metrics_by_split(split_metrics, figures_dir),
        plot_prediction_table(results_df, figures_dir, top_k=show_top_k),
    ]
    generated.extend(
        plot_error_and_confidence(results_df, figures_dir)
    )
    saliency_path = plot_saliency_examples(
        results_df,
        saliency_rows,
        figures_dir,
        top_n=saliency_plot_n,
    )
    if saliency_path is not None:
        generated.append(saliency_path)
    return generated


def print_preview(df: pd.DataFrame, top_k: int) -> None:
    preview_cols = [
        "split",
        "sequence_id",
        "target",
        "prediction",
        "confidence",
        "abs_error",
        "sequence_length",
        "top_tokens",
    ]
    print(df[preview_cols].head(top_k).to_string(index=False))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        results_df, all_saliency = load_saved_results(args.output_dir)
        metrics_path = args.output_dir / "metrics_summary.json"
        if metrics_path.exists():
            metrics_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
            split_metrics = metrics_summary["splits"]
            calibration_mae = metrics_summary.get(
                "confidence_calibration_mae",
                float(results_df["abs_error"].mean()),
            )
        else:
            split_metrics = {
                split: compute_split_metrics(group)
                for split, group in results_df.groupby("split")
            }
            calibration_mae = (
                split_metrics["val"]["mae"]
                if "val" in split_metrics
                else float(results_df["abs_error"].mean())
            )
        if "confidence" not in results_df.columns:
            results_df["confidence"] = compute_confidence(
                results_df["abs_error"].to_numpy(),
                calibration_mae=calibration_mae,
            )
    else:
        vocab_df = load_vocab()
        index2token = vocab_df["TOKEN"].to_dict()

        model, dm, cfg = load_model_and_datamodule(
            experiment=args.experiment,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        attr_fn = build_saliency_fn(model)

        all_samples: List[Dict[str, Any]] = []
        all_saliency: List[Dict[str, Any]] = []
        split_metrics: Dict[str, Dict[str, float]] = {}

        for split in args.splits:
            sample_rows, saliency_rows = run_split(
                model=model,
                dm=dm,
                split=split,
                device=args.device,
                index2token=index2token,
                attr_fn=attr_fn,
                top_token_k=args.top_token_k,
            )
            split_df = pd.DataFrame(sample_rows)
            all_samples.extend(sample_rows)
            all_saliency.extend(saliency_rows)
            split_metrics[split] = compute_split_metrics(split_df)

            split_df.to_csv(args.output_dir / f"predictions_{split}.csv", index=False)
            save_jsonl(args.output_dir / f"saliency_{split}.jsonl", saliency_rows)

        results_df = pd.DataFrame(all_samples)
        if "val" in args.splits:
            calibration_mae = split_metrics["val"]["mae"]
        else:
            calibration_mae = float(results_df["abs_error"].mean())

        results_df["confidence"] = compute_confidence(
            results_df["abs_error"].to_numpy(),
            calibration_mae=calibration_mae,
        )

        for split in args.splits:
            mask = results_df["split"] == split
            split_path = args.output_dir / f"predictions_{split}.csv"
            results_df.loc[mask].to_csv(split_path, index=False)

        results_df.to_csv(args.output_dir / "predictions_all.csv", index=False)
        save_jsonl(args.output_dir / "saliency_all.jsonl", all_saliency)

        metrics_summary = {
            "checkpoint": str(args.checkpoint),
            "experiment": args.experiment,
            "device": args.device,
            "confidence_calibration_mae": calibration_mae,
            "confidence_formula": "exp(-abs_error / val_mae)",
            "splits": split_metrics,
        }
        (args.output_dir / "metrics_summary.json").write_text(
            json.dumps(metrics_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (args.output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "experiment": args.experiment,
                    "output_dir": str(args.output_dir),
                    "splits": args.splits,
                    "loss_type": cfg.model.hparams.loss_type,
                    "max_length": int(cfg.model.hparams.max_length),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print("\n[berry2vec 예측 요약]")
    for split, metrics in split_metrics.items():
        print(
            f"- {split}: n={metrics['n_samples']}, "
            f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"
        )
    if not args.plot_only:
        print(f"- confidence calibration (val MAE): {calibration_mae:.4f}")

    print("\n[예측 결과 샘플]")
    print_preview(results_df, args.show_top_k)

    figure_paths: List[Path] = []
    if not args.no_plots:
        figure_paths = generate_visualizations(
            results_df=results_df,
            saliency_rows=all_saliency,
            split_metrics=split_metrics,
            output_dir=args.output_dir,
            saliency_plot_n=args.saliency_plot_n,
            show_top_k=args.show_top_k,
        )

    print("\n[저장 파일]")
    print(f"- 전체 예측: {args.output_dir / 'predictions_all.csv'}")
    print(f"- 전체 saliency: {args.output_dir / 'saliency_all.jsonl'}")
    print(f"- 성능 요약: {args.output_dir / 'metrics_summary.json'}")
    if figure_paths:
        print("- 시각화:")
        for path in figure_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
