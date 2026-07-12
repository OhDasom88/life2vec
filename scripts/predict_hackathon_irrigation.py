#!/usr/bin/env python3
"""Run irrigation inference on hackathon test_X.csv with 30-minute seq_id windows."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_new.ops import concat_columns_dask
from src.data_new.sources.hackathon_environment import (  # noqa: E402
    HackathonEnvironmentTokens,
    _parse_hackathon_time,
)
from src.data_new.datamodule import collate_encoded_documents  # noqa: E402


DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "rawdata" / "hackathon" / "test_X.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hackathon_irrigation_predictions"
DEFAULT_EXPERIMENT = "finetune_hackathon"
DEFAULT_CHECKPOINT_GLOB = "checkpoints/hackathon/hackathon_cls/l2v/**/*.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="30분 단위 seq_id를 생성해 hackathon irrigation 여부를 예측합니다."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="분류 모델 체크포인트(.ckpt). 미지정 시 checkpoints/hackathon 아래 최신 파일을 탐색합니다.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--seq-minutes",
        type=int,
        default=30,
        help="한 seq_id로 묶을 시간 단위(분)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="추론 배치 크기",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.5,
        help="irrigation=1 판정 확률 임계값",
    )
    return parser.parse_args()


def _config_dir() -> str:
    return str((PROJECT_ROOT / "conf").resolve())


def discover_checkpoint() -> Path:
    candidates = sorted(
        PROJECT_ROOT.glob(DEFAULT_CHECKPOINT_GLOB),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "체크포인트를 찾지 못했습니다. "
            "--checkpoint 로 .ckpt 경로를 직접 지정해 주세요."
        )
    return candidates[0]


def load_checkpoint_hparams(checkpoint: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    if hasattr(hparams, "items"):
        return dict(hparams)
    return hparams


def load_experiment_config(experiment: str, vocab_size: int, checkpoint: Path):
    ckpt_hparams = load_checkpoint_hparams(checkpoint)
    overrides = [
        f"experiment={experiment}",
        f"model.hparams.vocab_size={int(ckpt_hparams.get('vocab_size', vocab_size))}",
        "model.hparams.pretrained_model_path=none",
    ]
    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def load_model_and_datamodule(experiment: str, checkpoint: Path, device: str):
    base_cfg = load_experiment_config(
        experiment=experiment,
        vocab_size=570,
        checkpoint=checkpoint,
    )
    dm = instantiate(base_cfg.datamodule, _convert_="all")
    dm.vocabulary.prepare()

    cfg = load_experiment_config(
        experiment=experiment,
        vocab_size=dm.vocabulary.size(),
        checkpoint=checkpoint,
    )
    dm = instantiate(cfg.datamodule, _convert_="all")
    dm.vocabulary.prepare()

    model = instantiate(cfg.model, _convert_="all")
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model, dm, cfg


def assign_seq_ids(raw_df: pd.DataFrame, seq_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw_df.copy()
    df["timestamp"] = df["time"].map(_parse_hackathon_time)
    df["window_start"] = df["timestamp"].dt.floor(f"{seq_minutes}min")
    df["window_end"] = df["window_start"] + pd.Timedelta(minutes=seq_minutes - 1)

    window_codes, unique_windows = pd.factorize(df["window_start"], sort=True)
    df["seq_id"] = window_codes.astype(int) + 1

    windows_df = (
        df.groupby("seq_id", as_index=False)
        .agg(
            window_start=("window_start", "min"),
            window_end=("window_end", "max"),
            n_rows=("time", "size"),
        )
        .sort_values("seq_id")
    )
    assert len(unique_windows) == len(windows_df)
    return df, windows_df


def build_tokenized_sentences(
    input_with_seq: Path,
    dm,
    source_name: str = "hackathon_environment_infer",
) -> pd.DataFrame:
    infer_source = HackathonEnvironmentTokens(name=source_name, input_csv=input_with_seq)
    tokenized = infer_source.tokenized()

    train_source = dm.corpus.sources[0]
    fitted_fields = dm.corpus.fitted_fields(train_source)
    for field in fitted_fields:
        tokenized[field.field_label] = field.transform(tokenized[field.field_label])

    field_labels = infer_source.field_labels()
    tokenized = tokenized.astype({label: "string" for label in field_labels}).assign(
        SENTENCE=concat_columns_dask(tokenized, columns=field_labels)
    )[["START_DATE", "AGE", "SENTENCE"]]

    sentences = tokenized.compute().reset_index()
    sentences = sentences.sort_values(["PERSON_ID", "START_DATE"]).set_index("PERSON_ID")

    reference_date = pd.Timestamp(dm.corpus.reference_date)
    sentences["START_DATE"] = (sentences["START_DATE"] - reference_date).dt.days.astype(int)
    sentences["AFTER_THRESHOLD"] = False
    sentences["RES_ORIGIN"] = "GH_1"
    sentences["GENDER"] = "LN_0"
    sentences["BIRTHDAY"] = pd.Timestamp("2024-01-01")
    sentences["TARGET"] = 0
    return sentences


def encode_documents(sentences_df: pd.DataFrame, dm) -> list[Any]:
    preprocessor = dm.task.get_preprocessor(is_train=False)
    encoded_docs = []
    for person_id, group in sentences_df.groupby(level=0, sort=True):
        group.name = person_id
        document = dm.task.get_document(group)
        encoded_docs.append(preprocessor(document))
    return encoded_docs


@torch.no_grad()
def predict_encoded_documents(
    *,
    model,
    encoded_docs: list[Any],
    device: str,
    batch_size: int,
    positive_threshold: float,
) -> pd.DataFrame:
    loader = DataLoader(
        encoded_docs,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_encoded_documents,
        num_workers=0,
    )

    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        pos_probs = probs[:, 1]
        preds = (pos_probs >= positive_threshold).long()

        sequence_ids = batch["sequence_id"].detach().cpu().numpy().reshape(-1)
        pos_probs_np = pos_probs.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        neg_probs_np = probs[:, 0].detach().cpu().numpy()

        for seq_id, neg_prob, pos_prob, pred in zip(
            sequence_ids,
            neg_probs_np,
            pos_probs_np,
            preds_np,
        ):
            rows.append(
                {
                    "seq_id": int(seq_id),
                    "irrigation_probability": float(pos_prob),
                    "non_irrigation_probability": float(neg_prob),
                    "irrigation_prediction": int(pred),
                }
            )

    return pd.DataFrame(rows).sort_values("seq_id").reset_index(drop=True)


def write_outputs(
    *,
    output_dir: Path,
    assigned_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    args: argparse.Namespace,
    checkpoint: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    assigned_path = output_dir / "test_X_with_seq_id.csv"
    assigned_df.drop(columns=["timestamp"]).to_csv(assigned_path, index=False)

    window_predictions = windows_df.merge(predictions_df, on="seq_id", how="left")
    predictions_path = output_dir / "irrigation_predictions.csv"
    window_predictions.to_csv(predictions_path, index=False)

    submission_path = output_dir / "submission_irrigation.csv"
    window_predictions[["seq_id", "irrigation_prediction"]].to_csv(
        submission_path, index=False
    )

    run_config = {
        "input_csv": str(args.input_csv),
        "checkpoint": str(checkpoint),
        "experiment": args.experiment,
        "seq_minutes": args.seq_minutes,
        "batch_size": args.batch_size,
        "positive_threshold": args.positive_threshold,
        "n_sequences": int(len(window_predictions)),
        "n_rows": int(len(assigned_df)),
    }
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "assigned_path": assigned_path,
        "predictions_path": predictions_path,
        "submission_path": submission_path,
        "run_config_path": run_config_path,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or discover_checkpoint()
    if not checkpoint.exists():
        raise FileNotFoundError(f"체크포인트가 없습니다: {checkpoint}")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {args.input_csv}")

    raw_df = pd.read_csv(args.input_csv)
    assigned_df, windows_df = assign_seq_ids(raw_df, seq_minutes=args.seq_minutes)

    model, dm, cfg = load_model_and_datamodule(
        experiment=args.experiment,
        checkpoint=checkpoint,
        device=args.device,
    )

    with tempfile.TemporaryDirectory(prefix="hackathon_infer_", dir=PROJECT_ROOT) as tmp_dir:
        temp_input = Path(tmp_dir) / "test_X_seq.csv"
        assigned_df.drop(columns=["timestamp", "window_start", "window_end"]).to_csv(
            temp_input,
            index=False,
        )

        sentences_df = build_tokenized_sentences(temp_input, dm)
        encoded_docs = encode_documents(sentences_df, dm)

    predictions_df = predict_encoded_documents(
        model=model,
        encoded_docs=encoded_docs,
        device=args.device,
        batch_size=args.batch_size,
        positive_threshold=args.positive_threshold,
    )
    output_paths = write_outputs(
        output_dir=args.output_dir,
        assigned_df=assigned_df,
        windows_df=windows_df,
        predictions_df=predictions_df,
        args=args,
        checkpoint=checkpoint,
    )

    positive_rate = float(predictions_df["irrigation_prediction"].mean())
    print("\n[hackathon irrigation inference]")
    print(f"- checkpoint: {checkpoint}")
    print(f"- experiment: {args.experiment}")
    print(f"- device: {args.device}")
    print(f"- rows: {len(assigned_df)}")
    print(f"- sequences({args.seq_minutes}m): {len(predictions_df)}")
    print(f"- predicted positive rate: {positive_rate:.4f}")
    print("\n[files]")
    for key, path in output_paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
