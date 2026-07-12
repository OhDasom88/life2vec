#!/usr/bin/env python3
"""Build row-level hackathon result CSVs using the best finetuned checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from predict_hackathon_irrigation import (
    assign_seq_ids,
    build_tokenized_sentences,
    encode_documents,
    load_model_and_datamodule,
    predict_encoded_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "hackathon" / "hackathon_cls" / "l2v" / "sweep_p251zrwk" / "sweep_p251zrwk-epoch=09.ckpt"
TRAIN_X = PROJECT_ROOT / "data" / "rawdata" / "hackathon" / "train_X.csv"
TEST_X = PROJECT_ROOT / "data" / "rawdata" / "hackathon" / "test_X.csv"
TARGETS = PROJECT_ROOT / "data" / "rawdata" / "hackathon" / "targets.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hackathon_best_model_results"
TEMP_DIR = PROJECT_ROOT / "data" / "interim" / "hackathon_inference_inputs"
EXPERIMENT = "finetune_hackathon"
DEVICE = "cuda"
BATCH_SIZE = 64
POSITIVE_THRESHOLD = 0.5


def ensure_seq_id(df: pd.DataFrame) -> pd.DataFrame:
    if "seq_id" in df.columns:
        return df.copy()
    assigned_df, _ = assign_seq_ids(df, seq_minutes=30)
    return assigned_df.drop(columns=["timestamp", "window_start", "window_end"])


def predict_for_dataframe(model, dm, input_df: pd.DataFrame, stem: str) -> pd.DataFrame:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_csv = TEMP_DIR / f"{stem}_with_seq_id.csv"
    input_df.to_csv(temp_csv, index=False)
    sentences_df = build_tokenized_sentences(
        temp_csv,
        dm,
        source_name=f"hackathon_environment_infer_{stem}",
    )
    encoded_docs = encode_documents(sentences_df, dm)
    return predict_encoded_documents(
        model=model,
        encoded_docs=encoded_docs,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        positive_threshold=POSITIVE_THRESHOLD,
    )


def attach_predictions_to_last_rows(
    row_df: pd.DataFrame,
    seq_predictions: pd.DataFrame,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result_df = row_df.copy()
    result_df["prediction"] = pd.NA
    if targets is not None:
        result_df["target"] = pd.NA

    last_row_index = result_df.groupby("seq_id", sort=False).tail(1).index
    seq_to_last_idx = result_df.loc[last_row_index, ["seq_id"]].reset_index().set_index("seq_id")["index"]

    for row in seq_predictions.itertuples(index=False):
        seq_id = int(row.seq_id)
        if seq_id not in seq_to_last_idx.index:
            continue
        result_df.loc[seq_to_last_idx.loc[seq_id], "prediction"] = int(row.irrigation_prediction)

    if targets is not None:
        target_map = targets.set_index("seq_id")["target"]
        for seq_id, idx in seq_to_last_idx.items():
            if seq_id in target_map.index:
                result_df.loc[idx, "target"] = int(target_map.loc[seq_id])

    return result_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, dm, _ = load_model_and_datamodule(
        experiment=EXPERIMENT,
        checkpoint=BEST_CHECKPOINT,
        device=DEVICE,
    )

    train_df = ensure_seq_id(pd.read_csv(TRAIN_X))
    test_df = ensure_seq_id(pd.read_csv(TEST_X))
    targets_df = pd.read_csv(TARGETS).astype({"seq_id": int, "target": int})

    train_predictions = predict_for_dataframe(model, dm, train_df, "train")
    test_predictions = predict_for_dataframe(model, dm, test_df, "test")

    result_train = attach_predictions_to_last_rows(
        row_df=train_df,
        seq_predictions=train_predictions,
        targets=targets_df,
    )
    result_test = attach_predictions_to_last_rows(
        row_df=test_df,
        seq_predictions=test_predictions,
        targets=None,
    )

    result_train_path = OUTPUT_DIR / "result_train.csv"
    result_test_path = OUTPUT_DIR / "result_test.csv"
    train_predictions_path = OUTPUT_DIR / "train_seq_predictions.csv"
    test_predictions_path = OUTPUT_DIR / "test_seq_predictions.csv"

    result_train.to_csv(result_train_path, index=False)
    result_test.to_csv(result_test_path, index=False)
    train_predictions.to_csv(train_predictions_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)

    summary = {
        "best_checkpoint": str(BEST_CHECKPOINT),
        "experiment": EXPERIMENT,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "positive_threshold": POSITIVE_THRESHOLD,
        "train_rows": int(len(result_train)),
        "test_rows": int(len(result_test)),
        "train_sequences": int(train_df["seq_id"].nunique()),
        "test_sequences": int(test_df["seq_id"].nunique()),
        "filled_train_predictions": int(result_train["prediction"].notna().sum()),
        "filled_train_targets": int(result_train["target"].notna().sum()),
        "filled_test_predictions": int(result_test["prediction"].notna().sum()),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("[best model]")
    print(BEST_CHECKPOINT)
    print("[outputs]")
    print(result_train_path)
    print(result_test_path)
    print(train_predictions_path)
    print(test_predictions_path)


if __name__ == "__main__":
    main()
