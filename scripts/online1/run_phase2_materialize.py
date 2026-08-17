#!/usr/bin/env python3
"""Phase 2: split-first materialization + overlap/leakage checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/dasom/life2vec")
sys.path.insert(0, str(ROOT))

from src.online1.paths import Online1Paths, load_pipeline_config, load_yaml, CONFIG_DIR, write_json
from src.online1.env_loader import load_clean_env, aggregate_to_5min
from src.online1.vocab import build_vocab, save_vocab
from src.online1.materialize import materialize_pretrain_sequences
from src.online1.regression_anchors import build_regression_anchors
from src.online1.splits import verify_no_overlap, attach_split_id
from src.online1.report import write_phase_report, write_markdown_summary
from src.online1.lineage import scan_dataframe_for_denied, load_deny_config


def _explode_cube_ids(df: pd.DataFrame):
    ids = []
    for v in df.get("cube_ids", []):
        if isinstance(v, list):
            ids.extend(v)
    return ids


def main():
    cfg = load_pipeline_config()
    paths = Online1Paths.from_cfg(cfg)
    out = paths.output_root / "phase2"
    out.mkdir(parents=True, exist_ok=True)

    split = json.loads((paths.output_root / "phase0" / "split_manifest.json").read_text())
    env, feat_cols, lineages = load_clean_env(paths.train_x, cfg, split="train")
    env = attach_split_id(env, split)
    # deny scan
    deny = load_deny_config()
    denied_hits = scan_dataframe_for_denied(env[feat_cols], deny)

    env5 = aggregate_to_5min(env, feat_cols)
    env5 = attach_split_id(env5, split)
    # fit vocab on train split only (inductive)
    train_env5 = env5[env5["split_id"] == "train"]
    vocab = build_vocab(train_env5, feat_cols, n_bins=16)
    save_vocab(vocab, out / "vocab_inductive.json")

    # cube index with env zone mapping
    cube_full = json.loads((paths.output_root / "phase0" / "cube_inventory_full.json").read_text())
    zone_map = {int(k): int(v) for k, v in cfg["zone"]["cube_zone_to_env"].items()}
    crecs = []
    for r in cube_full["train"]["records"]:
        crecs.append(
            {
                **r,
                "env_zone_id": zone_map[int(r["cube_zone"])],
                "capture_end_minute": int(r["capture_time"][:2]) * 60 + int(r["capture_time"][3:5]),
            }
        )
    cube_index = pd.DataFrame(crecs)
    # attach split by DAT
    cube_index["split_id"] = cube_index["dat"].astype(str).map(split["membership_by_dat"])

    # materialize only using rows after split attachment (split-first)
    seq_df, seq_report = materialize_pretrain_sequences(
        env5,
        feat_cols,
        vocab,
        split,
        max_per_narrative=cfg.get("materialize", {}).get("max_per_narrative"),
        cube_index=cube_index[cube_index["split_id"].notna()],
    )
    seq_df.to_pickle(out / "pretrain_sequences.pkl")
    write_json(out / "pretrain_sequences_report.json", seq_report)

    reg_df, reg_report = build_regression_anchors(
        paths.train_y,
        env5,
        feat_cols,
        vocab,
        split,
        cfg,
        cube_index=cube_index,
        scope="inductive",
    )
    reg_df.to_pickle(out / "regression_examples.pkl")
    write_json(out / "regression_examples_report.json", reg_report)

    # overlap checks across train/valid sequences and regression examples
    def split_parts(df, col="split_id"):
        return df[df[col] == "train"], df[df[col] == "valid"]

    tr_s, va_s = split_parts(seq_df)
    tr_r, va_r = split_parts(reg_df)

    overlaps = {
        "pretrain_dat": verify_no_overlap(tr_s["dat"], va_s["dat"], "pretrain_dat"),
        "pretrain_source_window": verify_no_overlap(tr_s["source_window_id"], va_s["source_window_id"], "pretrain_sw"),
        "regression_dat": verify_no_overlap(tr_r["dat"], va_r["dat"], "reg_dat"),
        "regression_source_window": verify_no_overlap(tr_r["source_window_id"], va_r["source_window_id"], "reg_sw"),
        "regression_example_id": verify_no_overlap(
            tr_r["regression_example_id"], va_r["regression_example_id"], "reg_id"
        ),
        "cube_ids": verify_no_overlap(_explode_cube_ids(tr_s) + _explode_cube_ids(tr_r), _explode_cube_ids(va_s) + _explode_cube_ids(va_r), "cube_ids"),
    }

    # strict past check on regression cubes
    strict_past_violations = 0
    for _, r in reg_df.iterrows():
        for cid in r["cube_ids"]:
            crow = cube_index[cube_index["cube_id"] == cid]
            if crow.empty:
                continue
            if int(crow.iloc[0]["capture_end_minute"]) >= int(r["target_minute"]):
                strict_past_violations += 1

    # future dependency / target in sequences token check: soil tokens absent
    soil_token_hits = 0
    soil_prefixes = ("soil_", "SM|", "SEC|", "ST|")
    token_list = vocab["tokens"]
    for t in token_list:
        if t.startswith(soil_prefixes) or "soil_" in t:
            soil_token_hits += 1

    leak_pass = all(v["pass"] for v in overlaps.values()) and strict_past_violations == 0 and soil_token_hits == 0 and len(denied_hits) == 0
    phase2_pass = leak_pass and len(seq_df) > 0 and len(reg_df) > 0

    write_json(out / "overlap_report.json", overlaps)
    write_json(out / "feature_cols.json", {"feature_cols": feat_cols, "lineages": lineages})

    report = write_phase_report(
        out,
        "phase2",
        {
            "pass": phase2_pass,
            "n_sequences": int(len(seq_df)),
            "n_regression_examples": int(len(reg_df)),
            "overlaps": overlaps,
            "strict_past_violations": strict_past_violations,
            "soil_token_hits": soil_token_hits,
            "denied_feature_hits": denied_hits,
            "seq_report": seq_report,
            "reg_report": reg_report,
            "vocab_checksum": vocab["checksum"],
            "vocab_n_tokens": vocab["n_tokens"],
        },
    )
    write_markdown_summary(
        out / "PHASE2_SUMMARY.md",
        "Online1 Phase 2 Materialization",
        [
            ("Verdict", f"PASS={phase2_pass}"),
            ("Sequences", str(seq_report)),
            ("Regression", str(reg_report)),
            ("Overlaps", str({k: v['pass'] for k, v in overlaps.items()})),
        ],
    )
    print("PHASE2", phase2_pass, report)


if __name__ == "__main__":
    main()
