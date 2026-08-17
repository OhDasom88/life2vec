#!/usr/bin/env python3
"""Phase 3–4: C0/C1 training smoke + length preflight lock."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/dasom/life2vec")
sys.path.insert(0, str(ROOT))

from src.online1.paths import Online1Paths, load_pipeline_config, write_json
from src.online1.pretrain_smoke import run_pretrain_smoke, compute_preflight_lengths, encode_sequences, Online1LifeVecTransformer
from src.online1.report import write_phase_report, write_markdown_summary


def load_cube_emb_map(sample_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for p in sample_dir.glob("*.npz"):
        data = np.load(p, allow_pickle=True)
        out[str(data["cube_id"])] = data["pooled_embedding"]
    return out


def main():
    cfg = load_pipeline_config()
    paths = Online1Paths.from_cfg(cfg)
    out = paths.output_root / "phase3_4"
    out.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((paths.output_root / "phase2" / "vocab_inductive.json").read_text())
    import pandas as pd

    seq_df = pd.read_pickle(paths.output_root / "phase2" / "pretrain_sequences.pkl")
    sequences = seq_df.to_dict(orient="records")
    cube_map = load_cube_emb_map(paths.output_root / "phase1" / "embeddings_sample")

    c0 = run_pretrain_smoke(
        [s for s in sequences if s.get("arm", "C0") == "C0"] or sequences,
        vocab,
        cfg,
        max_length=512,
        steps=int(cfg["pretrain"]["max_steps_smoke"]),
        arm="C0",
    )
    # strip heavy state for json
    c0_state = c0.pop("model_state")
    torch.save(c0_state, out / "smoke_c0_state.pt")

    c1_seqs = [s for s in sequences if s.get("arm") == "C1"] or sequences[:32]
    c1 = run_pretrain_smoke(
        c1_seqs,
        vocab,
        cfg,
        max_length=512,
        steps=int(cfg["pretrain"]["max_steps_smoke"]),
        arm="C1",
        cube_emb_map=cube_map,
    )
    c1_state = c1.pop("model_state")
    torch.save(c1_state, out / "smoke_c1_state.pt")

    preflight = compute_preflight_lengths(
        sequences[:64],
        vocab,
        cfg,
        lengths=[1024, 2048, 4096],
    )
    locked = preflight["locked_default_max_length"] or 1024
    write_json(out / "max_length_lock.json", {"locked_default_max_length": locked, "preflight": preflight})

    smoke_pass = (
        c0["nan_inf_steps"] == 0
        and c1["nan_inf_steps"] == 0
        and c0["resume_ok"]
        and c1["resume_ok"]
        and locked is not None
    )

    report = write_phase_report(
        out,
        "phase3_4",
        {
            "pass": smoke_pass,
            "c0": c0,
            "c1": c1,
            "preflight": preflight,
            "locked_default_max_length": locked,
            "wandb": {"enabled": cfg["wandb"]["enabled"], "mode": cfg["wandb"]["mode"]},
        },
    )
    write_markdown_summary(
        out / "PHASE3_4_SUMMARY.md",
        "Online1 Phase 3–4 Smoke & Preflight",
        [
            ("Verdict", f"PASS={smoke_pass}"),
            ("C0", f"loss={c0['mean_loss']} nan={c0['nan_inf_steps']}"),
            ("C1", f"loss={c1['mean_loss']} nan={c1['nan_inf_steps']}"),
            ("Length lock", str(locked)),
            ("Preflight", str(preflight["results"])),
        ],
    )
    print("PHASE3_4", smoke_pass, report)


if __name__ == "__main__":
    main()
