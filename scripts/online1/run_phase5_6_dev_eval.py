#!/usr/bin/env python3
"""Phase 5–6: development pretrain trials (arm champions) + regression comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/dasom/life2vec")
sys.path.insert(0, str(ROOT))

from src.online1.paths import Online1Paths, load_pipeline_config, write_json
from src.online1.pretrain_smoke import (
    run_pretrain_smoke,
    encode_sequences,
    Online1LifeVecTransformer,
    compute_alignment_recall_at_k,
)
from src.online1.regression_eval import run_regression_suite, life2vec_flat_decoder, attention_pooled_regression, _xy
from src.online1.report import write_phase_report, write_markdown_summary


def zscore(vals: list[float]) -> list[float]:
    arr = np.asarray(vals, dtype=float)
    mu, sd = float(np.mean(arr)), float(np.std(arr) + 1e-8)
    return list((arr - mu) / sd)


def champion_score(z: dict[str, float], weights: dict, arm: str) -> float:
    if arm == "C0":
        return weights["w_mlm"] * z["mlm"] + weights["w_sop"] * z["sop"]
    return (
        weights["w_mlm"] * z["mlm"]
        + weights["w_sop"] * z["sop"]
        + weights["w_cube"] * z["cube"]
        + weights["w_align"] * z["align"]
    )


def load_cube_map(path: Path):
    out = {}
    for p in path.glob("*.npz"):
        d = np.load(p, allow_pickle=True)
        out[str(d["cube_id"])] = d["pooled_embedding"]
    return out


def main():
    cfg = load_pipeline_config()
    paths = Online1Paths.from_cfg(cfg)
    out = paths.output_root / "phase5_6"
    out.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((paths.output_root / "phase2" / "vocab_inductive.json").read_text())
    lock = json.loads((paths.output_root / "phase3_4" / "max_length_lock.json").read_text())
    max_len = int(lock["locked_default_max_length"])
    seq_df = pd.read_pickle(paths.output_root / "phase2" / "pretrain_sequences.pkl")
    reg_df = pd.read_pickle(paths.output_root / "phase2" / "regression_examples.pkl")
    cube_map = load_cube_map(paths.output_root / "phase1" / "embeddings_sample")

    train_seqs = seq_df[seq_df["split_id"] == "train"].to_dict(orient="records")
    weights = cfg["champion_score"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Development sweep: small grid over lr / mask_ratio per arm (not full W&B online)
    grid = [
        {"lr": 1e-3, "mask_ratio": 0.15},
        {"lr": 5e-4, "mask_ratio": 0.15},
        {"lr": 1e-3, "mask_ratio": 0.20},
    ]
    # Train at cfg.pretrain.train_max_length (sized to real content, see pipeline.yaml) —
    # max_len (the Phase3-4 preflight lock) validates safety headroom separately and would
    # waste compute padding every batch to it when real sequences top out far shorter.
    dev_max_length = min(max_len, int(cfg["pretrain"].get("train_max_length", 1024)))
    arm_results = {}
    for arm in ["C0", "C1"]:
        trials = []
        seqs = [s for s in train_seqs if (arm == "C0" and s.get("arm", "C0") == "C0") or (arm == "C1" and s.get("arm") == "C1")]
        if not seqs:
            seqs = train_seqs
        for i, g in enumerate(grid):
            cfg_t = json.loads(json.dumps(cfg))
            cfg_t["pretrain"]["lr"] = g["lr"]
            cfg_t["pretrain"]["mask_ratio"] = g["mask_ratio"]
            out_t = run_pretrain_smoke(
                seqs,
                vocab,
                cfg_t,
                max_length=dev_max_length,
                steps=int(cfg["pretrain"]["max_steps_dev"]),
                arm=arm,
                cube_emb_map=cube_map if arm != "C0" else None,
            )
            state = out_t.pop("model_state")
            out_t["trial_id"] = f"{arm}_{i}"
            out_t["params"] = g
            # sop_accuracy / cube_loss_mean come back from run_pretrain_smoke already measured
            # (SOP head accuracy on the trained batches; mean cosine cube-alignment loss).
            align = None
            if arm != "C0":
                align_model = Online1LifeVecTransformer(
                    vocab_size=vocab["n_tokens"],
                    hidden=int(cfg["pretrain"]["hidden_size"]),
                    layers=int(cfg["pretrain"]["num_layers"]),
                    heads=int(cfg["pretrain"]["num_heads"]),
                ).to(device)
                align_model.load_state_dict(state)
                align = compute_alignment_recall_at_k(align_model, seqs, cube_map, dev_max_length, device, k=3)
                del align_model
                if device == "cuda":
                    torch.cuda.empty_cache()
            out_t["align_recall_at_k"] = align
            torch.save(state, out / f"champion_cand_{arm}_{i}.pt")
            trials.append(out_t)
            if device == "cuda":
                torch.cuda.empty_cache()

        # z-score each metric across this arm's trials before combining, per the documented
        # S = w1*Z(-L_mlm) + w2*Z(A_sop) + w3*Z(-L_cube) + w4*Z(R@K) formula (previously the
        # zscore() helper was defined but never called, and cube/align terms were hardcoded).
        mlm_z = zscore([-(t["mean_loss"] or 0.0) for t in trials])
        sop_z = zscore([t["sop_accuracy"] if t["sop_accuracy"] is not None else 0.0 for t in trials])
        if arm == "C0":
            for t, zm, zs in zip(trials, mlm_z, sop_z):
                t["score"] = champion_score({"mlm": zm, "sop": zs}, weights, arm)
        else:
            cube_z = zscore([-(t["cube_loss_mean"] if t["cube_loss_mean"] is not None else 0.0) for t in trials])
            align_z = zscore([t["align_recall_at_k"] if t["align_recall_at_k"] is not None else 0.0 for t in trials])
            for t, zm, zs, zc, za in zip(trials, mlm_z, sop_z, cube_z, align_z):
                t["score"] = champion_score({"mlm": zm, "sop": zs, "cube": zc, "align": za}, weights, arm)

        # arm-internal selection
        trials_sorted = sorted(trials, key=lambda t: (-t["score"], t.get("seconds", 0), t["trial_id"]))
        champ = trials_sorted[0]
        arm_results[arm] = {"trials": trials, "champion": champ}
        # reload champion state path marker
        write_json(out / f"champion_{arm}.json", champ)

    # Encode regression examples with C0 champion (frozen)
    champ_c0 = arm_results["C0"]["champion"]
    # retrain once to get state file index  from best trial_id
    best_idx = int(champ_c0["trial_id"].split("_")[-1])
    state = torch.load(out / f"champion_cand_C0_{best_idx}.pt", map_location="cpu")
    model = Online1LifeVecTransformer(
        vocab_size=vocab["n_tokens"],
        hidden=int(cfg["pretrain"]["hidden_size"]),
        layers=int(cfg["pretrain"]["num_layers"]),
        heads=int(cfg["pretrain"]["num_heads"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    tr = reg_df[reg_df["split_id"] == "train"].reset_index(drop=True)
    va = reg_df[reg_df["split_id"] == "valid"].reset_index(drop=True)
    # subsample only if the corpus is far larger than the dev-scale defaults were sized for
    if len(tr) > 20000:
        tr = tr.sample(20000, random_state=0)
    if len(va) > 5000:
        va = va.sample(5000, random_state=0)

    Xtr = encode_sequences(model, tr["token_ids"].tolist(), dev_max_length, device, pooling="cls")
    Xva = encode_sequences(model, va["token_ids"].tolist(), dev_max_length, device, pooling="cls")
    np.savez_compressed(out / "frozen_reps.npz", X_train=Xtr, X_valid=Xva, train_ids=tr["regression_example_id"].to_numpy(), valid_ids=va["regression_example_id"].to_numpy())

    reg_results = run_regression_suite(tr, va, Xtr, Xva)
    write_json(out / "regression_comparison.json", reg_results)

    # Summary-vector architecture comparison: same frozen C0-champion encoder, three ways to
    # turn its per-token hidden states into a fixed regression input — (1) CLS pooling (what
    # reg_results above used, life2vec's default), (2) masked mean-pooling, both scored with
    # the same life2vec_flat_decoder head so only the pooling strategy varies; (3) life2vec's
    # AttentionDecoderL, whose learned pooling + head train jointly on the raw hidden states.
    Xtr_mean = encode_sequences(model, tr["token_ids"].tolist(), dev_max_length, device, pooling="mean")
    Xva_mean = encode_sequences(model, va["token_ids"].tolist(), dev_max_length, device, pooling="mean")
    cls_Xtr, cls_ytr, cls_mtr = _xy(tr, Xtr)
    cls_Xva, cls_yva, cls_mva = _xy(va, Xva)
    mean_Xtr, mean_ytr, mean_mtr = _xy(tr, Xtr_mean)
    mean_Xva, mean_yva, mean_mva = _xy(va, Xva_mean)
    pooling_comparison = {
        "cls_pooling": life2vec_flat_decoder(cls_Xtr, cls_ytr, cls_mtr, cls_Xva, cls_yva, cls_mva),
        "mean_pooling": life2vec_flat_decoder(mean_Xtr, mean_ytr, mean_mtr, mean_Xva, mean_yva, mean_mva),
        "attention_pooling": attention_pooled_regression(model, tr, va, dev_max_length, device),
    }
    write_json(out / "pooling_comparison.json", pooling_comparison)

    # Final holdout evaluation: the holdout split must be touched exactly once, after
    # architecture/pooling/champion choices are already locked from train+valid. Refit
    # life2vec_flat_decoder on train+valid (same frozen encoder) and score only on holdout.
    ho = reg_df[reg_df["split_id"] == "holdout"].reset_index(drop=True)
    if len(ho) > 5000:
        ho = ho.sample(5000, random_state=0)
    if len(ho):
        Xho = encode_sequences(model, ho["token_ids"].tolist(), dev_max_length, device, pooling="cls")
        trva = pd.concat([tr, va], ignore_index=True)
        Xtrva = np.concatenate([Xtr, Xva], axis=0)
        trva_X, trva_y, trva_m = _xy(trva, Xtrva)
        ho_X, ho_y, ho_m = _xy(ho, Xho)
        holdout_result = life2vec_flat_decoder(trva_X, trva_y, trva_m, ho_X, ho_y, ho_m, epochs=200)
    else:
        holdout_result = None
    holdout_note = {
        "n_holdout": int(len(ho)),
        "note": "life2vec_flat_decoder refit on train+valid, scored once on the untouched holdout split.",
        "result": holdout_result,
    }
    write_json(out / "holdout_evaluation.json", holdout_note)

    phase_pass = (
        all(arm_results[a]["champion"]["nan_inf_steps"] == 0 for a in arm_results)
        and reg_results["n_train"] > 0
        and reg_results["n_valid"] > 0
    )

    report = write_phase_report(
        out,
        "phase5_6",
        {
            "pass": phase_pass,
            "score_formula": "S=w1*Z(-L_mlm)+w2*Z(A_sop)+w3*Z(-L_cube)+w4*Z(R@K) with arm-internal selection",
            "arm_results_summary": {
                a: {"champion_trial": arm_results[a]["champion"]["trial_id"], "score": arm_results[a]["champion"]["score"], "mean_loss": arm_results[a]["champion"]["mean_loss"]}
                for a in arm_results
            },
            "regression": reg_results,
            "pooling_comparison": pooling_comparison,
            "holdout": holdout_note,
            "excluded_baselines": reg_results["excluded_baselines"],
            "wandb_sweeps_defined": [
                "online1_pretrain_sweep",
                "online1_regression_life2vec_sweep",
                "online1_regression_table_sweep",
            ],
        },
    )
    # also dump full arm trials
    write_json(out / "arm_trials.json", {a: arm_results[a]["trials"] for a in arm_results})
    write_markdown_summary(
        out / "PHASE5_6_SUMMARY.md",
        "Online1 Phase 5–6 Dev Sweep & Regression",
        [
            ("Verdict", f"PASS={phase_pass}"),
            ("Champions", str({a: arm_results[a]['champion']['trial_id'] for a in arm_results})),
            ("Regression models", str([r.get('model') for r in reg_results['results']])),
            (
                "Pooling comparison (macro r2)",
                str({k: v["metrics"]["macro"]["r2"] for k, v in pooling_comparison.items()}),
            ),
        ],
    )
    print("PHASE5_6", phase_pass, report)


if __name__ == "__main__":
    main()
