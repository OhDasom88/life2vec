#!/usr/bin/env python3
"""Evaluation pipeline §§10–18 for 5-fold ensemble (v0.1 or v0.2).

Default auto-detects checkpoint version (state/cause heads → v02).
v0.1 baseline example: wandb 7czhwn2y / cv_20260713_144719

Runnable:
  - §10 probability ensemble + agreement / p_mean / p_std / h_case fold cosine
  - §11.1 / §13 / §14 event IxG + optional occlusion
  - §15 OOF-only saliency for example cases
  - §16 Structured Evidence JSON (diagnosis_support; state/cause filled when available)
  - §17–18 Gemma prompt stubs (+ generate_gemma_reports.py for live Gemma)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.diagnosis_dataset import (  # noqa: E402
    DiagnosisEventDataset,
    collate_diagnosis_batch,
    load_label_map,
)
from src.online2.v2.finetune_v02.checkpoint import (  # noqa: E402
    load_five_fold_models,
)
from src.online2.v2.finetune_v02.consensus import (  # noqa: E402
    consensus_table,
    select_report_events,
)
from src.online2.v2.finetune_v02.config import SaliencyConfig  # noqa: E402
from src.online2.v2.finetune_v02.evidence import build_structured_evidence  # noqa: E402
from src.online2.v2.finetune_v02.gemma_report import (  # noqa: E402
    GemmaReportInput,
    build_gemma_prompt,
    generate_report_stub,
)
from src.online2.v2.finetune_v02.metrics import (  # noqa: E402
    accuracy,
    average_probability_ensemble,
    macro_f1_all_classes,
)
from src.online2.v2.finetune_v02.saliency import (  # noqa: E402
    event_input_x_gradient,
    hierarchical_plan,
    select_top_event_indices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/runs/cv_20260713_144719",
    )
    p.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/event_embeddings",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/labels_example_score90.csv",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/label_map.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: outputs/online2/v2_finetune/evaluation/<stamp>",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-events", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--saliency-cases",
        type=str,
        default="sample",
        help="'all' | 'none' | 'sample' | comma-separated case_ids",
    )
    p.add_argument("--saliency-sample-n", type=int, default=3)
    p.add_argument("--run-occlusion", action="store_true")
    p.add_argument("--top-frac", type=float, default=0.08)
    p.add_argument("--wandb-run", type=str, default="7czhwn2y")
    p.add_argument(
        "--model-version",
        choices=["auto", "v01", "v02"],
        default="auto",
        help="auto-detect from ckpt (state/cause heads → v02); default auto",
    )
    p.add_argument(
        "--saliency-target",
        choices=["classification", "state", "cause"],
        default="classification",
        help="v02 can also target state/cause heads; v01 supports classification only",
    )
    return p.parse_args()


def load_fold_models(
    run_dir: Path,
    device: torch.device,
    *,
    model_version: str = "auto",
) -> Tuple[List[Any], List[Dict[str, Any]], str]:
    force = None if model_version == "auto" else model_version  # type: ignore[assignment]
    models, metas, ver = load_five_fold_models(run_dir, device, force_version=force)
    return models, metas, ver


def oof_fold_for_case(case_id: str, metas: Sequence[Dict[str, Any]]) -> Optional[int]:
    for m in metas:
        if case_id in set(m["val_cases"]):
            return int(m["fold"])
    return None


@torch.no_grad()
def forward_batch(
    model: Any, batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return model(
        event_mean=batch["event_mean"].to(device),
        event_max=batch["event_max"].to(device),
        case_age_hours=batch["case_age_hours"].to(device),
        view_id=batch["view_id"].to(device),
        zone_id=batch["zone_id"].to(device),
        local_hour=batch["local_hour"].to(device),
        padding_mask=batch["padding_mask"].to(device),
    )


def run_ensemble(
    models: Sequence[Any],
    ds: DiagnosisEventDataset,
    *,
    device: torch.device,
    max_events: int,
    batch_size: int,
    id_to_name: Dict[int, str],
) -> Dict[str, Any]:
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=False,
        collate_fn=lambda xs: collate_diagnosis_batch(xs, max_events=max_events),
        num_workers=0,
    )
    n = len(ds)
    c = models[0].cfg.num_classes
    fold_logits = [np.zeros((n, c), dtype=np.float64) for _ in models]
    fold_h = [np.zeros((n, models[0].cfg.hidden_dim), dtype=np.float64) for _ in models]
    y_true: List[int] = []
    case_ids: List[str] = []
    offset = 0
    for batch in loader:
        bsz = batch["label"].size(0)
        y_true.extend(batch["label"].tolist())
        case_ids.extend(batch["case_ids"])
        for f, model in enumerate(models):
            out = forward_batch(model, batch, device)
            fold_logits[f][offset : offset + bsz] = out["logits"].float().cpu().numpy()
            fold_h[f][offset : offset + bsz] = out["h_case"].float().cpu().numpy()
        offset += bsz

    fold_probs = [
        np.exp(x - x.max(axis=-1, keepdims=True)) for x in fold_logits
    ]
    fold_probs = [p / p.sum(axis=-1, keepdims=True) for p in fold_probs]
    ens = average_probability_ensemble(fold_probs)
    pred = ens.argmax(axis=1)
    top1 = np.stack([p.argmax(axis=1) for p in fold_probs], axis=0)  # [F,N]
    agreement = (top1 == pred[None, :]).sum(axis=0)

    p_stack = np.stack(fold_probs, axis=0)  # [F,N,C]
    p_mean = p_stack.mean(axis=0)
    p_std = p_stack.std(axis=0)

    # pairwise cosine of h_case across folds
    h_stack = np.stack(fold_h, axis=0)  # [F,N,D]
    h_n = h_stack / (np.linalg.norm(h_stack, axis=-1, keepdims=True) + 1e-8)
    # mean over fold pairs
    cos_pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            cos_pairs.append((h_n[i] * h_n[j]).sum(axis=-1))
    h_case_fold_cosine = np.mean(np.stack(cos_pairs, axis=0), axis=0)

    rows = []
    for i, cid in enumerate(case_ids):
        rows.append(
            {
                "case_id": cid,
                "label_id": int(y_true[i]),
                "label": id_to_name[int(y_true[i])],
                "pred_id": int(pred[i]),
                "pred": id_to_name[int(pred[i])],
                "correct": bool(pred[i] == y_true[i]),
                "ensemble_probability": float(ens[i, pred[i]]),
                "model_agreement": int(agreement[i]),
                "h_case_fold_cosine": float(h_case_fold_cosine[i]),
                "image_injection_delta": None,  # IMAGE not in Stage A cache
                **{f"p_mean_{k}": float(p_mean[i, k]) for k in range(c)},
                **{f"p_std_{k}": float(p_std[i, k]) for k in range(c)},
                **{f"fold{f}_top1": id_to_name[int(top1[f, i])] for f in range(5)},
            }
        )

    acc = accuracy(y_true, pred.tolist())
    mf1 = macro_f1_all_classes(y_true, pred.tolist(), num_classes=c)
    summary = {
        "n_cases": n,
        "n_folds": 5,
        "acc": acc,
        "macro_f1": mf1,
        "perfect": bool(acc >= 1.0 - 1e-12 and mf1 >= 1.0 - 1e-12),
        "mean_model_agreement": float(np.mean(agreement)),
        "n_full_agreement_5": int((agreement == 5).sum()),
        "mean_h_case_fold_cosine": float(np.mean(h_case_fold_cosine)),
        "image_injection_delta_status": "not_applicable_image_excluded_from_stage_a",
    }
    return {
        "summary": summary,
        "rows": rows,
        "y_true": y_true,
        "pred": pred.tolist(),
        "ens": ens,
        "fold_probs": fold_probs,
        "case_ids": case_ids,
    }


def event_ixg_classification(
    model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    class_id: int,
    target: str = "classification",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Input×Gradient on z_i (classification, or state/cause for v02)."""
    score = event_input_x_gradient(model, batch, target=target, class_id=class_id)
    with torch.no_grad():
        z = model.meta(
            batch["event_mean"].detach(),
            batch["event_max"].detach(),
            batch["case_age_hours"],
            batch["view_id"],
            batch["zone_id"],
            batch["local_hour"],
            batch["padding_mask"],
        )
        _h, attn = model.pad(z, batch["padding_mask"])
    aux_attn = attn.detach().cpu().numpy()[0, 0] if attn is not None else None
    return score.detach().cpu().numpy(), aux_attn


@torch.no_grad()
def event_occlusion_deltas(
    model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    class_id: int,
    indices: Sequence[int],
) -> np.ndarray:
    """Δlogit_y when masking each candidate event (set padding False)."""
    out0 = model(
        event_mean=batch["event_mean"],
        event_max=batch["event_max"],
        case_age_hours=batch["case_age_hours"],
        view_id=batch["view_id"],
        zone_id=batch["zone_id"],
        local_hour=batch["local_hour"],
        padding_mask=batch["padding_mask"],
    )
    base = float(out0["logits"][0, class_id].item())
    deltas = np.zeros(batch["padding_mask"].size(1), dtype=np.float64)
    for idx in indices:
        mask = batch["padding_mask"].clone()
        mask[0, idx] = False
        if not bool(mask.any()):
            continue
        out = model(
            event_mean=batch["event_mean"],
            event_max=batch["event_max"],
            case_age_hours=batch["case_age_hours"],
            view_id=batch["view_id"],
            zone_id=batch["zone_id"],
            local_hour=batch["local_hour"],
            padding_mask=mask,
        )
        deltas[idx] = base - float(out["logits"][0, class_id].item())
    return deltas


def load_case_meta(emb_dir: Path, case_id: str, n: int) -> pd.DataFrame:
    pdf = pd.read_parquet(emb_dir / f"{case_id}.parquet")
    pdf = pdf.sort_values("event_order").reset_index(drop=True).iloc[:n].copy()
    return pdf


def resolve_saliency_cases(arg: str, all_ids: Sequence[str], sample_n: int) -> List[str]:
    if arg == "none":
        return []
    if arg == "all":
        return list(all_ids)
    if arg == "sample":
        rng = np.random.RandomState(2023)
        n = min(sample_n, len(all_ids))
        return [str(x) for x in rng.choice(list(all_ids), size=n, replace=False)]
    return [c.strip() for c in arg.split(",") if c.strip()]


def run_case_saliency(
    case_id: str,
    *,
    models: Sequence[Any],
    metas: Sequence[Dict[str, Any]],
    ds_index: Dict[str, int],
    ds: DiagnosisEventDataset,
    emb_dir: Path,
    device: torch.device,
    max_events: int,
    pred_id: int,
    ens_prob: float,
    model_agreement: int,
    pred_name: str,
    sal_cfg: SaliencyConfig,
    out_root: Path,
    run_occlusion: bool,
    saliency_target: str = "classification",
) -> Dict[str, Any]:
    sample = ds[ds_index[case_id]]
    batch = collate_diagnosis_batch([sample], max_events=max_events)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    t = int(batch["padding_mask"][0].sum().item())
    meta_df = load_case_meta(emb_dir, case_id, t)
    event_ids = meta_df["event_id"].astype(str).tolist()

    fold_scores: List[np.ndarray] = []
    fold_attn: List[Optional[np.ndarray]] = []
    for f, model in enumerate(models):
        scores, attn = event_ixg_classification(
            model, batch, class_id=pred_id, target=saliency_target
        )
        scores = scores[:t]
        fold_scores.append(scores)
        fold_attn.append(attn[:t] if attn is not None else None)
        per = out_root / "saliency" / "per_model" / f"fold{f}"
        per.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "event_id": event_ids,
                "ixg_score": scores,
                "attn_weight": fold_attn[-1] if fold_attn[-1] is not None else np.nan,
            }
        ).to_parquet(per / f"{case_id}_event_scores.parquet", index=False)

    table = consensus_table(event_ids, fold_scores, cfg=sal_cfg)
    # enrich with event meta
    for col in ("timestamp", "view", "zone", "case_age_hours", "local_hour"):
        if col in meta_df.columns:
            table[col] = meta_df[col].astype(str).tolist() if col in ("timestamp", "view") else meta_df[col].tolist()

    splits = select_report_events(table, cfg=sal_cfg)
    plan = hierarchical_plan(sal_cfg, t)

    # candidate union for occlusion
    cand: set = set()
    for scores in fold_scores:
        cand.update(select_top_event_indices(scores, top_frac=sal_cfg.top_frac_per_model))
    cand_list = sorted(cand)[: sal_cfg.occlusion_candidate_cap]

    occlusion_note = "skipped"
    if run_occlusion and cand_list:
        occ_mat = []
        for model in models:
            deltas = event_occlusion_deltas(
                model, batch, class_id=pred_id, indices=cand_list
            )[:t]
            occ_mat.append(deltas)
        # store mean occlusion delta on candidates
        occ_mean = np.mean(np.stack(occ_mat, axis=0), axis=0)
        table["occlusion_delta_mean"] = occ_mean
        occlusion_note = f"candidates={len(cand_list)}"

    case_out = out_root / "saliency" / "consensus" / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(case_out / "event_consensus.parquet", index=False)
    for name, df in splits.items():
        df.to_parquet(case_out / f"event_{name}.parquet", index=False)

    # §15 OOF-only table for example cases
    oof = oof_fold_for_case(case_id, metas)
    oof_info: Dict[str, Any] = {"oof_fold": oof, "mode": None}
    if oof is not None:
        oof_scores = fold_scores[oof]
        oof_df = pd.DataFrame(
            {
                "event_id": event_ids,
                "oof_ixg_score": oof_scores,
                "oof_fold": oof,
            }
        )
        oof_df.to_parquet(case_out / "event_oof_only.parquet", index=False)
        oof_info["mode"] = "example_oof_single_fold"
    else:
        oof_info["mode"] = "problem_use_all_folds"

    # §16 diagnosis_support from primary/strong classification consensus
    primary = splits["primary"].sort_values("median_saliency", ascending=False).head(
        sal_cfg.consensus_events_max
    )
    evidence = build_structured_evidence(
        case_id=case_id,
        diagnosis=pred_name,
        ensemble_probability=ens_prob,
        model_agreement=model_agreement,
        diagnosis_support=primary,
        state_events=pd.DataFrame(),
        cause_events=pd.DataFrame(),
        image_events=None,
        disagreement=splits["disputed"]
        .sort_values("saliency_iqr", ascending=False)
        .head(20)
        .to_dict(orient="records"),
    )
    evidence["saliency_meta"] = {
        "target": saliency_target,
        "method": "input_x_gradient",
        "plan": plan,
        "candidate_union": len(cand_list),
        "occlusion": occlusion_note,
        "oof": oof_info,
        "attn_weight_role": "auxiliary_only",
        "token_saliency": "deferred",
        "image_saliency": "deferred",
    }

    ev_path = out_root / "evidence" / f"{case_id}.json"
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    ev_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "case_id": case_id,
        "n_events": t,
        "plan": plan,
        "n_primary": int(len(splits["primary"])),
        "n_strong": int(len(splits["strong"])),
        "n_disputed": int(len(splits["disputed"])),
        "oof": oof_info,
        "occlusion": occlusion_note,
    }
    (case_out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"meta": meta, "evidence": evidence}


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        ROOT / "outputs/online2/v2_finetune/evaluation" / f"{args.wandb_run}_{stamp}"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(args.label_map)
    id_to_name = {int(k): v for k, v in label_map["id_to_name"].items()}
    labels_df = pd.read_csv(args.labels)
    case_ids_all = [str(c) for c in labels_df["case_id"].tolist()]

    device = torch.device(args.device)
    models, metas, model_ver = load_fold_models(
        args.run_dir, device, model_version=args.model_version
    )
    ds = DiagnosisEventDataset(
        case_ids_all,
        labels_df,
        label_map,
        args.emb_dir,
        max_events=args.max_events,
        require_complete=False,
    )
    print(
        {
            "n_cached": len(ds),
            "device": str(device),
            "out_dir": str(out_dir),
            "model_version": model_ver,
        },
        flush=True,
    )

    t0 = time.time()
    ens = run_ensemble(
        models,
        ds,
        device=device,
        max_events=args.max_events,
        batch_size=args.batch_size,
        id_to_name=id_to_name,
    )
    ens_dir = out_dir / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    pdf = pd.DataFrame(ens["rows"])
    pdf.to_parquet(ens_dir / "predictions.parquet", index=False)
    pdf.to_csv(ens_dir / "predictions.csv", index=False)
    (ens_dir / "summary.json").write_text(
        json.dumps(ens["summary"], indent=2) + "\n", encoding="utf-8"
    )
    print({"ENSEMBLE": ens["summary"], "sec": round(time.time() - t0, 1)}, flush=True)

    # fold membership map
    (out_dir / "fold_membership.json").write_text(
        json.dumps(
            {
                "wandb_run": args.wandb_run,
                "run_dir": str(args.run_dir),
                "folds": [
                    {"fold": m["fold"], "val_cases": m["val_cases"], "epoch": m["epoch"]}
                    for m in metas
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    sal_cfg = SaliencyConfig(top_frac_per_model=float(args.top_frac))
    sal_cases = resolve_saliency_cases(args.saliency_cases, ds.case_ids, args.saliency_sample_n)
    ds_index = {cid: i for i, cid in enumerate(ds.case_ids)}
    pred_by_case = {r["case_id"]: r for r in ens["rows"]}

    sal_summaries = []
    gemma_recs = []
    for cid in sal_cases:
        print(f"[saliency] {cid}", flush=True)
        row = pred_by_case[cid]
        result = run_case_saliency(
            cid,
            models=models,
            metas=metas,
            ds_index=ds_index,
            ds=ds,
            emb_dir=args.emb_dir,
            device=device,
            max_events=args.max_events,
            pred_id=int(row["pred_id"]),
            ens_prob=float(row["ensemble_probability"]),
            model_agreement=int(row["model_agreement"]),
            pred_name=str(row["pred"]),
            sal_cfg=sal_cfg,
            out_root=out_dir,
            run_occlusion=bool(args.run_occlusion),
            saliency_target=str(args.saliency_target),
        )
        sal_summaries.append(result["meta"])
        probs = {
            id_to_name[k]: float(ens["ens"][ds_index[cid], k])
            for k in range(ens["ens"].shape[1])
        }
        gin = GemmaReportInput(
            case_id=cid,
            diagnosis=str(row["pred"]),
            probabilities=probs,
            model_agreement=int(row["model_agreement"]),
            structured_evidence=result["evidence"],
        )
        gemma_recs.append(
            {
                "case_id": cid,
                "prompt": build_gemma_prompt(gin),
                "report_stub": generate_report_stub(gin),
                "structured_evidence": result["evidence"],
            }
        )

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "gemma_inputs.jsonl").open("w", encoding="utf-8") as f:
        for rec in gemma_recs:
            f.write(json.dumps({"case_id": rec["case_id"], "prompt": rec["prompt"]}, ensure_ascii=False) + "\n")
    with (reports_dir / "reports_stub.jsonl").open("w", encoding="utf-8") as f:
        for rec in gemma_recs:
            f.write(
                json.dumps(
                    {"case_id": rec["case_id"], "report": rec["report_stub"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    master = {
        "wandb_run": args.wandb_run,
        "run_dir": str(args.run_dir),
        "out_dir": str(out_dir),
        "model_version": model_ver,
        "ensemble": ens["summary"],
        "saliency_cases": sal_cases,
        "saliency_summaries": sal_summaries,
        "deferred": [
            "token_saliency_H_target",
            "image_saliency",
            "state_cause_saliency_targets" if model_ver == "v01" else "full_state_cause_occlusion",
            "image_injection_before_after",
            "factor_catalog_mapping",
        ],
        "doc": "outputs/online2/v2_finetune/EVALUATION_PROCESS.md",
    }
    (out_dir / "EVAL_SUMMARY.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print({"DONE": master["ensemble"], "saliency_n": len(sal_cases), "out": str(out_dir)}, flush=True)


if __name__ == "__main__":
    main()
