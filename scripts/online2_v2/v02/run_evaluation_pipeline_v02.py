#!/usr/bin/env python3
"""v0.2 evaluation: ensemble + consensus saliency + evidence (IMAGE/DINO aware).

Writes ONLY under outputs/online2/v2_finetune_v02/. Does not modify v0.1
evaluation artifacts or shared report defaults.

Uses DiagnosisEventDatasetV02 + dino_vec/dino_mask so SharedImageAdapter is active.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v02.checkpoint import load_five_fold_models  # noqa: E402
from src.online2.v2.finetune_v02.consensus import (  # noqa: E402
    consensus_table,
    select_report_events,
)
from src.online2.v2.finetune_v02.config import SaliencyConfig  # noqa: E402
from src.online2.v2.finetune_v02.dataset import (  # noqa: E402
    DiagnosisEventDatasetV02,
    collate_diagnosis_batch_v02,
    load_label_map,
)
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
from src.online2.v2.finetune_v02.version import DEFAULT_OUTPUT_ROOT, FINETUNE_VERSION  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="v0.2 cv run with fold{0..4}_best.pt",
    )
    p.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / "event_embeddings",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / "labels_example35_problem20.csv",
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
        help=f"Default: {DEFAULT_OUTPUT_ROOT}/evaluation/<stamp>",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-events", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--saliency-cases",
        type=str,
        default="all",
        help="'all' | 'none' | 'sample' | comma-separated case_ids",
    )
    p.add_argument("--saliency-sample-n", type=int, default=3)
    p.add_argument("--run-occlusion", action="store_true")
    p.add_argument("--top-frac", type=float, default=0.08)
    p.add_argument("--wandb-run", type=str, default="v02")
    p.add_argument(
        "--saliency-target",
        choices=["classification", "state", "cause"],
        default="classification",
    )
    p.add_argument(
        "--skip-metrics-on-unlabeled",
        action="store_true",
        default=True,
        help="Exclude problem_set / [UNLABELED_PROBLEM] from acc/F1 (default on)",
    )
    return p.parse_args()


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def forward_batch_v02(model: Any, batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    batch = _move_batch(batch, device)
    kwargs = dict(
        event_mean=batch["event_mean"],
        event_max=batch["event_max"],
        case_age_hours=batch["case_age_hours"],
        view_id=batch["view_id"],
        zone_id=batch["zone_id"],
        local_hour=batch["local_hour"],
        padding_mask=batch["padding_mask"],
    )
    if "dino_vec" in batch and "dino_mask" in batch:
        kwargs["dino_vec"] = batch["dino_vec"]
        kwargs["dino_mask"] = batch["dino_mask"]
    return model(**kwargs)


def fuse_batch_events(model: Any, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply SharedImageAdapter residual in-place on a copy of mean/max."""
    event_mean = batch["event_mean"]
    event_max = batch["event_max"]
    if hasattr(model, "fuse_image_into_events") and "dino_vec" in batch:
        event_mean, event_max = model.fuse_image_into_events(
            event_mean,
            event_max,
            dino_vec=batch.get("dino_vec"),
            dino_mask=batch.get("dino_mask"),
        )
    fused = dict(batch)
    fused["event_mean"] = event_mean
    fused["event_max"] = event_max
    return fused


def oof_fold_for_case(case_id: str, metas: Sequence[Dict[str, Any]]) -> Optional[int]:
    for m in metas:
        if case_id in set(m["val_cases"]):
            return int(m["fold"])
    return None


@torch.no_grad()
def run_ensemble(
    models: Sequence[Any],
    ds: DiagnosisEventDatasetV02,
    *,
    device: torch.device,
    max_events: int,
    batch_size: int,
    id_to_name: Dict[int, str],
    labels_df: pd.DataFrame,
) -> Dict[str, Any]:
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, max(1, len(ds))),
        shuffle=False,
        collate_fn=lambda xs: collate_diagnosis_batch_v02(xs, max_events=max_events),
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
            out = forward_batch_v02(model, batch, device)
            fold_logits[f][offset : offset + bsz] = out["logits"].float().cpu().numpy()
            fold_h[f][offset : offset + bsz] = out["h_case"].float().cpu().numpy()
        offset += bsz

    fold_probs = [np.exp(x - x.max(axis=-1, keepdims=True)) for x in fold_logits]
    fold_probs = [p / p.sum(axis=-1, keepdims=True) for p in fold_probs]
    ens = average_probability_ensemble(fold_probs)
    pred = ens.argmax(axis=1)
    top1 = np.stack([p.argmax(axis=1) for p in fold_probs], axis=0)
    agreement = (top1 == pred[None, :]).sum(axis=0)
    p_stack = np.stack(fold_probs, axis=0)
    p_mean = p_stack.mean(axis=0)
    p_std = p_stack.std(axis=0)

    h_stack = np.stack(fold_h, axis=0)
    h_n = h_stack / (np.linalg.norm(h_stack, axis=-1, keepdims=True) + 1e-8)
    cos_pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            cos_pairs.append((h_n[i] * h_n[j]).sum(axis=-1))
    h_case_fold_cosine = np.mean(np.stack(cos_pairs, axis=0), axis=0)

    lab_idx = labels_df.set_index(labels_df["case_id"].astype(str))
    rows = []
    for i, cid in enumerate(case_ids):
        lab_row = lab_idx.loc[cid]
        if isinstance(lab_row, pd.DataFrame):
            lab_row = lab_row.iloc[0]
        set_name = str(lab_row["set"]) if "set" in lab_row.index else "unknown"
        raw = str(lab_row["diagnosis_raw"]) if "diagnosis_raw" in lab_row.index else ""
        unlabeled = set_name == "problem_set" or raw == "[UNLABELED_PROBLEM]"
        rows.append(
            {
                "case_id": cid,
                "set": set_name,
                "unlabeled": bool(unlabeled),
                "label_id": int(y_true[i]),
                "label": id_to_name[int(y_true[i])],
                "pred_id": int(pred[i]),
                "pred": id_to_name[int(pred[i])],
                "correct": bool(pred[i] == y_true[i]) if not unlabeled else None,
                "ensemble_probability": float(ens[i, pred[i]]),
                "model_agreement": int(agreement[i]),
                "h_case_fold_cosine": float(h_case_fold_cosine[i]),
                "image_injection_delta": None,
                **{f"p_mean_{k}": float(p_mean[i, k]) for k in range(c)},
                **{f"p_std_{k}": float(p_std[i, k]) for k in range(c)},
                **{f"fold{f}_top1": id_to_name[int(top1[f, i])] for f in range(len(models))},
            }
        )

    labeled_idx = [i for i, r in enumerate(rows) if not r["unlabeled"]]
    if labeled_idx:
        y_l = [y_true[i] for i in labeled_idx]
        p_l = [int(pred[i]) for i in labeled_idx]
        acc = accuracy(y_l, p_l)
        mf1 = macro_f1_all_classes(y_l, p_l, num_classes=c)
    else:
        acc, mf1 = float("nan"), float("nan")

    summary = {
        "finetune_version": FINETUNE_VERSION,
        "n_cases": n,
        "n_labeled": len(labeled_idx),
        "n_unlabeled": n - len(labeled_idx),
        "n_folds": len(models),
        "acc_labeled": acc,
        "macro_f1_labeled": mf1,
        "perfect_labeled": bool(
            labeled_idx and acc >= 1.0 - 1e-12 and mf1 >= 1.0 - 1e-12
        ),
        "mean_model_agreement": float(np.mean(agreement)),
        "n_full_agreement_5": int((agreement == len(models)).sum()),
        "mean_h_case_fold_cosine": float(np.mean(h_case_fold_cosine)),
        "image_path": "SharedImageAdapter_dino_residual",
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


def event_ixg_v02(
    model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    class_id: int,
    target: str = "classification",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    fused = fuse_batch_events(model, batch)
    score = event_input_x_gradient(model, fused, target=target, class_id=class_id)
    with torch.no_grad():
        z = model.meta(
            fused["event_mean"].detach(),
            fused["event_max"].detach(),
            fused["case_age_hours"],
            fused["view_id"],
            fused["zone_id"],
            fused["local_hour"],
            fused["padding_mask"],
        )
        _h, attn = model.pad(z, fused["padding_mask"])
    aux_attn = attn.detach().cpu().numpy()[0, 0] if attn is not None else None
    return score.detach().cpu().numpy(), aux_attn


@torch.no_grad()
def event_occlusion_deltas_v02(
    model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    class_id: int,
    indices: Sequence[int],
) -> np.ndarray:
    fused = fuse_batch_events(model, batch)
    out0 = forward_batch_v02(model, fused, fused["event_mean"].device)
    base = float(out0["logits"][0, class_id].item())
    deltas = np.zeros(fused["padding_mask"].size(1), dtype=np.float64)
    for idx in indices:
        mask = fused["padding_mask"].clone()
        mask[0, idx] = False
        if not bool(mask.any()):
            continue
        patched = dict(fused)
        patched["padding_mask"] = mask
        out = forward_batch_v02(model, patched, fused["event_mean"].device)
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
    ds: DiagnosisEventDatasetV02,
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
    batch = collate_diagnosis_batch_v02([sample], max_events=max_events)
    batch = _move_batch(batch, device)
    t = int(batch["padding_mask"][0].sum().item())
    meta_df = load_case_meta(emb_dir, case_id, t)
    event_ids = meta_df["event_id"].astype(str).tolist()

    fold_scores: List[np.ndarray] = []
    fold_attn: List[Optional[np.ndarray]] = []
    for f, model in enumerate(models):
        scores, attn = event_ixg_v02(
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
    for col in ("timestamp", "view", "zone", "case_age_hours", "local_hour"):
        if col in meta_df.columns:
            table[col] = (
                meta_df[col].astype(str).tolist()
                if col in ("timestamp", "view")
                else meta_df[col].tolist()
            )

    splits = select_report_events(table, cfg=sal_cfg)
    plan = hierarchical_plan(sal_cfg, t)

    cand: set = set()
    for scores in fold_scores:
        cand.update(select_top_event_indices(scores, top_frac=sal_cfg.top_frac_per_model))
    cand_list = sorted(cand)[: sal_cfg.occlusion_candidate_cap]

    occlusion_note = "skipped"
    if run_occlusion and cand_list:
        occ_mat = []
        for model in models:
            deltas = event_occlusion_deltas_v02(
                model, batch, class_id=pred_id, indices=cand_list
            )[:t]
            occ_mat.append(deltas)
        table["occlusion_delta_mean"] = np.mean(np.stack(occ_mat, axis=0), axis=0)
        occlusion_note = f"candidates={len(cand_list)}"

    case_out = out_root / "saliency" / "consensus" / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(case_out / "event_consensus.parquet", index=False)
    for name, df in splits.items():
        df.to_parquet(case_out / f"event_{name}.parquet", index=False)

    oof = oof_fold_for_case(case_id, metas)
    oof_info: Dict[str, Any] = {"oof_fold": oof, "mode": None}
    if oof is not None:
        oof_df = pd.DataFrame(
            {"event_id": event_ids, "oof_ixg_score": fold_scores[oof], "oof_fold": oof}
        )
        oof_df.to_parquet(case_out / "event_oof_only.parquet", index=False)
        oof_info["mode"] = "example_oof_single_fold"
    else:
        oof_info["mode"] = "problem_use_all_folds"

    image_events = None
    if "view" in table.columns:
        img = table[table["view"].astype(str) == "IMAGE"]
        if len(img):
            image_events = img.sort_values("median_saliency", ascending=False).head(10)

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
        image_events=image_events,
        disagreement=splits["disputed"]
        .sort_values("saliency_iqr", ascending=False)
        .head(20)
        .to_dict(orient="records"),
    )
    evidence["saliency_meta"] = {
        "target": saliency_target,
        "method": "input_x_gradient_after_dino_fuse",
        "plan": plan,
        "candidate_union": len(cand_list),
        "occlusion": occlusion_note,
        "oof": oof_info,
        "attn_weight_role": "auxiliary_only",
        "token_saliency": "deferred",
        "image_saliency": "view_filter_on_consensus" if image_events is not None else "none",
        "finetune_version": FINETUNE_VERSION,
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
        ROOT / DEFAULT_OUTPUT_ROOT / "evaluation" / f"{args.wandb_run}_{stamp}"
    )
    out_dir = Path(out_dir)
    # safety: never write into v0.1 evaluation tree
    forbidden = (ROOT / "outputs/online2/v2_finetune/evaluation").resolve()
    if out_dir.resolve() == forbidden or forbidden in out_dir.resolve().parents:
        raise SystemExit(f"refusing to write under v0.1 evaluation path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(args.label_map)
    id_to_name = {int(k): v for k, v in label_map["id_to_name"].items()}
    labels_df = pd.read_csv(args.labels)
    case_ids_all = [str(c) for c in labels_df["case_id"].tolist()]

    device = torch.device(args.device)
    models, metas, model_ver = load_five_fold_models(
        args.run_dir, device, force_version="v02"
    )
    ds = DiagnosisEventDatasetV02(
        case_ids_all,
        labels_df,
        label_map,
        args.emb_dir,
        max_events=args.max_events,
        require_complete=False,
    )
    print(
        {
            "finetune_version": FINETUNE_VERSION,
            "n_cached": len(ds),
            "device": str(device),
            "out_dir": str(out_dir),
            "model_version": model_ver,
            "emb_dir": str(args.emb_dir),
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
        labels_df=labels_df,
    )
    ens_dir = out_dir / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    pdf = pd.DataFrame(ens["rows"])
    pdf.to_parquet(ens_dir / "predictions.parquet", index=False)
    pdf.to_csv(ens_dir / "predictions.csv", index=False)
    # split convenience exports
    if "set" in pdf.columns:
        pdf[pdf["set"] == "example_set"].to_csv(ens_dir / "example_predictions.csv", index=False)
        pdf[pdf["set"] == "problem_set"].to_csv(ens_dir / "problem_predictions.csv", index=False)
        pdf[pdf["set"] == "problem_set"].to_parquet(
            ens_dir / "problem_predictions.parquet", index=False
        )
    (ens_dir / "summary.json").write_text(
        json.dumps(ens["summary"], indent=2) + "\n", encoding="utf-8"
    )
    print({"ENSEMBLE": ens["summary"], "sec": round(time.time() - t0, 1)}, flush=True)

    (out_dir / "fold_membership.json").write_text(
        json.dumps(
            {
                "wandb_run": args.wandb_run,
                "run_dir": str(args.run_dir),
                "finetune_version": FINETUNE_VERSION,
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
                "set": row.get("set"),
                "prompt": build_gemma_prompt(gin),
                "report_stub": generate_report_stub(gin),
                "structured_evidence": result["evidence"],
            }
        )

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "gemma_inputs.jsonl").open("w", encoding="utf-8") as f:
        for rec in gemma_recs:
            f.write(
                json.dumps(
                    {"case_id": rec["case_id"], "set": rec.get("set"), "prompt": rec["prompt"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (reports_dir / "reports_stub.jsonl").open("w", encoding="utf-8") as f:
        for rec in gemma_recs:
            f.write(
                json.dumps(
                    {"case_id": rec["case_id"], "set": rec.get("set"), "report": rec["report_stub"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    master = {
        "finetune_version": FINETUNE_VERSION,
        "wandb_run": args.wandb_run,
        "run_dir": str(args.run_dir),
        "out_dir": str(out_dir),
        "model_version": model_ver,
        "ensemble": ens["summary"],
        "saliency_cases": sal_cases,
        "saliency_summaries": sal_summaries,
        "note": "Live Gemma markdown via generate_gemma_reports.py --eval-dir <this> --out-dir <v02 reports>",
    }
    (out_dir / "EVAL_SUMMARY.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        {"DONE": master["ensemble"], "saliency_n": len(sal_cases), "out": str(out_dir)},
        flush=True,
    )


if __name__ == "__main__":
    main()
