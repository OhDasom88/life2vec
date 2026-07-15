#!/usr/bin/env python3
"""Post-hoc v0.1 layered saliency factor analysis (v0.2-safe).

Layer 1: factor strata (view/zone/hour/age) from existing consensus parquets.
Layer 2: Stage A reconnect → token IxG for primary/strong events only.

Does NOT modify v0.1 freeze training modules or v0.2 train scripts.

Examples:
  python scripts/online2_v2/analyze_v01_layer_saliency.py \\
    --eval-dir outputs/online2/v2_finetune/evaluation/7czhwn2y_20260713_212502 \\
    --layer1-only

  python scripts/online2_v2/analyze_v01_layer_saliency.py \\
    --eval-dir outputs/online2/v2_finetune/evaluation/7czhwn2y_20260713_212502 \\
    --case-id F995842_2025-02-12_2025-02-25 \\
    --with-token --max-events-per-case 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_new.vocabulary import RegistryVocabulary
from src.online2.v2.finetune_v02.checkpoint import load_five_fold_models
from src.online2.v2.v01_layer_saliency import (
    factor_summary_rows,
    load_case_batch_tensors,
    load_stage_a_module,
    pick_primary_events,
    render_factor_report_md,
    summarize_factor_distributions,
    summarize_token_frame,
    token_ixg_for_event,
    token_results_to_frame,
    write_cross_case_aggregates,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/evaluation/latest",
    )
    p.add_argument("--case-id", type=str, default=None, help="Single case; default=all with consensus")
    p.add_argument("--layer1-only", action="store_true")
    p.add_argument("--with-token", action="store_true", help="Enable Layer-2 Stage A token IxG")
    p.add_argument("--max-events-per-case", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override fold ckpts (default: EVAL_SUMMARY.json run_dir)",
    )
    p.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/event_embeddings",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt",
    )
    p.add_argument(
        "--events",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/case_manifest.csv",
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json",
    )
    p.add_argument(
        "--abspos-reference",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/abspos_reference.json",
    )
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--oof-only", action="store_true", help="Token IxG with OOF fold only")
    return p.parse_args()


def resolve_eval_dir(path: Path) -> Path:
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if path.is_symlink() or (path / "EVAL_SUMMARY.json").exists() or (path / "saliency").exists():
        return path
    raise FileNotFoundError(f"eval-dir not found or incomplete: {path}")


def list_consensus_cases(eval_dir: Path) -> List[str]:
    cons = eval_dir / "saliency" / "consensus"
    if not cons.is_dir():
        raise FileNotFoundError(cons)
    return sorted(p.name for p in cons.iterdir() if p.is_dir())


def load_eval_summary(eval_dir: Path) -> Dict[str, Any]:
    p = eval_dir / "EVAL_SUMMARY.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_evidence_meta(eval_dir: Path, case_id: str) -> Dict[str, Any]:
    p = eval_dir / "evidence" / f"{case_id}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def oof_fold_for_case(case_id: str, metas: List[Dict[str, Any]]) -> Optional[int]:
    for m in metas:
        if case_id in set(m.get("val_cases") or []):
            return int(m["fold"])
    return None


def run_layer1_case(eval_dir: Path, case_id: str, out_cases: Path) -> Dict[str, Any]:
    cdir = eval_dir / "saliency" / "consensus" / case_id
    cons = pd.read_parquet(cdir / "event_consensus.parquet")
    primary_path = cdir / "event_primary.parquet"
    primary = pd.read_parquet(primary_path) if primary_path.exists() else pd.DataFrame()
    strong_path = cdir / "event_strong.parquet"
    strong = pd.read_parquet(strong_path) if strong_path.exists() else pd.DataFrame()
    # Prefer strong∪primary for "primary_top" display
    if len(strong) and len(primary):
        pick_src = pd.concat([strong, primary], ignore_index=True).drop_duplicates("event_id")
    elif len(strong):
        pick_src = strong
    else:
        pick_src = primary

    dist = summarize_factor_distributions(cons, case_id=case_id, primary_df=pick_src)
    evidence = load_evidence_meta(eval_dir, case_id)
    diagnosis = evidence.get("diagnosis")

    case_out = out_cases / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    (case_out / "factor_dist.json").write_text(
        json.dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render_factor_report_md(dist, diagnosis=diagnosis)
    (case_out / "factor_report.md").write_text(md, encoding="utf-8")
    return {
        "case_id": case_id,
        "diagnosis": diagnosis,
        "n_events": dist["n_events"],
        "n_primary": int(len(primary)),
        "n_strong": int(len(strong)),
        "dist": dist,
        "primary_df": pick_src,
        "evidence": evidence,
    }


def run_layer2_case(
    *,
    case_info: Dict[str, Any],
    args: argparse.Namespace,
    stage_a_mod: Any,
    encoder: Any,
    vocab: Any,
    abspos_reference: Any,
    models: List[Any],
    metas: List[Dict[str, Any]],
    events_df: pd.DataFrame,
    manifest: pd.DataFrame,
    device: torch.device,
    out_cases: Path,
) -> Dict[str, Any]:
    case_id = case_info["case_id"]
    primary_df = pick_primary_events(case_info["primary_df"], max_events=args.max_events_per_case)
    if primary_df.empty:
        return {"case_id": case_id, "n_token_events": 0, "skipped": "no_primary"}

    emb_path = args.emb_dir / f"{case_id}.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(emb_path)

    batch, event_ids, _emb_df = load_case_batch_tensors(emb_path, device=device)
    eid_to_idx = {e: i for i, e in enumerate(event_ids)}

    mrow = manifest[manifest["case_id"].astype(str) == case_id]
    if mrow.empty:
        raise KeyError(f"manifest missing {case_id}")
    crow = mrow.iloc[0]
    farm_id = str(crow["farm_id"])
    period_start = pd.Timestamp(crow["period_start"])
    period_end = pd.Timestamp(crow["period_end"])
    case_events = stage_a_mod.case_events_from_frame(
        events_df, farm_id, period_start, period_end, include_image=False
    )
    order_to_idx = {ev.event_id: i for i, ev in enumerate(case_events)}
    case_t0 = case_events[0].timestamp if case_events else period_start

    evidence = case_info.get("evidence") or {}
    pred_name = evidence.get("diagnosis")
    class_id = 0
    # Prefer pred_id from ensemble if present in evidence keys later; map via label if needed
    # Evidence usually has diagnosis string only — load from ensemble parquet if available
    ens_path = args.eval_dir / "ensemble" / "predictions.parquet"
    if ens_path.exists():
        ens = pd.read_parquet(ens_path)
        row = ens[ens["case_id"].astype(str) == case_id]
        if not row.empty and "pred_id" in row.columns:
            class_id = int(row.iloc[0]["pred_id"])
        elif not row.empty and "pred" in row.columns:
            pred_name = str(row.iloc[0]["pred"])

    use_models = list(models)
    if args.oof_only:
        of = oof_fold_for_case(case_id, metas)
        if of is not None:
            use_models = [models[of]]

    meta_map = {
        str(r["event_id"]): {
            "view": r.get("view"),
            "zone": r.get("zone"),
            "median_saliency": float(r["median_saliency"]),
            "positive_agreement_count": int(r.get("positive_agreement_count", 0)),
        }
        for _, r in primary_df.iterrows()
    }

    results = []
    for _, r in primary_df.iterrows():
        eid = str(r["event_id"])
        if eid not in eid_to_idx:
            print(f"[warn] {case_id}: {eid} not in emb cache index", flush=True)
            continue
        if eid not in order_to_idx:
            print(f"[warn] {case_id}: {eid} not in Stage A event list", flush=True)
            continue
        t0 = time.time()
        res = token_ixg_for_event(
            encoder=encoder,
            diagnosis_models=use_models,
            stage_a_mod=stage_a_mod,
            events=case_events,
            target_idx=order_to_idx[eid],
            case_t0=case_t0,
            abspos_reference=abspos_reference,
            vocab=vocab,
            batch=batch,
            event_index=eid_to_idx[eid],
            class_id=class_id,
            device=device,
            max_length=args.max_length,
        )
        results.append(res)
        print(
            {
                "token_ixg": case_id,
                "event_id": eid[:24],
                "span": res.span_end - res.span_start,
                "pool_l2_err": round(res.pool_l2_err, 4),
                "sec": round(time.time() - t0, 2),
            },
            flush=True,
        )

    tok_df = token_results_to_frame(results, case_id=case_id, meta=meta_map)
    case_out = out_cases / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    if len(tok_df):
        tok_df.to_parquet(case_out / "token_saliency.parquet", index=False)
    tok_sum = summarize_token_frame(tok_df)
    (case_out / "token_summary.json").write_text(
        json.dumps(tok_sum, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # refresh markdown with token section
    md = render_factor_report_md(
        case_info["dist"],
        diagnosis=pred_name or case_info.get("diagnosis"),
        token_summary=tok_sum,
    )
    (case_out / "factor_report.md").write_text(md, encoding="utf-8")
    return {
        "case_id": case_id,
        "n_token_events": int(tok_sum.get("n_events") or 0),
        "n_tokens": int(tok_sum.get("n_tokens") or 0),
        "mean_pool_l2_err": tok_sum.get("mean_pool_l2_err"),
        "class_id": class_id,
        "diagnosis": pred_name,
    }


def main() -> None:
    args = parse_args()
    if args.with_token and args.layer1_only:
        raise SystemExit("use either --layer1-only or --with-token, not both")
    do_token = bool(args.with_token) and not bool(args.layer1_only)

    eval_dir = resolve_eval_dir(args.eval_dir)
    args.eval_dir = eval_dir
    summary = load_eval_summary(eval_dir)
    run_dir = args.run_dir
    if run_dir is None:
        rd = summary.get("run_dir")
        run_dir = Path(rd) if rd else ROOT / "outputs/online2/v2_finetune/runs/cv_20260713_144719"
    run_dir = run_dir if run_dir.is_absolute() else (ROOT / run_dir)

    out_root = eval_dir / "layer_analysis"
    out_cases = out_root / "cases"
    out_cases.mkdir(parents=True, exist_ok=True)

    case_ids = [args.case_id] if args.case_id else list_consensus_cases(eval_dir)
    single_case = args.case_id is not None
    print({"eval_dir": str(eval_dir), "n_cases": len(case_ids), "token": do_token}, flush=True)

    # --- Layer 1 ---
    all_summary_rows: List[Dict[str, Any]] = []
    dists_by_case: Dict[str, Dict[str, Any]] = {}
    case_infos: List[Dict[str, Any]] = []
    for cid in case_ids:
        info = run_layer1_case(eval_dir, cid, out_cases)
        case_infos.append(info)
        all_summary_rows.extend(factor_summary_rows(info["dist"]))
        dists_by_case[cid] = {k: v for k, v in info["dist"].items()}
        print({"layer1": cid, "n_events": info["n_events"], "n_primary": info["n_primary"]}, flush=True)

    summary_csv = out_root / "factor_summary.csv"
    jsonl_path = out_root / "factor_distributions.jsonl"

    if single_case and summary_csv.exists() and jsonl_path.exists():
        # Merge so a 1-case L2 smoke does not wipe the all-case L1 corpus.
        prev = pd.read_csv(summary_csv)
        prev = prev[prev["case_id"].astype(str) != str(args.case_id)]
        merged = pd.concat([prev, pd.DataFrame(all_summary_rows)], ignore_index=True)
        merged.to_csv(summary_csv, index=False)
        kept: Dict[str, Dict[str, Any]] = {}
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = str(rec.get("case_id"))
                if cid != str(args.case_id):
                    kept[cid] = rec
        kept.update(dists_by_case)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for cid in sorted(kept):
                f.write(json.dumps(kept[cid], ensure_ascii=False) + "\n")
    else:
        pd.DataFrame(all_summary_rows).to_csv(summary_csv, index=False)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for cid in sorted(dists_by_case):
                f.write(json.dumps(dists_by_case[cid], ensure_ascii=False) + "\n")

    layer2_stats: List[Dict[str, Any]] = []
    if do_token:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
            print("[warn] cuda unavailable → cpu", flush=True)

        stage_a_mod = load_stage_a_module()
        abspos_reference = stage_a_mod.load_abspos_reference(args.abspos_reference)
        vocab = RegistryVocabulary(registry_path=str(args.vocab), registry_version="v2")
        encoder, _, _ = stage_a_mod.load_frozen_encoder(args.ckpt, vocab, device)

        models, metas, ver = load_five_fold_models(run_dir, device, force_version="v01")
        print({"folds": len(models), "version": ver, "run_dir": str(run_dir)}, flush=True)

        farm_ids = set()
        manifest = pd.read_csv(args.manifest)
        for cid in case_ids:
            rows = manifest[manifest["case_id"].astype(str) == cid]
            if not rows.empty:
                farm_ids.add(str(rows.iloc[0]["farm_id"]))
        print(f"loading events for {len(farm_ids)} farms…", flush=True)
        events_df = stage_a_mod.load_events_frame(args.events, farm_ids)

        for info in case_infos:
            st = run_layer2_case(
                case_info=info,
                args=args,
                stage_a_mod=stage_a_mod,
                encoder=encoder,
                vocab=vocab,
                abspos_reference=abspos_reference,
                models=models,
                metas=metas,
                events_df=events_df,
                manifest=manifest,
                device=device,
                out_cases=out_cases,
            )
            layer2_stats.append(st)

    # When single-case L2, preserve prior layer2 case stats in summary if present.
    if single_case and do_token and (out_root / "LAYER_ANALYSIS_SUMMARY.json").exists():
        try:
            prev_master = json.loads((out_root / "LAYER_ANALYSIS_SUMMARY.json").read_text(encoding="utf-8"))
            prev_cases = {
                str(c.get("case_id")): c
                for c in (prev_master.get("layer2") or {}).get("cases") or []
                if c.get("case_id")
            }
            for st in layer2_stats:
                prev_cases[str(st["case_id"])] = st
            layer2_stats = [prev_cases[k] for k in sorted(prev_cases)]
        except Exception:
            pass

    # Rebuild case_infos corpus for aggregates when single-case (read sibling reports).
    agg_infos = case_infos
    if single_case:
        agg_infos = []
        for d in sorted((out_cases).glob("*/factor_dist.json")):
            dist = json.loads(d.read_text(encoding="utf-8"))
            cid = dist.get("case_id") or d.parent.name
            ev = load_evidence_meta(eval_dir, cid)
            agg_infos.append(
                {
                    "case_id": cid,
                    "diagnosis": ev.get("diagnosis"),
                    "dist": dist,
                }
            )

    agg_paths = write_cross_case_aggregates(
        out_root, case_infos=agg_infos, factor_summary_csv=summary_csv
    )

    master = {
        "eval_dir": str(eval_dir),
        "out_dir": str(out_root),
        "n_cases_this_run": len(case_ids),
        "n_cases_in_corpus": len(agg_infos),
        "layer1": {
            "factor_summary_csv": str(summary_csv),
            "factor_distributions_jsonl": str(jsonl_path),
        },
        "layer2": {
            "enabled": do_token or any((out_cases / c / "token_saliency.parquet").exists() for c in [i["case_id"] for i in agg_infos]),
            "max_events_per_case": args.max_events_per_case,
            "oof_only": bool(args.oof_only),
            "n_cases_with_tokens": sum(
                1
                for i in agg_infos
                if (out_cases / str(i["case_id"]) / "token_saliency.parquet").exists()
            ),
            "cases": layer2_stats,
        },
        "aggregates": agg_paths,
        "gemma_reports": "deferred (GPU reserved / follow-up)",
        "notes": [
            "v0.1 post-hoc analysis; does not mutate freeze training modules.",
            "Layer1=event saliency factor strata; Layer2=Stage A reconnect token IxG.",
            "Saliency is model sensitivity, not proven causation.",
            "Gemma report generation left as follow-up.",
        ],
    }
    (out_root / "LAYER_ANALYSIS_SUMMARY.json").write_text(
        json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print({"wrote": str(out_root / "LAYER_ANALYSIS_SUMMARY.json"), "aggregates": agg_paths}, flush=True)


if __name__ == "__main__":
    main()
