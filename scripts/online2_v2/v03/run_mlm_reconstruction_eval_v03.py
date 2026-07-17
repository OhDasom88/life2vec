#!/usr/bin/env python3
"""MLM reconstruction selection + final metrics (expansion by recoverable only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_new.vocabulary import RegistryVocabulary
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_index import (
    annotate_bundles_with_token_ids,
    build_bank_query_set_manifest,
    load_two_tier_bank,
    query_unique_bundles_from_bank,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_RECONSTRUCTION_EVAL,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_loader import (
    load_continuous_feature_names,
    resolve_training_case_ids,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.constrained_mlm import (
    mlm_reconstruction_metrics,
    score_bundles_joint,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.mlm_window_lift import (
    annotate_sentence_tokens,
    extract_value_bundle,
    lift_local_mask_to_window,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    dump_runtime_imports_from_env,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.metric_eligibility import (
    classify_metric_eligibility,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    MIN_RECONSTRUCTION_CANDIDATES,
    build_selected_mg_flag_rows,
    exact_match_selection_metrics,
    frozen_final_manifest_sha256,
    mg_key,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths
from src.online2.v2.finetune_v03.counterfactual.pipeline_m2 import _load_case_events
from src.online2.v2.finetune_v03.counterfactual.stage_a_loader import load_stage_a_module


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_eligible_pool(cfg: Dict[str, Any], *, limit_cases: int = 0) -> List[Dict[str, Any]]:
    continuous = load_continuous_feature_names(Path(cfg["feature_schema_path"]))
    train_ids = list(resolve_training_case_ids(Path(cfg["labels_path"])))
    if int(limit_cases) > 0:
        train_ids = train_ids[: int(limit_cases)]
    pool: List[Dict[str, Any]] = []
    for cid in train_ids:
        try:
            _sa, events, _farm, _ps = _load_case_events(cfg, cid)
        except Exception:
            continue
        for ev in events:
            ann = annotate_sentence_tokens(list(ev.sentence_tokens))
            seen = set()
            for g, f in zip(ann.group_ids, ann.features):
                if g in seen:
                    continue
                seen.add(g)
                if str(f).lower() not in continuous and str(f) not in continuous:
                    continue
                v_toks, _, v_roles = extract_value_bundle(ann, measurement_group_id=g)
                if not v_toks:
                    continue
                pool.append(
                    {
                        "case_id": cid,
                        "event_id": str(ev.event_id),
                        "measurement_group_id": str(g),
                        "feature": str(f),
                        "original_tokens": list(v_toks),
                        "original_roles": list(v_roles),
                        "timestamp": pd.Timestamp(ev.timestamp),
                    }
                )
    return pool


def evaluate_recoverable(
    selected_rows: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    encoder,
    stage_a,
    vocab,
    abspos_ts,
    device: torch.device,
    observations,
    uniques,
) -> int:
    """Count MGs where original ranks in scored bank (recoverable reconstruction)."""
    mask_id = int(vocab.token2index["[MASK]"])
    unk = int(vocab.token2index.get("[UNK]", 0))
    n_ok = 0
    for row in selected_rows:
        try:
            _sa, events, _farm, period_start = _load_case_events(cfg, row["case_id"])
        except Exception:
            continue
        eid_map = {str(ev.event_id): i for i, ev in enumerate(events)}
        if row["event_id"] not in eid_map:
            continue
        tidx = eid_map[row["event_id"]]
        ev = events[tidx]
        ann = annotate_sentence_tokens(
            list(ev.sentence_tokens),
            vocab_token2index=vocab.token2index,
            unk_id=unk,
        )
        bundles, _executed = query_unique_bundles_from_bank(
            observations,
            uniques,
            feature=row["feature"],
            expected_roles=row["original_roles"],
            target_event_id=row["event_id"],
            target_timestamp=row["timestamp"],
            case_id=row["case_id"],
            mode=BANK_MODE_RECONSTRUCTION_EVAL,
        )
        bundles = annotate_bundles_with_token_ids(
            bundles, vocab_token2index=vocab.token2index, unk_id=unk
        )
        if len(bundles) < MIN_RECONSTRUCTION_CANDIDATES:
            continue
        window = stage_a.construct_target_window(events, tidx, max_length=1024)
        x, pad, _L = stage_a.window_to_tensors(
            events,
            window,
            vocab=vocab,
            abspos_reference=abspos_ts,
            case_t0=period_start,
            max_length=1024,
        )
        try:
            lift = lift_local_mask_to_window(
                local_token_ids=ann.token_ids,
                local_group_ids=ann.group_ids,
                local_roles=ann.roles,
                local_tokens=ann.tokens,
                target_group_id=row["measurement_group_id"],
                mask_id=mask_id,
                window_input_ids_4ch=x,
                window_padding_mask=pad,
                target_token_start=int(window.target_token_start),
            )
            scored = score_bundles_joint(
                encoder,
                input_ids_4ch=lift.input_ids_4ch,
                padding_mask=lift.padding_mask.long(),
                mask_result=lift.mask_result,
                bundles=bundles,
                device=device,
            )
        except Exception:
            continue
        if not scored:
            continue
        # recoverable if original appears among scored
        orig = list(row["original_tokens"])
        if any(list(c.tokens) == orig for c in scored):
            n_ok += 1
    return n_ok


def _append_per_mg_row(
    per_mg: List[Dict[str, Any]],
    *,
    k: str,
    row: Dict[str, Any],
    candidate_count: Any,
    original_rank: Any,
    scoring_completed: bool,
    manifest_recoverable: bool,
    early_ineligible_reason: str | None = None,
    recall_at_1: int = 0,
    recall_at_3: int = 0,
    reciprocal_rank: float = 0.0,
    random_recall_at_3: float = 0.0,
    original_tokens: List[Any] | None = None,
) -> None:
    """Build one per-MG row; recoverable always mirrors frozen manifest (never re-inferred)."""
    clf = classify_metric_eligibility(
        manifest_recoverable=manifest_recoverable,
        original_rank=original_rank,
        scoring_completed=scoring_completed,
        candidate_count=candidate_count,
        early_ineligible_reason=early_ineligible_reason,
    )
    per_mg.append(
        {
            "mg_key": k,
            "feature": row["feature"],
            "case_id": row["case_id"],
            "event_id": row["event_id"],
            "candidate_count": candidate_count,
            "manifest_recoverable": bool(manifest_recoverable),
            "recoverable": bool(manifest_recoverable),
            "original_rank": original_rank,
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "reciprocal_rank": reciprocal_rank,
            "random_recall_at_3": random_recall_at_3,
            "original_tokens": list(original_tokens or row["original_tokens"]),
            "scoring_completed": bool(scoring_completed),
            "metric_eligible": clf["metric_eligible"],
            "metric_ineligible_reason": clf["metric_ineligible_reason"],
            "metric_ineligible_reasons": clf["metric_ineligible_reasons"],
            "metric_audit_error": clf["metric_audit_error"],
            "metric_audit_errors": clf["metric_audit_errors"],
        }
    )


def final_metrics_for_keys(
    keys: List[str],
    pool: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    encoder,
    stage_a,
    vocab,
    abspos_ts,
    device: torch.device,
    observations,
    uniques,
    manifest_recoverable_by_key: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    key_set = set(keys)
    by_key = {mg_key(r): r for r in pool}
    rows = [by_key[k] for k in keys if k in by_key]
    mask_id = int(vocab.token2index["[MASK]"])
    unk = int(vocab.token2index.get("[UNK]", 0))
    metrics_list = []
    per_mg: List[Dict[str, Any]] = []
    evaluated_mg_keys: List[str] = []
    scored_mg_count = 0
    ranked_original_mg_count = 0
    manifest_map = dict(manifest_recoverable_by_key or {})
    for k in keys:
        row = by_key.get(k)
        if row is None:
            # selected key missing from pool → not evaluated (A8-2 must fail)
            continue
        manifest_recoverable = bool(manifest_map.get(k, False))
        try:
            _sa, events, _farm, period_start = _load_case_events(cfg, row["case_id"])
        except Exception:
            continue
        eid_map = {str(ev.event_id): i for i, ev in enumerate(events)}
        if row["event_id"] not in eid_map:
            continue
        tidx = eid_map[row["event_id"]]
        ev = events[tidx]
        ann = annotate_sentence_tokens(
            list(ev.sentence_tokens),
            vocab_token2index=vocab.token2index,
            unk_id=unk,
        )
        bundles, _executed = query_unique_bundles_from_bank(
            observations,
            uniques,
            feature=row["feature"],
            expected_roles=row["original_roles"],
            target_event_id=row["event_id"],
            target_timestamp=row["timestamp"],
            case_id=row["case_id"],
            mode=BANK_MODE_RECONSTRUCTION_EVAL,
        )
        bundles = annotate_bundles_with_token_ids(
            bundles, vocab_token2index=vocab.token2index, unk_id=unk
        )
        # Always count as evaluated once we attempt this selected key in the pool
        evaluated_mg_keys.append(k)
        if len(bundles) < MIN_RECONSTRUCTION_CANDIDATES:
            _append_per_mg_row(
                per_mg,
                k=k,
                row=row,
                candidate_count=len(bundles),
                original_rank=None,
                scoring_completed=False,
                manifest_recoverable=manifest_recoverable,
                early_ineligible_reason="insufficient_candidates",
            )
            continue
        window = stage_a.construct_target_window(events, tidx, max_length=1024)
        x, pad, _L = stage_a.window_to_tensors(
            events,
            window,
            vocab=vocab,
            abspos_reference=abspos_ts,
            case_t0=period_start,
            max_length=1024,
        )
        try:
            lift = lift_local_mask_to_window(
                local_token_ids=ann.token_ids,
                local_group_ids=ann.group_ids,
                local_roles=ann.roles,
                local_tokens=ann.tokens,
                target_group_id=row["measurement_group_id"],
                mask_id=mask_id,
                window_input_ids_4ch=x,
                window_padding_mask=pad,
                target_token_start=int(window.target_token_start),
            )
            scored = score_bundles_joint(
                encoder,
                input_ids_4ch=lift.input_ids_4ch,
                padding_mask=lift.padding_mask.long(),
                mask_result=lift.mask_result,
                bundles=bundles,
                device=device,
            )
        except Exception:
            _append_per_mg_row(
                per_mg,
                k=k,
                row=row,
                candidate_count=len(bundles),
                original_rank=None,
                scoring_completed=False,
                manifest_recoverable=manifest_recoverable,
                early_ineligible_reason="scoring_failed",
            )
            continue
        if not scored:
            _append_per_mg_row(
                per_mg,
                k=k,
                row=row,
                candidate_count=len(bundles),
                original_rank=None,
                scoring_completed=False,
                manifest_recoverable=manifest_recoverable,
                early_ineligible_reason="scoring_empty",
            )
            continue
        scored_mg_count += 1
        m = mlm_reconstruction_metrics(
            scored,
            original_tokens=row["original_tokens"],
            random_baseline_recall_at_3=min(3, len(scored)) / float(max(len(scored), 1)),
        )
        metrics_list.append(m)
        orig = list(row["original_tokens"])
        ranks = [i for i, c in enumerate(scored, start=1) if list(c.tokens) == orig]
        original_rank = ranks[0] if ranks else None
        if original_rank is not None:
            ranked_original_mg_count += 1
        _append_per_mg_row(
            per_mg,
            k=k,
            row=row,
            candidate_count=len(scored),
            original_rank=original_rank,
            scoring_completed=True,
            manifest_recoverable=manifest_recoverable,
            recall_at_1=int(m["recall@1"]),
            recall_at_3=int(m["recall@3"]),
            reciprocal_rank=float(m["MRR"]),
            random_recall_at_3=float(m["random_baseline_recall@3"]),
            original_tokens=orig,
        )

    n = max(len(metrics_list), 1)
    recall3 = sum(m["recall@3"] for m in metrics_list) / n if metrics_list else 0.0
    recall1 = sum(m["recall@1"] for m in metrics_list) / n if metrics_list else 0.0
    mrr = sum(m["MRR"] for m in metrics_list) / n if metrics_list else 0.0
    rand3 = (
        sum(m["random_baseline_recall@3"] for m in metrics_list) / n if metrics_list else 0.0
    )
    lift = (recall3 / rand3) if rand3 > 0 else 0.0
    selected_cov = scored_mg_count / float(max(len(rows), 1))
    recoverable_mg_count = sum(1 for k in keys if bool(manifest_map.get(k, False)))
    metric_eligible_mg_count = sum(1 for r in per_mg if r.get("metric_eligible") is True)
    return {
        "eligible_mg_count": len(pool),
        "selected_mg_count": len(keys),
        "recoverable_mg_count": recoverable_mg_count,
        "scored_mg_count": scored_mg_count,
        "scored_candidate_available_mg_count": scored_mg_count,
        "ranked_original_mg_count": ranked_original_mg_count,
        "metric_eligible_mg_count": metric_eligible_mg_count,
        "selected_bank_coverage": selected_cov,
        "full_pool_bank_coverage": scored_mg_count / float(max(len(pool), 1)),
        "diagnostic_raw_metrics": {
            "recall_at_3": recall3,
            "recall_at_1": recall1,
            "MRR": mrr,
            "mean_random_recall_at_3": rand3,
            "lift": lift,
        },
        "recall@1": recall1,
        "recall@3": recall3,
        "MRR": mrr,
        "mean_random_recall_at_3": rand3,
        "Lift": lift,
        "n_metrics": len(metrics_list),
        "per_mg": per_mg,
        "evaluated_mg_keys": evaluated_mg_keys,
    }



def main() -> int:
    exit_code = 1
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
        ap.add_argument("--out-root", type=Path, default=None)
        ap.add_argument("--limit-cases", type=int, default=0, help="0=full pool")
        args = ap.parse_args()
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        out_root = Path(args.out_root or cfg["output_root"])
        paths = M1Paths(out_root)
        reports = out_root / "reports"
        manifests = out_root / "manifests" / "mlm_selection"
        reports.mkdir(parents=True, exist_ok=True)

        device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        stage_a = load_stage_a_module()
        vocab = RegistryVocabulary(
            registry_path=str(cfg.get("vocabulary_path") or cfg["tokenizer_path"]),
            registry_version="v2",
        )
        encoder, _hp, abspos = stage_a.load_frozen_encoder(
            Path(cfg["stage_a_ckpt"]), vocab, device
        )
        if abspos is None:
            abspos = stage_a.load_abspos_reference(Path(cfg["abspos_reference_path"]))
        abspos_ts = pd.Timestamp(abspos)

        print("[1] build eligible pool", flush=True)
        pool = build_eligible_pool(cfg, limit_cases=int(args.limit_cases))
        print(json.dumps({"eligible_mg_count": len(pool)}), flush=True)

        print("[2] load persisted bank", flush=True)
        observations, uniques, bank_manifest = load_two_tier_bank(
            paths.artifacts / "mlm_bundle_bank"
        )
        bank_content = bank_manifest.get("bundle_bank_content_sha256")

        print("[3] load frozen selection (read-only)", flush=True)
        final_path = manifests / "selection_manifest_final.json"
        chain_path = manifests / "selection_manifest_chain.json"
        if not final_path.exists():
            print(
                "ERROR: selection_manifest_final.json missing — run reconstruction_selection first",
                flush=True,
            )
            return 1
        final_doc = json.loads(final_path.read_text(encoding="utf-8"))
        expected_sha = None
        if chain_path.exists():
            chain_doc = json.loads(chain_path.read_text(encoding="utf-8"))
            expected_sha = chain_doc.get("selection_manifest_final_sha256")
        actual_sha = frozen_final_manifest_sha256(final_doc)
        if expected_sha and actual_sha != expected_sha:
            print("ERROR: frozen manifest SHA mismatch", flush=True)
            write_json(
                reports / "mlm_reconstruction_metrics.json",
                {
                    "a8_2_pass": False,
                    "mlm_reconstruction_quality": "NOT_EVALUATED",
                    "reconstruction_metric_audit_status": "FAIL",
                    "errors": ["frozen_manifest_sha_mismatch"],
                },
            )
            return 1

        selected_keys = list(final_doc.get("selected_mg_keys") or [])
        manifest_flags = list(final_doc.get("selected_mg_flags") or [])
        if not selected_keys or not manifest_flags:
            write_json(
                reports / "mlm_reconstruction_metrics.json",
                {
                    "a8_2_pass": False,
                    "mlm_reconstruction_quality": "NOT_EVALUATED",
                    "reconstruction_metric_audit_status": "FAIL",
                    "errors": ["missing_frozen_flags_or_keys"],
                },
            )
            return 1

        manifest_recoverable_by_key = {
            str(f.get("mg_key")): bool(f.get("is_recoverable")) for f in manifest_flags
        }

        print("[4] final metrics after freeze (manifest read-only)", flush=True)
        # Independent recoverable flags — do NOT write back to selection_manifest_final.json
        key_set = set(selected_keys)
        selected_rows = [r for r in pool if mg_key(r) in key_set]
        recoverable_keys: List[str] = []
        for row in selected_rows:
            n = evaluate_recoverable(
                [row],
                cfg=cfg,
                encoder=encoder,
                stage_a=stage_a,
                vocab=vocab,
                abspos_ts=abspos_ts,
                device=device,
                observations=observations,
                uniques=uniques,
            )
            if n >= 1:
                recoverable_keys.append(mg_key(row))
        metrics_flags = build_selected_mg_flag_rows(selected_keys, recoverable_keys=recoverable_keys)

        metrics = final_metrics_for_keys(
            selected_keys,
            pool,
            cfg=cfg,
            encoder=encoder,
            stage_a=stage_a,
            vocab=vocab,
            abspos_ts=abspos_ts,
            device=device,
            observations=observations,
            uniques=uniques,
            manifest_recoverable_by_key=manifest_recoverable_by_key,
        )
        evaluated_mg_keys = list(metrics.get("evaluated_mg_keys") or [])
        # A8-2 keys: independent evaluated set (not a copy of selected_keys)
        # Include selected keys missing from pool as unevaluated → mismatch if missing
        for k in selected_keys:
            if k not in evaluated_mg_keys and k not in set(mg_key(r) for r in pool):
                # key absent from pool entirely — still must appear in metrics_keys for fail
                pass
        metrics_keys_for_a8 = list(evaluated_mg_keys)
        # If selected key not in pool, it was never evaluated — omit from metrics_keys → A8-2 FAIL
        metrics["selected_mg_keys"] = list(selected_keys)
        metrics["evaluated_mg_keys"] = metrics_keys_for_a8
        metrics["metrics_recoverable_flags"] = metrics_flags
        metrics["manifest_recoverable_flags"] = manifest_flags
        a8_2 = exact_match_selection_metrics(
            manifest_keys=selected_keys,
            metrics_keys=metrics_keys_for_a8,
            manifest_recoverable_flags=manifest_flags,
            metrics_recoverable_flags=metrics_flags,
        )

        # Write per-MG JSONL then recompute Q4/Q5 from file
        per_mg_path = reports / "mlm_reconstruction_per_mg.jsonl"
        with per_mg_path.open("w", encoding="utf-8") as f:
            for row in metrics.get("per_mg") or []:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        per_mg_rows = []
        if per_mg_path.exists():
            for line in per_mg_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    per_mg_rows.append(json.loads(line))
        from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_q4_q5 import (
            aggregate_q4_q5_from_per_mg,
        )

        q45_agg = aggregate_q4_q5_from_per_mg(per_mg_rows)
        mean_r3 = q45_agg.get("mean_recall_at_3")
        mean_rand = q45_agg.get("mean_random_recall_at_3")
        lift_from_file = q45_agg.get("lift")

        manifest_recoverable = sum(1 for f in manifest_flags if f.get("is_recoverable"))
        q1 = metrics["eligible_mg_count"] >= 20
        q2 = manifest_recoverable >= 20
        q3 = (manifest_recoverable / float(max(len(selected_keys), 1))) >= 0.5
        q4 = bool(q45_agg.get("q4")) if q45_agg.get("q4_q5_evaluable") else False
        q5 = bool(q45_agg.get("q5")) if q45_agg.get("q4_q5_evaluable") else False
        quality_pass = bool(
            q1 and q2 and q3 and q4 and q5 and q45_agg.get("q4_q5_evaluable")
        )
        if not a8_2["a8_2_pass"]:
            quality_enum = "NOT_EVALUATED"
            audit = "FAIL"
        elif q45_agg.get("reconstruction_metric_audit_status") != "PASS":
            quality_enum = "NOT_EVALUATED"
            audit = "FAIL"
        elif not q45_agg.get("q4_q5_evaluable"):
            quality_enum = "NOT_EVALUATED"
            audit = str(q45_agg.get("reconstruction_metric_audit_status") or "FAIL")
        else:
            quality_enum = "PASS" if quality_pass else "FAIL"
            audit = "PASS"

        chain_hash = None
        if chain_path.exists():
            chain_hash = json.loads(chain_path.read_text()).get("selection_manifest_chain_hash")

        bank_query = {}
        if selected_rows:
            executed_entries = []
            for row in selected_rows:
                _b, executed = query_unique_bundles_from_bank(
                    observations,
                    uniques,
                    feature=row["feature"],
                    expected_roles=row["original_roles"],
                    target_event_id=row["event_id"],
                    target_timestamp=row["timestamp"],
                    case_id=row["case_id"],
                    mode=BANK_MODE_RECONSTRUCTION_EVAL,
                )
                executed_entries.append(
                    {"mg_key": mg_key(row), "executed_query_manifest": executed}
                )
            bank_query = build_bank_query_set_manifest(
                run_id="reconstruction_eval",
                bank_manifest=bank_manifest,
                mode=BANK_MODE_RECONSTRUCTION_EVAL,
                executed_entries=executed_entries,
            )
            bank_query["expected_mg_keys"] = [mg_key(r) for r in selected_rows]
            bank_query["query_set_manifest"] = {
                "run_id": bank_query["run_id"],
                "query_count": bank_query["query_count"],
                "query_records": bank_query["query_records"],
                "query_set_sha256": bank_query["query_set_sha256"],
            }

        out = {
            **{k: v for k, v in metrics.items() if k != "per_mg"},
            "quality_pass": quality_pass,
            "mlm_reconstruction_quality": quality_enum,
            "reconstruction_metric_audit_status": audit,
            "official_strict_metrics": {
                "audit_status": audit,
                "recall_at_3": mean_r3,
                "lift": lift_from_file,
                "mean_random_recall_at_3": mean_rand,
                "q4": q4,
                "q5": q5,
                "q4_q5_evaluable": q45_agg.get("q4_q5_evaluable"),
                "metric_eligible_mg_count": q45_agg.get("metric_eligible_mg_count"),
            },
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "Q4": q4,
            "Q5": q5,
            "manifest_recoverable_mg_count": manifest_recoverable,
            "selected_bank_coverage_diagnostic": metrics.get("selected_bank_coverage"),
            "mean_recall_at_3_from_per_mg": mean_r3,
            "mean_random_recall_at_3_from_per_mg": mean_rand,
            "lift_from_per_mg": lift_from_file,
            "selection_manifest_chain_hash": chain_hash,
            "selection_manifest_final_sha256": actual_sha,
            "expansion_uses_only_recoverable": True,
            "a8_2_pass": a8_2["a8_2_pass"],
            "a8_2_detail": a8_2,
            "bundle_bank_content_sha256": bank_content,
            "manifest_frozen_read_only": True,
            "bank_query_record": bank_query,
            **{k: bank_query[k] for k in bank_query if k != "query_manifest"},
            "mlm_reconstruction_per_mg_path": str(per_mg_path),
        }
        write_json(reports / "mlm_reconstruction_metrics.json", out)
        write_json(reports / "a8_2_selection_exact_match.json", a8_2)
        if bank_query:
            write_json(reports / "bank_query_record_reconstruction_eval.json", bank_query)
            write_json(
                reports / "bank_query_set_manifest_reconstruction_eval.json",
                bank_query.get("query_set_manifest") or {},
            )
        # Do NOT rewrite selection_manifest_final.json
        print(json.dumps(out, indent=2), flush=True)
        exit_code = 0 if a8_2["a8_2_pass"] else 1
        return exit_code
    finally:
        dump_runtime_imports_from_env(ROOT, entrypoint=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
