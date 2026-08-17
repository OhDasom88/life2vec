#!/usr/bin/env python3
"""Measure real Search-baseline (fold 0,1) repeat-noise and identity-reconstruction
error for Development3, and lock the cohort threshold/calibration artifact from it.

This is a measurement-only pre-step: no candidate search, no selection, no
promotion, no official-result claims. Its output binds calibration_artifact_sha256
into the PRE-stage full stable lock (a separate, later step)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _abnormal_logit(row: Mapping[str, Any], check_risk_logit_contract) -> float:
    logits = list(row.get("logit") or [])
    if len(logits) != 1:
        raise RuntimeError("abnormal binary logit must contain exactly one value")
    value = float(logits[0])
    check_risk_logit_contract(float(row["risk"]), value)
    return value


def main() -> int:
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
        canonical_json_sha256,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
        CoreContractError,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_production import (
        ForwardCounters,
        ForwardKind,
        lock_thresholds_from_search_baseline_noise,
        make_production_fold_forward_fn,
        run_cold_forward_pipeline,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_raw_transaction import (
        build_identity_transaction,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_runtime import (
        apply_exact_byte_deterministic_runtime,
        build_stage_a_reencoder,
        case_fold_provenance,
        discover_critic_checkpoints,
        load_case_events,
        load_fold_routing_manifest,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_scientific import (
        check_risk_logit_contract,
        max_search_identity_reconstruction_error,
        runtime_repeat_noise,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_trace import (
        GlobalForwardTrace,
    )
    from src.online2.v2.finetune_v03.counterfactual.evaluation.case_batch import (
        load_embedding_sidecar,
    )

    cfg_path = ROOT / "conf/m1/cf1s_core_smoke.yaml"
    cfg = _load_yaml(cfg_path)
    fold_routing_path = ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_FOLD_ROUTING_V1.json"
    man_path = ROOT / str(cfg["development_manifest_path"])
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    case_ids = list(manifest["ordered_case_ids"])
    if len(case_ids) != 3:
        raise CoreContractError("Development3 requires exactly 3 cases")
    routing = load_fold_routing_manifest(fold_routing_path)

    apply_exact_byte_deterministic_runtime(before_cuda_init=True)

    emb_dir = Path(cfg["embeddings_dir"])
    ckpts = discover_critic_checkpoints(cfg)
    global_trace = GlobalForwardTrace()
    fold_forward = make_production_fold_forward_fn(
        normal_class_id=0,
        gpu_fraction=float(cfg.get("gpu_memory_fraction") or 0.4),
        global_trace=global_trace,
    )
    counters = ForwardCounters()

    per_case_measurements: List[Dict[str, Any]] = []
    all_runtime_noise: List[float] = []
    all_identity_err: List[float] = []

    for case_id in case_ids:
        case_entry = (routing.get("cases") or {}).get(case_id) or {}
        prov = case_fold_provenance(case_entry)
        search_folds = list(prov["search_checkpoint_ids"])

        events = load_case_events(cfg, case_id)
        for ev in events:
            if not getattr(ev, "sentence_tokens", None):
                toks = list(getattr(ev, "tokens", None) or [])
                ev.sentence_tokens = toks
                ev.token_count = len(toks)
        side = load_embedding_sidecar(emb_dir, case_id)
        sidecar: Dict[str, Dict[str, Any]] = {}
        for row in side.itertuples(index=False):
            d = row._asdict() if hasattr(row, "_asdict") else dict(zip(side.columns, row))
            eid = str(d.get("event_id"))
            sidecar[eid] = {
                "view": d.get("view", "UNKNOWN"),
                "zone": d.get("zone", 0),
                "case_age_hours": float(d.get("case_age_hours") or 0.0),
                "local_hour": int(float(d.get("local_hour") or 0)) % 24,
            }
        reencoder = build_stage_a_reencoder(cfg, case_id=case_id, global_trace=global_trace)

        payload = []
        for ev in events:
            tokens = list(
                getattr(ev, "sentence_tokens", None) or getattr(ev, "tokens", None) or []
            )
            payload.append({"event_id": str(getattr(ev, "event_id", "")), "tokens": tokens})
        if not any(p["tokens"] for p in payload):
            raise RuntimeError(f"no tokenized events for identity on {case_id}")
        identity_tx = build_identity_transaction(
            case_id=case_id,
            original_caseevents_sha=canonical_json_sha256(payload),
        )

        baseline_risks_by_fold: Dict[str, List[float]] = {str(f): [] for f in search_folds}
        for repeat in range(3):
            base = run_cold_forward_pipeline(
                events=events,
                validated_transaction=identity_tx,
                reencoder=reencoder,
                fold_forward_fn=fold_forward,
                checkpoint_paths=ckpts,
                fold_ids=[0, 1, 2],
                requested_fold_ids=search_folds,
                sidecar_by_event_id=sidecar,
                scope="search",
                transaction_mode="FORCED_IDENTITY",
                forward_kind=ForwardKind.BASELINE_FORWARD,
                counters=counters,
                global_trace=global_trace,
                case_id=case_id,
                candidate_id="CALIBRATION_BASELINE",
                phase="SEARCH_BASELINE_CALIBRATION",
                invocation_nonce=f"calib-{repeat}",
            )
            for fid, row in base["scores"]["by_fold"].items():
                baseline_risks_by_fold[str(fid)].append(float(row["risk"]))

        ident = run_cold_forward_pipeline(
            events=events,
            validated_transaction=identity_tx,
            reencoder=reencoder,
            fold_forward_fn=fold_forward,
            checkpoint_paths=ckpts,
            fold_ids=[0, 1, 2],
            requested_fold_ids=search_folds,
            sidecar_by_event_id=sidecar,
            scope="search",
            transaction_mode="FORCED_IDENTITY",
            forward_kind=ForwardKind.FORCED_IDENTITY_FORWARD,
            counters=counters,
            global_trace=global_trace,
            case_id=case_id,
            candidate_id="CALIBRATION_IDENTITY",
            phase="SEARCH_BASELINE_CALIBRATION",
        )
        identity_risk_by_fold = {
            str(fid): float(row["risk"]) for fid, row in ident["scores"]["by_fold"].items()
        }
        baseline_med_by_fold = {
            fid: sorted(vals)[len(vals) // 2] for fid, vals in baseline_risks_by_fold.items()
        }
        case_runtime_noise = max(
            (runtime_repeat_noise(vals) for vals in baseline_risks_by_fold.values()),
            default=0.0,
        )
        case_identity_err = max_search_identity_reconstruction_error(
            identity_by_fold=identity_risk_by_fold,
            baseline_by_fold=baseline_med_by_fold,
        )
        all_runtime_noise.append(case_runtime_noise)
        all_identity_err.append(case_identity_err)
        per_case_measurements.append(
            {
                "case_id": case_id,
                "baseline_risks_by_fold": baseline_risks_by_fold,
                "identity_risk_by_fold": identity_risk_by_fold,
                "baseline_median_by_fold": baseline_med_by_fold,
                "runtime_repeat_noise": case_runtime_noise,
                "identity_reconstruction_error": case_identity_err,
            }
        )

    cohort_runtime_noise = max(all_runtime_noise)
    cohort_identity_err = max(all_identity_err)
    threshold_lock = lock_thresholds_from_search_baseline_noise(
        cohort_runtime_noise,
        max_search_identity_reconstruction_error=cohort_identity_err,
    )

    trace_summary = global_trace.summarize()
    artifact = {
        "artifact_kind": "CF1S_DEVELOPMENT3_SEARCH_BASELINE_CALIBRATION_V1",
        "cohort": "DEVELOPMENT3",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "case_ids": case_ids,
        "per_case_measurements": per_case_measurements,
        "cohort_max_search_baseline_pure_repeat_risk_noise": cohort_runtime_noise,
        "cohort_max_search_identity_reconstruction_error": cohort_identity_err,
        "threshold_lock": threshold_lock,
        "trace_event_count": trace_summary["trace_event_count"],
        "global_trace_chain_root_sha": trace_summary["global_trace_chain_root_sha"],
        "official_result": False,
        "measurement_only": True,
    }
    artifact["calibration_artifact_sha256"] = canonical_json_sha256(artifact)

    out_dir = ROOT / "outputs/cf1s_core/calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"CF1S_DEVELOPMENT3_SEARCH_BASELINE_CALIBRATION_{ts}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(artifact) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out_dir)

    trace_out = out_dir / f"trace_events_{ts}.json"
    trace_tmp = trace_out.with_suffix(".json.tmp")
    trace_tmp.write_text(
        canonical_json_dumps({"events": global_trace.events}) + "\n", encoding="utf-8"
    )
    fsync_file(trace_tmp)
    rename_noreplace(trace_tmp, trace_out)
    fsync_dir(out_dir)

    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "out": str(out),
                "calibration_artifact_sha256": artifact["calibration_artifact_sha256"],
                "cohort_max_search_baseline_pure_repeat_risk_noise": cohort_runtime_noise,
                "cohort_max_search_identity_reconstruction_error": cohort_identity_err,
                "threshold_lock": threshold_lock,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
