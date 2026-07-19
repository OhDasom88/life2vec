#!/usr/bin/env python3
"""Development3-only production evidence runner. No cohort argument. Fail-closed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_production_hooks(cfg: Mapping[str, Any]):
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_orchestrator import CaseRuntimeHooks
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_runtime import (
        build_stage_a_reencoder,
        discover_critic_checkpoints,
        load_case_events,
        parse_case_id,
    )
    from src.online2.v2.finetune_v03.counterfactual.evaluation.case_batch import (
        load_embedding_sidecar,
    )

    emb_dir = Path(cfg["embeddings_dir"])
    # Cache one reencoder per case (case_t0 differs).
    _reenc_cache: Dict[str, Any] = {}
    _active_case = {"id": None}
    _trace = {"writer": None}

    def bind_global_trace(writer) -> None:
        _trace["writer"] = writer

    def list_checkpoint_paths() -> Sequence[Path]:
        return discover_critic_checkpoints(cfg)

    def _load_events(case_id: str) -> list:
        return load_case_events(cfg, case_id)

    def load_sidecar_by_event_id(case_id: str) -> Mapping[str, Mapping[str, Any]]:
        side = load_embedding_sidecar(emb_dir, case_id)
        out: Dict[str, Dict[str, Any]] = {}
        for row in side.itertuples(index=False):
            d = row._asdict() if hasattr(row, "_asdict") else dict(zip(side.columns, row))
            eid = str(d.get("event_id"))
            out[eid] = {
                "view": d.get("view", "UNKNOWN"),
                "zone": d.get("zone", 0),
                "case_age_hours": float(d.get("case_age_hours") or 0.0),
                "local_hour": int(float(d.get("local_hour") or 0)) % 24,
            }
        return out

    def build_reencoder():
        cid = _active_case["id"]
        if cid is None:
            raise RuntimeError("active case unset before reencoder build")
        if cid not in _reenc_cache:
            _reenc_cache[cid] = build_stage_a_reencoder(
                cfg,
                case_id=cid,
                global_trace=_trace["writer"],
            )
        return _reenc_cache[cid]

    def load_case_events_wrapped(case_id: str) -> list:
        _active_case["id"] = case_id
        events = _load_events(case_id)
        # Ensure sentence_tokens attribute for Core edit path
        for ev in events:
            if not getattr(ev, "sentence_tokens", None):
                toks = list(getattr(ev, "tokens", None) or [])
                ev.sentence_tokens = toks
                ev.token_count = len(toks)
        return events

    def build_candidates(case_id: str) -> List[Mapping[str, Any]]:
        from src.online2.v2.finetune_v03.counterfactual.cf1s.core_candidates import (
            build_multievent_candidates_for_case,
        )

        _active_case["id"] = case_id
        events = load_case_events_wrapped(case_id)
        return build_multievent_candidates_for_case(
            cfg,
            case_id=case_id,
            events=events,
            max_two_event=min(
                2,
                int((cfg.get("cf1s_core") or {}).get("max_two_event_families_per_case") or 2),
            ),
            top_k_events=int((cfg.get("attribution") or {}).get("event_preselect_top_k") or 8),
            global_trace=_trace["writer"],
        )

    def score_identity_transaction(case_id: str):
        from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
            canonical_json_sha256,
        )
        from src.online2.v2.finetune_v03.counterfactual.cf1s.core_raw_transaction import (
            build_identity_transaction,
        )

        _active_case["id"] = case_id
        events = load_case_events_wrapped(case_id)
        payload = []
        for ev in events:
            tokens = list(
                getattr(ev, "sentence_tokens", None)
                or getattr(ev, "tokens", None)
                or []
            )
            payload.append(
                {
                    "event_id": str(getattr(ev, "event_id", "")),
                    "tokens": tokens,
                }
            )
        if not any(p["tokens"] for p in payload):
            raise RuntimeError(f"no tokenized events for identity on {case_id}")
        return build_identity_transaction(
            case_id=case_id,
            original_caseevents_sha=canonical_json_sha256(payload),
        )

    return CaseRuntimeHooks(
        load_case_events=load_case_events_wrapped,
        load_sidecar_by_event_id=load_sidecar_by_event_id,
        build_reencoder=build_reencoder,
        list_checkpoint_paths=list_checkpoint_paths,
        fold_ids=[0, 1, 2],
        build_candidates=build_candidates,
        score_identity_transaction=score_identity_transaction,
        bind_global_trace=bind_global_trace,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf1s_core_smoke.yaml")
    parser.add_argument(
        "--authorization",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trust-root",
        type=Path,
        default=ROOT / "conf/m1/cf1s_policies/CF1S_DEVELOPMENT_TRUST_ROOT_V1.json",
    )
    parser.add_argument(
        "--fold-routing",
        type=Path,
        default=ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_FOLD_ROUTING_V1.json",
    )
    parser.add_argument(
        "--dry-authority-check",
        action="store_true",
        help="Validate authorization gates only; do not create output directories.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    auth_path = (
        args.authorization if args.authorization.is_absolute() else ROOT / args.authorization
    )
    trust_path = args.trust_root if args.trust_root.is_absolute() else ROOT / args.trust_root

    if not auth_path.exists():
        raise SystemExit(
            "development production authorization artifact missing; refusing to create outputs"
        )

    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    if auth.get("artifact_kind") != "PRE_EXECUTION_AUTHORIZATION":
        raise SystemExit("authorization must be signed PRE_EXECUTION_AUTHORIZATION")
    if not auth.get("development_production_execution_authorized"):
        raise SystemExit("development_production_execution_authorized=false")
    if not auth.get("evidence_output_authorized"):
        raise SystemExit("evidence_output_authorized=false")
    if auth.get("recommendation_authorized"):
        raise SystemExit("recommendation must remain unauthorized for this runner")
    if auth.get("primary32_execution_authorized") or auth.get("problem20_execution_authorized"):
        raise SystemExit(
            "Primary32/Problem20 must not be authorized by development smoke artifact"
        )

    cfg = _load_yaml(cfg_path)
    core = cfg.get("cf1s_core") or {}
    if core.get("primary32", {}).get("execution_allowed") or core.get("primary32_allowed"):
        raise SystemExit("policy primary32 execution must remain false")
    if core.get("problem20", {}).get("execution_allowed") or core.get("problem20_allowed"):
        raise SystemExit("policy problem20 execution must remain false")

    man = json.loads((ROOT / str(cfg["development_manifest_path"])).read_text(encoding="utf-8"))
    case_ids = list(man["ordered_case_ids"])
    if len(case_ids) != 3:
        raise SystemExit("Development3 must have exactly 3 cases")

    if args.dry_authority_check:
        from src.online2.v2.finetune_v03.counterfactual.cf1s.core_authorization import (
            consume_pre_execution_authorization,
            load_trust_root,
            verify_trust_root_env,
        )
        from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
            recompute_stable_lock_manifest,
        )

        tr_sha = verify_trust_root_env(trust_path)
        observed_stable = recompute_stable_lock_manifest(
            auth.get("stable_lock_manifest") or {},
            root=ROOT,
        )
        consume_pre_execution_authorization(
            auth,
            trust_root=load_trust_root(trust_path),
            trust_root_sha256=tr_sha,
            observed_stable_lock=observed_stable,
            expected_run_id=str(auth.get("run_id") or ""),
        )
        print(
            json.dumps(
                {
                    "status": "AUTHORITY_GATE_PASS",
                    "cohort": "development3",
                    "case_ids": case_ids,
                    "authorization": str(auth_path),
                    "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
                },
                indent=2,
            )
        )
        return 0

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_orchestrator import (
        run_development3_orchestrator,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_runtime import (
        apply_exact_byte_deterministic_runtime,
    )

    # Lock determinism before any CUDA work in hooks.
    apply_exact_byte_deterministic_runtime(before_cuda_init=True)

    out_root = Path(args.out) if args.out else Path(cfg.get("output_root") or (ROOT / "outputs/cf1s_core"))
    hooks = _build_production_hooks(cfg)
    result = run_development3_orchestrator(
        config=cfg,
        authorization_artifact=auth,
        trust_root_path=trust_path,
        fold_routing_path=args.fold_routing
        if args.fold_routing.is_absolute()
        else ROOT / args.fold_routing,
        hooks=hooks,
        out_root=out_root,
        signing_private_key_hex=os.environ.get("CF1S_DEVELOPMENT_SIGNING_KEY_HEX"),
    )
    print(
        json.dumps(
            {
                "status": (
                    "FINAL_SUCCESS"
                    if all(value == "PASS" for value in result.gate_results.values())
                    else "EXECUTION_FINISHED_NOT_FINAL"
                ),
                "package_dir": str(result.package_dir),
                "package_state": result.package_state,
                "development_execution_readiness": result.execution_readiness,
                "computed_evaluation_coverage": result.evaluation_coverage,
                "attested_evaluation_coverage": result.attested_evaluation_coverage,
                "development_scientific_result": result.scientific_counts,
                "gate_results": result.gate_results,
                "report": result.report_lines,
                "execution_scope": "TWO_EVENT_ONLY",
                "three_event_execution_status": "OUT_OF_SCOPE",
            },
            indent=2,
        )
    )
    return 0 if all(value == "PASS" for value in result.gate_results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
