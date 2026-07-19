"""CF-1S Core preflight — does not authorize readiness or smoke by itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .core_contract import (
    CoreContractError,
    apply_runtime_lock,
    assert_core_import_graph_clean,
    build_legacy_path_manifest,
    deprecate_legacy_readiness,
    inspect_runtime_lock,
    required_observed_gate,
    sha256_file,
)
from .production_preflight import check_fold_partition, resolve_prediction_cutoff_time

CORE_MODULE_RELPATHS = [
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_contract.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_preflight.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_acceptance.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_identity.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_candidate_family.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_manifest.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_execution.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_edit_proposal.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_transaction.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_output.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/core_production.py",
    "src/online2/v2/finetune_v03/counterfactual/pipeline_cf1s_core.py",
    "scripts/online2_v2/v03/run_cf1s_core_v03.py",
    "scripts/online2_v2/v03/package_cf1s_core_v03.py",
    "scripts/online2_v2/v03/promote_cf1s_primary32_lock.py",
    "scripts/online2_v2/v03/authorize_cf1s_development_smoke.py",
    "scripts/online2_v2/v03/run_cf1s_core_development_production_v03.py",
]
REQUIRED_MEASURED_CHECKS = {
    "cold_rebuild_equivalence",
    "official_batch_field_parity",
    "stage_a_critic_pairing",
    "fold_isolation",
    "deterministic_runtime",
    "risk_logit_contract",
}


def validate_measured_preflight_contract(
    checks: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    normalized = [dict(check) for check in checks]
    names = {str(check.get("check_name")) for check in normalized}
    missing = sorted(REQUIRED_MEASURED_CHECKS - names)
    if missing:
        raise CoreContractError(
            "preflight measured checks missing: " + ",".join(missing)
        )
    for check in normalized:
        for field in (
            "check_name",
            "target_sha256",
            "observations",
            "tolerance",
            "status",
            "producer_code_sha256",
        ):
            if check.get(field) is None:
                raise CoreContractError(
                    f"preflight {check.get('check_name')} missing {field}"
                )
        if check["status"] != "PASS":
            raise CoreContractError(
                f"preflight measured check failed: {check['check_name']}"
            )
    return normalized


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cohort_manifests(
    *,
    root: Path,
    development_path: Path,
    primary32_path: Path,
    problem20_path: Path,
) -> Dict[str, Any]:
    dev = _load_json(development_path)
    pri = _load_json(primary32_path)
    prob = _load_json(problem20_path)
    d_ids = list(dev["ordered_case_ids"])
    p_ids = list(pri["ordered_case_ids"])
    b_ids = list(prob["ordered_case_ids"])
    errors = []
    if len(d_ids) != 3:
        errors.append(f"development count {len(d_ids)} != 3")
    if len(p_ids) != 32:
        errors.append(f"primary32 count {len(p_ids)} != 32")
    if len(b_ids) != 20:
        errors.append(f"problem20 count {len(b_ids)} != 20")
    if set(d_ids) & set(p_ids) or set(d_ids) & set(b_ids) or set(p_ids) & set(b_ids):
        errors.append("cohort manifests are not pairwise disjoint")
    if len(set(d_ids) | set(p_ids)) != 35:
        errors.append("development+primary must equal Example35")
    if prob.get("ground_truth_value_in_runtime_manifest") is not False:
        errors.append("problem20 runtime manifest must not carry ground truth values")
    for case in prob.get("cases") or []:
        if "label" in case or "diagnosis_normalized" in case:
            errors.append(f"problem20 runtime case leaked label: {case.get('case_id')}")
    if errors:
        raise CoreContractError("; ".join(errors))
    return {
        "ok": True,
        "development_case_count": 3,
        "primary32_case_count": 32,
        "problem20_case_count": 20,
        "pairwise_disjoint": True,
        "development_plus_primary_equals_example35": True,
        "problem20_ground_truth_in_runtime_manifest": False,
        "manifest_sha256": {
            "development": sha256_file(development_path),
            "primary32": sha256_file(primary32_path),
            "problem20": sha256_file(problem20_path),
        },
    }


def core_preflight(
    cfg: Mapping[str, Any],
    *,
    root: Path,
    apply_runtime: bool = False,
) -> Dict[str, Any]:
    root = Path(root)
    path_keys = [
        "run_dir",
        "embeddings_dir",
        "labels_path",
        "label_map_path",
        "stage_a_ckpt",
        "abspos_reference_path",
        "tokenizer_path",
        "vocabulary_path",
        "feature_schema_path",
        "binning_registry_path",
        "events_path",
        "cells_path",
    ]
    missing = []
    path_hashes = {}
    for key in path_keys:
        raw = cfg.get(key)
        if not raw:
            missing.append(key)
            continue
        p = Path(str(raw))
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            missing.append(f"{key}:{p}")
        elif p.is_file():
            path_hashes[key] = sha256_file(p)

    critic = cfg.get("critic") or {}
    folds = check_fold_partition(
        search_fold_ids=list(critic.get("search_fold_ids") or []),
        holdout_fold_ids=list(critic.get("holdout_fold_ids") or []),
    )

    dev_m = root / str(cfg.get("development_manifest_path"))
    pri_m = root / str(cfg.get("primary32_manifest_path"))
    prob_m = root / str(cfg.get("problem20_manifest_path"))
    cohorts = validate_cohort_manifests(
        root=root,
        development_path=dev_m,
        primary32_path=pri_m,
        problem20_path=prob_m,
    )

    core_paths = [root / rel for rel in CORE_MODULE_RELPATHS if (root / rel).exists()]
    import_graph = assert_core_import_graph_clean(core_paths)

    legacy = build_legacy_path_manifest(root=root)
    legacy_path = root / "outputs/cf1s/CF1S_LEGACY_PATH_MANIFEST.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(legacy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    deprecate_legacy_readiness(root / "outputs/cf1s/CF1S_PRODUCTION_READINESS.json")

    if apply_runtime:
        runtime = apply_runtime_lock(seed=0)
    else:
        runtime = inspect_runtime_lock()

    acceptance_path = root / str(cfg.get("acceptance_policy_path"))
    measured_checks = validate_measured_preflight_contract(
        list(cfg.get("qualification_observations") or [])
    )
    ok = (
        not missing
        and folds.get("fold_partition_disjoint")
        and folds.get("fold_partition_complete")
        and cohorts.get("ok")
        and import_graph.get("ok")
        and bool(measured_checks)
    )
    result = {
        "artifact": "CF1S_CORE_PREFLIGHT",
        "ok": ok,
        "is_readiness": False,
        "development_smoke_allowed": False,
        "primary32_allowed": False,
        "problem20_allowed": False,
        "missing_paths": missing,
        "path_sha256": path_hashes,
        "folds": folds,
        "cohorts": cohorts,
        "import_graph": import_graph,
        "legacy_manifest_path": str(legacy_path),
        "legacy_status": "FROZEN_REFERENCE_ONLY",
        "acceptance_policy_sha256": sha256_file(acceptance_path) if acceptance_path.exists() else None,
        "runtime_lock": runtime,
        "checks": measured_checks,
        "deterministic_runtime_state_verified": required_observed_gate(
            required_value=True,
            observed_value=bool(runtime.get("lock_satisfied")) if apply_runtime else None,
            evidence_artifact="CF1S_CORE_PREFLIGHT.json" if apply_runtime else None,
        ),
        "cold_rebuild_equivalence_test_pass": required_observed_gate(required_value=True),
    }
    if not ok:
        raise CoreContractError(f"core preflight failed: missing={missing} folds={folds}")
    return result


def write_core_preflight(result: Mapping[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
