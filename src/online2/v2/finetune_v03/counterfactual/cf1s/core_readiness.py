"""Evidence-based Core readiness — never hardcode observed gate values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .core_acceptance import development_readiness_status, evaluate_cohort_status
from .core_contract import (
    CoreContractError,
    gate_observed_pass,
    required_observed_gate,
    sha256_file,
    sha256_json,
)


def build_development_smoke_readiness(
    *,
    execution_status: str,
    case_scientific_statuses: Sequence[str],
    noise_ceiling_pass: bool,
    preflight_path: Path,
    test_log_path: Optional[Path],
    package_dir: Path,
    code_sha256: str,
    policy_sha256: str,
    cold_rebuild_equivalence: Mapping[str, Any],
    runtime_verified: Mapping[str, Any],
) -> Dict[str, Any]:
    """Issue readiness only from recomputed evidence; does not authorize Primary32."""
    status = development_readiness_status(
        execution_status=execution_status,
        case_statuses=case_scientific_statuses,
        noise_ceiling_pass=noise_ceiling_pass,
    )
    preflight_sha = sha256_file(Path(preflight_path)) if Path(preflight_path).exists() else None
    test_sha = sha256_file(Path(test_log_path)) if test_log_path and Path(test_log_path).exists() else None
    cold_gate = dict(cold_rebuild_equivalence)
    runtime_gate = dict(runtime_verified)
    evidence_ok = (
        status["development_readiness_status"] == "READY"
        and gate_observed_pass(cold_gate)
        and gate_observed_pass(runtime_gate)
        and preflight_sha is not None
        and test_sha is not None
    )
    payload = {
        "artifact": "CF1S_CORE_DEVELOPMENT_SMOKE_READINESS",
        **status,
        "hardcoded_invariant_count": 0,
        "primary32_allowed": False,
        "problem20_allowed": False,
        "final_primary_lock": False,
        "preflight_sha256": preflight_sha,
        "test_log_sha256": test_sha,
        "code_sha256": code_sha256,
        "policy_sha256": policy_sha256,
        "package_dir": str(package_dir),
        "cold_rebuild_equivalence_test_pass": cold_gate,
        "deterministic_runtime_state_verified": runtime_gate,
        "readiness_issued": bool(evidence_ok),
    }
    if not evidence_ok:
        payload["development_readiness_status"] = "BLOCKED"
        payload["readiness_issued"] = False
    return payload


def build_primary32_lock_readiness_candidate(
    *,
    development_package_dir: Path,
    expected_code_sha: str,
    expected_policy_sha: str,
    observed_code_sha: str,
    observed_policy_sha: str,
    development_readiness: Mapping[str, Any],
    complete_lock_dependency_closure: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create Primary32 lock readiness only via explicit promotion command."""
    errors = []
    if development_readiness.get("development_readiness_status") != "READY":
        errors.append("development_readiness_not_ready")
    if development_readiness.get("primary32_allowed") is True:
        errors.append("development_readiness_must_not_auto_allow_primary32")
    if observed_code_sha != expected_code_sha:
        errors.append("code_sha_mismatch")
    if observed_policy_sha != expected_policy_sha:
        errors.append("policy_sha_mismatch")
    if not complete_lock_dependency_closure.get("complete_lock_dependency_closure"):
        errors.append("incomplete_dependency_closure")
    ok = not errors
    return {
        "artifact": "CF1S_PRIMARY32_LOCK_READINESS",
        "ok": ok,
        "errors": errors,
        "primary32_allowed": False,  # this stage never auto-enables; promotion sets explicit flag separately
        "promotion_verified": ok,
        "development_package_dir": str(development_package_dir),
        "expected_code_sha": expected_code_sha,
        "expected_policy_sha": expected_policy_sha,
        "observed_code_sha": observed_code_sha,
        "observed_policy_sha": observed_policy_sha,
        "complete_lock_dependency_closure": complete_lock_dependency_closure,
        # Explicit promotion must flip this in promote script after verification.
        "primary32_execution_authorized": False,
    }


def assert_primary32_authorized(lock_path: Path) -> Dict[str, Any]:
    if not Path(lock_path).exists():
        raise CoreContractError("Primary32 lock readiness missing")
    doc = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if doc.get("artifact_kind") != "CF1S_PRIMARY32_EXECUTION_AUTHORIZATION_V1":
        raise CoreContractError("legacy Primary32 readiness artifact is not authoritative")
    if doc.get("scope") != ["PRIMARY32", "TWO_EVENT_ONLY"]:
        raise CoreContractError("Primary32 authorization scope mismatch")
    if not doc.get("primary32_execution_authorized"):
        raise CoreContractError("Primary32 not authorized; run promote_cf1s_primary32_lock.py")
    if doc.get("threshold_change_authorized"):
        raise CoreContractError("Primary32 authorization must not permit threshold changes")
    if not doc.get("source_development_final_verdict_sha256"):
        raise CoreContractError("Primary32 authorization missing Development3 FINAL link")
    if not doc.get("source_validation20_final_verdict_sha256"):
        raise CoreContractError("Primary32 authorization missing Validation20 FINAL link")
    return doc


def assert_problem20_authorized(marker_path: Path) -> Dict[str, Any]:
    if not Path(marker_path).exists():
        raise CoreContractError("Problem20 authorization marker missing")
    doc = json.loads(Path(marker_path).read_text(encoding="utf-8"))
    if not doc.get("problem20_execution_authorized"):
        raise CoreContractError("Problem20 not authorized")
    return doc
