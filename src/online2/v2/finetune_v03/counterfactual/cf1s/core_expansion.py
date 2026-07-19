"""Fail-closed Development3 → Validation20 → Primary32 expansion contracts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError
from .core_verifier import validate_completion_projection


def require_all_pass_final(
    verdict: Mapping[str, Any],
    *,
    expected_run_id: str | None = None,
) -> None:
    if verdict.get("verdict_kind") != "CF1S_FINAL_VERIFIER_VERDICT_V1":
        raise CoreContractError("expansion requires canonical FINAL verifier verdict")
    if expected_run_id is not None and verdict.get("run_id") != expected_run_id:
        raise CoreContractError("expansion source run_id mismatch")
    gates = dict(verdict.get("gate_results") or {})
    if set(gates) != {f"G{index}" for index in range(1, 13)}:
        raise CoreContractError("expansion source gate map incomplete")
    if any(value != "PASS" for value in gates.values()):
        raise CoreContractError("expansion source has non-PASS gate")
    if not verdict.get("final_report_allowed"):
        raise CoreContractError("expansion source final_report_allowed=false")


def authorize_validation20(
    *,
    development_final_verdict: Mapping[str, Any],
    development_completion: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
    problem20_manifest_sha256: str,
) -> Dict[str, Any]:
    require_all_pass_final(development_final_verdict)
    validate_completion_projection(
        development_completion,
        development_final_verdict,
    )
    case_ids = list(validation_manifest.get("ordered_case_ids") or [])
    if len(case_ids) != 20 or len(set(case_ids)) != 20:
        raise CoreContractError("Validation20 requires exactly 20 unique cases")
    if (
        validation_manifest.get("source_problem20_manifest_sha256")
        != problem20_manifest_sha256
    ):
        raise CoreContractError("Validation20 Problem20 provenance mismatch")
    if not validation_manifest.get("selection_rule"):
        raise CoreContractError("Validation20 selection_rule missing")
    cases = list(validation_manifest.get("cases") or [])
    by_id = {str(case.get("case_id")): case for case in cases}
    for case_id in case_ids:
        row = by_id.get(str(case_id)) or {}
        value = row.get("input_artifact_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise CoreContractError(
                f"Validation20 input_artifact_sha256 missing: {case_id}"
            )
    body = {
        "artifact_kind": "CF1S_VALIDATION20_EXECUTION_AUTHORIZATION_V1",
        "scope": ["VALIDATION20", "TWO_EVENT_ONLY"],
        "source_development_final_verdict_sha256": development_final_verdict[
            "verdict_sha256"
        ],
        "source_development_completion_sha256": development_completion[
            "completion_sha256"
        ],
        "validation_manifest_sha256": canonical_json_sha256(validation_manifest),
        "source_problem20_manifest_sha256": problem20_manifest_sha256,
        "threshold_change_authorized": False,
        "primary32_execution_authorized": False,
        "recommendation_authorized": False,
        "validation20_execution_authorized": True,
    }
    body["authorization_sha256"] = canonical_json_sha256(body)
    return body


def authorize_primary32(
    *,
    development_final_verdict: Mapping[str, Any],
    development_completion: Mapping[str, Any],
    validation20_final_verdict: Mapping[str, Any],
    validation20_completion: Mapping[str, Any],
    primary32_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    require_all_pass_final(development_final_verdict)
    validate_completion_projection(
        development_completion,
        development_final_verdict,
    )
    require_all_pass_final(validation20_final_verdict)
    validate_completion_projection(
        validation20_completion,
        validation20_final_verdict,
    )
    case_ids = list(primary32_manifest.get("ordered_case_ids") or [])
    if len(case_ids) != 32 or len(set(case_ids)) != 32:
        raise CoreContractError("Primary32 requires exactly 32 unique cases")
    body = {
        "artifact_kind": "CF1S_PRIMARY32_EXECUTION_AUTHORIZATION_V1",
        "scope": ["PRIMARY32", "TWO_EVENT_ONLY"],
        "source_development_final_verdict_sha256": development_final_verdict[
            "verdict_sha256"
        ],
        "source_validation20_final_verdict_sha256": validation20_final_verdict[
            "verdict_sha256"
        ],
        "primary32_manifest_sha256": canonical_json_sha256(primary32_manifest),
        "threshold_change_authorized": False,
        "primary32_execution_authorized": True,
        "recommendation_authorized": False,
    }
    body["authorization_sha256"] = canonical_json_sha256(body)
    return body


def build_train35_reference_report(
    *,
    development3_evidence_root_sha256: str,
    primary32_evidence_root_sha256: str,
    development3_case_ids: Sequence[str],
    primary32_case_ids: Sequence[str],
) -> Dict[str, Any]:
    development_ids = list(development3_case_ids)
    primary_ids = list(primary32_case_ids)
    if len(development_ids) != 3 or len(primary_ids) != 32:
        raise CoreContractError("Train35 requires Development3 + Primary32")
    if set(development_ids) & set(primary_ids):
        raise CoreContractError("Train35 source cohorts overlap")
    report = {
        "artifact_kind": "CF1S_TRAIN35_REFERENCE_AGGREGATE_V1",
        "reporting_set": "TRAIN35",
        "case_count": 35,
        "source_packages": [
            {
                "cohort": "Development3",
                "evidence_root_sha256": development3_evidence_root_sha256,
                "case_ids": development_ids,
            },
            {
                "cohort": "Primary32",
                "evidence_root_sha256": primary32_evidence_root_sha256,
                "case_ids": primary_ids,
            },
        ],
        "copies_source_evidence": False,
        "creates_new_case_evidence_root": False,
    }
    report["aggregate_report_sha256"] = canonical_json_sha256(report)
    return report
