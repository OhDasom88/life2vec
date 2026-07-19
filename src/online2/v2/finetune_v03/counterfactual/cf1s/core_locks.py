"""Stable lock vs runtime observation; public-key-only POST verification."""

from __future__ import annotations

import os
import platform
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .core_authorization import (
    POST_EXECUTION_KIND,
    load_trust_root,
    payload_sha256,
    tree_manifest_sha,
    verify_signature,
)
from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError, sha256_file


REPO_ROOT = Path("/home/dasom/life2vec")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_STABLE_LOCK_SHA_FIELDS = (
    "code_tree_sha256",
    "test_tree_sha256",
    "policy_tree_sha256",
    "qualification_test_log_sha256",
    "qualification_preflight_sha256",
    "qualification_node_id_manifest_sha256",
    "development_manifest_sha256",
    "case_input_manifest_root_sha256",
    "fold_routing_sha256",
    "stage_a_checkpoint_manifest_sha256",
    "critic_checkpoint_manifest_sha256",
    "tokenizer_sha256",
    "vocab_sha256",
    "feature_schema_sha256",
    "binning_registry_sha256",
    "label_map_sha256",
    "deterministic_runtime_policy_sha256",
    "risk_head_contract_sha256",
    "calibration_artifact_sha256",
)


def default_code_tree_paths(root: Path = REPO_ROOT) -> list:
    return sorted(
        list((root / "src/online2/v2/finetune_v03/counterfactual/cf1s").glob("core_*.py"))
        + [
            root / "src/online2/v2/finetune_v03/counterfactual/pipeline_cf1s_core.py",
            root / "src/online2/v2/finetune_v03/counterfactual/attribution/token_ixg_v03.py",
            root / "src/online2/v2/finetune_v03/counterfactual/attribution/token_attribution.py",
            root / "scripts/online2_v2/v03/run_cf1s_core_v03.py",
            root / "scripts/online2_v2/v03/run_cf1s_core_development_production_v03.py",
            root / "scripts/online2_v2/v03/run_cf1s_stage_gpu_smoke_v03.py",
            root / "scripts/online2_v2/v03/build_cf1s_stable_lock_v03.py",
            root / "scripts/online2_v2/v03/authorize_cf1s_development_smoke.py",
            root / "scripts/online2_v2/v03/authorize_cf1s_validation20_v03.py",
            root / "scripts/online2_v2/v03/complete_cf1s_run_v03.py",
            root / "scripts/online2_v2/v03/promote_cf1s_primary32_lock.py",
            root / "scripts/online2_v2/v03/build_cf1s_train35_report_v03.py",
        ]
    )


def default_policy_tree_paths(root: Path = REPO_ROOT) -> list:
    return [
        root / "conf/m1/cf1s_policies/CF1S_CORE_ACCEPTANCE_POLICY.yaml",
        root / "conf/m1/cf1s_core_smoke.yaml",
        root / "conf/m1/cf1s_policies/CF1S_EDIT_PROPOSAL_SCHEMA_V1.json",
        root / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_MANIFEST.json",
        root / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_FOLD_ROUTING_V1.json",
        root / "conf/m1/cf1s_policies/CF1S_DEVELOPMENT_TRUST_ROOT_V1.json",
    ]


def default_test_tree_paths(root: Path = REPO_ROOT) -> list:
    return sorted(
        (root / "tests/v2/counterfactual_cf1s").glob("test_cf1s_*.py")
    ) + [root / "tests/v2/counterfactual_cf1s/cf1s_test_tx.py"]


def compute_stable_lock_manifest(
    *,
    root: Path = REPO_ROOT,
    qualification_test_log_sha: Optional[str] = None,
    qualification_preflight_sha: Optional[str] = None,
    qualification_test_node_id_manifest_sha: Optional[str] = None,
    artifact_sha256: Optional[Mapping[str, str]] = None,
    artifact_sources: Optional[Mapping[str, str]] = None,
    fold_routing_path: Optional[Path] = None,
    development_manifest_path: Optional[Path] = None,
    runtime_versions: Optional[Mapping[str, Any]] = None,
    risk_head_contract: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Exact-comparable stable lock — excludes volatile runtime observations."""
    code_sha, _ = tree_manifest_sha(default_code_tree_paths(root), root=root, required=True)
    policy_sha, _ = tree_manifest_sha(default_policy_tree_paths(root), root=root, required=True)
    test_paths = [p for p in default_test_tree_paths(root) if p.is_file()]
    test_sha, _ = tree_manifest_sha(test_paths, root=root, required=False) if test_paths else (
        canonical_json_sha256({"files": {}}),
        {},
    )

    man_path = development_manifest_path or (
        root / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_MANIFEST.json"
    )
    routing_path = fold_routing_path or (
        root / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_FOLD_ROUTING_V1.json"
    )
    body = {
        "version": "CF1S_STABLE_LOCK_MANIFEST_V2",
        "code_tree_sha256": code_sha,
        "test_tree_sha256": test_sha,
        "policy_tree_sha256": policy_sha,
        "development_manifest_sha256": sha256_file(man_path),
        "fold_routing_sha256": sha256_file(routing_path),
        "qualification_test_log_sha256": qualification_test_log_sha,
        "qualification_preflight_sha256": qualification_preflight_sha,
        "qualification_node_id_manifest_sha256": qualification_test_node_id_manifest_sha,
        "runtime_versions": dict(runtime_versions or {}),
        "risk_head_contract": dict(risk_head_contract or {}),
        "cohort": "development3",
        "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
        "execution_scope": "TWO_EVENT_ONLY",
        "artifact_sources": dict(sorted((artifact_sources or {}).items())),
    }
    body.update(dict(artifact_sha256 or {}))
    body["deterministic_runtime_policy_sha256"] = body.get(
        "deterministic_runtime_policy_sha256"
    ) or canonical_json_sha256(dict(runtime_versions or {}))
    body["risk_head_contract_sha256"] = body.get(
        "risk_head_contract_sha256"
    ) or canonical_json_sha256(dict(risk_head_contract or {}))
    if extra:
        body["extra"] = dict(extra)
    body["stable_lock_sha256"] = canonical_json_sha256(
        {k: v for k, v in body.items() if k != "stable_lock_sha256"}
    )
    return body


def validate_stable_lock_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    body = dict(manifest)
    if body.get("version") != "CF1S_STABLE_LOCK_MANIFEST_V2":
        raise CoreContractError("stable lock version must be CF1S_STABLE_LOCK_MANIFEST_V2")
    if body.get("cohort") != "development3":
        raise CoreContractError("stable lock cohort must be development3")
    if body.get("execution_scope") != "TWO_EVENT_ONLY":
        raise CoreContractError("stable lock execution_scope must be TWO_EVENT_ONLY")
    for field_name in REQUIRED_STABLE_LOCK_SHA_FIELDS:
        value = body.get(field_name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise CoreContractError(
                f"stable lock required SHA invalid or placeholder: {field_name}"
            )
    supplied = body.get("stable_lock_sha256")
    computed = canonical_json_sha256(
        {key: value for key, value in body.items() if key != "stable_lock_sha256"}
    )
    if supplied != computed:
        raise CoreContractError("stable_lock_sha256 mismatch")
    return body


def recompute_stable_lock_manifest(
    expected: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Freshly hash all declared sources; never accept expected digest as observed."""
    observed = dict(expected)
    observed["code_tree_sha256"], _ = tree_manifest_sha(
        default_code_tree_paths(root), root=root, required=True
    )
    test_paths = [path for path in default_test_tree_paths(root) if path.is_file()]
    observed["test_tree_sha256"], _ = (
        tree_manifest_sha(test_paths, root=root, required=True)
        if test_paths
        else (canonical_json_sha256({"files": {}}), {})
    )
    observed["policy_tree_sha256"], _ = tree_manifest_sha(
        default_policy_tree_paths(root), root=root, required=True
    )
    sources = dict(expected.get("artifact_sources") or {})
    source_required = set(REQUIRED_STABLE_LOCK_SHA_FIELDS) - {
        "code_tree_sha256",
        "test_tree_sha256",
        "policy_tree_sha256",
    }
    missing_sources = sorted(source_required - set(sources))
    if missing_sources:
        raise CoreContractError(
            "stable lock artifact source missing: " + ",".join(missing_sources)
        )
    for field_name in sorted(source_required):
        raw_path = Path(str(sources[field_name]))
        path = raw_path if raw_path.is_absolute() else root / raw_path
        if not path.is_file():
            raise CoreContractError(f"stable lock source missing for {field_name}: {path}")
        observed[field_name] = sha256_file(path)
    observed["stable_lock_sha256"] = canonical_json_sha256(
        {key: value for key, value in observed.items() if key != "stable_lock_sha256"}
    )
    return validate_stable_lock_manifest(observed)


def compute_runtime_observation(
    *,
    run_id: str,
    temporary_path: Optional[str] = None,
    gpu_free_memory_bytes: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    trace_sequence: Optional[int] = None,
) -> Dict[str, Any]:
    """Volatile observations — never used in stable lock exact compare."""
    body = {
        "version": "CF1S_RUNTIME_OBSERVATION_V1",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "run_id": run_id,
        "temporary_path": temporary_path,
        "gpu_free_memory_bytes": gpu_free_memory_bytes,
        "duration_seconds": duration_seconds,
        "trace_sequence": trace_sequence,
    }
    body["runtime_observation_sha256"] = canonical_json_sha256(
        {k: v for k, v in body.items() if k != "runtime_observation_sha256"}
    )
    return body


def build_final_rerun_observation_manifest(
    *,
    qualification_node_id_manifest_path: Path,
    final_rerun_test_log_path: Path,
    final_rerun_node_id_manifest_path: Path,
    qualification_node_ids: Sequence[str],
    final_rerun_node_ids: Sequence[str],
    qualification_test_selection: Sequence[str],
    final_rerun_test_selection: Sequence[str],
    exit_code: int,
    passed_count: int,
    failed_count: int,
    executed_at: str,
    node_identity: str,
    host_identity: str,
    final_rerun_test_log_relpath: str = "final_rerun/final_rerun_test.log",
    final_rerun_node_id_manifest_relpath: str = (
        "final_rerun/final_rerun_node_id_manifest.json"
    ),
) -> Dict[str, Any]:
    """Describe immutable final-rerun snapshots included in the evidence root."""
    body = {
        "artifact_kind": "CF1S_FINAL_RERUN_OBSERVATION_MANIFEST_V1",
        "qualification_node_id_manifest_sha256": sha256_file(
            qualification_node_id_manifest_path
        ),
        "final_rerun_test_log_sha256": sha256_file(final_rerun_test_log_path),
        "final_rerun_node_id_manifest_sha256": sha256_file(
            final_rerun_node_id_manifest_path
        ),
        "qualification_node_ids": list(qualification_node_ids),
        "final_rerun_node_ids": list(final_rerun_node_ids),
        "qualification_test_selection": list(qualification_test_selection),
        "final_rerun_test_selection": list(final_rerun_test_selection),
        "final_rerun_test_log_relpath": final_rerun_test_log_relpath,
        "final_rerun_node_id_manifest_relpath": (
            final_rerun_node_id_manifest_relpath
        ),
        "exit_code": int(exit_code),
        "passed_count": int(passed_count),
        "failed_count": int(failed_count),
        "executed_at": executed_at,
        "node_identity": node_identity,
        "host_identity": host_identity,
    }
    body["final_rerun_observation_manifest_sha256"] = canonical_json_sha256(body)
    return body


def assert_stable_locks_identical(pre: Mapping[str, Any], post: Mapping[str, Any]) -> bool:
    validated_pre = validate_stable_lock_manifest(pre)
    validated_post = validate_stable_lock_manifest(post)
    pre_sha = validated_pre["stable_lock_sha256"]
    post_sha = validated_post["stable_lock_sha256"]
    if pre_sha != post_sha:
        raise CoreContractError(
            f"pre/post stable lock mismatch: pre={pre_sha} post={post_sha}"
        )
    if validated_pre != validated_post:
        raise CoreContractError("pre/post stable lock canonical bodies differ")
    return True


def verify_post_attestation_public_key_only(
    artifact: Mapping[str, Any],
    *,
    trust_root: Mapping[str, Any],
    trust_root_sha256: str,
    expected_evidence_root: str,
    expected_pre_payload_sha: str,
    expected_run_id: Optional[str] = None,
    expected_stable_lock_sha256: Optional[str] = None,
    expected_final_destination: Optional[str] = None,
    expected_final_rerun_observation_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Independent verifier — never requires signing private key."""
    if artifact.get("artifact_kind") != POST_EXECUTION_KIND:
        raise CoreContractError("not a POST_EXECUTION_EVIDENCE_ATTESTATION")
    if artifact.get("schema_version") != "CF1S_POST_ATTESTATION_V2":
        raise CoreContractError("POST schema version mismatch")
    if artifact.get("cohort_id") != "DEVELOPMENT3":
        raise CoreContractError("POST cohort mismatch")
    if artifact.get("scope") != ["DEVELOPMENT3", "TWO_EVENT_ONLY"]:
        raise CoreContractError("POST scope mismatch")
    if artifact.get("trust_root_sha256") != trust_root_sha256:
        raise CoreContractError("POST trust_root_sha256 mismatch")
    key_id = str(artifact["key_id"])
    keys = trust_root.get("keys") or {}
    if key_id not in keys or keys[key_id].get("revoked"):
        raise CoreContractError(f"POST key invalid: {key_id}")
    pub = keys[key_id]["public_key_hex"]
    psha = payload_sha256(
        artifact, exclude_keys=("signature_hex", "payload_sha256")
    )
    if psha != artifact.get("payload_sha256"):
        raise CoreContractError("POST payload_sha256 mismatch")
    if not verify_signature(
        psha, signature_hex=str(artifact["signature_hex"]), public_key_hex=pub
    ):
        raise CoreContractError("POST signature invalid")
    if artifact.get("evidence_root_sha") != expected_evidence_root:
        raise CoreContractError("POST evidence_root mismatch")
    if artifact.get("pre_execution_payload_sha256") != expected_pre_payload_sha:
        raise CoreContractError("POST pre-link mismatch")
    if not artifact.get("pre_post_lock_identical"):
        raise CoreContractError("POST pre_post_lock_identical=false")
    if expected_run_id is not None and artifact.get("run_id") != expected_run_id:
        raise CoreContractError("POST run_id mismatch")
    if expected_stable_lock_sha256 is not None:
        if artifact.get("pre_stable_lock_sha256") != expected_stable_lock_sha256:
            raise CoreContractError("POST PRE stable lock mismatch")
        if artifact.get("post_stable_lock_sha256") != expected_stable_lock_sha256:
            raise CoreContractError("POST observed stable lock mismatch")
    if (
        expected_final_destination is not None
        and artifact.get("intended_final_destination") != expected_final_destination
    ):
        raise CoreContractError("POST final destination mismatch")
    if (
        expected_final_rerun_observation_manifest_sha256 is not None
        and artifact.get("final_rerun_observation_manifest_sha256")
        != expected_final_rerun_observation_manifest_sha256
    ):
        raise CoreContractError("POST final-rerun observation link mismatch")
    return {"ok": True, "payload_sha256": psha, "public_key_only": True}


def derive_attested_coverage(
    *,
    computed_coverage: str,
    post_ok: bool,
    promotion_receipt_valid: bool,
    final_readback_ok: bool,
    pre_post_lock_identical: bool,
) -> str:
    """Attested coverage is never stored inside the signed package."""
    if not (
        post_ok
        and promotion_receipt_valid
        and final_readback_ok
        and pre_post_lock_identical
    ):
        return "UNVERIFIED"
    if computed_coverage not in ("COMPLETE", "PARTIAL", "NONE"):
        return "UNVERIFIED"
    return str(computed_coverage)
