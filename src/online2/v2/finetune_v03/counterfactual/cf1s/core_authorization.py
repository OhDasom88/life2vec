"""Two-phase Development authorization: PRE_EXECUTION + POST_EXECUTION attestation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .core_canonical import canonical_json_dumps, canonical_json_sha256
from .core_contract import CoreContractError, sha256_file

TRUST_ROOT_ENV = "CF1S_TRUST_ROOT_SHA256"
TRUST_ROOT_RELPATH = "conf/m1/cf1s_policies/CF1S_DEVELOPMENT_TRUST_ROOT_V1.json"
PRE_EXECUTION_KIND = "PRE_EXECUTION_AUTHORIZATION"
POST_EXECUTION_KIND = "POST_EXECUTION_EVIDENCE_ATTESTATION"
ALLOWED_COHORT_IDS = ("DEVELOPMENT3", "VALIDATION20", "PRIMARY32")


def _cohort_execution_kind(cohort_id: str) -> str:
    return f"{cohort_id}_PRODUCTION_TWO_EVENT"


def repository_relative(path: Path, root: Path) -> str:
    p = Path(path).resolve()
    r = Path(root).resolve()
    rel = p.relative_to(r)
    return str(rel).replace("\\", "/")


def tree_manifest_sha(
    paths: Sequence[Path],
    *,
    root: Path,
    required: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """Canonical relative-path tree hash; missing required files fail closed."""
    entries: Dict[str, str] = {}
    for raw in paths:
        p = Path(raw)
        if not p.is_file():
            if required:
                raise CoreContractError(f"required tree file missing: {p}")
            continue
        rel = repository_relative(p, root)
        if rel in entries:
            raise CoreContractError(f"duplicate relative path in tree hash: {rel}")
        entries[rel] = sha256_file(p)
    manifest = {"version": "CF1S_TREE_MANIFEST_V1", "files": dict(sorted(entries.items()))}
    return canonical_json_sha256(manifest), entries


def load_trust_root(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("scope") != "DEVELOPMENT_ONLY":
        raise CoreContractError("trust root scope must be DEVELOPMENT_ONLY")
    return data


def verify_trust_root_env(trust_root_path: Path) -> str:
    """Protected env SHA must match repository trust-root file bytes."""
    expected = os.environ.get(TRUST_ROOT_ENV)
    if not expected:
        raise CoreContractError(
            f"{TRUST_ROOT_ENV} missing; production authorization issuance/consumption forbidden"
        )
    observed = sha256_file(Path(trust_root_path))
    if observed != expected:
        raise CoreContractError(
            f"trust root SHA mismatch: env={expected} file={observed}"
        )
    return observed


def payload_sha256(payload: Mapping[str, Any], *, exclude_keys: Sequence[str] = ()) -> str:
    body = {k: v for k, v in payload.items() if k not in set(exclude_keys)}
    return canonical_json_sha256(body)


def _ed25519_available() -> bool:
    try:
        from nacl.signing import SigningKey  # type: ignore

        return True
    except Exception:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: F401
                Ed25519PrivateKey,
            )

            return True
        except Exception:
            return False


def sign_payload(payload_sha: str, *, private_key_hex: str) -> str:
    """Detached Ed25519 signature over UTF-8 hex of payload_sha string."""
    msg = payload_sha.encode("utf-8")
    key_bytes = bytes.fromhex(private_key_hex)
    try:
        from nacl.signing import SigningKey

        sig = SigningKey(key_bytes).sign(msg).signature
        return sig.hex()
    except Exception:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.from_private_bytes(key_bytes)
        return key.sign(msg).hex()


def verify_signature(
    payload_sha: str,
    *,
    signature_hex: str,
    public_key_hex: str,
) -> bool:
    msg = payload_sha.encode("utf-8")
    sig = bytes.fromhex(signature_hex)
    pub = bytes.fromhex(public_key_hex)
    try:
        from nacl.signing import VerifyKey

        VerifyKey(pub).verify(msg, sig)
        return True
    except Exception:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)
            return True
        except Exception:
            return False


def issue_pre_execution_authorization(
    *,
    trust_root: Mapping[str, Any],
    trust_root_sha256: str,
    issuer_id: str,
    key_id: str,
    private_key_hex: str,
    expires_at: datetime,
    stable_lock_manifest: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    locks: Optional[Mapping[str, Any]] = None,
    cohort_id: str = "DEVELOPMENT3",
) -> Dict[str, Any]:
    from .core_locks import validate_stable_lock_manifest

    if cohort_id not in ALLOWED_COHORT_IDS:
        raise CoreContractError(f"unknown cohort_id: {cohort_id}")
    verify_issuer(trust_root, issuer_id=issuer_id, key_id=key_id)
    stable = dict(stable_lock_manifest or locks or {})
    validate_stable_lock_manifest(stable, expected_cohort=cohort_id.lower())
    if not run_id:
        raise CoreContractError("PRE authorization run_id required")
    pub = trust_root["keys"][key_id]["public_key_hex"]
    artifact = {
        "artifact_kind": PRE_EXECUTION_KIND,
        "schema_version": "CF1S_PRE_AUTHORIZATION_V2",
        "authorization_type": "DEVELOPMENT_PRODUCTION_EXECUTION",
        "run_id": str(run_id),
        "cohort_id": cohort_id,
        "scope": [cohort_id, "TWO_EVENT_ONLY"],
        "stable_lock_sha256": stable["stable_lock_sha256"],
        "stable_lock_manifest": stable,
        "allowed_execution_kinds": [_cohort_execution_kind(cohort_id)],
        "issuer_id": issuer_id,
        "key_id": key_id,
        "trust_root_sha256": trust_root_sha256,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "development_production_execution_authorized": True,
        "evidence_output_authorized": True,
        "recommendation_authorized": False,
        "primary32_execution_authorized": False,
        "validation20_execution_authorized": False,
        "problem20_execution_authorized": False,
        "trajectory_authorized": False,
        "causal_effect_claim_authorized": False,
        "real_world_action_authorized": False,
        "interpretation_scope": "Development3 selection-blind contract verification evidence",
        "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
    }
    psha = payload_sha256(artifact, exclude_keys=("signature_hex", "payload_sha256"))
    artifact["payload_sha256"] = psha
    artifact["signature_hex"] = sign_payload(psha, private_key_hex=private_key_hex)
    # Do not trust embedded public key for verification — store for diagnostics only
    artifact["issuer_public_key_hex_diagnostic_only"] = pub
    return artifact


def verify_issuer(trust_root: Mapping[str, Any], *, issuer_id: str, key_id: str) -> None:
    allow = set(trust_root.get("issuer_allowlist") or [])
    if issuer_id not in allow:
        raise CoreContractError(f"issuer not allowlisted: {issuer_id}")
    keys = trust_root.get("keys") or {}
    if key_id not in keys:
        raise CoreContractError(f"unknown key_id: {key_id}")
    meta = keys[key_id]
    if meta.get("algorithm") != "Ed25519":
        raise CoreContractError("only Ed25519 keys accepted")
    if meta.get("revoked"):
        raise CoreContractError(f"key revoked: {key_id}")


def consume_pre_execution_authorization(
    artifact: Mapping[str, Any],
    *,
    trust_root: Mapping[str, Any],
    trust_root_sha256: str,
    observed_stable_lock: Optional[Mapping[str, Any]] = None,
    expected_run_id: Optional[str] = None,
    expected_locks: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
    expected_cohort_id: str = "DEVELOPMENT3",
) -> Dict[str, Any]:
    from .core_locks import validate_stable_lock_manifest

    if expected_cohort_id not in ALLOWED_COHORT_IDS:
        raise CoreContractError(f"unknown expected cohort_id: {expected_cohort_id}")
    if expected_locks is not None:
        raise CoreContractError("AUTH_SELF_REFERENCE_FORBIDDEN")
    if artifact.get("artifact_kind") != PRE_EXECUTION_KIND:
        raise CoreContractError("not a PRE_EXECUTION_AUTHORIZATION artifact")
    if artifact.get("trust_root_sha256") != trust_root_sha256:
        raise CoreContractError("authorization trust_root_sha256 mismatch")
    verify_issuer(
        trust_root,
        issuer_id=str(artifact["issuer_id"]),
        key_id=str(artifact["key_id"]),
    )
    pub = trust_root["keys"][str(artifact["key_id"])]["public_key_hex"]
    # Never trust artifact-embedded public key
    psha = payload_sha256(artifact, exclude_keys=("signature_hex", "payload_sha256", "issuer_public_key_hex_diagnostic_only"))
    if psha != artifact.get("payload_sha256"):
        raise CoreContractError("authorization payload_sha256 mismatch")
    if not verify_signature(
        psha,
        signature_hex=str(artifact["signature_hex"]),
        public_key_hex=pub,
    ):
        raise CoreContractError("authorization signature invalid")
    now = now or datetime.now(timezone.utc)
    exp = datetime.fromisoformat(str(artifact["expires_at"]).replace("Z", "+00:00"))
    if now > exp:
        raise CoreContractError("authorization expired")
    if artifact.get("schema_version") != "CF1S_PRE_AUTHORIZATION_V2":
        raise CoreContractError("authorization schema version mismatch")
    if artifact.get("authorization_type") != "DEVELOPMENT_PRODUCTION_EXECUTION":
        raise CoreContractError("authorization type mismatch")
    if artifact.get("cohort_id") != expected_cohort_id:
        raise CoreContractError("authorization cohort mismatch")
    if artifact.get("scope") != [expected_cohort_id, "TWO_EVENT_ONLY"]:
        raise CoreContractError("authorization scope mismatch")
    if artifact.get("allowed_execution_kinds") != [
        _cohort_execution_kind(expected_cohort_id)
    ]:
        raise CoreContractError("authorization execution kinds mismatch")
    if expected_run_id is not None and artifact.get("run_id") != expected_run_id:
        raise CoreContractError("authorization run_id mismatch")
    for flag in (
        "recommendation_authorized",
        "primary32_execution_authorized",
        "validation20_execution_authorized",
        "problem20_execution_authorized",
        "trajectory_authorized",
        "causal_effect_claim_authorized",
        "real_world_action_authorized",
    ):
        if artifact.get(flag):
            raise CoreContractError(f"{flag} must be false")
    if not artifact.get("development_production_execution_authorized"):
        raise CoreContractError("development_production_execution_authorized=false")
    signed_stable = validate_stable_lock_manifest(
        artifact.get("stable_lock_manifest") or {},
        expected_cohort=expected_cohort_id.lower(),
    )
    if artifact.get("stable_lock_sha256") != signed_stable["stable_lock_sha256"]:
        raise CoreContractError("authorization stable lock payload mismatch")
    observed = dict(observed_stable_lock or {})
    if not observed:
        raise CoreContractError("fresh observed stable lock required")
    observed = validate_stable_lock_manifest(
        observed, expected_cohort=expected_cohort_id.lower()
    )
    if observed["stable_lock_sha256"] != signed_stable["stable_lock_sha256"]:
        raise CoreContractError("observed stable lock mismatch")
    return {
        "ok": True,
        "payload_sha256": psha,
        "run_id": artifact["run_id"],
        "stable_lock_sha256": signed_stable["stable_lock_sha256"],
    }


def issue_post_execution_attestation(
    *,
    trust_root: Mapping[str, Any],
    trust_root_sha256: str,
    issuer_id: str,
    key_id: str,
    private_key_hex: str,
    evidence_root_sha: str,
    pre_execution_payload_sha256: str,
    acceptance_summary: Mapping[str, Any],
    pre_post_lock_identical: bool,
    run_id: Optional[str] = None,
    pre_stable_lock_sha256: Optional[str] = None,
    post_stable_lock_sha256: Optional[str] = None,
    intended_final_destination: Optional[str] = None,
    final_rerun_observation_manifest_sha256: Optional[str] = None,
    cohort_id: str = "DEVELOPMENT3",
) -> Dict[str, Any]:
    if cohort_id not in ALLOWED_COHORT_IDS:
        raise CoreContractError(f"unknown cohort_id: {cohort_id}")
    verify_issuer(trust_root, issuer_id=issuer_id, key_id=key_id)
    if not pre_post_lock_identical:
        raise CoreContractError("pre/post lock mismatch; attestation forbidden")
    for name, value in (
        ("run_id", run_id),
        ("pre_stable_lock_sha256", pre_stable_lock_sha256),
        ("post_stable_lock_sha256", post_stable_lock_sha256),
        ("intended_final_destination", intended_final_destination),
    ):
        if not value:
            raise CoreContractError(f"POST {name} required")
    if pre_stable_lock_sha256 != post_stable_lock_sha256:
        raise CoreContractError("pre/post lock mismatch; attestation forbidden")
    artifact = {
        "artifact_kind": POST_EXECUTION_KIND,
        "schema_version": "CF1S_POST_ATTESTATION_V2",
        "run_id": str(run_id),
        "cohort_id": cohort_id,
        "scope": [cohort_id, "TWO_EVENT_ONLY"],
        "trust_root_sha256": trust_root_sha256,
        "issuer_id": issuer_id,
        "key_id": key_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_root_sha": evidence_root_sha,
        "pre_execution_payload_sha256": pre_execution_payload_sha256,
        "pre_stable_lock_sha256": pre_stable_lock_sha256,
        "post_stable_lock_sha256": post_stable_lock_sha256,
        "intended_final_destination": intended_final_destination,
        "final_rerun_observation_manifest_sha256": final_rerun_observation_manifest_sha256,
        "acceptance_summary": dict(acceptance_summary),
        "pre_post_lock_identical": True,
        # attestation excluded from package root it signs
        "signs_package_root": True,
        "included_in_signed_package_root": False,
    }
    psha = payload_sha256(artifact, exclude_keys=("signature_hex", "payload_sha256"))
    artifact["payload_sha256"] = psha
    artifact["signature_hex"] = sign_payload(psha, private_key_hex=private_key_hex)
    return artifact


def compute_evidence_package_root(
    artifact_paths: Mapping[str, Path],
    *,
    root: Path,
    exclude_names: Sequence[str] = ("POST_EXECUTION_EVIDENCE_ATTESTATION.json",),
) -> str:
    paths = [p for name, p in sorted(artifact_paths.items()) if name not in set(exclude_names)]
    sha, _ = tree_manifest_sha(paths, root=root, required=True)
    return sha
