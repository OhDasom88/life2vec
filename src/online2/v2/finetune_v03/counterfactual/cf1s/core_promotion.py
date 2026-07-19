"""Atomic same-filesystem promotion with fsync durability and external receipts."""

from __future__ import annotations

import json
import os
import ctypes
import errno
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .core_canonical import canonical_json_dumps, canonical_json_sha256
from .core_contract import CoreContractError, sha256_file


PROMOTED = "PROMOTED"
PROMOTED_BUT_UNVERIFIED = "PROMOTED_BUT_UNVERIFIED"
PROMOTED_BUT_UNATTESTED_BY_RECEIPT = "PROMOTED_BUT_UNATTESTED_BY_RECEIPT"
QUARANTINE_ONLY = "QUARANTINE_ONLY"
RENAME_NOREPLACE = 1
AT_FDCWD = -100


def fsync_file(path: Path) -> None:
    p = Path(path)
    fd = os.open(str(p), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    p = Path(path)
    fd = os.open(str(p), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def flush_tree_files(root: Path) -> None:
    """Best-effort fsync of every regular file under root (depth-first)."""
    root = Path(root)
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            fsync_file(Path(dirpath) / name)
        fsync_dir(Path(dirpath))


def same_filesystem(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def rename_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic no-replace rename. No unsafe compatibility fallback."""
    if os.name != "posix":
        raise CoreContractError("RENAME_NOREPLACE requires POSIX/Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CoreContractError("renameat2 unavailable; promotion forbidden")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise CoreContractError(f"destination already exists: {destination}")
        if error == errno.EXDEV:
            raise CoreContractError("cross-filesystem promotion forbidden")
        raise CoreContractError(
            f"renameat2(RENAME_NOREPLACE) failed errno={error}: {os.strerror(error)}"
        )


def atomic_promote_directory(
    *,
    quarantine_dir: Path,
    final_dir: Path,
) -> Dict[str, Any]:
    """
    Same-filesystem rename only. Never copytree. Never overwrite existing final.
    Sequence: flush files → quarantine dir fsync → rename → final parent fsync.
    """
    q = Path(quarantine_dir).resolve()
    f = Path(final_dir).resolve()
    if not q.is_dir():
        raise CoreContractError(f"quarantine missing: {q}")
    parent = f.parent
    source_parent = q.parent
    parent.mkdir(parents=True, exist_ok=True)

    if f.exists():
        raise CoreContractError(f"final path already exists; refuse overwrite: {f}")

    if not same_filesystem(q, parent):
        raise CoreContractError(
            f"cross-filesystem promotion forbidden: quarantine st_dev={os.stat(q).st_dev} "
            f"final_parent st_dev={os.stat(parent).st_dev}"
        )

    flush_tree_files(q)
    fsync_dir(q)
    fsync_dir(source_parent)

    rename_noreplace(q, f)
    fsync_dir(source_parent)
    fsync_dir(parent)

    return {
        "promotion_method": "renameat2(RENAME_NOREPLACE)",
        "quarantine_path": str(q),
        "final_path": str(f),
        "source_st_dev": os.stat(f).st_dev,
        "destination_st_dev": os.stat(f).st_dev,
        "same_filesystem": True,
    }


def write_promotion_receipt(
    *,
    receipts_dir: Path,
    run_id: str,
    final_path: Path,
    quarantine_path: str,
    evidence_root_sha: str,
    post_attestation_sha: Optional[str],
    promotion_meta: Mapping[str, Any],
    final_readback: Mapping[str, Any],
    previous_receipt_sha: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """External append-only receipt — never written inside the signed package."""
    receipts_dir = Path(receipts_dir)
    receipts_dir.mkdir(parents=True, exist_ok=True)
    resolved_final_path = str(Path(final_path).resolve())
    resolved_quarantine_path = str(Path(quarantine_path).resolve())
    receipt = {
        "artifact_kind": "CF1S_PROMOTION_RECEIPT_V1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "final_path": resolved_final_path,
        "quarantine_path": resolved_quarantine_path,
        "evidence_root_sha": evidence_root_sha,
        "post_attestation_sha": post_attestation_sha,
        "promotion_method": promotion_meta.get("promotion_method"),
        "source_st_dev": promotion_meta.get("source_st_dev"),
        "destination_st_dev": promotion_meta.get("destination_st_dev"),
        "final_readback": dict(final_readback),
        "previous_receipt_sha": previous_receipt_sha,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    out = receipts_dir / f"{run_id}.json"
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        supplied_sha = existing.get("receipt_sha256")
        observed_sha = canonical_json_sha256(
            {k: v for k, v in existing.items() if k != "receipt_sha256"}
        )
        if supplied_sha != observed_sha:
            raise CoreContractError(f"existing promotion receipt invalid: {out}")
        semantic_keys = set(receipt) - {"created_at", "receipt_sha256"}
        if any(existing.get(key) != receipt.get(key) for key in semantic_keys):
            raise CoreContractError("receipt retry semantic fields differ")
        return out, existing
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(canonical_json_dumps(receipt) + "\n", encoding="utf-8")
    fsync_file(tmp)
    try:
        rename_noreplace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    fsync_dir(receipts_dir)
    return out, receipt


def validate_promotion_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_final_path: Path,
    expected_evidence_root: str,
    expected_post_attestation_sha: str,
    artifact_relpaths: Mapping[str, str],
) -> Dict[str, Any]:
    """Verify the observable promotion outcome and receipt bindings."""
    if receipt.get("artifact_kind") != "CF1S_PROMOTION_RECEIPT_V1":
        raise CoreContractError("promotion receipt artifact kind mismatch")
    supplied_sha = receipt.get("receipt_sha256")
    observed_sha = canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    if supplied_sha != observed_sha:
        raise CoreContractError("promotion receipt SHA invalid")
    if receipt.get("run_id") != expected_run_id:
        raise CoreContractError("receipt run_id mismatch")
    if receipt.get("evidence_root_sha") != expected_evidence_root:
        raise CoreContractError("receipt evidence root mismatch")

    final_path = Path(expected_final_path).resolve()
    if Path(str(receipt.get("final_path"))).resolve() != final_path:
        raise CoreContractError("receipt final path mismatch")
    if receipt.get("post_attestation_sha") != expected_post_attestation_sha:
        raise CoreContractError("receipt POST attestation SHA mismatch")
    if receipt.get("promotion_method") != "renameat2(RENAME_NOREPLACE)":
        raise CoreContractError("promotion method is not RENAME_NOREPLACE")

    source_st_dev = receipt.get("source_st_dev")
    destination_st_dev = receipt.get("destination_st_dev")
    if source_st_dev != destination_st_dev:
        raise CoreContractError("receipt filesystem device mismatch")
    if not final_path.is_dir():
        raise CoreContractError("promoted final package missing")
    actual_st_dev = os.stat(final_path).st_dev
    if destination_st_dev != actual_st_dev:
        raise CoreContractError("receipt destination device differs from filesystem")

    observed_readback = readback_final_package(
        final_dir=final_path,
        expected_evidence_root=expected_evidence_root,
        artifact_relpaths=artifact_relpaths,
    )
    if not observed_readback.get("ok"):
        raise CoreContractError(
            f"final package readback failed: {observed_readback.get('reason')}"
        )
    if dict(receipt.get("final_readback") or {}) != observed_readback:
        raise CoreContractError("receipt final readback differs from recomputation")
    return {
        "ok": True,
        "receipt_sha256": supplied_sha,
        "final_readback": observed_readback,
        "claim": "atomic promotion outcome and receipt consistency verified",
    }


def readback_final_package(
    *,
    final_dir: Path,
    expected_evidence_root: str,
    artifact_relpaths: Mapping[str, str],
) -> Dict[str, Any]:
    """Immutable readback of promoted package (no mutation)."""
    from .core_authorization import compute_evidence_package_root

    final_dir = Path(final_dir)
    if not final_dir.is_dir():
        return {"ok": False, "reason": "FINAL_MISSING"}
    paths = {name: final_dir / rel for name, rel in artifact_relpaths.items()}
    for name, p in paths.items():
        if not p.is_file():
            return {"ok": False, "reason": f"MISSING_ARTIFACT:{name}"}
    observed = compute_evidence_package_root(paths, root=final_dir)
    if observed != expected_evidence_root:
        return {
            "ok": False,
            "reason": "EVIDENCE_ROOT_MISMATCH",
            "expected": expected_evidence_root,
            "observed": observed,
        }
    return {
        "ok": True,
        "evidence_root_sha": observed,
        "artifact_count": len(paths),
    }


def package_state_after_receipt(
    *,
    promoted: bool,
    receipt_valid: bool,
) -> str:
    if not promoted:
        return QUARANTINE_ONLY
    if receipt_valid:
        return PROMOTED
    return PROMOTED_BUT_UNATTESTED_BY_RECEIPT
