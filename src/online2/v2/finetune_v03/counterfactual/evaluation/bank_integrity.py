"""Bank file / content / mode-policy hashing and 2-tier relational integrity (A8-1)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


BANK_MODE_DEPLOYMENT = "deployment"
BANK_MODE_RECONSTRUCTION_EVAL = "reconstruction_eval"
BANK_SCHEMA_VERSION = "mlm_bundle_bank_v1"

SOURCE_PARTITION_TRAINING = "TRAINING"
SOURCE_PARTITION_PROBLEM = "PROBLEM"

PROBLEM_SCOPE_TARGET_CASE_ONLY = "TARGET_CASE_ONLY"
PROBLEM_SCOPE_NONE = "NONE"

EXPECTED_POLICY_BY_MODE: Dict[str, Dict[str, Any]] = {
    BANK_MODE_DEPLOYMENT: {
        "exclude_target_event": True,
        "exclude_same_case_future": True,
        "exclude_target_case": False,
        "exclude_duplicate_event_fingerprints": True,
        "same_timestamp_policy": "EXCLUDE",
        "allowed_source_partitions": [SOURCE_PARTITION_TRAINING, SOURCE_PARTITION_PROBLEM],
        "allowed_problem_case_scope": PROBLEM_SCOPE_TARGET_CASE_ONLY,
    },
    BANK_MODE_RECONSTRUCTION_EVAL: {
        "exclude_target_event": True,
        "exclude_same_case_future": True,
        "exclude_target_case": True,
        "exclude_duplicate_event_fingerprints": True,
        "same_timestamp_policy": "EXCLUDE",
        "allowed_source_partitions": [SOURCE_PARTITION_TRAINING],
        "allowed_problem_case_scope": PROBLEM_SCOPE_NONE,
    },
}


def canonical_policy_fields_for_mode(mode: str) -> Dict[str, Any]:
    mode_s = str(mode)
    if mode_s not in EXPECTED_POLICY_BY_MODE:
        raise ValueError(f"unknown_bank_mode:{mode_s}")
    pol = EXPECTED_POLICY_BY_MODE[mode_s]
    return {
        "exclude_target_event": bool(pol["exclude_target_event"]),
        "exclude_same_case_future": bool(pol["exclude_same_case_future"]),
        "exclude_target_case": bool(pol["exclude_target_case"]),
        "exclude_duplicate_event_fingerprints": bool(
            pol["exclude_duplicate_event_fingerprints"]
        ),
        "same_timestamp_policy": str(pol["same_timestamp_policy"]),
        "allowed_source_partitions": list(pol["allowed_source_partitions"]),
        "allowed_problem_case_scope": str(pol["allowed_problem_case_scope"]),
    }



def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def observation_index_content_sha256(observations: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for o in observations:
        rows.append(
            {
                "bundle_id": str(o.get("bundle_id") or ""),
                "feature": str(o.get("feature") or ""),
                "role_signature": str(o.get("role_signature") or o.get("signature") or ""),
                "case_id": str(o.get("case_id") or ""),
                "event_id": str(o.get("event_id") or ""),
                "timestamp_utc": str(o.get("timestamp_utc") or o.get("timestamp") or ""),
                "event_fingerprint": str(o.get("event_fingerprint") or o.get("fingerprint") or ""),
                "source_partition": str(o.get("source_partition") or ""),
            }
        )
    rows.sort(
        key=lambda r: (
            r["bundle_id"],
            r["case_id"],
            r["event_id"],
            r["timestamp_utc"],
            r["event_fingerprint"],
            r["source_partition"],
        )
    )
    return sha256_bytes(canonical_json_bytes({"observations": rows}))


def unique_bundle_index_content_sha256(uniques: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for u in uniques:
        tokens = list(u.get("canonical_full_mg_bundle") or u.get("tokens") or [])
        rows.append(
            {
                "bundle_id": str(u.get("bundle_id") or ""),
                "feature": str(u.get("feature") or ""),
                "role_signature": str(u.get("role_signature") or u.get("signature") or ""),
                "canonical_full_mg_bundle": tokens,
                "raw_observation_count": int(u.get("raw_observation_count") or 0),
                "deduplicated_observation_count": int(
                    u.get("deduplicated_observation_count")
                    or u.get("observation_count_used_for_provenance")
                    or u.get("observation_count")
                    or 0
                ),
                "observation_count_used_for_provenance": int(
                    u.get("observation_count_used_for_provenance")
                    or u.get("deduplicated_observation_count")
                    or u.get("observation_count")
                    or 0
                ),
            }
        )
    rows.sort(key=lambda r: (r["feature"], r["bundle_id"], tuple(r["canonical_full_mg_bundle"])))
    return sha256_bytes(canonical_json_bytes({"unique_bundles": rows}))


def composite_bundle_bank_content_sha256(
    *,
    observation_index_content_sha256: str,
    unique_bundle_index_content_sha256: str,
    bank_schema_version: str = BANK_SCHEMA_VERSION,
) -> str:
    payload = {
        "bank_schema_version": bank_schema_version,
        "observation_index_content_sha256": observation_index_content_sha256,
        "unique_bundle_index_content_sha256": unique_bundle_index_content_sha256,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def bank_mode_policy_payload(
    *,
    mode: str,
    exclude_target_event: bool,
    exclude_same_case_future: bool,
    continuous_features_only: bool = True,
    exclude_target_case: bool = False,
    exclude_duplicate_event_fingerprints: bool = True,
    same_timestamp_policy: str = "EXCLUDE",
    allowed_source_partitions: Optional[Sequence[str]] = None,
    allowed_problem_case_scope: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if str(mode) in EXPECTED_POLICY_BY_MODE:
        ep = EXPECTED_POLICY_BY_MODE[str(mode)]
        if allowed_source_partitions is None:
            allowed_source_partitions = list(ep["allowed_source_partitions"])
        if allowed_problem_case_scope is None:
            allowed_problem_case_scope = str(ep["allowed_problem_case_scope"])
    payload = {
        "mode": str(mode),
        "exclude_target_event": bool(exclude_target_event),
        "exclude_same_case_future": bool(exclude_same_case_future),
        "exclude_target_case": bool(exclude_target_case),
        "exclude_duplicate_event_fingerprints": bool(exclude_duplicate_event_fingerprints),
        "same_timestamp_policy": str(same_timestamp_policy),
        "allowed_source_partitions": list(allowed_source_partitions or []),
        "allowed_problem_case_scope": str(allowed_problem_case_scope or ""),
        "continuous_features_only": bool(continuous_features_only),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def bank_mode_policy_sha256(**kwargs: Any) -> str:
    return sha256_bytes(canonical_json_bytes(bank_mode_policy_payload(**kwargs)))


def bank_query_manifest_sha256(
    *,
    target_event_id: str,
    case_id: str,
    feature: str,
    mode: str,
    exclude_target_event: bool,
    exclude_same_case_future: bool,
    expected_signature: Optional[Sequence[str]] = None,
    exclude_target_case: bool = False,
    exclude_duplicate_event_fingerprints: bool = True,
    same_timestamp_policy: str = "EXCLUDE",
    allowed_source_partitions: Optional[Sequence[str]] = None,
    allowed_problem_case_scope: Optional[str] = None,
    target_timestamp_utc: str = "",
) -> str:
    if str(mode) in EXPECTED_POLICY_BY_MODE:
        ep = EXPECTED_POLICY_BY_MODE[str(mode)]
        if allowed_source_partitions is None:
            allowed_source_partitions = list(ep["allowed_source_partitions"])
        if allowed_problem_case_scope is None:
            allowed_problem_case_scope = str(ep["allowed_problem_case_scope"])
    payload = {
        "target_event_id": str(target_event_id),
        "case_id": str(case_id),
        "feature": str(feature),
        "mode": str(mode),
        "exclude_target_event": bool(exclude_target_event),
        "exclude_same_case_future": bool(exclude_same_case_future),
        "exclude_target_case": bool(exclude_target_case),
        "exclude_duplicate_event_fingerprints": bool(exclude_duplicate_event_fingerprints),
        "same_timestamp_policy": str(same_timestamp_policy),
        "allowed_source_partitions": list(allowed_source_partitions or []),
        "allowed_problem_case_scope": str(allowed_problem_case_scope or ""),
        "target_timestamp_utc": str(target_timestamp_utc or ""),
        "expected_signature": list(expected_signature or []),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def query_set_manifest_sha256(
    *,
    run_id: str,
    query_count: int,
    query_records: Sequence[Mapping[str, Any]],
) -> str:
    rows = []
    for r in query_records:
        rows.append(
            {
                "mg_key": str(r.get("mg_key") or ""),
                "query_manifest_sha256": str(r.get("query_manifest_sha256") or ""),
            }
        )
    rows.sort(key=lambda x: x["mg_key"])
    payload = {
        "run_id": str(run_id),
        "query_count": int(query_count),
        "query_records": rows,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def recorded_policy_matches_canonical(
    *,
    mode: str,
    recorded: Mapping[str, Any],
) -> Tuple[bool, List[str]]:
    """Compare recorded policy fields to EXPECTED_POLICY_BY_MODE."""
    errs: List[str] = []
    try:
        canon = canonical_policy_fields_for_mode(mode)
    except ValueError as e:
        return False, [str(e)]
    qm = dict(recorded.get("query_manifest") or {})

    def _get(key: str, default: Any = None) -> Any:
        if key in recorded and recorded[key] is not None:
            return recorded[key]
        if key in qm and qm[key] is not None:
            return qm[key]
        return default

    for key, expected in canon.items():
        got = _get(key)
        if got is None:
            errs.append(f"missing_policy_field:{key}")
            continue
        if key == "allowed_source_partitions":
            if list(got) != list(expected):
                errs.append(f"canonical_policy_mismatch:{key}")
        elif key in {"same_timestamp_policy", "allowed_problem_case_scope"}:
            if str(got) != str(expected):
                errs.append(f"canonical_policy_mismatch:{key}")
        else:
            if bool(got) != bool(expected):
                errs.append(f"canonical_policy_mismatch:{key}")
    return len(errs) == 0, errs


def validate_bank_relational_integrity(
    observations: Sequence[Mapping[str, Any]],
    uniques: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """A8-1 relational invariants between observation and unique indexes."""
    obs_ids = [str(o.get("bundle_id") or "") for o in observations]
    uniq_ids = [str(u.get("bundle_id") or "") for u in uniques]
    uniq_set = set(uniq_ids)
    obs_set = set(obs_ids)

    orphan_obs = sorted(i for i in obs_set if i and i not in uniq_set)
    orphan_uniq = sorted(i for i in uniq_set if i and i not in obs_set)

    # deduplicated observation counts by bundle_id (unique event_fingerprint per bundle)
    dedup_counts: Dict[str, int] = defaultdict(int)
    raw_counts: Counter = Counter()
    seen_fp: Dict[str, set] = defaultdict(set)
    for o in observations:
        bid = str(o.get("bundle_id") or "")
        if not bid:
            continue
        raw_counts[bid] += 1
        fp = str(o.get("event_fingerprint") or o.get("fingerprint") or "")
        key = fp or f"{o.get('case_id')}|{o.get('event_id')}|{o.get('timestamp_utc')}"
        if key not in seen_fp[bid]:
            seen_fp[bid].add(key)
            dedup_counts[bid] += 1

    count_mismatches = []
    for u in uniques:
        bid = str(u.get("bundle_id") or "")
        used = int(
            u.get("observation_count_used_for_provenance")
            or u.get("deduplicated_observation_count")
            or u.get("observation_count")
            or 0
        )
        expected = int(dedup_counts.get(bid, 0))
        if used != expected:
            count_mismatches.append(
                {"bundle_id": bid, "declared": used, "expected_deduplicated": expected}
            )

    fk_ok = len(orphan_obs) == 0 and len(orphan_uniq) == 0
    counts_ok = len(count_mismatches) == 0
    return {
        "observation_unique_foreign_keys_valid": fk_ok,
        "observation_counts_exact_match": counts_ok,
        "orphan_observation_count": len(orphan_obs),
        "orphan_unique_bundle_count": len(orphan_uniq),
        "orphan_observation_ids": orphan_obs[:50],
        "orphan_unique_ids": orphan_uniq[:50],
        "count_mismatches": count_mismatches[:50],
        "a8_1_relational_pass": fk_ok and counts_ok,
    }


def build_two_tier_integrity_record(
    *,
    observations: Sequence[Mapping[str, Any]],
    uniques: Sequence[Mapping[str, Any]],
    mode: str,
    observation_file: Optional[Path] = None,
    unique_file: Optional[Path] = None,
    exclude_target_event: bool = True,
    exclude_same_case_future: bool = True,
) -> Dict[str, Any]:
    obs_sha = observation_index_content_sha256(observations)
    uniq_sha = unique_bundle_index_content_sha256(uniques)
    content_sha = composite_bundle_bank_content_sha256(
        observation_index_content_sha256=obs_sha,
        unique_bundle_index_content_sha256=uniq_sha,
    )
    relational = validate_bank_relational_integrity(observations, uniques)
    policy_sha = bank_mode_policy_sha256(
        mode=mode,
        exclude_target_event=exclude_target_event,
        exclude_same_case_future=exclude_same_case_future,
    )
    obs_f = (
        file_sha256(Path(observation_file))
        if observation_file and Path(observation_file).exists()
        else None
    )
    uniq_f = (
        file_sha256(Path(unique_file)) if unique_file and Path(unique_file).exists() else None
    )
    file_sha = None
    if obs_f and uniq_f:
        file_sha = sha256_bytes(
            canonical_json_bytes(
                {
                    "observation_file_sha256": obs_f,
                    "unique_file_sha256": uniq_f,
                }
            )
        )
    return {
        "bank_schema_version": BANK_SCHEMA_VERSION,
        "observation_index_content_sha256": obs_sha,
        "unique_bundle_index_content_sha256": uniq_sha,
        "bundle_bank_content_sha256": content_sha,
        "bundle_bank_policy_sha256": policy_sha,
        "bundle_bank_mode": str(mode),
        "observation_file_sha256": obs_f,
        "unique_file_sha256": uniq_f,
        "bundle_bank_file_sha256": file_sha,
        "n_observations": len(observations),
        "n_unique_bundles": len(uniques),
        **relational,
    }


def verify_query_manifest_sha256(
    *,
    recorded_sha: str,
    target_event_id: str,
    case_id: str,
    feature: str,
    mode: str,
    exclude_target_event: bool,
    exclude_same_case_future: bool,
    expected_signature: Optional[Sequence[str]] = None,
    exclude_target_case: bool = False,
    exclude_duplicate_event_fingerprints: bool = True,
    same_timestamp_policy: str = "EXCLUDE",
    allowed_source_partitions: Optional[Sequence[str]] = None,
    allowed_problem_case_scope: Optional[str] = None,
    target_timestamp_utc: str = "",
) -> bool:
    """Per-run query hash check — does NOT require cross-run equality."""
    expected = bank_query_manifest_sha256(
        target_event_id=target_event_id,
        case_id=case_id,
        feature=feature,
        mode=mode,
        exclude_target_event=exclude_target_event,
        exclude_same_case_future=exclude_same_case_future,
        expected_signature=expected_signature,
        exclude_target_case=exclude_target_case,
        exclude_duplicate_event_fingerprints=exclude_duplicate_event_fingerprints,
        same_timestamp_policy=same_timestamp_policy,
        allowed_source_partitions=allowed_source_partitions,
        allowed_problem_case_scope=allowed_problem_case_scope,
        target_timestamp_utc=target_timestamp_utc,
    )
    return str(recorded_sha) == str(expected)


A8_1_REQUIRED_CONSUMER_RUN_IDS = (
    "functional_smoke",
    "curated_fixture",
    "natural_path_a",
    "reconstruction_selection",
    "reconstruction_eval",
)

A8_1_REQUIRED_REF_KEYS = (
    "run_id",
    "bundle_bank_content_sha256",
    "observation_file_sha256",
    "unique_file_sha256",
    "bank_mode",
    "bundle_bank_policy_sha256",
    "bundle_bank_query_manifest_sha256",
)


def evaluate_a8_1_bank_integrity(
    *,
    bank_dir: Path,
    bank_meta: Mapping[str, Any],
    consumer_bank_refs: Optional[Sequence[Mapping[str, Any]]] = None,
    query_records: Optional[Sequence[Mapping[str, Any]]] = None,
    required_consumer_run_ids: Sequence[str] = A8_1_REQUIRED_CONSUMER_RUN_IDS,
    expected_mode_by_run: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """A8-1 fail-closed: parquet/FK + required consumers + per-run query hashes."""
    errs: List[str] = []
    bank_dir = Path(bank_dir)
    man_path = bank_dir / "mlm_bundle_bank_manifest.json"
    if not man_path.is_file():
        return {"a8_1_pass": False, "errors": ["missing_bank_manifest"]}

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    obs_path = Path(manifest.get("observation_path") or (bank_dir / "mlm_bundle_bank_observations.parquet"))
    uniq_path = Path(manifest.get("unique_path") or (bank_dir / "mlm_bundle_bank_unique.parquet"))
    if not obs_path.is_file() or not uniq_path.is_file():
        errs.append("missing_parquet")
        return {"a8_1_pass": False, "errors": errs}

    import pandas as pd

    observations = pd.read_parquet(obs_path).to_dict(orient="records")
    uniques = pd.read_parquet(uniq_path).to_dict(orient="records")

    obs_file_sha = file_sha256(obs_path)
    uniq_file_sha = file_sha256(uniq_path)
    if obs_file_sha != str(manifest.get("observation_file_sha256") or ""):
        errs.append("observation_file_sha_mismatch")
    if uniq_file_sha != str(manifest.get("unique_file_sha256") or ""):
        errs.append("unique_file_sha_mismatch")

    recomputed = build_two_tier_integrity_record(
        observations=observations,
        uniques=uniques,
        mode=str(manifest.get("bundle_bank_mode") or manifest.get("mode") or "index_neutral"),
        observation_file=obs_path,
        unique_file=uniq_path,
    )
    content_sha = str(recomputed.get("bundle_bank_content_sha256") or "")
    file_sha = str(recomputed.get("bundle_bank_file_sha256") or "")
    if content_sha != str(manifest.get("bundle_bank_content_sha256") or ""):
        errs.append("content_sha_mismatch_vs_manifest")
    if file_sha != str(manifest.get("bundle_bank_file_sha256") or ""):
        errs.append("file_sha_mismatch_vs_manifest")

    meta_path_hint = bank_meta.get("_meta_path") or bank_meta.get("meta_path")
    if meta_path_hint and Path(str(meta_path_hint)).is_file():
        if file_sha == file_sha256(Path(str(meta_path_hint))):
            errs.append("meta_json_used_as_bank_file_hash")

    relational = validate_bank_relational_integrity(observations, uniques)
    if not relational.get("a8_1_relational_pass"):
        errs.append("relational_integrity_fail")

    # Partition completeness (fail-closed)
    unknown_parts = 0
    empty_parts = 0
    for o in observations:
        part = str(o.get("source_partition") or "").upper()
        if not part:
            empty_parts += 1
        elif part not in {SOURCE_PARTITION_TRAINING, SOURCE_PARTITION_PROBLEM}:
            unknown_parts += 1
    if empty_parts or unknown_parts:
        errs.append("partition_assignment_incomplete")
    man_complete = manifest.get("source_partition_assignment_complete")
    if man_complete is False:
        errs.append("partition_assignment_incomplete_manifest")
    overlap = int(manifest.get("training_problem_case_overlap_count") or 0)
    if overlap != 0:
        errs.append("training_problem_case_overlap")

    mode_by_run = dict(
        expected_mode_by_run
        or {
            "functional_smoke": BANK_MODE_DEPLOYMENT,
            "curated_fixture": BANK_MODE_DEPLOYMENT,
            "natural_path_a": BANK_MODE_DEPLOYMENT,
            "reconstruction_selection": BANK_MODE_RECONSTRUCTION_EVAL,
            "reconstruction_eval": BANK_MODE_RECONSTRUCTION_EVAL,
        }
    )

    refs = list(consumer_bank_refs or [])
    # Prefer query_records as authoritative consumer set when provided
    records = list(query_records) if query_records is not None else list(refs)
    by_run: Dict[str, Mapping[str, Any]] = {}
    for r in records:
        rid = str(r.get("run_id") or "")
        if rid:
            by_run[rid] = r

    for req_id in required_consumer_run_ids:
        if req_id not in by_run:
            errs.append(f"missing_required_consumer:{req_id}")

    for rid, ref in by_run.items():
        for k in A8_1_REQUIRED_REF_KEYS:
            # bank_mode alias
            if k == "bank_mode":
                if not (ref.get("bank_mode") or ref.get("bundle_bank_mode")):
                    errs.append(f"consumer[{rid}].missing_bank_mode")
                continue
            if ref.get(k) in (None, ""):
                errs.append(f"consumer[{rid}].missing_{k}")

        ref_content = str(ref.get("bundle_bank_content_sha256") or "")
        if not ref_content:
            errs.append(f"consumer[{rid}].missing_content_sha")
        elif ref_content != content_sha:
            errs.append(f"consumer[{rid}].content_sha_mismatch")

        ref_obs = str(ref.get("observation_file_sha256") or "")
        ref_uniq = str(ref.get("unique_file_sha256") or "")
        if not ref_obs:
            errs.append(f"consumer[{rid}].missing_observation_file_sha")
        elif ref_obs != obs_file_sha:
            errs.append(f"consumer[{rid}].observation_file_sha_mismatch")
        if not ref_uniq:
            errs.append(f"consumer[{rid}].missing_unique_file_sha")
        elif ref_uniq != uniq_file_sha:
            errs.append(f"consumer[{rid}].unique_file_sha_mismatch")

        mode = str(ref.get("bank_mode") or ref.get("bundle_bank_mode") or "")
        expected_mode = mode_by_run.get(rid)
        if expected_mode and mode and mode != expected_mode:
            errs.append(f"consumer[{rid}].mode_mismatch")
        if not mode:
            continue

        ok_canon, canon_errs = recorded_policy_matches_canonical(mode=mode, recorded=ref)
        for ce in canon_errs:
            errs.append(f"consumer[{rid}].{ce}")

        try:
            canon = canonical_policy_fields_for_mode(mode)
        except ValueError:
            errs.append(f"consumer[{rid}].unknown_mode")
            continue

        expected_policy = bank_mode_policy_sha256(
            mode=mode,
            exclude_target_event=bool(canon["exclude_target_event"]),
            exclude_same_case_future=bool(canon["exclude_same_case_future"]),
            exclude_target_case=bool(canon["exclude_target_case"]),
            exclude_duplicate_event_fingerprints=bool(
                canon["exclude_duplicate_event_fingerprints"]
            ),
            same_timestamp_policy=str(canon["same_timestamp_policy"]),
            allowed_source_partitions=list(canon["allowed_source_partitions"]),
            allowed_problem_case_scope=str(canon["allowed_problem_case_scope"]),
        )
        recorded_policy = str(ref.get("bundle_bank_policy_sha256") or "")
        if not recorded_policy:
            errs.append(f"consumer[{rid}].missing_policy_sha")
        elif recorded_policy != expected_policy:
            errs.append(f"consumer[{rid}].policy_sha_mismatch")

        qm = dict(ref.get("query_manifest") or {})
        recorded_query = str(ref.get("bundle_bank_query_manifest_sha256") or "")
        if not recorded_query:
            errs.append(f"consumer[{rid}].missing_query_sha")
        else:
            ok = verify_query_manifest_sha256(
                recorded_sha=recorded_query,
                target_event_id=str(
                    qm.get("target_event_id") or ref.get("target_event_id") or ""
                ),
                case_id=str(qm.get("case_id") or ref.get("case_id") or ""),
                feature=str(qm.get("feature") or ref.get("feature") or ""),
                mode=mode,
                exclude_target_event=bool(canon["exclude_target_event"]),
                exclude_same_case_future=bool(canon["exclude_same_case_future"]),
                expected_signature=qm.get("expected_signature") or ref.get("expected_signature"),
                exclude_target_case=bool(canon["exclude_target_case"]),
                exclude_duplicate_event_fingerprints=bool(
                    canon["exclude_duplicate_event_fingerprints"]
                ),
                same_timestamp_policy=str(canon["same_timestamp_policy"]),
                allowed_source_partitions=list(canon["allowed_source_partitions"]),
                allowed_problem_case_scope=str(canon["allowed_problem_case_scope"]),
                target_timestamp_utc=str(
                    qm.get("target_timestamp_utc")
                    or ref.get("target_timestamp_utc")
                    or ""
                ),
            )
            if not ok:
                errs.append(f"consumer[{rid}].query_manifest_sha_mismatch")

        # Reconstruction consumers require query-set evidence
        if mode == BANK_MODE_RECONSTRUCTION_EVAL:
            qset = dict(ref.get("query_set_manifest") or {})
            if not qset and ref.get("query_records") is not None:
                qset = {
                    "run_id": rid,
                    "query_count": ref.get("query_count"),
                    "query_records": ref.get("query_records"),
                    "query_set_sha256": ref.get("query_set_sha256"),
                }
            if not qset:
                errs.append(f"consumer[{rid}].missing_query_set_manifest")
            else:
                expected_keys = list(ref.get("expected_mg_keys") or [])
                qrecs = list(qset.get("query_records") or ref.get("query_records") or [])
                qcount = int(qset.get("query_count") or ref.get("query_count") or 0)
                if expected_keys and qcount != len(expected_keys):
                    errs.append(f"consumer[{rid}].query_count_mismatch")
                if expected_keys:
                    got_keys = [str(r.get("mg_key") or "") for r in qrecs]
                    if sorted(got_keys) != sorted(str(k) for k in expected_keys):
                        errs.append(f"consumer[{rid}].query_mg_keys_mismatch")
                for qi, qr in enumerate(qrecs):
                    qqm = dict(qr.get("query_manifest") or {})
                    for k, expected in canon.items():
                        if k == "allowed_source_partitions":
                            if list(qqm.get(k) or []) != list(expected):
                                errs.append(
                                    f"consumer[{rid}].query[{qi}].canonical_policy_mismatch:{k}"
                                )
                        elif k in {"same_timestamp_policy", "allowed_problem_case_scope"}:
                            if str(qqm.get(k) or "") != str(expected):
                                errs.append(
                                    f"consumer[{rid}].query[{qi}].canonical_policy_mismatch:{k}"
                                )
                        else:
                            if k not in qqm:
                                errs.append(
                                    f"consumer[{rid}].query[{qi}].missing_policy_field:{k}"
                                )
                            elif bool(qqm.get(k)) != bool(expected):
                                errs.append(
                                    f"consumer[{rid}].query[{qi}].canonical_policy_mismatch:{k}"
                                )
                    if "target_timestamp_utc" not in qqm:
                        errs.append(
                            f"consumer[{rid}].query[{qi}].missing_target_timestamp_utc"
                        )
                    qsha = str(qr.get("query_manifest_sha256") or "")
                    if not verify_query_manifest_sha256(
                        recorded_sha=qsha,
                        target_event_id=str(qqm.get("target_event_id") or ""),
                        case_id=str(qqm.get("case_id") or ""),
                        feature=str(qqm.get("feature") or ""),
                        mode=mode,
                        exclude_target_event=bool(canon["exclude_target_event"]),
                        exclude_same_case_future=bool(canon["exclude_same_case_future"]),
                        expected_signature=qqm.get("expected_signature"),
                        exclude_target_case=bool(canon["exclude_target_case"]),
                        exclude_duplicate_event_fingerprints=bool(
                            canon["exclude_duplicate_event_fingerprints"]
                        ),
                        same_timestamp_policy=str(canon["same_timestamp_policy"]),
                        allowed_source_partitions=list(canon["allowed_source_partitions"]),
                        allowed_problem_case_scope=str(canon["allowed_problem_case_scope"]),
                        target_timestamp_utc=str(qqm.get("target_timestamp_utc") or ""),
                    ):
                        errs.append(f"consumer[{rid}].query[{qi}].query_manifest_sha_mismatch")
                recorded_set = str(
                    qset.get("query_set_sha256") or ref.get("query_set_sha256") or ""
                )
                expected_set = query_set_manifest_sha256(
                    run_id=rid,
                    query_count=qcount,
                    query_records=qrecs,
                )
                if not recorded_set:
                    errs.append(f"consumer[{rid}].missing_query_set_sha")
                elif recorded_set != expected_set:
                    errs.append(f"consumer[{rid}].query_set_sha_mismatch")

    meta_content = str(bank_meta.get("bundle_bank_content_sha256") or "")
    if meta_content and meta_content != content_sha:
        errs.append("bank_meta_content_sha_mismatch")

    passed = len(errs) == 0 and bool(content_sha) and bool(file_sha)
    return {
        "a8_1_pass": passed,
        "errors": errs,
        "bundle_bank_content_sha256": content_sha,
        "bundle_bank_file_sha256": file_sha,
        "observation_file_sha256": obs_file_sha,
        "unique_file_sha256": uniq_file_sha,
        "relational": relational,
        "required_consumers": list(required_consumer_run_ids),
        "present_consumers": sorted(by_run.keys()),
    }


# --- legacy helpers kept for older call sites / unit tests ---


def bank_content_payload(
    bundles: Sequence[Mapping[str, Any]],
    *,
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Legacy query-slice payload — must NOT be used as base bank content hash."""
    rows = []
    for b in bundles:
        rows.append(
            {
                "feature": str(b.get("feature") or ""),
                "tokens": list(b.get("tokens") or []),
                "roles": list(b.get("roles") or []),
                "fingerprint": str(b.get("fingerprint") or ""),
                "is_original": bool(b.get("is_original")),
                "event_id": str(b.get("event_id") or ""),
                "case_id": str(b.get("case_id") or ""),
            }
        )
    rows.sort(
        key=lambda r: (
            r["feature"],
            r["fingerprint"],
            r["case_id"],
            r["event_id"],
            tuple(r["tokens"]),
        )
    )
    return {"bundles": rows, "note": "legacy_query_slice_not_base_bank"}


def bundle_bank_content_sha256(
    bundles: Sequence[Mapping[str, Any]],
    *,
    meta: Optional[Mapping[str, Any]] = None,
) -> str:
    return sha256_bytes(canonical_json_bytes(bank_content_payload(bundles, meta=meta)))


def build_bank_integrity_record(
    *,
    bundles: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    mode: str,
    bank_file_path: Optional[Path] = None,
    continuous_features_only: bool = True,
) -> Dict[str, Any]:
    """Legacy wrapper — prefer build_two_tier_integrity_record for A8-1."""
    content_sha = bundle_bank_content_sha256(bundles, meta=meta)
    policy_sha = bank_mode_policy_sha256(
        mode=mode,
        exclude_target_event=bool(meta.get("exclude_target_event", True)),
        exclude_same_case_future=bool(meta.get("exclude_same_case_future", True)),
        continuous_features_only=continuous_features_only,
    )
    file_sha = (
        file_sha256(Path(bank_file_path)) if bank_file_path and Path(bank_file_path).exists() else None
    )
    return {
        "bundle_bank_file_sha256": file_sha,
        "bundle_bank_content_sha256": content_sha,
        "bundle_bank_mode": str(mode),
        "bundle_bank_policy_sha256": policy_sha,
        "n_bundles": len(bundles),
        "legacy_query_slice_hash": True,
    }
