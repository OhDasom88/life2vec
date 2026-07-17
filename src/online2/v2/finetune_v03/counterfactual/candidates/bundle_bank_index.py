"""Build and query immutable 2-tier MLM bundle bank (observations + unique)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..evaluation.bank_integrity import (
    BANK_MODE_DEPLOYMENT,
    BANK_MODE_RECONSTRUCTION_EVAL,
    BANK_SCHEMA_VERSION,
    SOURCE_PARTITION_PROBLEM,
    SOURCE_PARTITION_TRAINING,
    build_two_tier_integrity_record,
    canonical_policy_fields_for_mode,
)


def normalize_utc_timestamp(value: Any) -> pd.Timestamp:
    """Compare-safe UTC timestamp: naive → localize UTC; aware → convert UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def canonical_timestamp_utc(ts: Any) -> str:
    """Normalize to YYYY-MM-DDTHH:MM:SS.ffffffZ."""
    if ts is None or ts == "":
        return ""
    t = normalize_utc_timestamp(ts).tz_localize(None)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(t.microsecond):06d}Z"


def make_bundle_id(
    feature: str,
    tokens: Sequence[str],
    role_signature: str = "",
) -> str:
    """sha256(feature + role_signature + canonical_full_mg_bundle)."""
    payload = json.dumps(
        {
            "feature": str(feature),
            "role_signature": str(role_signature),
            "canonical_full_mg_bundle": list(tokens),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_event_fingerprint(
    *,
    case_id: str,
    event_id: str,
    measurement_group_id: str,
    timestamp_utc: str,
    feature: str,
    bundle_id: str,
) -> str:
    """Observation-unit fingerprint (distinct from bundle_id)."""
    payload = "|".join(
        [
            str(case_id),
            str(event_id),
            str(measurement_group_id),
            str(timestamp_utc),
            str(feature),
            str(bundle_id),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def observations_and_uniques_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert flat event-MG rows into observation + unique indexes."""
    observations: List[Dict[str, Any]] = []
    by_bundle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        feat = str(row.get("feature") or "")
        tokens = list(row.get("tokens") or [])
        if not feat or not tokens:
            continue
        roles = list(row.get("roles") or [])
        sig = str(row.get("signature") or "|".join(str(r) for r in roles))
        bid = make_bundle_id(feat, tokens, role_signature=sig)
        ts_utc = canonical_timestamp_utc(row.get("timestamp") or row.get("timestamp_utc"))
        mg_id = str(row.get("measurement_group_id") or "")
        case_id = str(row.get("case_id") or "")
        event_id = str(row.get("event_id") or "")
        efp = make_event_fingerprint(
            case_id=case_id,
            event_id=event_id,
            measurement_group_id=mg_id,
            timestamp_utc=ts_utc,
            feature=feat,
            bundle_id=bid,
        )
        part = str(row.get("source_partition") or "").upper()
        if part not in {SOURCE_PARTITION_TRAINING, SOURCE_PARTITION_PROBLEM}:
            raise ValueError(
                f"invalid_or_missing_source_partition:{row.get('source_partition')!r}"
                f":case_id={case_id}:event_id={event_id}"
            )
        obs = {
            "bundle_id": bid,
            "feature": feat,
            "role_signature": sig,
            "case_id": case_id,
            "event_id": event_id,
            "measurement_group_id": mg_id,
            "timestamp_utc": ts_utc,
            "event_fingerprint": efp,
            "source_partition": part,
        }
        observations.append(obs)
        by_bundle[bid].append(
            {
                **obs,
                "tokens": tokens,
                "roles": roles,
                "token_ids": list(row.get("token_ids") or []),
            }
        )

    uniques: List[Dict[str, Any]] = []
    for bid, items in sorted(by_bundle.items()):
        tokens = list(items[0]["tokens"])
        roles = list(items[0]["roles"])
        token_ids = list(items[0].get("token_ids") or [])
        raw_n = len(items)
        seen = set()
        dedup = 0
        for it in items:
            key = it["event_fingerprint"]
            if key in seen:
                continue
            seen.add(key)
            dedup += 1
        uniques.append(
            {
                "bundle_id": bid,
                "feature": items[0]["feature"],
                "role_signature": items[0]["role_signature"],
                "canonical_full_mg_bundle": tokens,
                "tokens": tokens,
                "roles": roles,
                "token_ids": token_ids,
                "raw_observation_count": raw_n,
                "deduplicated_observation_count": dedup,
                "observation_count_used_for_provenance": dedup,
                "observation_count": dedup,
            }
        )
    return observations, uniques


def write_two_tier_bank(
    out_dir: Path,
    *,
    observations: Sequence[Mapping[str, Any]],
    uniques: Sequence[Mapping[str, Any]],
    mode: str = BANK_MODE_DEPLOYMENT,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_path = out_dir / "mlm_bundle_bank_observations.parquet"
    uniq_path = out_dir / "mlm_bundle_bank_unique.parquet"
    pd.DataFrame(list(observations)).to_parquet(obs_path, index=False)
    pd.DataFrame(list(uniques)).to_parquet(uniq_path, index=False)

    integrity = build_two_tier_integrity_record(
        observations=observations,
        uniques=uniques,
        mode=mode,
        observation_file=obs_path,
        unique_file=uniq_path,
    )
    manifest = {
        "bank_schema_version": BANK_SCHEMA_VERSION,
        "mode": mode,
        "observation_path": str(obs_path),
        "unique_path": str(uniq_path),
        **integrity,
    }
    man_path = out_dir / "mlm_bundle_bank_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(man_path)
    return manifest


def load_two_tier_bank(bank_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    bank_dir = Path(bank_dir)
    man_path = bank_dir / "mlm_bundle_bank_manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"missing bank manifest: {man_path}")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    obs_path = Path(manifest.get("observation_path") or (bank_dir / "mlm_bundle_bank_observations.parquet"))
    uniq_path = Path(manifest.get("unique_path") or (bank_dir / "mlm_bundle_bank_unique.parquet"))
    if not obs_path.is_file() or not uniq_path.is_file():
        raise FileNotFoundError(f"missing bank parquet under {bank_dir}")
    observations = pd.read_parquet(obs_path).to_dict(orient="records")
    uniques = pd.read_parquet(uniq_path).to_dict(orient="records")
    return observations, uniques, manifest


def annotate_bundles_with_token_ids(
    bundles: Sequence[Mapping[str, Any]],
    *,
    vocab_token2index: Mapping[str, int],
    unk_id: int = 0,
) -> List[Dict[str, Any]]:
    """Ensure each bundle has token_ids for scoring."""
    out: List[Dict[str, Any]] = []
    for b in bundles:
        row = dict(b)
        tokens = list(row.get("tokens") or row.get("canonical_full_mg_bundle") or [])
        row["tokens"] = tokens
        tids = list(row.get("token_ids") or [])
        if len(tids) != len(tokens):
            tids = [int(vocab_token2index.get(t, unk_id)) for t in tokens]
        row["token_ids"] = tids
        if "roles" not in row or row["roles"] is None:
            sig = str(row.get("role_signature") or "")
            row["roles"] = [x for x in sig.split("|") if x]
        out.append(row)
    return out


def _observation_partition_allowed(
    *,
    source_partition: str,
    oid_case: str,
    case_id: Optional[str],
    mode: str,
    allowed_source_partitions: Sequence[str],
    allowed_problem_case_scope: str,
) -> bool:
    """Apply partition + problem-case scope from canonical policy."""
    part = str(source_partition or "").upper()
    allowed = {str(p).upper() for p in allowed_source_partitions}
    scope = str(allowed_problem_case_scope or "")
    if mode == BANK_MODE_RECONSTRUCTION_EVAL or scope == "NONE":
        return part == SOURCE_PARTITION_TRAINING and SOURCE_PARTITION_TRAINING in allowed
    # deployment TARGET_CASE_ONLY
    if part == SOURCE_PARTITION_TRAINING and SOURCE_PARTITION_TRAINING in allowed:
        return True
    if (
        part == SOURCE_PARTITION_PROBLEM
        and SOURCE_PARTITION_PROBLEM in allowed
        and scope == "TARGET_CASE_ONLY"
        and case_id
        and oid_case == str(case_id)
    ):
        return True
    return False


def _resolve_canonical_policy(
    mode: str,
    *,
    exclude_target_event: Optional[bool] = None,
    exclude_same_case_future: Optional[bool] = None,
    exclude_target_case: Optional[bool] = None,
    exclude_duplicate_event_fingerprints: Optional[bool] = None,
    same_timestamp_policy: Optional[str] = None,
    allowed_source_partitions: Optional[Sequence[str]] = None,
    allowed_problem_case_scope: Optional[str] = None,
) -> Dict[str, Any]:
    """Return canonical policy; raise if any provided override differs."""
    canon = canonical_policy_fields_for_mode(mode)
    checks = {
        "exclude_target_event": exclude_target_event,
        "exclude_same_case_future": exclude_same_case_future,
        "exclude_target_case": exclude_target_case,
        "exclude_duplicate_event_fingerprints": exclude_duplicate_event_fingerprints,
        "same_timestamp_policy": same_timestamp_policy,
        "allowed_source_partitions": allowed_source_partitions,
        "allowed_problem_case_scope": allowed_problem_case_scope,
    }
    for key, got in checks.items():
        if got is None:
            continue
        expected = canon[key]
        if key == "allowed_source_partitions":
            if list(got) != list(expected):
                raise ValueError(f"noncanonical_policy_override:{key}")
        elif key in {"same_timestamp_policy", "allowed_problem_case_scope"}:
            if str(got) != str(expected):
                raise ValueError(f"noncanonical_policy_override:{key}")
        else:
            if bool(got) != bool(expected):
                raise ValueError(f"noncanonical_policy_override:{key}")
    return canon


def query_unique_bundles_from_bank(
    observations: Sequence[Mapping[str, Any]],
    uniques: Sequence[Mapping[str, Any]],
    *,
    feature: str,
    expected_roles: Optional[Sequence[str]] = None,
    target_event_id: Optional[str] = None,
    target_timestamp: Optional[Any] = None,
    case_id: Optional[str] = None,
    mode: str = "deployment",
    exclude_target_event: Optional[bool] = None,
    exclude_same_case_future: Optional[bool] = None,
    exclude_target_case: Optional[bool] = None,
    exclude_duplicate_event_fingerprints: Optional[bool] = None,
    same_timestamp_policy: Optional[str] = None,
    allowed_source_partitions: Optional[Sequence[str]] = None,
    allowed_problem_case_scope: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter by canonical policy. Returns (bundles, executed_query_manifest).

    Non-canonical policy kwargs raise ValueError.
    """
    from ..evaluation.bank_integrity import bank_query_manifest_sha256

    mode_s = str(mode)
    canon = _resolve_canonical_policy(
        mode_s,
        exclude_target_event=exclude_target_event,
        exclude_same_case_future=exclude_same_case_future,
        exclude_target_case=exclude_target_case,
        exclude_duplicate_event_fingerprints=exclude_duplicate_event_fingerprints,
        same_timestamp_policy=same_timestamp_policy,
        allowed_source_partitions=allowed_source_partitions,
        allowed_problem_case_scope=allowed_problem_case_scope,
    )
    exclude_target_event = bool(canon["exclude_target_event"])
    exclude_same_case_future = bool(canon["exclude_same_case_future"])
    exclude_target_case = bool(canon["exclude_target_case"])
    exclude_duplicate_event_fingerprints = bool(
        canon["exclude_duplicate_event_fingerprints"]
    )
    same_timestamp_policy = str(canon["same_timestamp_policy"])
    allowed_source_partitions = list(canon["allowed_source_partitions"])
    allowed_problem_case_scope = str(canon["allowed_problem_case_scope"])

    feat = str(feature)
    target_ts_utc_str = (
        canonical_timestamp_utc(target_timestamp) if target_timestamp is not None else ""
    )
    target_ts_utc = (
        normalize_utc_timestamp(target_timestamp) if target_timestamp is not None else None
    )
    kept_obs: List[Mapping[str, Any]] = []
    seen_fps: set = set()
    for o in observations:
        if str(o.get("feature")) != feat and str(o.get("feature")).lower() != feat.lower():
            continue
        oid_case = str(o.get("case_id") or "")
        part = str(o.get("source_partition") or "").upper()
        if not part:
            continue
        if not _observation_partition_allowed(
            source_partition=part,
            oid_case=oid_case,
            case_id=case_id,
            mode=mode_s,
            allowed_source_partitions=allowed_source_partitions,
            allowed_problem_case_scope=allowed_problem_case_scope,
        ):
            continue
        if exclude_target_case and case_id and oid_case == str(case_id):
            continue
        eid = str(o.get("event_id") or "")
        if exclude_target_event and target_event_id and eid == str(target_event_id):
            continue
        ts_raw = o.get("timestamp_utc") or o.get("timestamp")
        if exclude_same_case_future and case_id and target_ts_utc is not None and ts_raw:
            obs_ts = normalize_utc_timestamp(ts_raw)
            if oid_case == str(case_id) and obs_ts >= target_ts_utc:
                continue
        if (
            same_timestamp_policy == "EXCLUDE"
            and case_id
            and target_ts_utc is not None
            and oid_case == str(case_id)
            and ts_raw
        ):
            if normalize_utc_timestamp(ts_raw) == target_ts_utc:
                continue
        if expected_roles is not None:
            sig = str(o.get("role_signature") or "")
            if sig != "|".join(str(r) for r in expected_roles) and sig != str(
                tuple(expected_roles)
            ):
                role_list = [x for x in sig.split("|") if x]
                if role_list != list(expected_roles):
                    continue
        efp = str(o.get("event_fingerprint") or "")
        if exclude_duplicate_event_fingerprints and efp:
            if efp in seen_fps:
                continue
            seen_fps.add(efp)
        kept_obs.append(o)

    eligible_counts: Counter = Counter(str(o.get("bundle_id") or "") for o in kept_obs)
    kept_ids = set(eligible_counts.keys()) - {""}

    out = []
    for u in uniques:
        bid = str(u.get("bundle_id") or "")
        if bid not in kept_ids:
            continue
        if str(u.get("feature")) != feat and str(u.get("feature")).lower() != feat.lower():
            continue
        row = dict(u)
        row["tokens"] = list(u.get("canonical_full_mg_bundle") or u.get("tokens") or [])
        base = int(
            u.get("observation_count_used_for_provenance")
            or u.get("deduplicated_observation_count")
            or u.get("observation_count")
            or 0
        )
        eligible = int(eligible_counts.get(bid, 0))
        row["base_observation_count"] = base
        row["eligible_observation_count"] = eligible
        row["observation_count_used_for_provenance"] = eligible
        row["observation_count"] = eligible
        out.append(row)

    executed = {
        "target_event_id": str(target_event_id or ""),
        "case_id": str(case_id or ""),
        "feature": str(feature),
        "mode": mode_s,
        "exclude_target_event": exclude_target_event,
        "exclude_same_case_future": exclude_same_case_future,
        "exclude_target_case": exclude_target_case,
        "exclude_duplicate_event_fingerprints": exclude_duplicate_event_fingerprints,
        "same_timestamp_policy": same_timestamp_policy,
        "allowed_source_partitions": allowed_source_partitions,
        "allowed_problem_case_scope": allowed_problem_case_scope,
        "target_timestamp_utc": target_ts_utc_str,
        "expected_signature": list(expected_roles or []),
    }
    executed["query_manifest_sha256"] = bank_query_manifest_sha256(**executed)
    return out, executed


def wrap_executed_query_record(
    *,
    run_id: str,
    bank_manifest: Mapping[str, Any],
    executed_query_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Persist executed query evidence only (no post-hoc policy rewrite)."""
    from ..evaluation.bank_integrity import bank_mode_policy_sha256

    qm = dict(executed_query_manifest)
    mode = str(qm.get("mode") or "")
    canon = canonical_policy_fields_for_mode(mode)
    # executed must already match canonical
    _resolve_canonical_policy(
        mode,
        exclude_target_event=qm.get("exclude_target_event"),
        exclude_same_case_future=qm.get("exclude_same_case_future"),
        exclude_target_case=qm.get("exclude_target_case"),
        exclude_duplicate_event_fingerprints=qm.get("exclude_duplicate_event_fingerprints"),
        same_timestamp_policy=qm.get("same_timestamp_policy"),
        allowed_source_partitions=qm.get("allowed_source_partitions"),
        allowed_problem_case_scope=qm.get("allowed_problem_case_scope"),
    )
    query_sha = str(qm.get("query_manifest_sha256") or "")
    policy_sha = bank_mode_policy_sha256(
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
    clean_qm = {k: v for k, v in qm.items() if k != "query_manifest_sha256"}
    return {
        "run_id": str(run_id),
        "bundle_bank_content_sha256": bank_manifest.get("bundle_bank_content_sha256"),
        "observation_file_sha256": bank_manifest.get("observation_file_sha256"),
        "unique_file_sha256": bank_manifest.get("unique_file_sha256"),
        "bank_mode": mode,
        "bundle_bank_mode": mode,
        "bundle_bank_policy_sha256": policy_sha,
        "query_manifest": clean_qm,
        "bundle_bank_query_manifest_sha256": query_sha,
        "target_timestamp_utc": clean_qm.get("target_timestamp_utc"),
        "exclude_target_event": clean_qm.get("exclude_target_event"),
        "exclude_same_case_future": clean_qm.get("exclude_same_case_future"),
        "exclude_target_case": clean_qm.get("exclude_target_case"),
        "exclude_duplicate_event_fingerprints": clean_qm.get(
            "exclude_duplicate_event_fingerprints"
        ),
        "same_timestamp_policy": clean_qm.get("same_timestamp_policy"),
        "allowed_source_partitions": clean_qm.get("allowed_source_partitions"),
        "allowed_problem_case_scope": clean_qm.get("allowed_problem_case_scope"),
    }


def build_bank_query_record(
    *,
    run_id: str,
    bank_manifest: Mapping[str, Any],
    mode: str,
    target_event_id: str,
    case_id: str,
    feature: str,
    expected_signature: Optional[Sequence[str]] = None,
    target_timestamp: Optional[Any] = None,
    executed_query_manifest: Optional[Mapping[str, Any]] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Wrap an executed query manifest only — no post-hoc synthesis.

    ``executed_query_manifest`` is required (from ``query_unique_bundles_from_bank``).
    """
    del mode, target_event_id, case_id, feature, expected_signature, target_timestamp
    if executed_query_manifest is None:
        raise ValueError("executed_query_manifest_required")
    return wrap_executed_query_record(
        run_id=run_id,
        bank_manifest=bank_manifest,
        executed_query_manifest=executed_query_manifest,
    )


def build_bank_query_set_manifest(
    *,
    run_id: str,
    bank_manifest: Mapping[str, Any],
    mode: str,
    executed_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build query-set from executed manifests collected during the run.

    Each entry: {"mg_key": ..., "executed_query_manifest": {...}}
    """
    from ..evaluation.bank_integrity import query_set_manifest_sha256

    query_records = []
    first_full: Optional[Dict[str, Any]] = None
    for ent in executed_entries:
        executed = dict(ent.get("executed_query_manifest") or {})
        rec = wrap_executed_query_record(
            run_id=run_id,
            bank_manifest=bank_manifest,
            executed_query_manifest=executed,
        )
        if first_full is None:
            first_full = rec
        query_records.append(
            {
                "mg_key": str(ent.get("mg_key") or ""),
                "query_manifest": rec["query_manifest"],
                "query_manifest_sha256": rec["bundle_bank_query_manifest_sha256"],
            }
        )
    doc: Dict[str, Any] = {
        "run_id": str(run_id),
        "bank_mode": str(mode),
        "query_count": len(query_records),
        "query_records": query_records,
        "bundle_bank_content_sha256": bank_manifest.get("bundle_bank_content_sha256"),
        "observation_file_sha256": bank_manifest.get("observation_file_sha256"),
        "unique_file_sha256": bank_manifest.get("unique_file_sha256"),
    }
    if first_full is not None:
        doc.update(
            {
                "bundle_bank_policy_sha256": first_full["bundle_bank_policy_sha256"],
                "query_manifest": first_full["query_manifest"],
                "bundle_bank_query_manifest_sha256": first_full[
                    "bundle_bank_query_manifest_sha256"
                ],
                "target_timestamp_utc": first_full.get("target_timestamp_utc"),
                "exclude_target_event": first_full["exclude_target_event"],
                "exclude_same_case_future": first_full["exclude_same_case_future"],
                "exclude_target_case": first_full["exclude_target_case"],
                "exclude_duplicate_event_fingerprints": first_full[
                    "exclude_duplicate_event_fingerprints"
                ],
                "same_timestamp_policy": first_full["same_timestamp_policy"],
                "allowed_source_partitions": first_full["allowed_source_partitions"],
                "allowed_problem_case_scope": first_full["allowed_problem_case_scope"],
            }
        )
    doc["query_set_sha256"] = query_set_manifest_sha256(
        run_id=run_id,
        query_count=doc["query_count"],
        query_records=query_records,
    )
    return doc
