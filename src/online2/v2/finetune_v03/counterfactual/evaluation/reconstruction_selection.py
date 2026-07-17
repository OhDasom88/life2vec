"""Deterministic reconstruction MG selection with recoverable-only expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_EXPANSION_SCHEDULE = (
    {"max_mg_per_feature": 5},
    {"max_mg_per_feature": 10},
    {"max_mg_per_feature": 20},
)

SELECTION_ORDER_VERSION = "hash_order_v1"
DEFAULT_SELECTION_SEED = 20250717
MIN_RECONSTRUCTION_CANDIDATES = 5


def _stable_json_sha(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mg_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("case_id") or ""),
            str(row.get("event_id") or ""),
            str(row.get("measurement_group_id") or ""),
            str(row.get("feature") or ""),
        ]
    )


def eligible_pool_sha256(pool: Sequence[Mapping[str, Any]]) -> str:
    keys = sorted(mg_key(r) for r in pool)
    return _stable_json_sha({"eligible_mg_keys": keys})


def select_mg_keys_deterministic(
    pool: Sequence[Mapping[str, Any]],
    *,
    max_mg_per_feature: int,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    selection_order_version: str = SELECTION_ORDER_VERSION,
) -> List[str]:
    """Fixed-seed hash ordering (hash_order_v1), then per-feature cap."""
    def _order_key(r: Mapping[str, Any]) -> Tuple[str, str]:
        k = mg_key(r)
        h = hashlib.sha256(f"{selection_seed}|{selection_order_version}|{k}".encode()).hexdigest()
        return (h, k)

    rows = sorted(pool, key=_order_key)
    per_feat: Dict[str, int] = {}
    keys: List[str] = []
    for r in rows:
        feat = str(r.get("feature") or "")
        if per_feat.get(feat, 0) >= int(max_mg_per_feature):
            continue
        k = mg_key(r)
        if k in keys:
            continue
        keys.append(k)
        per_feat[feat] = per_feat.get(feat, 0) + 1
    return keys


FINAL_MANIFEST_HASH_EXCLUDE = frozenset(
    {
        "stage_sha256",
        "selection_manifest_final_sha256",
        "selection_manifest_chain_hash",
    }
)


def frozen_final_manifest_sha256(final_doc: Mapping[str, Any]) -> str:
    """SHA of frozen final payload excluding self-referential hash fields."""
    payload = {
        k: v for k, v in dict(final_doc).items() if k not in FINAL_MANIFEST_HASH_EXCLUDE
    }
    return _stable_json_sha(payload)


def build_selection_chain_document(
    *,
    stage_artifacts: Sequence[Mapping[str, Any]],
    selection_manifest_final_sha256: str,
    final_stage: int,
    stopped_reason: str,
    recoverable_mg_count: int,
) -> Dict[str, Any]:
    """Non-circular chain: final SHA first, then chain hash of payload (no self-ref)."""
    stage_manifests = [
        {"stage": int(a.get("stage") or i + 1), "sha256": str(a.get("stage_sha256") or "")}
        for i, a in enumerate(stage_artifacts)
    ]
    payload = {
        "chain_version": "selection_chain_v1",
        "stage_manifests": stage_manifests,
        "selection_manifest_final_sha256": str(selection_manifest_final_sha256),
    }
    chain_hash = _stable_json_sha(payload)
    return {
        **payload,
        "selection_manifest_chain_hash": chain_hash,
        "stage_hashes": [m["sha256"] for m in stage_manifests] + [str(selection_manifest_final_sha256)],
        "final_stage": int(final_stage),
        "stopped_reason": str(stopped_reason),
        "recoverable_mg_count": int(recoverable_mg_count),
        "frozen": True,
    }


def run_expansion_stages(
    eligible_pool: Sequence[Mapping[str, Any]],
    *,
    recoverable_fn,
    schedule: Sequence[Mapping[str, Any]] = DEFAULT_EXPANSION_SCHEDULE,
    min_recoverable_mg: int = 20,
    out_dir: Optional[Path] = None,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    selection_order_version: str = SELECTION_ORDER_VERSION,
) -> Dict[str, Any]:
    """Expand stages until recoverable_mg_count >= threshold.

    Expansion decisions use ONLY recoverable_mg_count (never coverage/Recall/Lift).
    recoverable_fn(selected_rows) -> int

    Note: provisional final/chain written here are pre-freeze. Callers that freeze
    flags must recompute final SHA + chain via build_selection_chain_document.
    """
    pool_sha = eligible_pool_sha256(eligible_pool)
    stage_artifacts: List[Dict[str, Any]] = []
    previous_hash = None
    final_keys: List[str] = []
    stopped_reason = "SCHEDULE_EXHAUSTED"
    final_stage = 0
    recoverable = 0

    for i, step in enumerate(schedule, start=1):
        max_per = int(step["max_mg_per_feature"])
        keys = select_mg_keys_deterministic(
            eligible_pool,
            max_mg_per_feature=max_per,
            selection_seed=selection_seed,
            selection_order_version=selection_order_version,
        )
        # prefix/superset of previous
        if final_keys and not set(final_keys).issubset(set(keys)):
            # enforce monotonic growth by union preserving order
            merged = list(final_keys)
            for k in keys:
                if k not in merged:
                    merged.append(k)
            keys = merged
        key_set = set(keys)
        selected_rows = [r for r in eligible_pool if mg_key(r) in key_set]
        recoverable = int(recoverable_fn(selected_rows))
        art = {
            "stage": i,
            "max_mg_per_feature": max_per,
            "previous_stage_hash": previous_hash,
            "selected_mg_keys": keys,
            "recoverable_mg_count": recoverable,
            "eligible_mg_count": len(eligible_pool),
            "unique_candidate_count": len(keys),
            "expansion_reason": (
                "MIN_RECOVERABLE_MET"
                if recoverable >= int(min_recoverable_mg)
                else "MIN_RECOVERABLE_NOT_MET"
            ),
        }
        art["stage_sha256"] = _stable_json_sha(
            {k: v for k, v in art.items() if k != "stage_sha256"}
        )
        previous_hash = art["stage_sha256"]
        stage_artifacts.append(art)
        final_keys = keys
        final_stage = i
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"selection_manifest_stage_{i:02d}.json").write_text(
                json.dumps(art, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        if recoverable >= int(min_recoverable_mg):
            stopped_reason = "MIN_RECOVERABLE_MET"
            break

    # Provisional final WITHOUT self-referential hash fields in payload used for SHA
    final = {
        "stage": "final",
        "final_stage": final_stage,
        "previous_stage_hash": previous_hash,
        "selected_mg_keys": final_keys,
        "recoverable_mg_count": recoverable,
        "eligible_mg_count": len(eligible_pool),
        "stopped_reason": stopped_reason,
        "min_recoverable_mg": int(min_recoverable_mg),
        "selection_seed": int(selection_seed),
        "selection_order_version": str(selection_order_version),
        "eligible_pool_sha256": pool_sha,
    }
    final_sha = frozen_final_manifest_sha256(final)
    final["selection_manifest_final_sha256"] = final_sha
    # stage_sha256 on final is alias of final content sha (not circular with chain)
    final["stage_sha256"] = final_sha
    if out_dir is not None:
        (Path(out_dir) / "selection_manifest_final.json").write_text(
            json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    chain_doc = build_selection_chain_document(
        stage_artifacts=stage_artifacts,
        selection_manifest_final_sha256=final_sha,
        final_stage=final_stage,
        stopped_reason=stopped_reason,
        recoverable_mg_count=recoverable,
    )
    # provisional chain is not frozen yet
    chain_doc["frozen"] = False
    if out_dir is not None:
        (Path(out_dir) / "selection_manifest_chain.json").write_text(
            json.dumps(chain_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return {
        "stages": stage_artifacts,
        "final": final,
        "chain": chain_doc,
        "selected_mg_keys": final_keys,
        "quality_gate_ready": recoverable >= int(min_recoverable_mg),
    }


def compute_selection_manifest_chain_hash(
    stage_hashes: Sequence[str],
    *,
    selection_manifest_final_sha256: Optional[str] = None,
) -> str:
    """Legacy helper; prefer build_selection_chain_document for A8-2 binding."""
    if selection_manifest_final_sha256 is not None:
        stage_manifests = [
            {"stage": i + 1, "sha256": str(h)} for i, h in enumerate(stage_hashes)
        ]
        payload = {
            "chain_version": "selection_chain_v1",
            "stage_manifests": stage_manifests,
            "selection_manifest_final_sha256": str(selection_manifest_final_sha256),
        }
        return _stable_json_sha(payload)
    chain = hashlib.sha256()
    for h in stage_hashes:
        chain.update(str(h).encode("utf-8"))
        chain.update(b"\n")
    return chain.hexdigest()


def build_selected_mg_flag_rows(
    selected_mg_keys: Sequence[str],
    *,
    recoverable_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    rec_set = set(str(k) for k in recoverable_keys)
    return [
        {"mg_key": str(k), "is_recoverable": str(k) in rec_set}
        for k in selected_mg_keys
    ]


def exact_match_selection_metrics(
    *,
    manifest_keys: Sequence[str],
    metrics_keys: Sequence[str],
    manifest_recoverable_flags: Sequence[Mapping[str, Any]],
    metrics_recoverable_flags: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """A8-2: selected keys and recoverable flags must match exactly."""
    m_keys = [str(k) for k in manifest_keys]
    q_keys = [str(k) for k in metrics_keys]
    keys_match = m_keys == q_keys

    def _flag_map(rows: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
        out: Dict[str, bool] = {}
        for r in rows:
            out[str(r.get("mg_key"))] = bool(r.get("is_recoverable"))
        return out

    m_flags = _flag_map(manifest_recoverable_flags)
    q_flags = _flag_map(metrics_recoverable_flags)
    flags_match = m_flags == q_flags and set(m_flags) == set(m_keys) == set(q_keys)

    passed = keys_match and flags_match
    return {
        "a8_2_pass": passed,
        "selected_keys_exact_match": keys_match,
        "recoverable_flags_exact_match": flags_match,
        "manifest_key_count": len(m_keys),
        "metrics_key_count": len(q_keys),
        "errors": (
            []
            if passed
            else (
                (["selected_keys_mismatch"] if not keys_match else [])
                + (["recoverable_flags_mismatch"] if not flags_match else [])
            )
        ),
    }
