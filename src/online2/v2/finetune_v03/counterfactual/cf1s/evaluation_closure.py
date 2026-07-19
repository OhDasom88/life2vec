"""Selection / evaluation-closure manifests and identity projections."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..candidates.cf1s_composer import multi_event_candidate_id


CANDIDATE_IDENTITY_PROJECTION_VERSION = "CF1S_CANDIDATE_IDENTITY_V1"


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_edit_payload_hash(edit: Mapping[str, Any]) -> str:
    payload = {
        "event_id": str(edit.get("event_id") or ""),
        "feature_id": str(edit.get("feature_id") or edit.get("feature") or ""),
        "mg_id": str(edit.get("mg_id") or ""),
        "observed_raw": edit.get("observed_raw"),
        "target_raw": edit.get("target_raw"),
        "schema_kind": str(edit.get("schema_kind") or ""),
    }
    return sha256_text(_stable_json(payload))


def candidate_identity_projection(
    *,
    candidate_id: str,
    case_id: str,
    selector: str,
    edits: Sequence[Mapping[str, Any]],
    parent_candidate_ids: Optional[Sequence[str]] = None,
    retokenization_input_hash: Optional[str] = None,
) -> Dict[str, Any]:
    hashes = [atomic_edit_payload_hash(e) for e in edits]
    hashes_sorted = sorted(hashes)
    if retokenization_input_hash is None:
        retokenization_input_hash = sha256_text(_stable_json(hashes_sorted))
    parents = sorted(str(x) for x in (parent_candidate_ids or []))
    return {
        "candidate_identity_projection_version": CANDIDATE_IDENTITY_PROJECTION_VERSION,
        "candidate_id": str(candidate_id),
        "case_id": str(case_id),
        "selector": str(selector),
        "atomic_edit_payload_hashes": hashes_sorted,
        "retokenization_input_hash": retokenization_input_hash,
        "parent_candidate_ids": parents,
    }


def identity_projection_sha256(projection: Mapping[str, Any]) -> str:
    return sha256_text(_stable_json(dict(projection)))


def build_selected_manifest_rows(
    bundles: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    selector: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, bundle in enumerate(bundles):
        edits = list(bundle.get("edits") or [])
        cid = str(
            bundle.get("multi_event_candidate_id")
            or multi_event_candidate_id(case_id=case_id, edits=edits)
        )
        parents = list(bundle.get("atomic_parent_candidate_ids") or [])
        proj = candidate_identity_projection(
            candidate_id=cid,
            case_id=case_id,
            selector=selector,
            edits=edits,
            parent_candidate_ids=parents,
        )
        rows.append(
            {
                **proj,
                "search_selection_rank": rank,
                "search_selection_reason": "BEAM_SELECTED",
                "n_events": int(bundle.get("n_events") or len(edits)),
                "identity_sha256": identity_projection_sha256(proj),
            }
        )
    return rows


def _subset_edits(edits: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [dict(edits[i]) for i in indices]


def parent_family_members(
    bundle: Mapping[str, Any],
    *,
    case_id: str,
) -> List[Dict[str, Any]]:
    """Return all parent-family candidates for a selected 2/3-event bundle."""
    edits = list(bundle.get("edits") or [])
    n = len(edits)
    members: List[Dict[str, Any]] = []
    if n <= 1:
        return members

    def _member(subset: Sequence[int], role: str) -> Dict[str, Any]:
        sub = _subset_edits(edits, subset)
        cid = multi_event_candidate_id(case_id=case_id, edits=sub)
        parent_ids = [
            multi_event_candidate_id(case_id=case_id, edits=[edits[i]]) for i in subset
        ] if len(subset) > 1 else []
        return {
            "candidate_id": cid,
            "role": role,
            "edits": sub,
            "n_events": len(sub),
            "parent_candidate_ids": parent_ids,
            "from_selected_bundle_id": str(
                bundle.get("multi_event_candidate_id")
                or multi_event_candidate_id(case_id=case_id, edits=edits)
            ),
        }

    # Atomics
    for i in range(n):
        members.append(_member([i], f"ATOMIC_{i}"))
    if n == 2:
        members.append(_member([0, 1], "PAIR_AB"))
    elif n >= 3:
        members.append(_member([0, 1], "PAIR_AB"))
        members.append(_member([0, 2], "PAIR_AC"))
        members.append(_member([1, 2], "PAIR_BC"))
        members.append(_member([0, 1, 2], "TRIPLE_ABC"))
    return members


def build_evaluation_closure(
    selected_bundles: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    selector: str,
    already_scored: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build parent-family evaluation closure for selected bundles (analysis only)."""
    scored = dict(already_scored or {})
    closure_candidates: Dict[str, Dict[str, Any]] = {}
    closure_only_ids: List[str] = []

    for bundle in selected_bundles:
        for member in parent_family_members(bundle, case_id=case_id):
            cid = str(member["candidate_id"])
            if cid in closure_candidates:
                continue
            proj = candidate_identity_projection(
                candidate_id=cid,
                case_id=case_id,
                selector=selector,
                edits=member["edits"],
                parent_candidate_ids=member.get("parent_candidate_ids"),
            )
            in_beam = cid in scored
            row = {
                **proj,
                "role": member["role"],
                "n_events": member["n_events"],
                "edits": member["edits"],
                "from_selected_bundle_id": member["from_selected_bundle_id"],
                "identity_sha256": identity_projection_sha256(proj),
                "present_in_beam_or_selection": in_beam,
                "closure_only": not in_beam,
            }
            closure_candidates[cid] = row
            if not in_beam:
                closure_only_ids.append(cid)

    ordered = sorted(closure_candidates.values(), key=lambda r: r["candidate_id"])
    identity_rows = [
        {
            "candidate_id": r["candidate_id"],
            "case_id": r["case_id"],
            "selector": r["selector"],
            "atomic_edit_payload_hashes": r["atomic_edit_payload_hashes"],
            "retokenization_input_hash": r["retokenization_input_hash"],
            "parent_candidate_ids": r["parent_candidate_ids"],
            "candidate_identity_projection_version": r[
                "candidate_identity_projection_version"
            ],
        }
        for r in ordered
    ]
    identity_sha = sha256_text(_stable_json(identity_rows))
    return {
        "candidates": ordered,
        "closure_only_candidate_ids": sorted(closure_only_ids),
        "evaluation_closure_identity_sha256": identity_sha,
        "n_candidates": len(ordered),
        "n_closure_only": len(closure_only_ids),
    }


def lock_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    """Return (jsonl_text, sha256 of identity projection list)."""
    identity_rows = []
    lines = []
    for r in rows:
        lines.append(_stable_json(r))
        identity_rows.append(
            {
                "candidate_id": r.get("candidate_id"),
                "case_id": r.get("case_id"),
                "selector": r.get("selector"),
                "atomic_edit_payload_hashes": r.get("atomic_edit_payload_hashes"),
                "retokenization_input_hash": r.get("retokenization_input_hash"),
                "parent_candidate_ids": r.get("parent_candidate_ids"),
                "candidate_identity_projection_version": r.get(
                    "candidate_identity_projection_version",
                    CANDIDATE_IDENTITY_PROJECTION_VERSION,
                ),
            }
        )
    jsonl = "\n".join(lines) + ("\n" if lines else "")
    return jsonl, sha256_text(_stable_json(identity_rows))


def holdout_set_matches_closure(
    holdout_identity_sha: str,
    evaluation_closure_identity_sha: str,
) -> bool:
    return str(holdout_identity_sha) == str(evaluation_closure_identity_sha)


def assert_no_parent_added_after_freeze(
    frozen_ids: Iterable[str],
    proposed_ids: Iterable[str],
) -> int:
    frozen = set(str(x) for x in frozen_ids)
    proposed = set(str(x) for x in proposed_ids)
    added = proposed - frozen
    return len(added)
