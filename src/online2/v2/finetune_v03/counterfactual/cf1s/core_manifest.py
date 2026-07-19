"""Search freeze / evaluation closure for CF-1S Core."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .core_contract import CoreContractError, sha256_bytes, sha256_json


def candidate_identity_projection(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Effect-free identity projection used for holdout set agreement."""
    return {
        "candidate_id": candidate.get("candidate_id"),
        "case_id": candidate.get("case_id"),
        "family_id": candidate.get("family_id"),
        "label": candidate.get("label"),
        "event_ids": list(candidate.get("event_ids") or []),
        "atomic_payload_hashes": list(candidate.get("atomic_payload_hashes") or []),
        "raw_transaction_hash": candidate.get("raw_transaction_hash"),
        "parent_ids": list(candidate.get("parent_ids") or []),
        "retokenization_hash": candidate.get("retokenization_hash"),
    }


def freeze_evaluation_closure(
    *,
    cohort: str,
    case_id: str,
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    projections = [candidate_identity_projection(c) for c in candidates]
    projections = sorted(projections, key=lambda x: str(x.get("candidate_id")))
    identity_sha = sha256_json(projections)
    return {
        "cohort": cohort,
        "case_id": case_id,
        "n_candidates": len(projections),
        "candidates": projections,
        "evaluation_closure_identity_sha256": identity_sha,
        "frozen": True,
        "parent_candidate_added_after_freeze_count": 0,
    }


def assert_holdout_matches_closure(
    *,
    closure: Mapping[str, Any],
    holdout_candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    holdout_proj = [candidate_identity_projection(c) for c in holdout_candidates]
    holdout_proj = sorted(holdout_proj, key=lambda x: str(x.get("candidate_id")))
    holdout_sha = sha256_json(holdout_proj)
    ok = holdout_sha == closure.get("evaluation_closure_identity_sha256")
    if not ok:
        raise CoreContractError(
            "holdout candidate set does not match frozen evaluation closure"
        )
    return {
        "holdout_set_matches_evaluation_closure": True,
        "holdout_identity_projection_sha256": holdout_sha,
        "evaluation_closure_identity_sha256": closure.get(
            "evaluation_closure_identity_sha256"
        ),
    }


def write_closure_jsonl(
    closures: Sequence[Mapping[str, Any]],
    out_path: Path,
) -> Dict[str, str]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(c, sort_keys=True, ensure_ascii=False) for c in closures
    ]
    text = "\n".join(lines) + ("\n" if lines else "")
    out_path.write_text(text, encoding="utf-8")
    digest = sha256_bytes(text.encode("utf-8"))
    sha_path = out_path.with_suffix(out_path.suffix + ".sha256")
    sha_path.write_text(digest + "  " + out_path.name + "\n", encoding="utf-8")
    return {"path": str(out_path), "sha256": digest, "sha_path": str(sha_path)}


class FoldForwardTrace:
    """Records fold forwards to prove holdout isolation."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._closure_frozen = False

    def mark_closure_frozen(self) -> None:
        self._closure_frozen = True

    def record(self, *, scope: str, fold_ids: Sequence[int]) -> None:
        if scope in ("holdout", "selection_blind_reevaluation", "fold2") and not self._closure_frozen:
            raise CoreContractError(f"{scope} forward before freeze")
        self.events.append(
            {
                "scope": scope,
                "fold_ids": [int(x) for x in fold_ids],
                "closure_frozen": self._closure_frozen,
            }
        )

    def summarize(self, *, search_fold_ids: Sequence[int], holdout_fold_ids: Sequence[int]) -> Dict[str, Any]:
        search = set(int(x) for x in search_fold_ids)
        holdout = set(int(x) for x in holdout_fold_ids)
        holdout_before = 0
        nonrequested = 0
        for ev in self.events:
            fids = set(ev["fold_ids"])
            if ev["scope"] in ("holdout", "selection_blind_reevaluation", "fold2") and not ev["closure_frozen"]:
                holdout_before += 1
            if ev["scope"] == "search" and fids - search:
                nonrequested += len(fids - search)
            if ev["scope"] in ("holdout", "selection_blind_reevaluation", "fold2") and fids - holdout:
                nonrequested += len(fids - holdout)
        return {
            "holdout_forward_before_freeze_count": holdout_before,
            "nonrequested_fold_forward_count": nonrequested,
            "events": list(self.events),
        }
