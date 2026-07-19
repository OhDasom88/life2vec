"""Tests for fresh cold diagnosis batch and stale-batch rejection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import CoreContractError
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_execution import (
    production_apply_edits_and_score,
    score_requested_folds_only,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_production import (
    assert_batch_matches_cold,
    build_cold_diagnosis_batch,
)
from tests.v2.counterfactual_cf1s.cf1s_test_tx import make_test_candidate_transaction


class _FakeReencoder:
    def cold_rebuild_full_sequence(self, events, batch_size=8):
        n = len(events)
        # Distinct values so stale reuse is detectable
        mean = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
        return {
            "event_mean": mean,
            "event_max": mean + 1,
            "valid_event_ids": [str(getattr(e, "event_id")) for e in events],
            "window_ids": [f"w{i}" for i in range(n)],
            "target_membership": {},
            "context_membership": {},
            "pre_cap_valid_event_count": n,
            "post_cap_valid_event_count": n,
            "valid_event_cap_applied": False,
            "excluded_tail_event_count": 0,
            "reencode_mode": "FULL_EVENT_COVERAGE_COLD_REBUILD",
        }


def test_fold_forward_requires_batch():
    calls = []

    def fold_forward_fn(path, batch):
        calls.append((path, batch["valid_event_ids"]))
        return {"risk": 0.2, "logit": [0.1]}

    batch = {
        "valid_event_ids": ["e1"],
        "event_mean": torch.zeros(1, 1, 2),
        "event_max": torch.zeros(1, 1, 2),
        "padding_mask": torch.ones(1, 1, dtype=torch.bool),
        "mask_semantics": "TRUE_IS_VALID",
    }
    out = score_requested_folds_only(
        fold_forward_fn=fold_forward_fn,
        selected_checkpoint_paths=["ck0"],
        selected_fold_ids=[0],
        diagnosis_batch=batch,
        scope="search",
    )
    assert out["forwarded_fold_count"] == 1
    assert calls[0][0] == "ck0"
    assert calls[0][1] == ["e1"]


def test_production_apply_rebuilds_batch_from_cold():
    events = [
        SimpleNamespace(event_id="e1", sentence_tokens=["A"], token_count=1),
        SimpleNamespace(event_id="e2", sentence_tokens=["B"], token_count=1),
    ]
    tx = make_test_candidate_transaction(event_id="e1", to_tokens=["A2"])
    seen_batches = []

    def fold_forward_fn(path, batch):
        seen_batches.append(batch)
        return {"risk": 0.5, "logit": [0.0]}

    sidecar = {
        "e1": {"view": "UNKNOWN", "zone": 1, "case_age_hours": 1.0, "local_hour": 3},
        "e2": {"view": "UNKNOWN", "zone": 2, "case_age_hours": 2.0, "local_hour": 4},
    }
    result = production_apply_edits_and_score(
        events=events,
        validated_transaction=tx,
        reencoder=_FakeReencoder(),
        fold_forward_fn=fold_forward_fn,
        checkpoint_paths=["ck0", "ck1"],
        fold_ids=[0, 1],
        requested_fold_ids=[0],
        sidecar_by_event_id=sidecar,
        scope="search",
    )
    assert result["reencode_mode"] == "FULL_EVENT_COVERAGE_COLD_REBUILD"
    assert result["scores"]["forwarded_fold_count"] == 1
    assert len(seen_batches) == 1
    batch = seen_batches[0]
    assert batch["event_mean"].shape == (1, 2, 4)
    assert batch["padding_mask"].dtype == torch.bool
    assert bool(batch["padding_mask"].all())
    # cold mean must match batch [0]
    assert torch.equal(batch["event_mean"][0], result["event_mean"])


def test_completed_diagnosis_batch_injection_rejected():
    events = [
        SimpleNamespace(event_id="e1", sentence_tokens=["A"], token_count=1),
    ]
    tx = make_test_candidate_transaction(event_id="e1", to_tokens=["A2"])

    def fold_forward_fn(path, batch):
        return {"risk": 0.1, "logit": [0.0]}

    with pytest.raises(CoreContractError, match="STALE_DIAGNOSIS_BATCH"):
        production_apply_edits_and_score(
            events=events,
            validated_transaction=tx,
            reencoder=_FakeReencoder(),
            fold_forward_fn=fold_forward_fn,
            checkpoint_paths=["ck0"],
            fold_ids=[0],
            requested_fold_ids=[0],
            diagnosis_batch={"event_mean": torch.zeros(1, 1, 4)},
        )


def test_bare_edits_rejected():
    events = [SimpleNamespace(event_id="e1", sentence_tokens=["A"], token_count=1)]

    def fold_forward_fn(path, batch):
        return {"risk": 0.1, "logit": [0.0]}

    with pytest.raises(CoreContractError, match="unchecked to_tokens"):
        production_apply_edits_and_score(
            events=events,
            edits=[{"event_id": "e1", "to_tokens": ["A2"]}],
            reencoder=_FakeReencoder(),
            fold_forward_fn=fold_forward_fn,
            checkpoint_paths=["ck0"],
            fold_ids=[0],
            requested_fold_ids=[0],
        )


def test_build_cold_diagnosis_batch_mask_semantics():
    cold = {
        "event_mean": torch.zeros(2, 3),
        "event_max": torch.zeros(2, 3),
        "valid_event_ids": ["a", "b"],
    }
    sidecar = {
        "a": {"view": "UNKNOWN", "zone": 1, "case_age_hours": 0.0, "local_hour": 0},
        "b": {"view": "UNKNOWN", "zone": 2, "case_age_hours": 1.0, "local_hour": 1},
    }
    batch = build_cold_diagnosis_batch(
        cold,
        sidecar_by_event_id=sidecar,
        mask_semantics="TRUE_IS_VALID",
        mask_reference_builder_sha256="abc",
    )
    assert batch["valid_event_count"] == 2
    assert batch["padded_event_count"] == 0
    assert batch["mask_semantics"] == "TRUE_IS_VALID"
    assert batch["event_mean"].shape == (1, 2, 3)
    assert batch["case_age_hours"].shape == (1, 2)
    assert batch["padding_mask"].shape == (1, 2)
    assert bool(batch["padding_mask"].all())
    assert_batch_matches_cold(batch, cold)


def test_stale_batch_field_parity_fails():
    cold = {
        "event_mean": torch.ones(1, 2),
        "event_max": torch.ones(1, 2),
        "valid_event_ids": ["a"],
    }
    sidecar = {"a": {"view": "UNKNOWN", "zone": 0, "case_age_hours": 0.0, "local_hour": 0}}
    batch = build_cold_diagnosis_batch(cold, sidecar_by_event_id=sidecar)
    batch["event_mean"] = torch.zeros(1, 1, 2)
    with pytest.raises(CoreContractError, match="STALE_DIAGNOSIS_BATCH"):
        assert_batch_matches_cold(batch, cold)
