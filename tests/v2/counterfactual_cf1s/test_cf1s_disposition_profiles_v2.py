from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import CoreContractError
from src.online2.v2.finetune_v03.counterfactual.cf1s.disposition_profiles import (
    CONSTRUCTIBLE_ALL_EXCLUDED,
    CONSTRUCTIBLE_ALL_GATE_FAILED,
    CONSTRUCTIBLE_SELECTED,
    NOT_CONSTRUCTIBLE,
    NOT_EVALUABLE,
    NOT_EVALUATED,
    case_trace_events,
    classify_case_disposition,
    evaluate_case_trace_profile,
)


def _proposal(**overrides):
    base = {
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_EVALUATED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    base.update(overrides)
    return base


def _event(*, kind, state, case_id="c1", scope="search", fold_id=None, sequence=1,
           closure_frozen=False, closure_sha=None, extra=None):
    return {
        "kind": kind,
        "state": state,
        "case_id": case_id,
        "scope": scope,
        "fold_id": fold_id,
        "sequence": sequence,
        "closure_frozen": closure_frozen,
        "closure_sha": closure_sha,
        "extra": extra or {},
    }


def test_classify_not_constructible_from_construction_status():
    proposal = _proposal(construction_status="NOT_CONSTRUCTIBLE")
    assert classify_case_disposition(proposal) == NOT_CONSTRUCTIBLE


def test_classify_not_evaluable_from_scientific_status():
    proposal = _proposal(scientific_status="NOT_EVALUABLE")
    assert classify_case_disposition(proposal) == NOT_EVALUABLE


def test_classify_not_evaluated_when_integrity_flags_missing():
    proposal = _proposal(
        execution_evidence={
            "production_input_verified": False,
            "pipeline_integrity_verified": True,
        }
    )
    assert classify_case_disposition(proposal) == NOT_EVALUATED


def test_classify_constructible_selected_when_candidate_selected():
    proposal = _proposal(
        candidate_results=[{"disposition": "SELECTED_MATERIAL"}],
    )
    assert classify_case_disposition(proposal) == CONSTRUCTIBLE_SELECTED


def test_classify_all_gate_failed_when_nothing_selected():
    proposal = _proposal(scientific_status="NOT_SUPPORTED", candidate_results=[])
    assert classify_case_disposition(proposal) == CONSTRUCTIBLE_ALL_GATE_FAILED


def test_classify_all_excluded_when_ledger_nonempty_but_nothing_selected():
    # Regression: search ran and scored every constructed candidate (no
    # pre-selection by budget) but none survived selection (e.g. mixed/
    # partial-material across search folds) — a real Primary32 case hit
    # this for the first time. Must NOT collapse into CONSTRUCTIBLE_ALL_
    # GATE_FAILED (which models a candidate forward being denied/blocked
    # before it ran, i.e. an empty ledger).
    proposal = _proposal(
        scientific_status="NOT_EVALUATED",
        candidate_results=[{"disposition": None, "kind": "PAIR"}],
    )
    assert classify_case_disposition(proposal) == CONSTRUCTIBLE_ALL_EXCLUDED


def test_case_trace_events_filters_by_case_id():
    events = [_event(kind="CRITIC_LOAD", state="COMPLETED", case_id="c1"),
              _event(kind="CRITIC_LOAD", state="COMPLETED", case_id="c2")]
    rows = case_trace_events(events, "c1")
    assert len(rows) == 1 and rows[0]["case_id"] == "c1"


def test_case_trace_events_raises_when_case_missing():
    events = [_event(kind="CRITIC_LOAD", state="COMPLETED", case_id="c2")]
    with pytest.raises(CoreContractError):
        case_trace_events(events, "c1")


def _constructible_selected_events(case_id="c1", *, candidate_forward=True, fold2=True):
    events = [
        _event(kind="ATTRIBUTION_MODEL_LOAD", state="COMPLETED", case_id=case_id, scope="search"),
        _event(kind="ATTRIBUTION_FORWARD", state="COMPLETED", case_id=case_id, scope="search"),
        _event(kind="CHECKPOINT_FORWARD", state="COMPLETED", case_id=case_id, scope="search"),
    ]
    if candidate_forward:
        events.append(
            _event(
                kind="PIPELINE_INVOCATION",
                state="COMPLETED",
                case_id=case_id,
                scope="search",
                extra={"forward_kind": "CANDIDATE_EFFECT_FORWARD"},
            )
        )
    if fold2:
        events.append(
            _event(
                kind="CHECKPOINT_FORWARD",
                state="COMPLETED",
                case_id=case_id,
                scope="selection_blind_reevaluation",
                fold_id=2,
                closure_frozen=True,
                closure_sha="closure-sha",
            )
        )
    return events


def test_constructible_selected_happy_path_passes():
    evaluate_case_trace_profile(
        case_id="c1",
        disposition=CONSTRUCTIBLE_SELECTED,
        case_events=_constructible_selected_events(),
        closure_sha="closure-sha",
    )


def test_constructible_selected_missing_fold2_fails():
    with pytest.raises(CoreContractError):
        evaluate_case_trace_profile(
            case_id="c1",
            disposition=CONSTRUCTIBLE_SELECTED,
            case_events=_constructible_selected_events(fold2=False),
            closure_sha="closure-sha",
        )


def test_constructible_selected_zero_forward_fails():
    with pytest.raises(CoreContractError):
        evaluate_case_trace_profile(
            case_id="c1",
            disposition=CONSTRUCTIBLE_SELECTED,
            case_events=_constructible_selected_events(candidate_forward=False),
            closure_sha="closure-sha",
        )


def test_not_constructible_zero_forward_with_failed_terminal_passes():
    events = [
        _event(
            kind="ATTRIBUTION_MODEL_LOAD",
            state="FAILED",
            case_id="c1",
            extra={"terminal_failure": True},
        ),
    ]
    evaluate_case_trace_profile(
        case_id="c1", disposition=NOT_CONSTRUCTIBLE, case_events=events,
    )


def test_not_constructible_without_failed_terminal_fails():
    events = [
        _event(kind="ATTRIBUTION_MODEL_LOAD", state="DENIED", case_id="c1"),
    ]
    with pytest.raises(CoreContractError):
        evaluate_case_trace_profile(
            case_id="c1", disposition=NOT_CONSTRUCTIBLE, case_events=events,
        )


def test_not_evaluated_denied_then_side_effect_fails():
    events = [
        _event(kind="CHECKPOINT_FORWARD", state="DENIED", case_id="c1", sequence=1),
        _event(kind="CHECKPOINT_FORWARD", state="COMPLETED", case_id="c1", sequence=2),
    ]
    with pytest.raises(CoreContractError):
        evaluate_case_trace_profile(
            case_id="c1", disposition=NOT_EVALUATED, case_events=events,
        )


def test_not_evaluated_denied_without_side_effect_passes():
    events = [
        _event(kind="CHECKPOINT_FORWARD", state="DENIED", case_id="c1", sequence=1),
    ]
    evaluate_case_trace_profile(
        case_id="c1", disposition=NOT_EVALUATED, case_events=events,
    )


def test_not_evaluable_partial_scope_with_failed_terminal_passes():
    events = [
        _event(kind="ATTRIBUTION_MODEL_LOAD", state="COMPLETED", case_id="c1"),
        _event(
            kind="CHECKPOINT_FORWARD",
            state="FAILED",
            case_id="c1",
            extra={"terminal_failure": True},
        ),
    ]
    evaluate_case_trace_profile(
        case_id="c1", disposition=NOT_EVALUABLE, case_events=events,
    )


def test_not_evaluable_without_failed_terminal_fails():
    events = [
        _event(kind="ATTRIBUTION_MODEL_LOAD", state="COMPLETED", case_id="c1"),
    ]
    with pytest.raises(CoreContractError):
        evaluate_case_trace_profile(
            case_id="c1", disposition=NOT_EVALUABLE, case_events=events,
        )


def test_all_gate_failed_blocked_before_side_effect_passes():
    events = [
        _event(kind="ATTRIBUTION_MODEL_LOAD", state="COMPLETED", case_id="c1"),
        _event(kind="ATTRIBUTION_FORWARD", state="COMPLETED", case_id="c1"),
        _event(kind="PIPELINE_INVOCATION", state="COMPLETED", case_id="c1"),
        _event(kind="PIPELINE_INVOCATION", state="DENIED", case_id="c1", sequence=2),
    ]
    evaluate_case_trace_profile(
        case_id="c1", disposition=CONSTRUCTIBLE_ALL_GATE_FAILED, case_events=events,
    )


def test_all_gate_failed_with_side_effect_before_block_fails():
    events = [
        _event(kind="ATTRIBUTION_MODEL_LOAD", state="COMPLETED", case_id="c1"),
        _event(kind="ATTRIBUTION_FORWARD", state="COMPLETED", case_id="c1"),
        _event(kind="PIPELINE_INVOCATION", state="COMPLETED", case_id="c1"),
        _event(
            kind="PIPELINE_INVOCATION",
            state="COMPLETED",
            case_id="c1",
            sequence=2,
            extra={"forward_kind": "CANDIDATE_EFFECT_FORWARD"},
        ),
        _event(kind="PIPELINE_INVOCATION", state="DENIED", case_id="c1", sequence=3),
    ]
    with pytest.raises(CoreContractError):
        evaluate_case_trace_profile(
            case_id="c1", disposition=CONSTRUCTIBLE_ALL_GATE_FAILED, case_events=events,
        )


def test_all_excluded_full_search_no_denial_passes():
    events = [
        _event(kind="ATTRIBUTION_MODEL_LOAD", state="COMPLETED", case_id="c1"),
        _event(kind="ATTRIBUTION_FORWARD", state="COMPLETED", case_id="c1"),
        _event(kind="CHECKPOINT_FORWARD", state="COMPLETED", case_id="c1"),
        _event(
            kind="PIPELINE_INVOCATION",
            state="COMPLETED",
            case_id="c1",
            sequence=2,
            extra={"forward_kind": "CANDIDATE_EFFECT_FORWARD"},
        ),
    ]
    evaluate_case_trace_profile(
        case_id="c1", disposition=CONSTRUCTIBLE_ALL_EXCLUDED, case_events=events,
    )


def test_all_excluded_zero_forward_fails():
    # No pre-selection by budget means search always forward-scores every
    # constructed candidate — a zero-forward trace under this disposition
    # is a contract violation, not a legitimate "nothing selected" state.
    events = [
        _event(kind="ATTRIBUTION_MODEL_LOAD", state="COMPLETED", case_id="c1"),
        _event(kind="ATTRIBUTION_FORWARD", state="COMPLETED", case_id="c1"),
        _event(kind="CHECKPOINT_FORWARD", state="COMPLETED", case_id="c1"),
    ]
    with pytest.raises(CoreContractError):
        evaluate_case_trace_profile(
            case_id="c1", disposition=CONSTRUCTIBLE_ALL_EXCLUDED, case_events=events,
        )


def test_cross_case_borrowing_cannot_satisfy_missing_case():
    events = _constructible_selected_events(case_id="other-case")
    filtered = case_trace_events(events, "other-case")
    assert filtered
    with pytest.raises(CoreContractError):
        case_trace_events(events, "missing-case")
