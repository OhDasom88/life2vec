"""§7 cross-cohort failure injection: a signature/lock/routing scoped to one
cohort must never authorize or validate another cohort's execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_authorization import (
    TRUST_ROOT_ENV,
    consume_pre_execution_authorization,
    issue_pre_execution_authorization,
    load_trust_root,
    verify_trust_root_env,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
    canonical_json_sha256,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
    CoreContractError,
    sha256_file,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
    REQUIRED_STABLE_LOCK_SHA_FIELDS,
    assert_stable_locks_identical,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_runtime import (
    load_fold_routing_manifest,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_verifier import (
    _case_proposals,
)

ROOT = Path("/home/dasom/life2vec")
TRUST = ROOT / "conf/m1/cf1s_policies/CF1S_DEVELOPMENT_TRUST_ROOT_V1.json"
DEV_PRIV = "d3d340008b8a043b537d3a49093983854381f0863d5acba31ef712910162facc"


def _stable_lock(cohort: str) -> dict:
    body = {
        "version": "CF1S_STABLE_LOCK_MANIFEST_V2",
        "cohort": cohort,
        "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
        "execution_scope": "TWO_EVENT_ONLY",
        "runtime_versions": {},
        "risk_head_contract": {},
        "artifact_sources": {},
    }
    body.update({name: "a" * 64 for name in REQUIRED_STABLE_LOCK_SHA_FIELDS})
    body["stable_lock_sha256"] = canonical_json_sha256(body)
    return body


def _issue(monkeypatch, *, cohort_id: str, run_id: str) -> dict:
    monkeypatch.setenv(TRUST_ROOT_ENV, sha256_file(TRUST))
    root = load_trust_root(TRUST)
    tr_sha = verify_trust_root_env(TRUST)
    stable = _stable_lock(cohort_id.lower())
    return issue_pre_execution_authorization(
        trust_root=root,
        trust_root_sha256=tr_sha,
        issuer_id="cf1s-development-issuer-v1",
        key_id="dev-key-1",
        private_key_hex=DEV_PRIV,
        stable_lock_manifest=stable,
        run_id=run_id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        cohort_id=cohort_id,
    )


def test_dev3_authorization_rejected_by_validation20_consumer(monkeypatch):
    art = _issue(monkeypatch, cohort_id="DEVELOPMENT3", run_id="run-dev3-as-val20")
    with pytest.raises(CoreContractError, match="cohort mismatch"):
        consume_pre_execution_authorization(
            art,
            trust_root=load_trust_root(TRUST),
            trust_root_sha256=verify_trust_root_env(TRUST),
            observed_stable_lock=art["stable_lock_manifest"],
            expected_run_id="run-dev3-as-val20",
            expected_cohort_id="VALIDATION20",
        )


def test_validation20_authorization_rejected_by_primary32_consumer(monkeypatch):
    art = _issue(monkeypatch, cohort_id="VALIDATION20", run_id="run-val20-as-p32")
    with pytest.raises(CoreContractError, match="cohort mismatch"):
        consume_pre_execution_authorization(
            art,
            trust_root=load_trust_root(TRUST),
            trust_root_sha256=verify_trust_root_env(TRUST),
            observed_stable_lock=art["stable_lock_manifest"],
            expected_run_id="run-val20-as-p32",
            expected_cohort_id="PRIMARY32",
        )


def test_20_case_lock_with_19_case_package_fails_completeness(tmp_path):
    for i in range(19):
        case_dir = tmp_path / f"case-{i}"
        case_dir.mkdir()
        (case_dir / "case_edit_proposal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CoreContractError, match="requires exactly 20 case proposals"):
        _case_proposals(tmp_path, 20)


def test_primary32_routing_rejects_validation20_run(tmp_path):
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(
        '{"manifest_id": "CF1S_PRIMARY32_FOLD_ROUTING_V1", "cases": {}}',
        encoding="utf-8",
    )
    with pytest.raises(CoreContractError, match="unexpected fold routing manifest_id"):
        load_fold_routing_manifest(
            routing_path, expected_manifest_id="CF1S_VALIDATION20_FOLD_ROUTING_V1"
        )


def test_unknown_cohort_authorization_rejected(monkeypatch):
    monkeypatch.setenv(TRUST_ROOT_ENV, sha256_file(TRUST))
    root = load_trust_root(TRUST)
    tr_sha = verify_trust_root_env(TRUST)
    with pytest.raises(CoreContractError, match="unknown cohort_id"):
        issue_pre_execution_authorization(
            trust_root=root,
            trust_root_sha256=tr_sha,
            issuer_id="cf1s-development-issuer-v1",
            key_id="dev-key-1",
            private_key_hex=DEV_PRIV,
            stable_lock_manifest=_stable_lock("development3"),
            run_id="run-unknown-cohort",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            cohort_id="UNKNOWN_COHORT",
        )


def test_assert_stable_locks_identical_accepts_non_dev3_cohort():
    # Regression: assert_stable_locks_identical (used by the orchestrator's
    # pre/post lock-drift check) must not hardcode expected_cohort="development3"
    # — every non-Dev3 cohort run would otherwise always report lock drift and
    # get stuck in QUARANTINE_ONLY regardless of actual code/test/policy state.
    stable = _stable_lock("validation20")
    assert assert_stable_locks_identical(
        stable, stable, expected_cohort="validation20"
    )
    with pytest.raises(CoreContractError):
        assert_stable_locks_identical(stable, stable, expected_cohort="development3")
