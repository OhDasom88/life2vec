"""Trust-root authorization and attestation tests."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_authorization import (
    TRUST_ROOT_ENV,
    consume_pre_execution_authorization,
    issue_post_execution_attestation,
    issue_pre_execution_authorization,
    load_trust_root,
    verify_trust_root_env,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import CoreContractError, sha256_file
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import canonical_json_sha256
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
    REQUIRED_STABLE_LOCK_SHA_FIELDS,
)

ROOT = Path("/home/dasom/life2vec")
TRUST = ROOT / "conf/m1/cf1s_policies/CF1S_DEVELOPMENT_TRUST_ROOT_V1.json"
# Development test key matching trust root public key (not for production).
DEV_PRIV = "d3d340008b8a043b537d3a49093983854381f0863d5acba31ef712910162facc"


def _stable_lock():
    body = {
        "version": "CF1S_STABLE_LOCK_MANIFEST_V2",
        "cohort": "development3",
        "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
        "execution_scope": "TWO_EVENT_ONLY",
        "runtime_versions": {},
        "risk_head_contract": {},
        "artifact_sources": {},
    }
    body.update({name: "a" * 64 for name in REQUIRED_STABLE_LOCK_SHA_FIELDS})
    body["stable_lock_sha256"] = canonical_json_sha256(body)
    return body


def test_trust_root_env_gate(monkeypatch):
    monkeypatch.delenv(TRUST_ROOT_ENV, raising=False)
    with pytest.raises(CoreContractError, match="missing"):
        verify_trust_root_env(TRUST)
    monkeypatch.setenv(TRUST_ROOT_ENV, sha256_file(TRUST))
    assert verify_trust_root_env(TRUST) == sha256_file(TRUST)
    monkeypatch.setenv(TRUST_ROOT_ENV, "0" * 64)
    with pytest.raises(CoreContractError, match="mismatch"):
        verify_trust_root_env(TRUST)


def test_issue_and_consume_pre_execution(monkeypatch):
    monkeypatch.setenv(TRUST_ROOT_ENV, sha256_file(TRUST))
    root = load_trust_root(TRUST)
    tr_sha = verify_trust_root_env(TRUST)
    stable = _stable_lock()
    art = issue_pre_execution_authorization(
        trust_root=root,
        trust_root_sha256=tr_sha,
        issuer_id="cf1s-development-issuer-v1",
        key_id="dev-key-1",
        private_key_hex=DEV_PRIV,
        stable_lock_manifest=stable,
        run_id="test-run-1",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert art["artifact_kind"] == "PRE_EXECUTION_AUTHORIZATION"
    assert art["closure_kind"] == "SELECTION_BLIND_REEVALUATION_CLOSURE"
    out = consume_pre_execution_authorization(
        art,
        trust_root=root,
        trust_root_sha256=tr_sha,
        observed_stable_lock=stable,
        expected_run_id="test-run-1",
    )
    assert out["ok"]

    with pytest.raises(CoreContractError, match="AUTH_SELF_REFERENCE_FORBIDDEN"):
        consume_pre_execution_authorization(
            art,
            trust_root=root,
            trust_root_sha256=tr_sha,
            expected_locks=art["stable_lock_manifest"],
            expected_run_id="test-run-1",
        )

    with pytest.raises(CoreContractError, match="fresh observed stable lock required"):
        consume_pre_execution_authorization(
            art,
            trust_root=root,
            trust_root_sha256=tr_sha,
            expected_run_id="test-run-1",
        )

    # forged: mutate lock after sign
    bad = dict(art)
    bad["stable_lock_sha256"] = "f" * 64
    with pytest.raises(CoreContractError):
        consume_pre_execution_authorization(
            bad,
            trust_root=root,
            trust_root_sha256=tr_sha,
            observed_stable_lock=stable,
            expected_run_id="test-run-1",
        )


def test_post_execution_attestation_requires_lock_identity(monkeypatch):
    monkeypatch.setenv(TRUST_ROOT_ENV, sha256_file(TRUST))
    root = load_trust_root(TRUST)
    tr_sha = verify_trust_root_env(TRUST)
    with pytest.raises(CoreContractError, match="lock mismatch"):
        issue_post_execution_attestation(
            trust_root=root,
            trust_root_sha256=tr_sha,
            issuer_id="cf1s-development-issuer-v1",
            key_id="dev-key-1",
            private_key_hex=DEV_PRIV,
            evidence_root_sha="e" * 64,
            pre_execution_payload_sha256="p" * 64,
            acceptance_summary={"ok": True},
            pre_post_lock_identical=False,
        )
