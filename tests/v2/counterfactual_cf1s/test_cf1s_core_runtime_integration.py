"""Mock-hooks orchestrator smoke (no GPU / no production_bridge)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_authorization import (
    TRUST_ROOT_ENV,
    issue_pre_execution_authorization,
    load_trust_root,
    verify_trust_root_env,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import sha256_file
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
    compute_stable_lock_manifest,
    REQUIRED_STABLE_LOCK_SHA_FIELDS,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_orchestrator import (
    CaseRuntimeHooks,
    run_development3_orchestrator,
)
from tests.v2.counterfactual_cf1s.cf1s_test_tx import make_test_identity_transaction

ROOT = Path("/home/dasom/life2vec")
TRUST = ROOT / "conf/m1/cf1s_policies/CF1S_DEVELOPMENT_TRUST_ROOT_V1.json"
DEV_PRIV = "d3d340008b8a043b537d3a49093983854381f0863d5acba31ef712910162facc"


class _FakeReencoder:
    def cold_rebuild_full_sequence(self, events, batch_size=8):
        n = len(events)
        mean = torch.zeros(n, 8)
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
        }


@pytest.fixture()
def auth_artifact(monkeypatch):
    monkeypatch.setenv(TRUST_ROOT_ENV, sha256_file(TRUST))
    root = load_trust_root(TRUST)
    tr = verify_trust_root_env(TRUST)
    source_fields = set(REQUIRED_STABLE_LOCK_SHA_FIELDS) - {
        "code_tree_sha256",
        "test_tree_sha256",
        "policy_tree_sha256",
    }
    sources = {field: str(TRUST) for field in source_fields}
    stable = compute_stable_lock_manifest(
        root=ROOT,
        qualification_test_log_sha=tr,
        qualification_preflight_sha=tr,
        qualification_test_node_id_manifest_sha=tr,
        artifact_sha256={field: tr for field in source_fields},
        artifact_sources=sources,
        runtime_versions={},
        risk_head_contract={},
    )
    return issue_pre_execution_authorization(
        trust_root=root,
        trust_root_sha256=tr,
        issuer_id="cf1s-development-issuer-v1",
        key_id="dev-key-1",
        private_key_hex=DEV_PRIV,
        stable_lock_manifest=stable,
        run_id="test-runtime-integration",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def test_orchestrator_mock_hooks_empty_candidates(tmp_path, auth_artifact, monkeypatch):
    monkeypatch.setenv(TRUST_ROOT_ENV, sha256_file(TRUST))
    monkeypatch.setenv("CF1S_DEVELOPMENT_SIGNING_KEY_HEX", DEV_PRIV)

    def fold_forward(path, batch):
        return {"risk": 0.25, "logit": [0.0]}

    import src.online2.v2.finetune_v03.counterfactual.cf1s.core_orchestrator as orch

    monkeypatch.setattr(
        orch,
        "make_production_fold_forward_fn",
        lambda **kwargs: fold_forward,
    )
    monkeypatch.setattr(
        orch,
        "apply_exact_byte_deterministic_runtime",
        lambda **kwargs: __import__(
            "src.online2.v2.finetune_v03.counterfactual.cf1s.core_runtime",
            fromlist=["RuntimeLock"],
        ).RuntimeLock(torch_version="test"),
    )

    hooks = CaseRuntimeHooks(
        load_case_events=lambda cid: [
            SimpleNamespace(event_id="e1", sentence_tokens=["A"], token_count=1),
            SimpleNamespace(event_id="e2", sentence_tokens=["B"], token_count=1),
        ],
        load_sidecar_by_event_id=lambda cid: {
            "e1": {"view": "UNKNOWN", "zone": 0, "case_age_hours": 1.0, "local_hour": 1},
            "e2": {"view": "UNKNOWN", "zone": 1, "case_age_hours": 2.0, "local_hour": 2},
        },
        build_reencoder=lambda: _FakeReencoder(),
        list_checkpoint_paths=lambda: [Path("ck0"), Path("ck1"), Path("ck2")],
        fold_ids=[0, 1, 2],
        build_candidates=lambda cid: [],
        score_identity_transaction=lambda cid: make_test_identity_transaction(case_id=cid),
    )
    cfg = {
        "development_manifest_path": "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_MANIFEST.json",
        "gpu_memory_fraction": 0.1,
    }
    result = run_development3_orchestrator(
        config=cfg,
        authorization_artifact=auth_artifact,
        trust_root_path=TRUST,
        fold_routing_path=ROOT
        / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_FOLD_ROUTING_V1.json",
        hooks=hooks,
        out_root=tmp_path,
        signing_private_key_hex=DEV_PRIV,
        repo_root=ROOT,
    )
    assert result.package_dir is not None
    assert (result.package_dir / "runner_summary.json").is_file()
    assert result.evaluation_coverage in ("NONE", "PARTIAL", "COMPLETE")
    assert result.attested_evaluation_coverage in ("UNVERIFIED", "NONE", "PARTIAL", "COMPLETE")
    assert "Execution completed" in result.report_lines[0]
    assert result.package_state in (
        "PROMOTED",
        "PROMOTED_BUT_UNATTESTED_BY_RECEIPT",
        "QUARANTINE_ONLY",
    )
    # Empty candidate universe → NONE coverage
    assert result.evaluation_coverage == "NONE"
    if result.package_state == "PROMOTED":
        assert (result.package_dir / "POST_EXECUTION_EVIDENCE_ATTESTATION.json").is_file()
        receipts = list((tmp_path / "promotion_receipts").glob("*.json"))
        assert any(
            p.name.endswith(".json") and not p.name.endswith(".verdict.json")
            for p in receipts
        )
