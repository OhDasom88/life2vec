"""Full CF-1S code/policy/profile tree hashing for primary lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence


CF1S_CODE_FILES = (
    "src/online2/v2/finetune_v03/counterfactual/pipeline_cf1s.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/__init__.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/edit_policy.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/locus_universe.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/selectors.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/beam.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/interaction.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/fold_effect.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/status.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/identity.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/code_lock.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/production_bridge.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/execution_context.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/evaluation_closure.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/production_preflight.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/acceptance.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/readiness.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/schema_targets.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/selector_runtime.py",
    "src/online2/v2/finetune_v03/counterfactual/candidates/cf1s_composer.py",
    "src/online2/v2/finetune_v03/counterfactual/candidates/path_a_schema_dispatch.py",
    "src/online2/v2/finetune_v03/counterfactual/candidates/path_a_generator.py",
    "src/online2/v2/finetune_v03/counterfactual/retokenization/multi_event_retokenizer.py",
    "src/online2/v2/finetune_v03/counterfactual/retokenization/full_event_retokenizer.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/cf1s_funnel_accounting.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/case_batch.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/crossfit_critic.py",
    "src/online2/v2/finetune_v03/counterfactual/stage_a_reencoder.py",
    "src/online2/v2/finetune_v03/counterfactual/grounding/locus_raw.py",
    "src/online2/v2/finetune_v03/counterfactual/attribution/token_ixg_v03.py",
    "src/online2/v2/finetune_v03/counterfactual/pipeline_m2.py",
    "src/online2/v2/finetune_v03/counterfactual/manifest.py",
    "scripts/online2_v2/v03/run_cf1s_v03.py",
    "scripts/online2_v2/v03/package_cf1s_v03.py",
    "scripts/online2_v2/v03/write_cf1s_readiness_v03.py",
)

CF1S_POLICY_FILES = (
    "conf/m1/cf1s_policies/CF1S_EDIT_POLICY_V1.yaml",
    "conf/m1/cf1s_policies/CF1S_SALIENCY_POLICY_V1.yaml",
    "conf/m1/cf1s_policies/CF1S_SEARCH_POLICY_V1.yaml",
    "conf/m1/cf1s_policies/CF1S_ACCEPTANCE_POLICY_V1.yaml",
    "conf/m1/cf1s_smoke.yaml",
)

CF1S_TEST_FILES = (
    "tests/v2/counterfactual_cf1s/conftest.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_universe_selectors.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_interaction_fold_status.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_composer_retokenizer.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_pipeline_identity.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_budget_parent.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_beam_budget.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_production_integration.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(root: Path, relatives: Sequence[str]) -> Dict[str, Any]:
    files = []
    for rel in relatives:
        path = root / rel
        if not path.exists():
            files.append({"path": rel, "present": False, "sha256": None})
            continue
        files.append(
            {
                "path": rel,
                "present": True,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    present = [f for f in files if f["present"]]
    tree_payload = json.dumps(
        [{"path": f["path"], "sha256": f["sha256"]} for f in present],
        sort_keys=True,
    )
    return {
        "files": files,
        "present_count": len(present),
        "missing_count": len(files) - len(present),
        "tree_sha256": hashlib.sha256(tree_payload.encode("utf-8")).hexdigest(),
    }


def build_cf1s_lock_artifacts(
    root: Path,
    *,
    fixture_contract_snapshot: bool = True,
    final_primary_lock: bool = False,
) -> Dict[str, Any]:
    code = build_file_manifest(root, CF1S_CODE_FILES)
    policy = build_file_manifest(root, CF1S_POLICY_FILES)
    tests = build_file_manifest(root, CF1S_TEST_FILES)
    combined = hashlib.sha256(
        (
            code["tree_sha256"]
            + "|"
            + policy["tree_sha256"]
            + "|"
            + tests["tree_sha256"]
        ).encode("utf-8")
    ).hexdigest()
    return {
        "fixture_contract_snapshot": bool(fixture_contract_snapshot),
        "final_primary_lock": bool(final_primary_lock),
        "policy_locked_before_primary_32": bool(final_primary_lock),
        "code_locked_before_primary_32": bool(final_primary_lock),
        "production_dependency_closure_locked": code["missing_count"] == 0,
        "CODE_MANIFEST": code,
        "POLICY_MANIFEST": policy,
        "TEST_MANIFEST": tests,
        "CODE_TREE_SHA256": code["tree_sha256"],
        "POLICY_TREE_SHA256": policy["tree_sha256"],
        "TEST_TREE_SHA256": tests["tree_sha256"],
        "CF1S_LOCK_TREE_SHA256": combined,
    }
