#!/usr/bin/env python3
"""Locked P1 acceptance recovery orchestrator — single execution_attempt_id."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    append_execution_run,
    build_artifact_list,
    build_run_record,
    dump_runtime_imports_from_env,
    evaluate_a7_provenance,
    file_sha256,
    load_runtime_imports,
    write_execution_provenance,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock import (
    build_source_lock,
    collect_declared_dependency_paths,
    compute_dependency_tree_sha256,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths


RUN_SPECS = [
    {
        "run_id": "decoder_preflight",
        "entrypoint": "scripts/online2_v2/v03/run_mlm_decoder_preflight_v03.py",
        "artifacts": ["artifacts/mlm_decoder_preflight.json"],
        "needs_bank": False,
    },
    {
        "run_id": "bundle_bank_build",
        "entrypoint": "scripts/online2_v2/v03/build_mlm_bundle_bank_v03.py",
        "artifacts": [
            "artifacts/mlm_bundle_bank_meta.json",
            "artifacts/mlm_bundle_bank/mlm_bundle_bank_manifest.json",
            "artifacts/mlm_bundle_bank/mlm_bundle_bank_observations.parquet",
            "artifacts/mlm_bundle_bank/mlm_bundle_bank_unique.parquet",
        ],
        "needs_bank": True,
    },
    {
        "run_id": "functional_smoke",
        "entrypoint": "scripts/online2_v2/v03/run_mlm_functional_smoke_v03.py",
        "artifacts": ["artifacts/mlm_functional_smoke_result.json"],
        "needs_bank": True,
    },
    {
        "run_id": "curated_fixture",
        "entrypoint": "scripts/online2_v2/v03/run_curated_mlm_fixture_v03.py",
        "artifacts": ["reports/curated_fixture_result.json"],
        "needs_bank": True,
    },
    {
        "run_id": "natural_path_a",
        "entrypoint": "scripts/online2_v2/v03/run_cf_m2_p1_smoke_v03.py",
        "artifacts": [
            "artifacts/mlm_path_a_funnel.json",
            "artifacts/mlm_natural_outcome.json",
        ],
        "needs_bank": True,
    },
    {
        "run_id": "reconstruction_selection",
        "entrypoint": "scripts/online2_v2/v03/run_mlm_reconstruction_selection_v03.py",
        "artifacts": [
            "manifests/mlm_selection/selection_manifest_final.json",
            "manifests/mlm_selection/selection_manifest_chain.json",
            "manifests/mlm_selection/selection_manifest_stage_01.json",
            "reports/reconstruction_selection_result.json",
        ],
        "needs_bank": True,
        "artifact_globs": [
            "manifests/mlm_selection/selection_manifest_stage_*.json",
        ],
    },
    {
        "run_id": "reconstruction_eval",
        "entrypoint": "scripts/online2_v2/v03/run_mlm_reconstruction_eval_v03.py",
        "artifacts": [
            "reports/mlm_reconstruction_metrics.json",
            "reports/mlm_reconstruction_per_mg.jsonl",
        ],
        "needs_bank": True,
    },
    {
        "run_id": "pytest",
        "entrypoint": "pytest",
        "pytest_args": ["tests/v2/counterfactual_m1/", "-q", "--tb=line"],
        "artifacts": ["reports/pytest_counterfactual_m1.log"],
        "needs_bank": False,
    },
]


def _expand_artifacts(out_root: Path, spec: Dict[str, Any]) -> List[str]:
    rels = list(spec.get("artifacts") or [])
    for pattern in spec.get("artifact_globs") or []:
        for p in sorted(out_root.glob(pattern)):
            if p.is_file():
                rel = str(p.relative_to(out_root))
                if rel not in rels:
                    rels.append(rel)
    return rels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
    ap.add_argument("--dry-run-plan-only", action="store_true")
    ap.add_argument(
        "--skip-gpu-heavy",
        action="store_true",
        help="Stop after bank+functional for CI dry paths",
    )
    ap.add_argument(
        "--skip-static-gate",
        action="store_true",
        help="Debug only — do not use for locked acceptance",
    )
    args = ap.parse_args()

    cfg_text = args.config.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    config_sha = hashlib.sha256(cfg_text.encode()).hexdigest()
    out_root = Path(cfg["output_root"])
    paths = M1Paths(out_root)
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    # Static start lock (imports may be empty; nonempty checked at end)
    source_lock = build_source_lock(
        root=ROOT,
        config_path=args.config,
        config_sha256=config_sha,
        runtime_imported_project_paths=[],
        require_nonempty_runtime_imports=False,
    )

    attempt_id = (
        f"p1_recovery_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    invocation_id = uuid.uuid4().hex
    dep_sha = str(source_lock.get("dependency_tree_sha256") or "")
    final_lock_id = f"lock_{dep_sha[:16] if dep_sha else 'unknown'}"
    ckpt = Path(cfg["stage_a_ckpt"])
    vocab = Path(cfg.get("vocabulary_path") or cfg["tokenizer_path"])
    ckpt_sha = file_sha256(ckpt) if ckpt.exists() else "MISSING"
    vocab_sha = file_sha256(vocab) if vocab.exists() else "MISSING"
    commit = str(source_lock.get("git_commit") or "")
    declared = collect_declared_dependency_paths(ROOT)

    # Static gate before any GPU subprocess
    static_errs: List[str] = []
    if commit in ("", "UNKNOWN"):
        static_errs.append("missing_git_commit")
    if not source_lock.get("entrypoints_complete"):
        static_errs.append("missing_entrypoints")
    if source_lock.get("tracked_changes_in_scope") or source_lock.get("untracked_files_in_scope"):
        static_errs.append("scope_dirty")
    if not dep_sha:
        static_errs.append("missing_dependency_tree_sha256")
    if ckpt_sha == "MISSING":
        static_errs.append("missing_checkpoint")
    if vocab_sha == "MISSING":
        static_errs.append("missing_vocab")
    static_gate = {
        "static_gate_pass": len(static_errs) == 0,
        "errors": static_errs,
        "git_commit": commit,
        "dependency_tree_sha256": dep_sha,
        "entrypoints_complete": bool(source_lock.get("entrypoints_complete")),
        "config_sha256": config_sha,
        "stage_a_checkpoint_sha256": ckpt_sha,
        "vocab_sha256": vocab_sha,
    }
    write_json(paths.reports / "orchestrator_static_gate.json", static_gate)
    if static_errs and not args.skip_static_gate and not args.dry_run_plan_only:
        print(json.dumps({"static_gate": static_gate}, indent=2, ensure_ascii=False))
        return 1

    prov_path = paths.reports / "execution_provenance.json"
    write_execution_provenance(
        prov_path,
        {
            "active_execution_attempt_id": attempt_id,
            "orchestrator_invocation_id": invocation_id,
            "final_lock_id": final_lock_id,
            "git_commit": commit,
            "dependency_tree_sha256": dep_sha,
            "pre_dependency_tree_sha256": dep_sha,
            "post_dependency_tree_sha256": dep_sha,
            "pre_source_tree_sha256": dep_sha,
            "post_source_tree_sha256": dep_sha,
            "runs": [],
            "attempts": [],
            "rerun_flag_only": False,
            "runtime_imports_by_run": {},
            "runtime_imported_project_paths_union": [],
        },
    )

    plan = [{"run_sequence": i + 1, **s} for i, s in enumerate(RUN_SPECS)]
    write_json(
        paths.reports / "recovery_orchestrator_plan.json",
        {"attempt_id": attempt_id, "plan": plan},
    )
    if args.dry_run_plan_only:
        print(
            json.dumps(
                {"attempt_id": attempt_id, "n_steps": len(plan), "static_gate": static_gate},
                indent=2,
            )
        )
        return 0 if static_gate["static_gate_pass"] else 1

    dump_runtime_imports_from_env(ROOT, entrypoint=__file__)

    bank_content_sha: Optional[str] = None
    failed = False
    for i, spec in enumerate(RUN_SPECS, start=1):
        if args.skip_gpu_heavy and spec["run_id"] not in {
            "decoder_preflight",
            "bundle_bank_build",
            "functional_smoke",
            "pytest",
        }:
            continue

        log_path = paths.reports / f"orchestrator_{spec['run_id']}.log"
        imports_path = paths.reports / f"runtime_imports_{spec['run_id']}.json"
        if imports_path.exists():
            imports_path.unlink()

        if spec["run_id"] == "pytest":
            cmd = [sys.executable, "-m", "pytest", *spec["pytest_args"]]
        else:
            cmd = [sys.executable, str(ROOT / spec["entrypoint"]), "--config", str(args.config)]

        env = dict(os.environ)
        env["CF_M2_RUNTIME_IMPORTS_PATH"] = str(imports_path)
        env["CF_M2_RUN_ID"] = str(spec["run_id"])
        env["CF_M2_ENTRYPOINT"] = str(spec["entrypoint"])

        step_pre_sha, _ = compute_dependency_tree_sha256(ROOT, relative_paths=declared)
        print(f"[{i}/8] {spec['run_id']}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        step_post_sha, _ = compute_dependency_tree_sha256(ROOT, relative_paths=declared)
        log_path.write_text(
            (proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""),
            encoding="utf-8",
        )
        if spec["run_id"] == "pytest":
            (paths.reports / "pytest_counterfactual_m1.log").write_text(
                log_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

        if step_pre_sha != step_post_sha:
            print(
                f"FAIL at {spec['run_id']}: pre/post dependency tree hash mismatch",
                flush=True,
            )
            failed = True
            runtime_paths = load_runtime_imports(imports_path)
            arts = build_artifact_list(out_root, _expand_artifacts(out_root, spec))
            arts.append(
                {
                    "relative_path": str(log_path.relative_to(out_root)),
                    "sha256": file_sha256(log_path),
                }
            )
            rec = build_run_record(
                run_id=spec["run_id"],
                entrypoint=spec["entrypoint"],
                git_commit=commit,
                dependency_tree_sha256=dep_sha,
                exit_code=1,
                pre_dependency_tree_sha256=step_pre_sha,
                post_dependency_tree_sha256=step_post_sha,
                execution_attempt_id=attempt_id,
                orchestrator_invocation_id=invocation_id,
                final_lock_id=final_lock_id,
                run_sequence=i,
                log_sha256=file_sha256(log_path),
                artifacts=arts,
                config_sha256=config_sha,
                stage_a_checkpoint_sha256=ckpt_sha,
                vocab_sha256=vocab_sha,
                bundle_bank_content_sha256=bank_content_sha if spec.get("needs_bank") else None,
                runtime_imported_project_paths=runtime_paths,
                extra={"fail_reason": "pre_post_dependency_tree_mismatch"},
            )
            append_execution_run(prov_path, rec, active_execution_attempt_id=attempt_id)
            break

        if spec["run_id"] == "bundle_bank_build":
            meta_p = paths.artifacts / "mlm_bundle_bank_meta.json"
            if meta_p.exists():
                bank_content_sha = json.loads(meta_p.read_text()).get(
                    "bundle_bank_content_sha256"
                )

        arts = build_artifact_list(out_root, _expand_artifacts(out_root, spec))
        arts.append(
            {
                "relative_path": str(log_path.relative_to(out_root)),
                "sha256": file_sha256(log_path),
            }
        )
        runtime_paths = load_runtime_imports(imports_path)
        if imports_path.is_file():
            arts.append(
                {
                    "relative_path": str(imports_path.relative_to(out_root)),
                    "sha256": file_sha256(imports_path),
                }
            )

        rec = build_run_record(
            run_id=spec["run_id"],
            entrypoint=spec["entrypoint"],
            git_commit=commit,
            dependency_tree_sha256=dep_sha,
            exit_code=int(proc.returncode),
            pre_dependency_tree_sha256=step_pre_sha,
            post_dependency_tree_sha256=step_post_sha,
            execution_attempt_id=attempt_id,
            orchestrator_invocation_id=invocation_id,
            final_lock_id=final_lock_id,
            run_sequence=i,
            log_sha256=file_sha256(log_path),
            artifacts=arts,
            config_sha256=config_sha,
            stage_a_checkpoint_sha256=ckpt_sha,
            vocab_sha256=vocab_sha,
            bundle_bank_content_sha256=bank_content_sha if spec.get("needs_bank") else None,
            runtime_imported_project_paths=list(runtime_paths),
        )
        append_execution_run(prov_path, rec, active_execution_attempt_id=attempt_id)
        if proc.returncode != 0:
            print(f"FAIL at {spec['run_id']} exit={proc.returncode}", flush=True)
            failed = True
            break

    prov_doc = json.loads(prov_path.read_text(encoding="utf-8"))
    end_lock = build_source_lock(
        root=ROOT,
        config_path=args.config,
        config_sha256=config_sha,
        runtime_imported_project_paths=prov_doc.get("runtime_imported_project_paths_union")
        or [],
        require_nonempty_runtime_imports=True,
    )
    write_json(paths.reports / "orchestrator_end_source_lock.json", end_lock)

    a7 = evaluate_a7_provenance(
        prov_doc,
        final_lock={
            "git_commit": commit,
            "dependency_tree_sha256": dep_sha,
            "cf_source_clean": bool(end_lock.get("cf_source_clean")),
            "config_sha256": config_sha,
            "stage_a_checkpoint_sha256": ckpt_sha,
            "vocab_sha256": vocab_sha,
            "final_lock_id": final_lock_id,
        },
        root=out_root,
    )
    write_json(paths.reports / "a7_provenance_eval.json", a7)
    print(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "a7": a7,
                "end_lock_a6": end_lock.get("a6_pass"),
                "static_gate": static_gate,
                "failed": failed,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if (a7.get("a7_pass") and not failed and end_lock.get("a6_pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
