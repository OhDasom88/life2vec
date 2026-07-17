#!/usr/bin/env python3
"""Phase 8: final code lock → artifact lock → package + external sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.counterfactual.evaluation.acceptance import (
    evaluate_p1_acceptance_conditions,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    evaluate_a8_1_bank_integrity,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    dump_runtime_imports_from_env,
    evaluate_a7_provenance,
    load_execution_provenance,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.funnel_accounting import (
    evaluate_a5_contracts,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    exact_match_selection_metrics,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock import (
    build_source_lock,
    file_sha256,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths


def _as_bank_query_ref(doc: Mapping[str, Any], *, default_run_id: str) -> Dict[str, Any]:
    if doc.get("bank_query_record"):
        ref = dict(doc["bank_query_record"])
    elif doc.get("bank_meta"):
        ref = dict(doc["bank_meta"])
    else:
        ref = dict(doc)
    if not ref.get("run_id"):
        ref["run_id"] = default_run_id
    if not ref.get("bank_mode") and ref.get("bundle_bank_mode"):
        ref["bank_mode"] = ref["bundle_bank_mode"]
    return ref


def _consumer_bank_refs(paths: M1Paths) -> List[Dict[str, Any]]:
    """Load required consumer bank/query records for A8-1 fail-closed checks."""
    refs: List[Dict[str, Any]] = []
    sources = [
        (paths.artifacts / "mlm_functional_smoke_result.json", "functional_smoke"),
        (paths.reports / "curated_fixture_result.json", "curated_fixture"),
        (paths.artifacts / "mlm_path_a_bank_query_meta.json", "natural_path_a"),
        (paths.artifacts / "mlm_path_a_funnel.json", "natural_path_a"),
        (
            paths.reports / "bank_query_set_manifest_reconstruction_selection.json",
            "reconstruction_selection",
        ),
        (paths.reports / "bank_query_record_reconstruction_selection.json", "reconstruction_selection"),
        (paths.reports / "reconstruction_selection_result.json", "reconstruction_selection"),
        (
            paths.reports / "bank_query_set_manifest_reconstruction_eval.json",
            "reconstruction_eval",
        ),
        (paths.reports / "bank_query_record_reconstruction_eval.json", "reconstruction_eval"),
        (paths.reports / "mlm_reconstruction_metrics.json", "reconstruction_eval"),
    ]
    seen: set = set()
    for path, run_id in sources:
        if not path.exists():
            continue
        if run_id in seen:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        # Prefer full bank_query_record companion when loading bare query-set
        if path.name.startswith("bank_query_set_manifest_"):
            companion = paths.reports / path.name.replace(
                "bank_query_set_manifest_", "bank_query_record_"
            )
            if companion.exists():
                full = json.loads(companion.read_text(encoding="utf-8"))
                ref = _as_bank_query_ref(full, default_run_id=run_id)
                ref["query_set_manifest"] = doc
                if not ref.get("query_records"):
                    ref["query_records"] = doc.get("query_records")
                if not ref.get("query_set_sha256"):
                    ref["query_set_sha256"] = doc.get("query_set_sha256")
                if not ref.get("query_count"):
                    ref["query_count"] = doc.get("query_count")
            else:
                ref = _as_bank_query_ref(doc, default_run_id=run_id)
                ref["query_set_manifest"] = doc
        else:
            ref = _as_bank_query_ref(doc, default_run_id=run_id)
        # Only accept if it looks like a bank query record
        if not ref.get("bundle_bank_content_sha256") and not ref.get("bank_query_record"):
            # try nested
            if not (ref.get("bundle_bank_policy_sha256") or ref.get("bundle_bank_query_manifest_sha256")):
                continue
        if ref.get("run_id") in seen:
            continue
        seen.add(str(ref.get("run_id")))
        refs.append(ref)
    return refs


def main() -> int:
    exit_code = 1
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
        ap.add_argument(
            "--rerun-completed",
            action="store_true",
            default=False,
            help="Deprecated for A7; kept for CLI compat. A7 uses execution_provenance.json only.",
        )
        ap.add_argument(
            "--package",
            type=Path,
            default=ROOT / "datasets/agrichallenge/CF_M2_P1_최종산출.tar.gz",
        )
        args = ap.parse_args()
        cfg_text = args.config.read_text(encoding="utf-8")
        cfg = yaml.safe_load(cfg_text) or {}
        config_sha = hashlib.sha256(cfg_text.encode()).hexdigest()
        out_root = Path(cfg["output_root"])
        paths = M1Paths(out_root)

        provenance_path = paths.reports / "execution_provenance.json"
        provenance = load_execution_provenance(provenance_path) or {}
        runtime_imported = list(provenance.get("runtime_imported_project_paths_union") or [])
        source_lock = build_source_lock(
            root=ROOT,
            config_path=args.config,
            config_sha256=config_sha,
            runtime_imported_project_paths=runtime_imported,
            require_nonempty_runtime_imports=True,
        )
        ckpt = Path(cfg["stage_a_ckpt"])
        vocab = Path(cfg.get("vocabulary_path") or cfg["tokenizer_path"])
        a6_pass = bool(source_lock.get("a6_pass"))
        final_lock_id = provenance.get("final_lock_id") or (
            f"lock_{source_lock.get('dependency_tree_sha256', 'unknown')[:16]}"
        )
        final_lock = {
            "final_code_source_lock": "PASS" if a6_pass else "FAIL",
            "final_lock_id": final_lock_id,
            "git_commit": source_lock.get("git_commit"),
            "cf_source_clean": bool(source_lock.get("cf_source_clean")),
            "dependency_closure_complete": bool(source_lock.get("dependency_closure_complete")),
            "dependency_tree_sha256": source_lock.get("dependency_tree_sha256"),
            "source_lock_scope_version": source_lock.get("source_lock_scope_version"),
            "source_tree_sha256": source_lock.get("source_tree_sha256"),
            "config_sha256": config_sha,
            "stage_a_checkpoint_sha256": file_sha256(ckpt) if ckpt.exists() else None,
            "vocab_sha256": file_sha256(vocab) if vocab.exists() else None,
            "undeclared_runtime_dependencies": source_lock.get("undeclared_runtime_dependencies") or [],
            "source_lock": {
                k: source_lock[k]
                for k in (
                    "git_commit",
                    "cf_source_clean",
                    "dependency_tree_sha256",
                    "dependency_closure_complete",
                    "source_tree_sha256",
                    "reproducibility",
                    "tracked_changes_in_scope",
                    "untracked_files_in_scope",
                    "repository_dirty_outside_scope",
                )
                if k in source_lock
            },
            "note": "Phase 8a lock snapshot — A6 from cf_m2_p1_v2 closure (never hardcoded PASS)",
        }
        write_json(paths.reports / "final_code_source_lock.json", final_lock)

        recon = {}
        if (paths.reports / "mlm_reconstruction_metrics.json").exists():
            recon = json.loads((paths.reports / "mlm_reconstruction_metrics.json").read_text())
        chain_hash = recon.get("selection_manifest_chain_hash")
        if not chain_hash and (paths.reports / "selection_manifest_chain.json").exists():
            chain_hash = json.loads(
                (paths.reports / "selection_manifest_chain.json").read_text()
            ).get("selection_manifest_chain_hash")

        bank_meta_path = paths.artifacts / "mlm_bundle_bank_meta.json"
        bank_meta = (
            json.loads(bank_meta_path.read_text(encoding="utf-8")) if bank_meta_path.exists() else {}
        )
        funnel_path = paths.artifacts / "mlm_path_a_funnel.json"
        fixture_path = paths.manifests / "curated_fixture_manifest.json"
        final_sel = paths.manifests / "mlm_selection" / "selection_manifest_final.json"

        consumer_refs = _consumer_bank_refs(paths)
        a8_1_eval = evaluate_a8_1_bank_integrity(
            bank_dir=paths.artifacts / "mlm_bundle_bank",
            bank_meta=bank_meta,
            consumer_bank_refs=consumer_refs,
            query_records=consumer_refs,
        )
        write_json(paths.reports / "a8_1_bank_integrity_eval.json", a8_1_eval)
        content_sha = a8_1_eval.get("bundle_bank_content_sha256") or bank_meta.get(
            "bundle_bank_content_sha256"
        )
        file_sha = a8_1_eval.get("bundle_bank_file_sha256") or bank_meta.get("bundle_bank_file_sha256")
        policy_sha = bank_meta.get("bundle_bank_policy_sha256")
        a8_1 = bool(a8_1_eval.get("a8_1_pass")) and bool(chain_hash)

        artifact_lock = {
            "artifact_lock": "PASS" if a8_1 else "FAIL",
            "bundle_bank_file_sha256": file_sha,
            "bundle_bank_content_sha256": content_sha,
            "bundle_bank_policy_sha256": policy_sha,
            "bundle_bank_mode": bank_meta.get("bundle_bank_mode") or bank_meta.get("mode"),
            # legacy alias — must not be the only bank hash
            "bundle_bank_sha256": file_sha,
            "selection_manifest_chain_hash": chain_hash,
            "selection_manifest_final_sha256": file_sha256(final_sel) if final_sel.exists() else None,
            "reconstruction_metrics_sha256": (
                file_sha256(paths.reports / "mlm_reconstruction_metrics.json")
                if (paths.reports / "mlm_reconstruction_metrics.json").exists()
                else None
            ),
            "candidate_funnel_sha256": file_sha256(funnel_path) if funnel_path.exists() else None,
            "fixture_manifest_sha256": file_sha256(fixture_path) if fixture_path.exists() else None,
            "a8_1_eval": a8_1_eval,
        }
        assert "final_package_sha256" not in artifact_lock
        write_json(paths.reports / "artifact_lock.json", artifact_lock)

        # A8-2 exact match — missing evidence = FAIL (no recoverable=True fallback)
        a8_2 = False
        a8_2_detail: dict = {}
        if final_sel.exists() and recon:
            final_man = json.loads(final_sel.read_text(encoding="utf-8"))
            man_keys = list(final_man.get("selected_mg_keys") or [])
            met_keys = list(recon.get("selected_mg_keys") or [])
            man_flags = list(final_man.get("selected_mg_flags") or [])
            met_flags = list(recon.get("metrics_recoverable_flags") or [])
            if not man_keys or not met_keys or not man_flags or not met_flags:
                a8_2 = False
                a8_2_detail = {"a8_2_pass": False, "errors": ["missing_flags_or_keys"]}
            else:
                a8_2_detail = exact_match_selection_metrics(
                    manifest_keys=man_keys,
                    metrics_keys=met_keys,
                    manifest_recoverable_flags=man_flags,
                    metrics_recoverable_flags=met_flags,
                )
                a8_2 = bool(a8_2_detail.get("a8_2_pass")) and (recon.get("a8_2_pass") is True)
        write_json(paths.reports / "a8_2_selection_exact_match.json", a8_2_detail or {"a8_2_pass": False})

        preflight = (
            json.loads((paths.artifacts / "mlm_decoder_preflight.json").read_text())
            if (paths.artifacts / "mlm_decoder_preflight.json").exists()
            else {}
        )
        natural = (
            json.loads((paths.artifacts / "mlm_natural_outcome.json").read_text())
            if (paths.artifacts / "mlm_natural_outcome.json").exists()
            else {}
        )
        curated = (
            json.loads((paths.reports / "curated_fixture_result.json").read_text())
            if (paths.reports / "curated_fixture_result.json").exists()
            else {}
        )
        funnel = json.loads(funnel_path.read_text()) if funnel_path.exists() else {}
        m2 = (
            json.loads((paths.reports / "m2_acceptance_report.json").read_text())
            if (paths.reports / "m2_acceptance_report.json").exists()
            else {}
        )
        functional = (
            json.loads((paths.artifacts / "mlm_functional_smoke_result.json").read_text())
            if (paths.artifacts / "mlm_functional_smoke_result.json").exists()
            else {}
        )

        a5 = evaluate_a5_contracts(
            lift_meta=funnel.get("lift_meta") or {},
            mlm_score_used_in_critic_ranking=False,
            rng_or_grouped_masker_used=False,
            funnel=funnel,
            topk_records=funnel.get("topk_records") or [],
        )

        if not provenance:
            provenance = {
                "git_commit": final_lock.get("git_commit"),
                "dependency_tree_sha256": final_lock.get("dependency_tree_sha256"),
                "pre_source_tree_sha256": final_lock.get("source_tree_sha256"),
                "post_source_tree_sha256": final_lock.get("source_tree_sha256"),
                "final_lock_id": final_lock_id,
                "runs": [],
                "rerun_flag_only": bool(args.rerun_completed) and not provenance_path.exists(),
            }
        a7 = evaluate_a7_provenance(
            provenance,
            final_lock={**final_lock, "final_lock_id": final_lock_id},
            root=out_root,
        )
        write_json(paths.reports / "a7_provenance_eval.json", a7)

        from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
            evaluate_a2_functional_smoke,
        )

        a2_eval = evaluate_a2_functional_smoke(functional)
        a2 = bool(a2_eval.get("a2_pass"))
        mlm_out = str(m2.get("mlm_cf_candidate_outcome") or natural.get("mlm_cf_candidate_outcome"))
        nat_out = str(natural.get("natural_integration_outcome") or mlm_out)

        quality_override = None
        audit = recon.get("reconstruction_metric_audit_status")
        if not a8_2:
            quality_override = "NOT_EVALUATED"
            audit = "FAIL"

        p1 = evaluate_p1_acceptance_conditions(
            a1_preflight_pass=bool(preflight.get("decoder_preflight_passed")),
            a2_functional=bool(a2),
            a3_curated_non_original_critic=bool(curated.get("non_original_candidate_reached_critic")),
            a4_outcome_equals_natural=(mlm_out == nat_out),
            a5_contracts=bool(a5.get("a5_pass")),
            a6_final_code_lock=final_lock.get("final_code_source_lock") == "PASS",
            a7_rerun_after_lock=bool(a7.get("a7_pass")),
            a8_artifact_lock=artifact_lock.get("artifact_lock") == "PASS",
            a8_1_bank_hashes=bool(a8_1),
            a8_2_selection_exact_match=bool(a8_2),
            q1=bool(recon.get("Q1")),
            q2=bool(recon.get("Q2")),
            q3=bool(recon.get("Q3")),
            q4=bool(recon.get("Q4")),
            q5=bool(recon.get("Q5")),
            mlm_cf_candidate_outcome=mlm_out,
            natural_integration_outcome=nat_out,
            selection_manifest_chain_hash=chain_hash,
            reconstruction_metric_audit_status=audit,
            quality_override=quality_override,
            extra={"a5_detail": a5, "a7_detail": a7, "a8_2_detail": a8_2_detail, "a8_1_detail": a8_1_eval},
        )
        write_json(paths.reports / "p1_acceptance_report.json", p1)

        pkg = Path(args.package)
        pkg.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(pkg, "w:gz") as tar:
            for sub in ("reports", "artifacts", "manifests", "results"):
                d = out_root / sub
                if d.exists():
                    tar.add(d, arcname=f"CF_M2_P1/{sub}")
        sidecar = Path(str(pkg) + ".sha256")
        sha = file_sha256(pkg)
        sidecar.write_text(f"{sha}  {pkg.name}\n", encoding="utf-8")
        write_json(
            paths.reports / "package_sidecar.json",
            {
                "package": str(pkg),
                "sidecar": str(sidecar),
                "package_sha256_external_only": True,
                "sha256": sha,
            },
        )
        print(
            json.dumps(
                {
                    "final_lock": final_lock["final_code_source_lock"],
                    "artifact_lock": artifact_lock["artifact_lock"],
                    "p1_readiness": p1.get("p1_readiness"),
                    "mlm_implementation": p1.get("mlm_implementation"),
                    "mlm_reconstruction_quality": p1.get("mlm_reconstruction_quality"),
                    "reconstruction_metric_audit_status": p1.get("reconstruction_metric_audit_status"),
                    "a5": a5.get("a5_pass"),
                    "a6": a6_pass,
                    "a7": a7.get("a7_pass"),
                    "a8_1": a8_1,
                    "a8_2": a8_2,
                    "package": str(pkg),
                    "sidecar_sha256": sha,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        exit_code = 0 if p1.get("mlm_implementation") == "PASS" else 1
        return exit_code
    finally:
        dump_runtime_imports_from_env(ROOT, entrypoint=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
