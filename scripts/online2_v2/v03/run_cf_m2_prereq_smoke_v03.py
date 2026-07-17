#!/usr/bin/env python3
"""CF M2 P0 strict smoke: Problem crossfit + GT-normal Example OOF audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.counterfactual.evaluation.acceptance import build_acceptance_report
from src.online2.v2.finetune_v03.counterfactual.evaluation.over_edit import build_over_edit_report
from src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock import build_source_lock
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths
from src.online2.v2.finetune_v03.counterfactual.pipeline_m2 import (
    stage_m2_compute_attribution,
    stage_m2_path_a_smoke,
    stage_m2_preflight,
    verify_gt_normal,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
    ap.add_argument(
        "--source-archive",
        type=Path,
        default=ROOT / "datasets/agrichallenge/CF_M2_P0_구현산출_20260717.tar.gz",
    )
    args = ap.parse_args()
    cfg_text = args.config.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    config_sha = hashlib.sha256(cfg_text.encode()).hexdigest()
    out_root = Path(cfg["output_root"])
    paths = M1Paths(out_root)

    source_lock = build_source_lock(
        root=ROOT,
        config_path=args.config,
        config_sha256=config_sha,
        source_archive_path=args.source_archive if args.source_archive.exists() else None,
    )
    write_json(paths.reports / "source_lock.json", source_lock)
    if not source_lock.get("working_tree_clean"):
        write_json(paths.reports / "source_lock_dirty_files.json", {
            "files": source_lock.get("dirty_source_files") or [],
            "source_tree_sha256": source_lock.get("source_tree_sha256"),
        })
    print("[0] source lock", json.dumps({
        k: source_lock[k]
        for k in (
            "git_commit",
            "working_tree_clean",
            "config_sha256",
            "source_tree_sha256",
            "source_archive_sha256",
            "reproducibility",
        )
        if k in source_lock
    }), flush=True)

    print("[1/4] preflight (problem crossfit)", flush=True)
    man = stage_m2_preflight(cfg, paths)
    ctx = json.loads((paths.manifests / "fold_execution_context.json").read_text())
    assert ctx["cohort"] == "problem_set", ctx
    assert ctx["evaluation_mode"] == "crossfit_3fold", ctx
    print(json.dumps({"folds": man["fold_ids"], "ctx": {
        "cohort": ctx["cohort"], "mode": ctx["evaluation_mode"],
        "search": ctx["search_fold_ids"], "holdout": ctx["holdout_fold_ids"],
    }}))

    print("[2/4] token IxG attribution (problem)", flush=True)
    attr = stage_m2_compute_attribution(cfg, paths)
    print(json.dumps({"attribution_mode": attr["attribution_mode"]}))

    print("[3/4] Path A production retokenize + material/NO_OP", flush=True)
    path_a = stage_m2_path_a_smoke(cfg, paths)
    selected = path_a["selected"]
    print(json.dumps({
        "n": path_a["n"],
        "parity": path_a["parity"],
        "cf_valid": selected.get("cf_valid"),
        "is_noop": selected.get("is_noop"),
        "evaluation_mode": selected.get("evaluation_mode"),
        "gate0_ok": path_a.get("gate0_ok"),
        "retokenize_ok": path_a.get("retokenize_ok"),
        "leakage_free": path_a.get("leakage_free"),
        "no_eligible_locus": path_a.get("no_eligible_locus"),
    }))

    over_edit = {}
    normal_over_edit_smoke = "NOT_RUN"
    gt_info = {"gt_normal_verified": False}
    normal_id = cfg.get("normal_case_id")
    if normal_id:
        print("[4/4] GT-normal Example OOF over-edit audit", flush=True)
        gt_info = verify_gt_normal(cfg, normal_id)
        write_json(paths.reports / "gt_normal_verification.json", gt_info)
        if not gt_info.get("gt_normal_verified"):
            normal_over_edit_smoke = "FAIL"
            print(f"[fail] GT normal not verified: {gt_info}", flush=True)
        else:
            ncfg = dict(cfg)
            ncfg["case_id"] = normal_id
            ncfg["cohort"] = cfg.get("normal_cohort") or "example_set"
            npaths = M1Paths(out_root / "normal")
            try:
                stage_m2_preflight(ncfg, npaths)
                nctx = json.loads((npaths.manifests / "fold_execution_context.json").read_text())
                assert nctx["evaluation_mode"] == "oof_reference_only", nctx
                assert nctx["cohort"] == "example_set", nctx
                stage_m2_compute_attribution(ncfg, npaths)
                nres = stage_m2_path_a_smoke(ncfg, npaths)
                over_edit = build_over_edit_report(
                    gt_normal_results=[{
                        "is_normal": True,
                        "candidates": [{}] * int(nres["n"]),
                        "selected": nres["selected"],
                    }],
                    route_normal_results=[],
                )
                write_json(paths.reports / "over_edit_metrics.json", over_edit)
                gt_rate = (over_edit.get("over_edit_gt_normal") or {}).get("normal_over_edit_rate")
                if gt_rate is not None and float(gt_rate) > 0:
                    normal_over_edit_smoke = "FAIL"
                else:
                    normal_over_edit_smoke = "PASS"
                print(json.dumps(over_edit.get("over_edit_gt_normal")), flush=True)
            except Exception as e:
                write_json(paths.reports / "over_edit_metrics.json", {"error": str(e)})
                print(f"[fail] normal over-edit: {e}", flush=True)
                normal_over_edit_smoke = "FAIL"

    base_acc = dict(path_a.get("acceptance") or {})
    source_lock_status = str(source_lock.get("reproducibility") or "PARTIAL")
    common_kwargs = dict(
        structural_smoke="PASS" if path_a["parity"] else "FAIL",
        behavioral_cf_smoke=base_acc.get("behavioral_cf_smoke", "NO_VALID_CF"),
        normal_over_edit_smoke=normal_over_edit_smoke,
        scenario_outcome=base_acc.get("scenario_outcome", "NO_VALID_CF"),
        leakage_free=base_acc.get("leakage_free"),
        gate0_ok=base_acc.get("gate0_ok"),
        retokenize_ok=base_acc.get("retokenize_ok"),
        noop_policy_ok=base_acc.get("noop_policy_ok"),
        cohort_integrity=base_acc.get("cohort_integrity"),
        mg_retokenization_smoke=base_acc.get("mg_retokenization_smoke"),
        dependency_closure_retokenization=base_acc.get(
            "dependency_closure_retokenization", "NOT_EVALUATED"
        ),
        full_retokenization=base_acc.get("mg_retokenization_smoke"),
        calibration_ok=base_acc.get("calibration_ok"),
        gt_normal_verified=bool(gt_info.get("gt_normal_verified")),
        tokenizer_integrity=base_acc.get("tokenizer_integrity"),
        sign_agreement_hard_gate=base_acc.get("sign_agreement_hard_gate"),
        source_lock_status=source_lock_status,
        constrained_mlm_execution=base_acc.get("constrained_mlm_execution", "NOT_RUN"),
        mlm_reconstruction_quality=base_acc.get("mlm_reconstruction_quality", "NOT_EVALUATED"),
        mlm_cf_candidate_outcome=base_acc.get("mlm_cf_candidate_outcome", "NOT_EVALUATED"),
        over_edit_gt_normal=over_edit.get("over_edit_gt_normal"),
        over_edit_route_normal=over_edit.get("over_edit_route_normal"),
    )
    # P0: source_lock PARTIAL does not fail the core policy gate
    p0_kwargs = dict(common_kwargs)
    p0_kwargs["source_lock_status"] = None  # reported separately; not a P0 FAIL
    p0_acc = build_acceptance_report(
        **p0_kwargs,
        mlm_required_for_pass=False,
        extra={
            "p0_verdict": {
                "behavioral_policy_smoke": "PASS" if base_acc.get("noop_policy_ok") else "FAIL",
                "cohort_split_smoke": base_acc.get("cohort_integrity"),
                "mg_retokenization_smoke": base_acc.get("mg_retokenization_smoke"),
                "reproducibility_integrity_audit": source_lock_status,
                "acceptance_automation": (
                    "PASS"
                    if base_acc.get("leakage_free") is not None
                    and base_acc.get("tokenizer_integrity") == "PASS"
                    else "PARTIAL"
                ),
                "overall_v3": "INCOMPLETE",
            },
            "source_lock": source_lock,
        },
    )
    acceptance = build_acceptance_report(
        **common_kwargs,
        mlm_required_for_pass=True,
        extra={
            **dict(base_acc.get("extra") or {}),
            "source_lock": source_lock,
            "attribution_method": attr["attribution_mode"],
            "selected_is_noop": bool(selected.get("is_noop")),
            "cf_valid": selected.get("cf_valid"),
            "p0_gate_acceptance": p0_acc.get("implementation_acceptance"),
            "p0_verdict": p0_acc.get("extra", {}).get("p0_verdict"),
            "v3_completion_status": "INCOMPLETE",
            "scope_note": (
                "P0 residual fixes applied; MLM pipeline and 20/35 audits deferred to P1/P2. "
                "dependency_closure_retokenization=NOT_EVALUATED."
            ),
        },
    )
    write_json(paths.reports / "m2_acceptance_report.json", acceptance)
    write_json(paths.reports / "p0_gate_acceptance.json", p0_acc)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    # Exit 0 only when P0 core acceptance is PASS
    return 0 if p0_acc["implementation_acceptance"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
