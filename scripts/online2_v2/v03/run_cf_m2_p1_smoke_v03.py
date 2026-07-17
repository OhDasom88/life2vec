#!/usr/bin/env python3
"""CF M2 P1 smoke: Path A MLM wiring + natural outcome (Problem crossfit)."""

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

from src.online2.v2.finetune_v03.counterfactual.evaluation.acceptance import (
    build_acceptance_report,
    evaluate_p1_acceptance_conditions,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    dump_runtime_imports_from_env,
)
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
    try:
        return _main_body()
    finally:
        dump_runtime_imports_from_env(ROOT, entrypoint=__file__)


def _main_body() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
    ap.add_argument("--rerun-after-final-lock", action="store_true", default=False)
    ap.add_argument(
        "--source-archive",
        type=Path,
        default=ROOT / "datasets/agrichallenge/CF_M2_P1_최종산출.tar.gz",
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
    print("[0] source lock", flush=True)

    print("[1/4] preflight", flush=True)
    man = stage_m2_preflight(cfg, paths)
    print("[2/4] attribution", flush=True)
    attr = stage_m2_compute_attribution(cfg, paths)
    print("[3/4] Path A + MLM", flush=True)
    path_a = stage_m2_path_a_smoke(cfg, paths)
    selected = path_a["selected"]
    base_acc = dict(path_a.get("acceptance") or {})
    natural = json.loads((paths.artifacts / "mlm_natural_outcome.json").read_text()) if (
        paths.artifacts / "mlm_natural_outcome.json"
    ).exists() else {}
    preflight = json.loads((paths.artifacts / "mlm_decoder_preflight.json").read_text()) if (
        paths.artifacts / "mlm_decoder_preflight.json"
    ).exists() else {}
    funnel = json.loads((paths.artifacts / "mlm_path_a_funnel.json").read_text()) if (
        paths.artifacts / "mlm_path_a_funnel.json"
    ).exists() else {}

    over_edit = {}
    normal_over_edit_smoke = "NOT_RUN"
    gt_info = {"gt_normal_verified": False}
    normal_id = cfg.get("normal_case_id")
    if normal_id:
        print("[4/4] GT-normal OOF audit", flush=True)
        gt_info = verify_gt_normal(cfg, normal_id)
        write_json(paths.reports / "gt_normal_verification.json", gt_info)
        if gt_info.get("gt_normal_verified"):
            ncfg = dict(cfg)
            ncfg["case_id"] = normal_id
            ncfg["cohort"] = cfg.get("normal_cohort") or "example_set"
            npaths = M1Paths(out_root / "normal")
            try:
                stage_m2_preflight(ncfg, npaths)
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
                normal_over_edit_smoke = (
                    "FAIL" if gt_rate is not None and float(gt_rate) > 0 else "PASS"
                )
            except Exception as e:
                write_json(paths.reports / "over_edit_metrics.json", {"error": str(e)})
                normal_over_edit_smoke = "FAIL"
        else:
            normal_over_edit_smoke = "FAIL"

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
        source_lock_status=str(source_lock.get("reproducibility") or "PARTIAL"),
        constrained_mlm_execution=base_acc.get("constrained_mlm_execution", "NOT_RUN"),
        mlm_reconstruction_quality=base_acc.get("mlm_reconstruction_quality", "NOT_EVALUATED"),
        mlm_cf_candidate_outcome=base_acc.get("mlm_cf_candidate_outcome", "NOT_EVALUATED"),
        over_edit_gt_normal=over_edit.get("over_edit_gt_normal"),
        over_edit_route_normal=over_edit.get("over_edit_route_normal"),
    )
    acceptance = build_acceptance_report(
        **common_kwargs,
        mlm_required_for_pass=True,
        extra={
            **dict(base_acc.get("extra") or {}),
            "source_lock": source_lock,
            "natural_outcome": natural,
            "preflight": preflight,
            "funnel": funnel,
            "attribution_method": attr["attribution_mode"],
            "scope_note": "P1 MLM Path A wiring; Problem20/Example35 audits are P2.",
        },
    )
    write_json(paths.reports / "m2_acceptance_report.json", acceptance)

    # Functional evidence for A2 — dedicated artifact only
    from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
        evaluate_a2_functional_smoke,
        evaluate_a7_provenance,
        load_execution_provenance,
    )
    from src.online2.v2.finetune_v03.counterfactual.evaluation.funnel_accounting import (
        evaluate_a5_contracts,
    )

    functional = (
        json.loads((paths.artifacts / "mlm_functional_smoke_result.json").read_text())
        if (paths.artifacts / "mlm_functional_smoke_result.json").exists()
        else {}
    )
    a2 = bool(evaluate_a2_functional_smoke(functional).get("a2_pass"))
    # For natural NO_ELIGIBLE, curated may still be pending — A3 recorded separately by curated script
    curated_path = paths.reports / "curated_fixture_result.json"
    curated = json.loads(curated_path.read_text()) if curated_path.exists() else {}
    a3 = bool(curated.get("non_original_candidate_reached_critic"))
    natural_out = str(
        natural.get("natural_integration_outcome")
        or acceptance.get("mlm_cf_candidate_outcome")
        or "NOT_EVALUATED"
    )
    mlm_out = str(acceptance.get("mlm_cf_candidate_outcome") or "NOT_EVALUATED")
    a4 = mlm_out == natural_out
    recon = {}
    recon_path = paths.reports / "mlm_reconstruction_metrics.json"
    if recon_path.exists():
        recon = json.loads(recon_path.read_text())
    chain_hash = recon.get("selection_manifest_chain_hash") or (
        json.loads((paths.reports / "selection_manifest_chain.json").read_text()).get(
            "selection_manifest_chain_hash"
        )
        if (paths.reports / "selection_manifest_chain.json").exists()
        else None
    )
    artifact_lock_path = paths.reports / "artifact_lock.json"
    artifact_lock = (
        json.loads(artifact_lock_path.read_text()) if artifact_lock_path.exists() else {}
    )

    a5 = evaluate_a5_contracts(
        lift_meta=funnel.get("lift_meta") or {},
        funnel=funnel,
        topk_records=funnel.get("topk_records") or [],
    )
    provenance = load_execution_provenance(paths.reports / "execution_provenance.json")
    a7 = evaluate_a7_provenance(
        provenance
        or {
            "runs": [],
            "rerun_flag_only": bool(args.rerun_after_final_lock),
        }
    )
    p1 = evaluate_p1_acceptance_conditions(
        a1_preflight_pass=bool(preflight.get("decoder_preflight_passed")),
        a2_functional=bool(a2),
        a3_curated_non_original_critic=bool(a3),
        a4_outcome_equals_natural=bool(a4),
        a5_contracts=bool(a5.get("a5_pass")),
        a6_final_code_lock=bool(
            (paths.reports / "final_code_source_lock.json").exists()
            and json.loads((paths.reports / "final_code_source_lock.json").read_text()).get(
                "final_code_source_lock"
            )
            == "PASS"
        ),
        a7_rerun_after_lock=bool(a7.get("a7_pass")),
        a8_artifact_lock=str(artifact_lock.get("artifact_lock")) == "PASS",
        a8_1_bank_hashes=bool(artifact_lock.get("bundle_bank_content_sha256")),
        a8_2_selection_exact_match=(recon.get("a8_2_pass") is True),
        q1=bool(recon.get("Q1")),
        q2=bool(recon.get("Q2")),
        q3=bool(recon.get("Q3")),
        q4=bool(recon.get("Q4")),
        q5=bool(recon.get("Q5")),
        mlm_cf_candidate_outcome=mlm_out,
        natural_integration_outcome=natural_out,
        selection_manifest_chain_hash=chain_hash,
        reconstruction_metric_audit_status=recon.get("reconstruction_metric_audit_status"),
        extra={"m2_acceptance": acceptance.get("implementation_acceptance"), "a5": a5, "a7": a7},
    )
    write_json(paths.reports / "p1_acceptance_report.json", p1)
    print(json.dumps({"m2": acceptance.get("implementation_acceptance"), "p1": p1}, indent=2))
    # Smoke exit: MLM must not be NOT_RUN when enabled
    if acceptance.get("constrained_mlm_execution") == "NOT_RUN" and (cfg.get("mlm") or {}).get(
        "run_in_path_a_smoke"
    ):
        return 2
    return 0 if acceptance.get("constrained_mlm_execution") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
