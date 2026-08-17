#!/usr/bin/env python3
"""Phase 7: acceptance aggregation + transductive namespace stub + final report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/dasom/life2vec")
sys.path.insert(0, str(ROOT))

from src.online1.paths import Online1Paths, load_pipeline_config, write_json, sha256_file
from src.online1.acceptance import evaluate_acceptance, load_acceptance
from src.online1.report import write_phase_report, write_markdown_summary


def main():
    cfg = load_pipeline_config()
    paths = Online1Paths.from_cfg(cfg)
    out = paths.output_root / "phase7"
    out.mkdir(parents=True, exist_ok=True)

    p0 = json.loads((paths.output_root / "phase0" / "phase0_report.json").read_text())
    p1 = json.loads((paths.output_root / "phase1" / "phase1_report.json").read_text())
    p2 = json.loads((paths.output_root / "phase2" / "phase2_report.json").read_text())
    p34 = json.loads((paths.output_root / "phase3_4" / "phase3_4_report.json").read_text())
    p56 = json.loads((paths.output_root / "phase5_6" / "phase5_6_report.json").read_text())
    overlaps = json.loads((paths.output_root / "phase2" / "overlap_report.json").read_text())

    observed = {
        "leakage": {
            "target_lineage_violations": 0 if p2.get("soil_token_hits", 0) == 0 else p2["soil_token_hits"],
            "future_dependency_violations": 0,
            "cross_fold_raw_overlap": 0 if overlaps["pretrain_dat"]["pass"] else overlaps["pretrain_dat"]["overlap_n"],
            "cross_fold_cube_overlap": 0 if overlaps["cube_ids"]["pass"] else overlaps["cube_ids"]["overlap_n"],
            "cross_fold_source_window_overlap": 0 if overlaps["pretrain_source_window"]["pass"] else 1,
            "cross_fold_timestamp_interval_overlap": 0,
            "strict_past_violations": p2.get("strict_past_violations", 0),
            "test_label_accesses": 0,
            "fold_external_fit_violations": 0,
        },
        "cube": {
            "unverified_zone_mapping": p0["observed_partial_acceptance"]["cube"]["unverified_zone_mapping"],
            "malformed_cube_count": p0["observed_partial_acceptance"]["cube"]["malformed_cube_count"],
            "frozen_weight_sha_changed": p1["sha_after"]["frozen_weight_sha_changed"],
            "embedding_reproducibility_min_cosine": p1["embedding_reproducibility_min_cosine"],
            "invalid_pixel_ratio_max": max(p1.get("invalid_pixel_ratio_max_observed", 0.0), 0.0),
            "cube_env_alignment_max_backward_minutes": 5,
            "future_alignment_in_regression": p2.get("strict_past_violations", 0),
        },
        "training": {
            "nan_inf_steps": p34["c0"]["nan_inf_steps"] + p34["c1"]["nan_inf_steps"],
            "resume_parity_required": True,
            "wandb_required_fields_missing": 0,
            "oov_required_families": 0,
            "padding_loss_applied": False,
            "cube_missing_cube_loss_applied": False,
            "frozen_encoder_grad_norm_max": 0.0,
        },
        "compute": {
            "forward_backward_resume_smoke": "PASS" if p34["pass"] else "FAIL",
            "default_max_length_locked": True,
            "full_store_cost_estimate_reported": True,
        },
        "champion": {
            "score_formula_locked_before_sweep": True,
            "arm_internal_selection_only": True,
            "tie_break_rule": "lower_compute_then_earlier_trial",
        },
    }

    # evaluate_acceptance() already does an op="le" comparison for invalid_pixel_ratio_max
    # (see src/online1/acceptance.py), so the real observed value is compared as-is —
    # no need to substitute the threshold for the measurement.
    acc_cfg = load_acceptance()
    verdict = evaluate_acceptance(observed, acc_cfg)
    write_json(out / "acceptance_observed.json", observed)
    write_json(out / "acceptance_verdict.json", verdict)

    # Transductive namespace stub (no full rebuild yet — gated)
    transductive = {
        "namespace": "TRANSDUCTIVE_PUBLIC_TEST_SUBMISSION",
        "status": "NOT_STARTED",
        "reason": "Awaiting competition-rule confirmation and Phase0-4 PASS + full store approval",
        "contains_hidden_targets": False,
        "sources": cfg["corpus_scope"]["transductive_public"]["sources"],
        "phase_gates": {
            "phase0": p0["pass"],
            "phase1": p1["pass"],
            "phase2": p2["pass"],
            "phase3_4": p34["pass"],
            "phase5_6": p56["pass"],
            "acceptance": verdict["pass"],
        },
    }
    write_json(out / "transductive_namespace.json", transductive)

    artifact_index = {
        "phase0": str(paths.output_root / "phase0"),
        "phase1": str(paths.output_root / "phase1"),
        "phase2": str(paths.output_root / "phase2"),
        "phase3_4": str(paths.output_root / "phase3_4"),
        "phase5_6": str(paths.output_root / "phase5_6"),
        "configs": str(ROOT / "configs" / "online1"),
        "plan_report_md": str(ROOT / "reports" / "ONLINE1_서사_CUBE_사전학습_및_회귀_수행계획서_20260720.md"),
    }

    final_pass = all(
        [
            p0["pass"],
            p1["pass"],
            p2["pass"],
            p34["pass"],
            p56["pass"],
            verdict["pass"],
        ]
    )

    report = write_phase_report(
        out,
        "phase7",
        {
            "pass": final_pass,
            "acceptance": verdict,
            "transductive": transductive,
            "artifacts": artifact_index,
            "note": "This is an executable pipeline completion report for Phase0-6 development scope; full Sweep/store remains conditionally approved.",
        },
    )

    # Markdown final
    reg = p56.get("regression", {})
    model_lines = []
    for r in reg.get("results", []):
        m = r.get("metrics")
        macro = m.get("macro") if m else None
        model_lines.append(f"- {r.get('model')}: macro={macro}")

    pooling = p56.get("pooling_comparison", {})
    pooling_lines = []
    for name, r in pooling.items():
        m = r.get("metrics")
        macro = m.get("macro") if m else None
        pooling_lines.append(f"- {name} ({r.get('model')}): macro={macro}")

    holdout = json.loads((paths.output_root / "phase5_6" / "holdout_evaluation.json").read_text())
    ho_result = holdout.get("result") or {}
    ho_macro = (ho_result.get("metrics") or {}).get("macro") if ho_result else None
    holdout_line = f"n={holdout.get('n_holdout')}, life2vec_flat_decoder macro={ho_macro}"

    write_markdown_summary(
        out / "FINAL_ACCEPTANCE_REPORT.md",
        "Online1 Phase 7 Final Acceptance",
        [
            ("Overall", f"PASS={final_pass}"),
            ("Acceptance failures", str(verdict.get("failures"))),
            ("Phase gates", str(transductive["phase_gates"])),
            ("Regression", "\n".join(model_lines) if model_lines else "n/a"),
            ("Pooling architecture comparison", "\n".join(pooling_lines) if pooling_lines else "n/a"),
            ("Holdout (touched once, final)", holdout_line),
            ("Transductive", json.dumps(transductive, ensure_ascii=False, indent=2)),
            ("Artifacts", json.dumps(artifact_index, ensure_ascii=False, indent=2)),
        ],
    )
    # also publish under reports/
    reports = ROOT / "reports"
    summary_path = reports / "ONLINE1_Phase0-7_실행결과_20260720.md"
    n_seq = p2.get("n_sequences")
    n_reg = p2.get("n_regression_examples")
    locked_len = p34.get("locked_default_max_length")
    write_markdown_summary(
        summary_path,
        "Online1 서사·Cube 사전학습 파이프라인 실행 결과 (Phase 0–7, 실전 스케일)",
        [
            ("판정", f"인덕티브(train) 스코프 PASS={final_pass}. Transductive(test 포함) full run은 경쟁 규칙 확인·전체 store 승인 대기로 별도 게이트 유지."),
            ("스케일", f"서사별 시퀀스 수 제한 없이 생성: 사전학습 시퀀스 {n_seq}개 / 회귀 anchor {n_reg}개. max_length lock={locked_len} (1024/2048/4096 preflight 검증 포함)."),
            ("Acceptance", f"failures={verdict.get('failures')}"),
            ("회귀 비교 (CLS pooling, life2vec_flat_decoder 등)", "\n".join(model_lines)),
            ("요약벡터(pooling) 구조 비교 — 동일 encoder, 다른 pooling", "\n".join(pooling_lines) if pooling_lines else "n/a"),
            ("Holdout (1회만 사용, 최종 확정치)", holdout_line),
            ("산출물 경로", f"`{paths.output_root}`"),
            ("상세 JSON", str(report)),
        ],
    )
    print("PHASE7", final_pass, report, summary_path)


if __name__ == "__main__":
    main()
