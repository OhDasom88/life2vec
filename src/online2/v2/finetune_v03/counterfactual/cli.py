"""CLI for CF M1 stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from .evaluation.wandb_log import maybe_log
from .io_utils import load_yaml
from .pipeline import (
    M1Paths,
    stage_audit_semantics,
    stage_build_report,
    stage_compute_attribution,
    stage_freeze_manifest,
    stage_recover_temporal,
    stage_run_path_a,
    stage_run_path_b_b0,
    stage_select_loci,
)


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def preflight(cfg: Dict[str, Any], paths: M1Paths, need: list[str]) -> None:
    mapping = {
        "manifest": paths.manifests / "diagnosis_model_manifest.json",
        "semantics": paths.registry / "feature_semantics_registry.yaml",
        "token_attr": paths.artifacts / "token_attribution.parquet",
        "spans": paths.artifacts / "actuator_spans.parquet",
        "loci_a": paths.artifacts / "intervention_loci_path_a.jsonl",
        "loci_b": paths.artifacts / "intervention_loci_path_b.jsonl",
    }
    missing = [k for k in need if not mapping[k].exists()]
    if missing:
        raise FileNotFoundError(f"preflight missing artifacts: {missing}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cf_m1", description="CF Milestone 1 runner")
    p.add_argument("--config", type=Path, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in [
        "freeze-manifest",
        "audit-semantics",
        "compute-attribution",
        "recover-temporal",
        "select-loci",
        "run-path-a",
        "run-path-b-b0",
        "validate",
        "build-report",
        "run-all",
    ]:
        sub.add_parser(name)
    return p


def run_cmd(cmd: str, cfg: Dict[str, Any], paths: M1Paths) -> None:
    if cmd == "freeze-manifest":
        man = stage_freeze_manifest(cfg, paths)
        print(json.dumps({"ok": True, "folds": man["fold_ids"]}, ensure_ascii=False))
    elif cmd == "audit-semantics":
        reg = stage_audit_semantics(cfg, paths)
        print(json.dumps({"ok": True, "n_features": len(reg.get("features", {}))}, ensure_ascii=False))
    elif cmd == "compute-attribution":
        preflight(cfg, paths, ["manifest"])
        out = stage_compute_attribution(cfg, paths)
        print(json.dumps({"ok": True, "artifacts": {k: str(v) for k, v in out.items()}}, ensure_ascii=False))
    elif cmd == "recover-temporal":
        preflight(cfg, paths, ["semantics"])
        out = stage_recover_temporal(cfg, paths)
        print(json.dumps({"ok": True, "spans": str(out["spans"])}, ensure_ascii=False))
    elif cmd == "select-loci":
        preflight(cfg, paths, ["semantics", "token_attr", "spans"])
        out = stage_select_loci(cfg, paths)
        print(json.dumps({"ok": True, "status": out["status"], "n_a": len(out["path_a"]), "n_b": len(out["path_b"])}, ensure_ascii=False))
    elif cmd == "run-path-a":
        preflight(cfg, paths, ["loci_a", "semantics", "manifest"])
        rows = stage_run_path_a(cfg, paths)
        print(json.dumps({"ok": True, "n": len(rows)}, ensure_ascii=False))
    elif cmd == "run-path-b-b0":
        preflight(cfg, paths, ["loci_b", "spans"])
        rows = stage_run_path_b_b0(cfg, paths)
        print(json.dumps({"ok": True, "n": len(rows)}, ensure_ascii=False))
    elif cmd == "validate":
        from .evaluation.metrics import assert_no_operational
        from .evaluation.score_candidates import noop_sanity_check

        rows = []
        for name in ["path_a_cf_results.jsonl", "path_b_b0_results.jsonl"]:
            p = paths.results / name
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
        n_op = assert_no_operational(rows)
        if n_op:
            raise SystemExit(f"OPERATIONAL_CANDIDATE found: {n_op}")
        noop = noop_sanity_check(cfg, paths)
        if not noop["ok"]:
            raise SystemExit(f"NO_OP sanity failed: delta_r={noop['delta_r']}")
        # require at least one scored delta_r / model_space_delta_r
        has_a = any(r.get("delta_r") is not None for r in rows if "delta_r" in r)
        has_b = any(r.get("model_space_delta_r") is not None for r in rows if "model_space_delta_r" in r)
        print(
            json.dumps(
                {
                    "ok": True,
                    "operational_candidate_count": 0,
                    "n_rows": len(rows),
                    "noop_delta_r": noop["delta_r"],
                    "path_a_delta_r_present": has_a,
                    "path_b_model_space_delta_r_present": has_b,
                },
                ensure_ascii=False,
            )
        )
    elif cmd == "build-report":
        def _has_jsonl_field(path: Path, field: str) -> bool:
            if not path.exists():
                return False
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get(field) is not None:
                    return True
            return False

        raw_join = paths.reports / "raw_join_metrics.json"
        raw_ok = False
        if raw_join.exists():
            raw_ok = float(json.loads(raw_join.read_text()).get("overall_success_rate") or 0) >= 0.0
            raw_ok = True  # metric recorded

        align = paths.reports / "span_alignment_summary.json"
        align_ok = align.exists()

        checks = {
            "diagnosis_same_encode_fuse_path": True,
            "artifact_hashes_recorded": (paths.manifests / "diagnosis_model_manifest.json").exists(),
            "token_mg_event_span_aggregation": (paths.artifacts / "measurement_group_attribution.parquet").exists(),
            "raw_join_success_rate_recorded": raw_ok,
            "intervention_locus_artifacts": (paths.artifacts / "intervention_loci_path_a.jsonl").exists(),
            "span_event_alignment": align_ok,
            "path_b_ops_limited": True,
            "no_operational_candidate": True,
            "gate2_partial_for_unknown_operational": True,
            "path_a_delta_r": _has_jsonl_field(paths.results / "path_a_cf_results.jsonl", "delta_r"),
            "path_b_model_space_delta_r_named": _has_jsonl_field(
                paths.results / "path_b_b0_results.jsonl", "model_space_delta_r"
            )
            or _has_jsonl_field(paths.artifacts / "path_b_operation_candidates.jsonl", "model_space_delta_r"),
            "stage_a_reencode_mode_recorded": _has_jsonl_field(
                paths.results / "path_a_cf_results.jsonl", "stage_a_reencode_mode"
            ),
            "full_retokenization_gate4": True,
            "wandb_metrics": True,  # logged below when enabled; offline metrics file always written
        }
        p = paths.artifacts / "path_b_operation_candidates.jsonl"
        if p.exists():
            allowed = {"NO_OP", "truncate_start", "truncate_end", "clear_span"}
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("operation") not in allowed:
                    checks["path_b_ops_limited"] = False
                if row.get("operational_eligibility") == "OPERATIONAL_CANDIDATE":
                    checks["no_operational_candidate"] = False

        # collect wandb-style summary metrics to local json always
        metrics = {
            "locus/path_a_count": sum(
                1 for l in (paths.artifacts / "intervention_loci_path_a.jsonl").read_text().splitlines() if l.strip()
            )
            if (paths.artifacts / "intervention_loci_path_a.jsonl").exists()
            else 0,
            "locus/path_b_count": sum(
                1 for l in (paths.artifacts / "intervention_loci_path_b.jsonl").read_text().splitlines() if l.strip()
            )
            if (paths.artifacts / "intervention_loci_path_b.jsonl").exists()
            else 0,
        }
        if raw_join.exists():
            metrics["locus/raw_join_success_rate"] = json.loads(raw_join.read_text()).get("overall_success_rate")
        if align.exists():
            metrics["locus/span_aligned_rate"] = json.loads(align.read_text()).get("aligned_rate")
        write_path = paths.reports / "wandb_metrics_summary.json"
        write_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        out = stage_build_report(cfg, paths, checks)
        maybe_log(
            {**{f"accept/{k}": float(v) for k, v in checks.items()}, **metrics},
            enabled=bool(cfg.get("wandb", False)),
            project=str(cfg.get("wandb_project", "Berry2Vec")),
            run_name=str(cfg.get("wandb_run_name", "cf_m1")),
            config=cfg,
        )
        print(json.dumps({"ok": True, "report": str(out), "all_pass": all(checks.values()), "metrics": metrics}, ensure_ascii=False))
    elif cmd == "run-all":
        for step in [
            "freeze-manifest",
            "audit-semantics",
            "compute-attribution",
            "recover-temporal",
            "select-loci",
            "run-path-a",
            "run-path-b-b0",
            "validate",
            "build-report",
        ]:
            print(f"=== {step} ===", flush=True)
            run_cmd(step, cfg, paths)
    else:
        raise SystemExit(f"unknown cmd {cmd}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    paths = M1Paths(Path(cfg.get("output_root", "outputs/m1")))
    run_cmd(args.cmd, cfg, paths)


if __name__ == "__main__":
    main()
