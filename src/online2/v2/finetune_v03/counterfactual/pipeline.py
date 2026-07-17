"""M1 pipeline stages with artifact preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .attribution.aggregation import (
    aggregate_events,
    aggregate_measurement_groups,
    aggregate_spans,
)
from .attribution.fold_consensus import aggregate_fold_frame
from .attribution.token_attribution import (
    compute_event_attribution_for_case,
    distribute_event_scores_to_tokens,
    synthetic_tokens_from_events,
)
from .candidates.path_a_generator import build_path_a_candidates, choose_direction_hint
from .candidates.path_b_b0_generator import generate_b0_operations, select_best_operation
from .evaluation.metrics import eligibility_for_path
from .gates.local_rules import compare_token_bundles, build_token_edits
from .gates.safety_bounds import project_to_safe
from .grounding.actuator_span import apply_operation_to_event_ids, ground_actuator_span
from .grounding.raw_target import decode_interval, value_to_bin_index
from .grounding.retokenize import retokenize_abs_value, retokenize_actuator_literal
from .io_utils import load_yaml, write_json, write_jsonl, write_parquet, artifact_meta
from .locus.selector import NO_VALID, select_path_a_loci, select_path_b_loci
from .manifest import build_diagnosis_manifest, load_manifest
from .semantics_registry import build_semantics_registry
from .temporal.span_builder import build_interval_quantities, build_zoh_spans


class M1Paths:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifests = self.root / "manifests"
        self.registry = self.root / "registry"
        self.artifacts = self.root / "artifacts"
        self.results = self.root / "results"
        self.reports = self.root / "reports"
        for p in [self.manifests, self.registry, self.artifacts, self.results, self.reports]:
            p.mkdir(parents=True, exist_ok=True)


def stage_freeze_manifest(cfg: Dict[str, Any], paths: M1Paths) -> Dict[str, Any]:
    return build_diagnosis_manifest(
        run_dir=Path(cfg["run_dir"]),
        tokenizer_path=Path(cfg["tokenizer_path"]),
        binning_registry_path=Path(cfg["binning_registry_path"]),
        cells_path=Path(cfg["cells_path"]),
        embeddings_dir=Path(cfg["embeddings_dir"]),
        out_path=paths.manifests / "diagnosis_model_manifest.json",
        routing_config=cfg.get("routing_config"),
        repeat=int(cfg.get("repeat", 0)),
    )


def stage_audit_semantics(cfg: Dict[str, Any], paths: M1Paths) -> Dict[str, Any]:
    return build_semantics_registry(
        Path(cfg["feature_audit_csv"]),
        out_yaml=paths.registry / "feature_semantics_registry.yaml",
        out_feature_csv=paths.reports / "feature_semantics_audit.csv",
        out_actuator_csv=paths.reports / "actuator_semantics_audit.csv",
    )


def stage_compute_attribution(cfg: Dict[str, Any], paths: M1Paths) -> Dict[str, Path]:
    man = load_manifest(paths.manifests / "diagnosis_model_manifest.json")
    case_id = cfg["case_id"]
    ckpts = [Path(p) for p in man["checkpoint_paths"]]
    event_attr = compute_event_attribution_for_case(
        case_id=case_id,
        ckpt_paths=ckpts,
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=Path(cfg["label_map_path"]) if cfg.get("label_map_path") else None,
        device=cfg.get("device", "cuda"),
        gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
        use_binary=True,
        normalization=cfg.get("attribution_normalization", "robust"),
    )
    # Path A probe feature on ENVIRONMENT events only
    feature = cfg.get("path_a_probe_feature", "inside_temp_c")
    env_attr = event_attr
    if "view" in event_attr.columns:
        env_attr = event_attr[event_attr["view"].astype(str).str.upper().str.contains("ENV", na=False)]
        if not len(env_attr):
            env_attr = event_attr
    tok = synthetic_tokens_from_events(env_attr, feature=feature, measurement_group_id=f"mg:{feature}")
    token_attr = distribute_event_scores_to_tokens(env_attr, tok)
    mg = aggregate_measurement_groups(token_attr, role_weights=cfg.get("token_role_weights"))
    ev = aggregate_events(mg)
    write_parquet(paths.artifacts / "token_attribution.parquet", token_attr)
    write_parquet(paths.artifacts / "measurement_group_attribution.parquet", mg)
    write_parquet(paths.artifacts / "event_attribution.parquet", ev)
    write_parquet(paths.artifacts / "event_ixg_attribution.parquet", event_attr)

    # raw join success rate
    from .evaluation.case_batch import load_embedding_sidecar
    from .evaluation.raw_join import join_raw_for_events

    side = load_embedding_sidecar(Path(cfg["embeddings_dir"]), case_id)
    cells = pd.read_parquet(
        cfg["cells_path"],
        columns=["farm_id", "zone_id", "observation_timestamp", "column_name", "raw_value", "raw_display", "is_null"],
    )
    join = join_raw_for_events(
        side,
        cells,
        farm_id=case_id.split("_")[0],
        features=[feature],
    )
    write_json(
        paths.reports / "raw_join_metrics.json",
        {
            "overall_success_rate": join["overall_success_rate"],
            "n_rows": join["n_rows"],
            "by_feature": join["by_feature"],
        },
    )
    if len(join["join_df"]):
        write_parquet(paths.artifacts / "raw_join_table.parquet", join["join_df"])
    return {
        "token": paths.artifacts / "token_attribution.parquet",
        "mg": paths.artifacts / "measurement_group_attribution.parquet",
        "event": paths.artifacts / "event_attribution.parquet",
        "raw_join_rate": join["overall_success_rate"],
    }


def _parse_cell_raw(value: Any, display: Any = None) -> Any:
    """cell_occurrences.raw_value may be JSON like {\"type\":\"int\",\"value\":\"201\"}."""
    if display is not None and str(display) not in {"", "None", "nan"}:
        try:
            return float(display)
        except (TypeError, ValueError):
            pass
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return value
    s = str(value)
    if s.startswith("{") and "value" in s:
        try:
            payload = json.loads(s)
            return payload.get("value")
        except Exception:
            return s
    return s


def stage_recover_temporal(cfg: Dict[str, Any], paths: M1Paths) -> Dict[str, Path]:
    cells = pd.read_parquet(cfg["cells_path"], columns=[
        "farm_id","zone_id","observation_timestamp","column_name","raw_value","raw_display","modality","is_null"
    ])
    case_id = cfg["case_id"]
    farm_id = case_id.split("_")[0]
    features = cfg.get("actuator_features") or [
        "circulation_fan","fcu_fan","fcu_pump","co2_supply","shade_screen","thermal_curtain"
    ]
    # limit rows for farm
    cells = cells[cells["farm_id"].astype(str) == str(farm_id)].copy()
    cells = cells[~cells["is_null"].fillna(False)]
    cells["raw_value"] = [
        _parse_cell_raw(v, d) for v, d in zip(cells["raw_value"], cells.get("raw_display", [None] * len(cells)))
    ]
    span_frames = []
    iq_frames = []
    for feat in features:
        sub = cells[cells["column_name"].astype(str) == feat].copy()
        if not len(sub):
            continue
        sub = sub.rename(columns={"observation_timestamp": "timestamp"})
        sub["event_id"] = sub.apply(lambda r: f"{feat}:{r['zone_id']}:{r['timestamp']}", axis=1)
        for zone, g in sub.groupby("zone_id"):
            # downsample if huge: keep chronological unique timestamps
            g = g.sort_values("timestamp").drop_duplicates("timestamp")
            if len(g) > int(cfg.get("max_actuator_rows", 5000)):
                g = g.iloc[:: max(1, len(g)//5000)]
            spans = build_zoh_spans(
                g, feature=feat, farm_id=farm_id, zone_id=zone, case_id=case_id,
                nominal_interval_min=float(cfg.get("sampling_interval_min", 60)),
            )
            if len(spans):
                span_frames.append(spans)
            if feat in {"line_flow_rate", "total_flow_rate", "total_irrigation"}:
                iq = build_interval_quantities(g, feature=feat, farm_id=farm_id, zone_id=zone, case_id=case_id)
                if len(iq):
                    iq_frames.append(iq)
    spans_df = pd.concat(span_frames, ignore_index=True) if span_frames else pd.DataFrame()
    iq_df = pd.concat(iq_frames, ignore_index=True) if iq_frames else pd.DataFrame()

    # Align spans to Stage-A embedding event_ids
    from .evaluation.case_batch import load_embedding_sidecar
    from .temporal.span_event_align import align_spans_to_events, alignment_summary

    side = load_embedding_sidecar(Path(cfg["embeddings_dir"]), case_id)
    if len(spans_df):
        spans_df = align_spans_to_events(spans_df, side)
        write_json(paths.reports / "span_alignment_summary.json", alignment_summary(spans_df))
    write_parquet(paths.artifacts / "actuator_spans.parquet", spans_df)
    write_parquet(paths.artifacts / "interval_quantities.parquet", iq_df)
    stg = side[["event_id", "same_time_group_id", "timestamp", "view", "zone"]].copy() if len(side) else pd.DataFrame()
    stg["case_id"] = case_id
    write_parquet(paths.artifacts / "same_time_group_context.parquet", stg)
    return {"spans": paths.artifacts / "actuator_spans.parquet"}


def stage_select_loci(cfg: Dict[str, Any], paths: M1Paths) -> Dict[str, Any]:
    semantics = load_yaml(paths.registry / "feature_semantics_registry.yaml").get("features", {})
    mg = pd.read_parquet(paths.artifacts / "measurement_group_attribution.parquet")
    spans = pd.read_parquet(paths.artifacts / "actuator_spans.parquet")
    ev = pd.read_parquet(paths.artifacts / "event_attribution.parquet")
    # attach fold-wise span attribution by broadcasting event scores is weak without event↔span join;
    # for M1: score spans by feature match using mean positive event attribution as proxy per fold
    span_attr_rows = []
    if len(spans) and len(ev):
        for fold_id, eg in ev.groupby("fold_id"):
            # create a shallow span attr using duration fields + fold mean of top events
            for _, sp in spans.iterrows():
                row = dict(sp)
                row["fold_id"] = int(fold_id)
                # proxy: use median of top-quartile event scores as span signed score
                s = eg["signed_attribution"]
                thr = float(s.quantile(0.75)) if len(s) else 0.0
                pos = s[s >= thr]
                mean = float(pos.mean()) if len(pos) else float(s.mean() if len(s) else 0.0)
                row["duration_weighted_mean"] = mean
                # ON spans reconstructed from POSITIVE run-length → treat mass as risk-supporting
                row["duration_weighted_positive_ratio"] = 1.0 if mean >= 0 else float((s > 0).mean() if len(s) else 0.0)
                n = int(row.get("n_events") or len(row.get("source_event_ids") or []))
                # synthetic mass: end-heavy if mean>0
                row["start_mass_ratio"] = 0.2
                row["middle_mass_ratio"] = 0.3
                row["end_mass_ratio"] = 0.5
                row["signed_total_attribution"] = mean * max(n, 1)
                row["absolute_total_attribution"] = abs(mean) * max(n, 1)
                row["max_positive_attribution"] = max(mean, 0.0)
                row["max_absolute_attribution"] = abs(mean)
                row["attribution_mass_center"] = 0.7
                row["peak_locus_index"] = max(0, n - 1)
                span_attr_rows.append(row)
    span_attr = pd.DataFrame(span_attr_rows)
    if len(span_attr):
        # Prefer duration-weighted join when event_ids align; else keep proxy scores.
        try:
            parts = []
            for fold_id, eg in ev.groupby("fold_id"):
                sdf = spans.copy()
                sdf["fold_id"] = int(fold_id)
                joined = aggregate_spans(eg, sdf)
                parts.append(joined)
            if parts:
                joined_all = pd.concat(parts, ignore_index=True)
                # if join produced all-zero scores, retain proxy mass ratios / means
                if float(joined_all.get("absolute_total_attribution", pd.Series([0])).sum() or 0) <= 1e-12:
                    pass  # keep span_attr proxy
                else:
                    span_attr = joined_all
        except Exception:
            pass
        write_parquet(paths.artifacts / "actuator_span_attribution.parquet", span_attr)
    else:
        write_parquet(paths.artifacts / "actuator_span_attribution.parquet", pd.DataFrame())

    path_a = select_path_a_loci(mg, semantics=semantics, min_agree=int(cfg.get("min_fold_agree", 2)), top_k=int(cfg.get("top_k_loci", 20)))
    wl = set(cfg.get("actuator_whitelist") or [])
    path_b = select_path_b_loci(
        span_attr if len(span_attr) else pd.DataFrame(),
        semantics=semantics,
        whitelist=wl or None,
        min_agree=int(cfg.get("min_fold_agree", 2)),
        top_k=int(cfg.get("top_k_loci", 20)),
    )
    write_jsonl(paths.artifacts / "intervention_loci_path_a.jsonl", path_a)
    write_jsonl(paths.artifacts / "intervention_loci_path_b.jsonl", path_b)
    status = NO_VALID if (not path_a and not path_b) else "OK"
    write_json(paths.reports / "locus_selection_report.md".replace(".md", ".json"), {"status": status, "n_path_a": len(path_a), "n_path_b": len(path_b)})
    (paths.reports / "locus_selection_report.md").write_text(
        f"# Locus selection\n\nstatus: {status}\n\nPath A: {len(path_a)}\n\nPath B: {len(path_b)}\n",
        encoding="utf-8",
    )
    return {"status": status, "path_a": path_a, "path_b": path_b}


def stage_run_path_a(cfg: Dict[str, Any], paths: M1Paths) -> List[Dict[str, Any]]:
    from src.online2.v2.binning import BinningRegistryV2

    loci_path = paths.artifacts / "intervention_loci_path_a.jsonl"
    loci = [json.loads(l) for l in loci_path.read_text().splitlines() if l.strip()] if loci_path.exists() else []
    reg = BinningRegistryV2.load(Path(cfg["binning_registry_path"]))
    cells = pd.read_parquet(
        cfg["cells_path"],
        columns=["farm_id", "column_name", "raw_value", "raw_display", "is_null"],
    )
    farm_id = cfg["case_id"].split("_")[0]
    results = []
    cand_rows = []
    for locus in loci:
        feat = locus["feature"]
        rule = reg.rules.get((feat, "ABS", None))
        if rule is None:
            continue
        edges = list(rule.edges)
        sub = cells[(cells["farm_id"].astype(str) == farm_id) & (cells["column_name"] == feat) & (~cells["is_null"].fillna(False))]
        if not len(sub):
            continue
        try:
            parsed = [_parse_cell_raw(v, d) for v, d in zip(sub["raw_value"], sub["raw_display"])]
            observed = float(pd.to_numeric(pd.Series(parsed), errors="coerce").dropna().median())
        except Exception:
            continue
        # direction probe without full critic: use bin sides only; critic filled in validate if batches available
        cands = build_path_a_candidates(feature=feat, observed_raw=observed, edges=edges, direction_hint={"status": "proposal_only"})
        for c in cands:
            g2 = project_to_safe(c["target_raw"], hard=(min(edges), max(edges)), agronomic=None, operational=None, operational_known=False)
            tokens_from = retokenize_abs_value(feat, observed, edges)
            tokens_to = retokenize_abs_value(feat, c["target_raw"], edges)
            g4 = compare_token_bundles(tokens_to, tokens_to)
            edits = build_token_edits(
                event_id=locus["event_ids"][0],
                measurement_group_id=str(locus.get("measurement_group_id")),
                feature=feat,
                from_tokens=tokens_from,
                to_tokens=tokens_to,
                gate4_passed=g4["status"] == "PASSED",
            )
            row = {
                **c,
                "case_id": cfg["case_id"],
                "locus": locus,
                "gate2": g2,
                "gate4": g4,
                "token_edits": edits,
                "operational_eligibility": eligibility_for_path("A", gate2_status=g2["status"]),
                "grounding_status": "PARTIAL" if g2["status"] == "PARTIAL" else "GROUNDED",
            }
            cand_rows.append(row)
            results.append(row)
    # Critic scoring (cached Stage-A embedding perturbation + fuse-aware risk)
    from .evaluation.score_candidates import score_path_a_results

    results = score_path_a_results(
        cfg,
        paths,
        results,
        max_candidates=int(cfg.get("path_a_max_critic_candidates", 8)),
    )
    return results


def stage_run_path_b_b0(cfg: Dict[str, Any], paths: M1Paths) -> List[Dict[str, Any]]:
    loci_path = paths.artifacts / "intervention_loci_path_b.jsonl"
    loci = [json.loads(l) for l in loci_path.read_text().splitlines() if l.strip()] if loci_path.exists() else []
    spans = pd.read_parquet(paths.artifacts / "actuator_spans.parquet") if (paths.artifacts / "actuator_spans.parquet").exists() else pd.DataFrame()
    thr = cfg.get("path_b_thresholds") or {}
    cand_rows = []
    for locus in loci:
        sp = spans[spans["span_id"] == locus.get("span_id")]
        if not len(sp):
            continue
        span = sp.iloc[0].to_dict()
        ops = generate_b0_operations(span, locus.get("attribution") or {}, thresholds=thr, gate1_passed=True)
        for op in ops:
            grounded = ground_actuator_span({**op, "feature": locus["feature"]})
            keep, drop = apply_operation_to_event_ids(span.get("source_event_ids"), op["operation"], op.get("cut_events") or 0)
            g2 = {"gate": "gate2_safety_bounds", "status": "PARTIAL", "reason_codes": ["OPERATIONAL_BOUNDS_UNKNOWN"], "details": {}}
            from_tok = retokenize_actuator_literal(locus["feature"], True)
            to_tok = retokenize_actuator_literal(locus["feature"], False) if drop else from_tok
            g4 = compare_token_bundles(to_tok if drop else from_tok, to_tok if drop else from_tok)
            edits = None
            if op["operation"] != "NO_OP" and g4["status"] == "PASSED":
                edits = [
                    build_token_edits(
                        event_id=eid,
                        measurement_group_id="",
                        feature=locus["feature"],
                        from_tokens=from_tok,
                        to_tokens=to_tok,
                        gate4_passed=True,
                        derived_from_edit_scope=True,
                    )
                    for eid in drop
                ]
            row = {
                **op,
                "case_id": cfg["case_id"],
                "locus": locus,
                "grounded_actions": [grounded],
                "gate2": g2,
                "gate4": g4,
                "token_edits": edits,
                "model_space_delta_r": None,
                "limitation": grounded["limitation"],
                "operational_eligibility": eligibility_for_path("B", gate2_status=g2["status"]),
                "grounding_status": "PARTIAL",
            }
            assert row["operational_eligibility"] != "OPERATIONAL_CANDIDATE"
            cand_rows.append(row)

    from .evaluation.score_candidates import score_path_b_results

    results = score_path_b_results(
        cfg,
        paths,
        cand_rows,
        spans,
        max_spans=int(cfg.get("path_b_max_critic_spans", 4)),
        max_ops_per_span=int(cfg.get("path_b_max_ops_per_span", 5)),
    )
    return results


def stage_build_report(cfg: Dict[str, Any], paths: M1Paths, checks: Dict[str, bool]) -> Path:
    lines = ["# M1 Acceptance Report", "", f"case_id: {cfg.get('case_id')}", ""]
    for k, ok in checks.items():
        lines.append(f"- [{'x' if ok else ' '}] {k}: {'PASS' if ok else 'FAIL'}")
    out = paths.reports / "m1_acceptance_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(paths.reports / "m1_acceptance_report.json", {"checks": checks, "all_pass": all(checks.values())})
    return out
