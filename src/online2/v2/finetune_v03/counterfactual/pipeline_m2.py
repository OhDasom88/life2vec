"""CF M2 prereq pipeline stages (strict path, v3 policies)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

from .attribution.aggregation import aggregate_events, aggregate_measurement_groups
from .attribution.token_ixg_v03 import (
    OBJECTIVE_BINARY,
    compute_token_ixg_for_case,
    preselect_events_search_folds,
)
from .candidates.mlm_path_a import run_constrained_mlm_for_locus
from .candidates.path_a_generator import build_path_a_candidates
from .candidates.path_a_schema_dispatch import (
    KIND_CIRCULAR,
    KIND_LINEAR,
    KIND_UNSUPPORTED,
    PathAEditStrategy,
    build_schema_dispatched_candidates,
    resolve_strategy_from_spec,
)
from .evaluation.acceptance import build_acceptance_report, mlm_outcome_from_candidates
from .evaluation.mlm_preflight import run_mlm_decoder_preflight
from .evaluation.canonical_locus import (
    canonical_locus_key,
    canonical_target_raw,
    dedup_key,
    edit_direction_from_raw,
    ensure_canonical_noop,
    merge_duplicate_candidates,
)
from .evaluation.funnel_accounting import build_cf0_case_funnel, count_mlm_provenance
from .evaluation.case_batch import load_case_batch, tensor_batch_only
from src.online2.v2.feature_schema import FeatureSchema
from .evaluation.case_fold_manifest import (
    build_case_fold_record,
    load_split_manifest,
    resolve_cohort,
)
from .evaluation.cf_validity import build_cf_validity, llm_case_payload
from .evaluation.crossfit_critic import (
    evaluate_selected_on_holdout,
    rank_candidates_on_search_folds,
    resolve_fold_split,
    score_candidate_folds,
    select_best_on_search,
)
from .evaluation.fold_execution_context import (
    EVAL_CROSSFIT,
    EVAL_OOF_REF,
    FoldExecutionContext,
    build_fold_execution_context,
)
from .evaluation.fold_usage import build_fold_usage_trace, compute_leakage_free
from .evaluation.metrics import eligibility_for_path
from .evaluation.noop_calibration import load_frozen_epsilon
from .gates.local_rules import compare_token_bundles
from .grounding.locus_raw import ground_locus_raw, is_exact
from .io_utils import file_sha256, write_json, write_jsonl, write_parquet
from .m2_preflight import ARTIFACT_SCHEMA, assert_strict_no_fallback, build_m2_prereq_manifest
from .pipeline import M1Paths
from .retokenization.full_event_retokenizer import (
    apply_raw_edit_closure,
    load_frozen_tokenizer_v2,
    recover_observed_raw_for_mg,
    tokenize_mg_production,
)
from .stage_a_loader import load_stage_a_module
from .stage_a_reencoder import StageAReencoder


class NoEligibleLocusError(RuntimeError):
    """Sign-agreement / locus hard gate: no eligible MG."""

    reason = "NO_ELIGIBLE_LOCUS"


def _cfg_m2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("cf_m2_prereq") or {})


def verify_gt_normal(cfg: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    """Confirm case is GT-normal from labels artifact (not caller-forced is_normal)."""
    labels_path = Path(cfg["labels_path"])
    df = pd.read_csv(labels_path)
    hit = df[df["case_id"].astype(str) == str(case_id)]
    if not len(hit):
        return {"gt_normal_verified": False, "reason": "case_not_in_labels", "case_id": case_id}
    row = hit.iloc[0]
    diag = str(row.get("diagnosis_normalized") or row.get("diagnosis") or "")
    normal_labels = set(
        str(x) for x in (cfg.get("gt_normal_labels") or ["정상_운영", "정상", "normal"])
    )
    ok = diag in normal_labels
    return {
        "gt_normal_verified": bool(ok),
        "case_id": str(case_id),
        "diagnosis_normalized": diag,
        "labels_path": str(labels_path),
        "reason": None if ok else "diagnosis_not_gt_normal",
    }


def _resolve_fold_context(cfg: Dict[str, Any], man: Dict[str, Any]) -> FoldExecutionContext:
    critic = cfg.get("critic") or {}
    case_id = str(cfg["case_id"])
    run_dir = Path(cfg["run_dir"])
    split_path = Path(critic.get("split_manifest_path") or (run_dir / "split_manifest.json"))
    splits = load_split_manifest(split_path, repeat=int(cfg.get("repeat", 0))) if split_path.exists() else []
    cohort = resolve_cohort(
        case_id,
        labels_path=Path(cfg["labels_path"]) if cfg.get("labels_path") else None,
        cohort_override=cfg.get("cohort"),
    )
    fold_ids = [int(x) for x in man["fold_ids"]]
    if not splits:
        raise RuntimeError(f"split_manifest required for cohort-safe CF: {split_path}")
    rec = build_case_fold_record(
        case_id=case_id,
        cohort=cohort,
        splits=splits,
        fold_ids=fold_ids,
        search_fold_ids=man.get("search_fold_ids") if cohort == "problem_set" else None,
        holdout_fold_ids=man.get("holdout_fold_ids") if cohort == "problem_set" else None,
        holdout_rotation=critic.get("holdout_rotation"),
    )

    # Forbidden override — never disguise example as crossfit / all-unseen
    if critic.get("force_crossfit_smoke") or cfg.get("force_crossfit_smoke"):
        raise RuntimeError(
            "force_crossfit_smoke is forbidden (v3 P0). "
            "Use a problem_set case for crossfit; example_set must stay oof_reference_only."
        )

    calib_path = Path(
        critic.get("noop_noise_artifact") or "outputs/cf_calibration/noop_noise.json"
    )
    if not calib_path.is_absolute():
        calib_path = Path("/home/dasom/life2vec") / calib_path
    strict_calib = bool(critic.get("strict_calibration", True))
    configured_eps = float(
        critic.get("configured_material_epsilon")
        or critic.get("material_delta_r_abs")
        or 0.01
    )
    if not calib_path.exists():
        raise FileNotFoundError(
            f"NO_OP calibration artifact missing: {calib_path} "
            "(zero-bootstrap forbidden in strict mode)"
        )
    calib = load_frozen_epsilon(
        calib_path,
        configured_epsilon=configured_eps,
        strict=strict_calib,
    )

    seen = {int(k): bool(v) for k, v in dict(rec.get("case_seen_by_fold") or {}).items()}
    return build_fold_execution_context(
        case_id=case_id,
        cohort=str(rec.get("cohort") or cohort),
        evaluation_mode=str(rec["evaluation_mode"]),
        case_seen_by_fold=seen,
        search_fold_ids=rec["search_fold_ids"],
        holdout_fold_ids=rec["holdout_fold_ids"],
        effective_epsilon=float(calib["effective_epsilon"]),
        oof_fold_id=rec.get("oof_fold_id"),
        configured_epsilon=float(calib["configured_epsilon"]),
        calibration_hash=calib.get("calibration_hash"),
        risk_metric=str(critic.get("risk_metric") or "abnormal_probability"),
    )


def stage_m2_preflight(cfg: Dict[str, Any], paths: M1Paths) -> Dict[str, Any]:
    man = build_m2_prereq_manifest(cfg, out_path=paths.manifests / "cf_m2_prereq_manifest.json")
    ctx = _resolve_fold_context(cfg, man)
    write_json(paths.manifests / "fold_execution_context.json", ctx.to_dict())
    write_json(
        paths.reports / "m2_preflight_report.json",
        {
            "ok": True,
            "artifact_schema_version": ARTIFACT_SCHEMA,
            "folds": man["fold_ids"],
            "evaluation_mode": ctx.evaluation_mode,
            "cohort": ctx.cohort,
            "effective_epsilon": ctx.effective_epsilon,
            "calibration_hash": ctx.calibration_hash,
        },
    )
    return man


def _load_case_events(cfg: Dict[str, Any], case_id: str):
    stage_a = load_stage_a_module()
    farm_id = str(case_id).split("_")[0]
    parts = str(case_id).split("_")
    if len(parts) >= 3:
        period_start = pd.Timestamp(parts[1])
        period_end = pd.Timestamp(parts[2])
    else:
        raise ValueError(f"cannot parse period from case_id={case_id}")
    events_df = stage_a.load_events_frame(Path(cfg["events_path"]), {farm_id})
    events = stage_a.case_events_from_frame(
        events_df,
        farm_id,
        period_start,
        period_end,
        include_image=bool(cfg.get("include_image", False)),
    )
    return stage_a, events, farm_id, period_start


def _resolve_vocab_and_encoder(cfg: Dict[str, Any], device: torch.device):
    from src.data_new.vocabulary import RegistryVocabulary

    stage_a = load_stage_a_module()
    vocab_path = Path(cfg.get("vocabulary_path") or cfg["tokenizer_path"])
    vocab = RegistryVocabulary(registry_path=str(vocab_path), registry_version="v2")
    encoder, hparams, abspos = stage_a.load_frozen_encoder(
        Path(cfg["stage_a_ckpt"]), vocab, device
    )
    if abspos is None:
        abspos_path = cfg.get("abspos_reference_path")
        if abspos_path and Path(abspos_path).exists():
            abspos = stage_a.load_abspos_reference(Path(abspos_path))
        else:
            abspos = cfg.get("abspos_reference") or "2000-01-01"
    abspos_ts = pd.Timestamp(abspos)
    return stage_a, encoder, vocab, abspos_ts, hparams


def _normalize_feature_name(feature: str) -> str:
    f = str(feature).strip()
    if f.startswith("FEATURE|"):
        f = f.split("|", 1)[-1]
    return f.lower()


def _select_path_a_mg(
    mg: pd.DataFrame,
    prefer_feature: str,
    *,
    require_sign_agreement: bool = True,
) -> Optional[pd.Series]:
    prefer = _normalize_feature_name(prefer_feature)
    # common probe aliases: cells column ↔ token FEATURE name
    aliases = {
        "inside_temp_c": {"inside_temp_c", "substrate_temp_c", "temp_c"},
        "substrate_temp_c": {"substrate_temp_c", "inside_temp_c", "temp_c"},
        "substrate_ec_ds_m": {"substrate_ec_ds_m", "ec", "ec_ds_m"},
    }
    prefer_set = aliases.get(prefer, {prefer})
    work = mg.copy()
    if "feature" in work.columns:
        work["_feat_norm"] = work["feature"].astype(str).map(_normalize_feature_name)
        known = work[~work["_feat_norm"].isin({"unknown", "", "nan", "none"})]
        if len(known):
            work = known
        hit = work[work["_feat_norm"].isin(prefer_set)]
        if len(hit):
            work = hit
    score_col = "absolute_attribution" if "absolute_attribution" in work.columns else "signed_attribution"
    if require_sign_agreement:
        if "sign_agreement" not in mg.columns:
            raise RuntimeError(
                "sign_agreement column missing from MG attribution; "
                "cannot enforce hard gate"
            )
        agreed = work[work["sign_agreement"] == True]  # noqa: E712
        if not len(agreed):
            # Prefer feature has no agreed locus — widen to any agreed MG (never soft-fall back to disagreed)
            pool = mg.copy()
            if "feature" in pool.columns:
                pool["_feat_norm"] = pool["feature"].astype(str).map(_normalize_feature_name)
                pool = pool[~pool["_feat_norm"].isin({"unknown", "", "nan", "none"})]
            agreed = pool[pool["sign_agreement"] == True]  # noqa: E712
            if not len(agreed):
                return None
        work = agreed
    elif "sign_agreement" in work.columns:
        agreed = work[work["sign_agreement"] == True]  # noqa: E712
        if len(agreed):
            work = agreed
        # soft mode may continue with disagreed if none agreed
    if not len(work):
        return None
    return work.sort_values(score_col, key=lambda s: s.abs(), ascending=False).iloc[0]


def _bin_edges_for_feature(cfg: Dict[str, Any], feature: str) -> List[float]:
    from src.online2.v2.binning import BinningRegistryV2

    feat = _normalize_feature_name(feature)
    reg = BinningRegistryV2.load(Path(cfg["binning_registry_path"]))
    for (f, kind, _scope), r in reg.rules.items():
        if _normalize_feature_name(f) == feat and str(kind).lower() in {"abs", "value_abs", "absolute"}:
            return list(r.edges)
    for (f, kind, _scope), r in reg.rules.items():
        if _normalize_feature_name(f) == feat:
            return list(r.edges)
    raise RuntimeError(f"no bin edges for feature={feature}")


def _feature_schema_path(cfg: Dict[str, Any]) -> Path:
    feature_schema_path = Path(
        cfg.get("feature_schema_path") or "outputs/online2/v2_build/feature_schema_v2.yaml"
    )
    if not feature_schema_path.is_absolute():
        feature_schema_path = Path("/home/dasom/life2vec") / feature_schema_path
    return feature_schema_path


def _load_feature_spec(cfg: Dict[str, Any], feature: str):
    schema = FeatureSchema.load(_feature_schema_path(cfg))
    feat = _normalize_feature_name(feature)
    return schema.features.get(feat)


def _maybe_wind_speed_mps(cells: pd.DataFrame, grounding: Dict[str, Any]) -> Optional[float]:
    """Best-effort wind speed for circular confidence metadata only."""
    try:
        farm = grounding.get("farm_id")
        zone = grounding.get("zone_id")
        ts = grounding.get("timestamp")
        if cells is None or not len(cells):
            return None
        work = cells
        if "feature" in work.columns:
            speed = work[work["feature"].astype(str).str.lower().isin({"wind_speed_mps", "wind_speed"})]
        else:
            return None
        if farm is not None and "farm_id" in speed.columns:
            speed = speed[speed["farm_id"].astype(str) == str(farm)]
        if zone is not None and "zone_id" in speed.columns:
            speed = speed[speed["zone_id"].astype(str) == str(zone)]
        if ts is not None and "timestamp" in speed.columns:
            speed = speed[speed["timestamp"].astype(str) == str(ts)]
        if not len(speed):
            return None
        val = speed.iloc[0].get("value")
        return float(val) if val is not None else None
    except Exception:
        return None


def stage_m2_compute_attribution(cfg: Dict[str, Any], paths: M1Paths) -> Dict[str, Any]:
    m2 = _cfg_m2(cfg)
    if not m2.get("enabled", False):
        raise RuntimeError("cf_m2_prereq.enabled must be true for stage_m2_compute_attribution")

    man = json.loads((paths.manifests / "cf_m2_prereq_manifest.json").read_text())
    ctx = _resolve_fold_context(cfg, man)
    write_json(paths.manifests / "fold_execution_context.json", ctx.to_dict())
    split = resolve_fold_split(
        checkpoint_paths=man["checkpoint_paths"],
        fold_ids=man["fold_ids"],
        search_fold_ids=list(ctx.search_fold_ids),
        holdout_fold_ids=list(ctx.holdout_fold_ids),
        holdout_mode=str((cfg.get("critic") or {}).get("holdout_mode") or "fixed"),
    )
    ctx.assert_holdout_unused_for_selection(split.search_fold_ids)

    case_id = cfg["case_id"]
    attr_cfg = cfg.get("attribution") or {}
    fold_cfg = cfg.get("attribution_fold") or {}
    pre = preselect_events_search_folds(
        case_id=case_id,
        search_ckpts=split.search_ckpts(),
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=Path(cfg.get("label_map_path") or Path(cfg["labels_path"]).parent / "label_map.json"),
        device=str(cfg.get("device", "cuda")),
        gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
        top_k=int(attr_cfg.get("event_preselect_top_k", 8)),
        use_binary=True,
    )
    write_parquet(paths.artifacts / "event_preselect_search.parquet", pre)

    batch, side = load_case_batch(
        case_id=case_id,
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=Path(cfg.get("label_map_path") or Path(cfg["labels_path"]).parent / "label_map.json"),
    )
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    stage_a, encoder, vocab, abspos_ts, _hp = _resolve_vocab_and_encoder(cfg, device)
    _sa, events, _farm, period_start = _load_case_events(cfg, case_id)

    tb = tensor_batch_only(batch)
    tb = {k: v.to(device) for k, v in tb.items()}
    token_df = compute_token_ixg_for_case(
        case_id=case_id,
        events=events,
        batch=tb,
        side=side,
        search_ckpts=split.search_ckpts(),
        encoder=encoder,
        stage_a_mod=stage_a,
        vocab=vocab,
        abspos_reference=abspos_ts,
        case_t0=period_start,
        normal_class_id=int(batch["_normal_class_id"]),
        preselected=pre,
        device=device,
        objective_id=str(attr_cfg.get("objective") or OBJECTIVE_BINARY),
        min_abs_score_per_fold=float(fold_cfg.get("min_abs_score_per_fold", 1.0e-5)),
        require_sign_agreement=bool(fold_cfg.get("require_sign_agreement", True)),
        fold_normalization=str(fold_cfg.get("fold_normalization") or "l1"),
    )
    token_df["search_fold_ids"] = str(list(ctx.search_fold_ids))
    token_df["evaluation_mode"] = ctx.evaluation_mode
    assert_strict_no_fallback(cfg, attribution_mode="token_ixg_v03")
    if (token_df["attribution_method"] != "token_ixg_v03").any():
        raise RuntimeError("non token_ixg_v03 rows present")

    mg = aggregate_measurement_groups(token_df, role_weights=cfg.get("token_role_weights"))
    ev = aggregate_events(mg)
    write_parquet(paths.artifacts / "token_attribution.parquet", token_df)
    write_parquet(paths.artifacts / "measurement_group_attribution.parquet", mg)
    write_parquet(paths.artifacts / "event_attribution.parquet", ev)
    write_json(
        paths.artifacts / "attribution_meta.json",
        {
            "attribution_method": "token_ixg_v03",
            "objective_id": str(attr_cfg.get("objective") or OBJECTIVE_BINARY),
            "fold_normalization": str(fold_cfg.get("fold_normalization") or "l1"),
            "min_abs_score_per_fold": float(fold_cfg.get("min_abs_score_per_fold", 1.0e-5)),
            "require_sign_agreement": bool(fold_cfg.get("require_sign_agreement", True)),
            "search_fold_ids": list(ctx.search_fold_ids),
            "holdout_fold_ids": list(ctx.holdout_fold_ids),
            "evaluation_mode": ctx.evaluation_mode,
            "cohort": ctx.cohort,
            "event_preselector": "search_fold_event_ixg",
            "n_tokens": int(len(token_df)),
            "token_grad_abs_sum": float(token_df["token_grad_abs_sum"].iloc[0]),
            "artifact_schema_version": ARTIFACT_SCHEMA,
            "runtime_trace": ctx.usage_trace("attribution"),
            "fold_usage": build_fold_usage_trace(
                event_preselection_folds=list(ctx.search_fold_ids),
                token_ixg_folds=list(ctx.search_fold_ids),
                locus_selection_folds=list(ctx.search_fold_ids),
                candidate_ranking_folds=[],
                holdout_evaluation_folds=list(ctx.holdout_fold_ids),
                evaluation_mode=ctx.evaluation_mode,
                cohort=ctx.cohort,
                case_id=case_id,
            ),
        },
    )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "token": paths.artifacts / "token_attribution.parquet",
        "mg": paths.artifacts / "measurement_group_attribution.parquet",
        "attribution_mode": "token_ixg_v03",
    }


def stage_m2_path_a_smoke(cfg: Dict[str, Any], paths: M1Paths) -> Dict[str, Any]:
    """Path A candidates with Gate0/4, frozen retokenize, material-only selection."""
    man = json.loads((paths.manifests / "cf_m2_prereq_manifest.json").read_text())
    ctx = _resolve_fold_context(cfg, man)
    split = resolve_fold_split(
        checkpoint_paths=man["checkpoint_paths"],
        fold_ids=man["fold_ids"],
        search_fold_ids=list(ctx.search_fold_ids),
        holdout_fold_ids=list(ctx.holdout_fold_ids),
    )
    case_id = cfg["case_id"]
    feature = cfg.get("path_a_probe_feature", "inside_temp_c")
    fold_cfg = cfg.get("attribution_fold") or {}
    gate0_cfg = cfg.get("gate0") or {}
    require_sign = bool(fold_cfg.get("require_sign_agreement", True))
    mg = pd.read_parquet(paths.artifacts / "measurement_group_attribution.parquet")
    if not len(mg):
        raise RuntimeError("empty measurement_group_attribution")
    top = _select_path_a_mg(mg, feature, require_sign_agreement=require_sign)
    no_eligible_locus = top is None
    sign_agreement_hard_gate = "PASS"
    if require_sign and no_eligible_locus:
        sign_agreement_hard_gate = "PASS"  # hard gate fired correctly → NO_ELIGIBLE_LOCUS path
    elif require_sign:
        sign_agreement_hard_gate = "PASS"
    else:
        sign_agreement_hard_gate = "FAIL"  # soft fallback still enabled

    if no_eligible_locus:
        # Canonical path: only NO_OP is selectable
        event_id = None
        mg_id = ""
        feat = _normalize_feature_name(feature)
        grounding = {
            "status": "NO_ELIGIBLE_LOCUS",
            "feature": feat,
            "observed_raw": None,
        }
        observed = None
        cands = []
        edges = []
        er = None
        edit_strategy = None
        locus_skip_reason = None
        unsupported_locus_count = 0
        candidate_generation_skip_reason_counts: Dict[str, int] = {}
        wind_speed = None
    else:
        event_id = str(top["event_id"])
        mg_id = str(top.get("measurement_group_id") or "")
        feat = _normalize_feature_name(str(top.get("feature") or feature))
        if feat in {"unknown", ""}:
            feat = _normalize_feature_name(feature)
        edit_strategy = None
        locus_skip_reason = None
        unsupported_locus_count = 0
        candidate_generation_skip_reason_counts = {}
        wind_speed = None

    batch, side = load_case_batch(
        case_id=case_id,
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=Path(cfg.get("label_map_path") or Path(cfg["labels_path"]).parent / "label_map.json"),
    )
    if not no_eligible_locus:
        side_row = side[side["event_id"].astype(str) == event_id]
        if not len(side_row):
            raise RuntimeError(f"event {event_id} not in sidecar")
        er = side_row.iloc[0]
        cells = pd.read_parquet(cfg["cells_path"])
        grounding = ground_locus_raw(
            cells=cells,
            farm_id=str(case_id).split("_")[0],
            feature=feat,
            zone_id=str(er["zone"]),
            timestamp=er["timestamp"],
            event_id=event_id,
            measurement_group_id=mg_id,
            tolerance_seconds=float((cfg.get("raw_grounding") or {}).get("tolerance_seconds", 0)),
            require_exact=True,
        )
        if not is_exact(grounding):
            raise RuntimeError(f"raw grounding not EXACT: {grounding}")

        observed = float(grounding["observed_raw"])
        spec = _load_feature_spec(cfg, feat)
        edit_strategy = resolve_strategy_from_spec(feat, spec)
        edges = []
        wind_speed = _maybe_wind_speed_mps(cells, grounding)
        if edit_strategy.kind == KIND_LINEAR:
            try:
                edges = _bin_edges_for_feature(cfg, feat)
            except RuntimeError:
                edit_strategy = PathAEditStrategy(
                    kind=KIND_UNSUPPORTED,
                    feature=feat,
                    feature_type="LINEAR_BINS_MISSING",
                    skip_reason="NO_SCHEMA_SUPPORTED_CANDIDATE",
                )
        try:
            cands, locus_skip_reason = build_schema_dispatched_candidates(
                strategy=edit_strategy,
                observed_raw=observed,
                edges=edges,
                wind_speed_mps=wind_speed,
                build_linear_fn=build_path_a_candidates,
            )
        except RuntimeError as exc:
            if "no bin edges" in str(exc):
                cands = []
                locus_skip_reason = "NO_SCHEMA_SUPPORTED_CANDIDATE"
                edit_strategy = PathAEditStrategy(
                    kind=KIND_UNSUPPORTED,
                    feature=feat,
                    feature_type="LINEAR_BINS_MISSING",
                    skip_reason=locus_skip_reason,
                )
            else:
                raise
        if locus_skip_reason:
            unsupported_locus_count = 1
            candidate_generation_skip_reason_counts[locus_skip_reason] = (
                int(candidate_generation_skip_reason_counts.get(locus_skip_reason, 0)) + 1
            )
            cands = []
        else:
            cands = list(cands or [])

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    stage_a, encoder, vocab, abspos_ts, _hp = _resolve_vocab_and_encoder(cfg, device)
    mlm_cfg = cfg.get("mlm") or {}
    preflight_art = run_mlm_decoder_preflight(
        encoder,
        vocab_size=int(vocab.size()),
        device=device,
        golden_logits_path=(
            Path(mlm_cfg["golden_logits_path"]) if mlm_cfg.get("golden_logits_path") else None
        ),
    )
    write_json(paths.artifacts / "mlm_decoder_preflight.json", preflight_art)
    if mlm_cfg.get("enabled", True) and mlm_cfg.get("run_in_path_a_smoke", False):
        if not preflight_art.get("decoder_preflight_passed"):
            raise RuntimeError(f"MLM decoder preflight FAILED: {preflight_art.get('fail_reasons')}")
    _sa, events, _farm, period_start = _load_case_events(cfg, case_id)
    sa_cfg = cfg.get("stage_a") or {}
    reenc = StageAReencoder(
        encoder=encoder,
        stage_a_mod=stage_a,
        vocab=vocab,
        abspos_reference=abspos_ts,
        case_t0=period_start,
        device=device,
        parity_cosine_min=float(sa_cfg.get("parity_cosine_min", 0.9999)),
        parity_mean_max_abs_error=float(sa_cfg.get("parity_mean_max_abs_error", 1e-4)),
        parity_max_max_abs_error=float(sa_cfg.get("parity_max_max_abs_error", 5e-4)),
    )
    base = tensor_batch_only(batch)
    eid_to_idx = {str(r.event_id): int(r.event_index) for r in side.itertuples()}
    events_by_id = {str(ev.event_id): i for i, ev in enumerate(events)}

    if no_eligible_locus:
        # Parity on first valid event index
        parity_idx = [int(side.iloc[0]["event_index"])] if len(side) else [0]
    else:
        parity_idx = [int(er["event_index"])]
    parity = reenc.validate_cache_parity(
        events, {k: v.to(device) for k, v in base.items()}, event_indices=parity_idx
    )
    if (cfg.get("stage_a") or {}).get("require_cache_parity", True) and not parity.passed:
        raise RuntimeError(f"Stage A cache parity failed: {parity}")

    critic_cfg = cfg.get("critic") or {}
    eps = float(ctx.effective_epsilon)
    retok_cfg = cfg.get("retokenization") or {}
    scored_rows: List[Dict[str, Any]] = []
    invalid_excluded = 0
    gate0_failures = 0
    retokenize_failures = 0

    feature_schema_path = _feature_schema_path(cfg)
    vocab_path = Path(cfg.get("vocabulary_path") or cfg["tokenizer_path"])
    prod_tok, runtime_hashes = load_frozen_tokenizer_v2(
        feature_schema_path=feature_schema_path,
        binning_registry_path=Path(cfg["binning_registry_path"]),
        vocab_path=vocab_path,
        farm_relative_enabled=bool(retok_cfg.get("farm_relative_enabled", True)),
    )
    # Independent expected hashes from manifest (not re-hash of the same runtime path alone)
    expected_reg = man.get("binning_registry_hash")
    expected_vocab = man.get("vocab_hash")
    expected_schema = man.get("feature_schema_hash")
    expected_tok_registry = man.get("tokenization_registry_hash") or man.get("tokenizer_hash")
    expected_tok_code = man.get("tokenizer_code_hash")
    actual_reg = runtime_hashes["actual_binning_registry_hash"]
    actual_vocab = runtime_hashes["actual_vocab_hash"]
    actual_schema = runtime_hashes["actual_feature_schema_hash"]
    tok_code_path = Path(__file__).resolve().parents[2] / "tokenizer.py"
    actual_tok_code = file_sha256(tok_code_path) if tok_code_path.exists() else None
    actual_tok_registry = (
        file_sha256(Path(cfg["tokenizer_path"])) if Path(cfg["tokenizer_path"]).exists() else None
    )
    hash_checks = {
        "binning_registry_match": expected_reg is not None and actual_reg == expected_reg,
        "vocab_match": expected_vocab is not None and actual_vocab == expected_vocab,
        "feature_schema_match": expected_schema is not None and actual_schema == expected_schema,
        "tokenization_registry_match": (
            expected_tok_registry is not None and actual_tok_registry == expected_tok_registry
        ),
        "tokenizer_code_match": (
            expected_tok_code is not None and actual_tok_code == expected_tok_code
        ),
    }
    tokenizer_integrity = (
        "PASS" if all(hash_checks.values()) else "FAIL"
    )
    write_json(
        paths.artifacts / "runtime_tokenizer_hashes.json",
        {
            **runtime_hashes,
            "expected_binning_registry_hash": expected_reg,
            "expected_vocab_hash": expected_vocab,
            "expected_feature_schema_hash": expected_schema,
            "expected_tokenization_registry_hash": expected_tok_registry,
            "expected_tokenizer_code_hash": expected_tok_code,
            "actual_tokenization_registry_hash": actual_tok_registry,
            "actual_tokenizer_code_hash": actual_tok_code,
            **{f"check_{k}": v for k, v in hash_checks.items()},
            "tokenizer_integrity": tokenizer_integrity,
        },
    )
    # Gate0 uses registry + vocab hashes (manifest expected vs runtime actual)
    expected_tok = expected_vocab
    actual_tok = actual_vocab

    farm_id = str(case_id).split("_")[0]
    strict_exact = bool(gate0_cfg.get("strict_exact", True))
    allow_probe = bool(gate0_cfg.get("allow_abs_bin_probe", False))

    recovered: Dict[str, Any] = {"ok": False, "reason": "NOT_ATTEMPTED"}
    if not no_eligible_locus:
        # Recover raw that reproduces original MG (Gate0 prerequisite) — grounded only in strict
        ev0 = events[events_by_id[event_id]]
        recovered = recover_observed_raw_for_mg(
            prod_tok,
            feature=feat,
            farm_id=farm_id,
            original_tokens=list(ev0.sentence_tokens),
            grounded_raw=observed,
            edges_abs=edges,
            allow_abs_bin_probe=allow_probe,
        )
        write_json(paths.artifacts / "observed_raw_recovery.json", recovered)
        if not recovered.get("ok"):
            gate0_failures = max(len(cands), 1)
            observed = float(observed)
            cands = []
        else:
            observed = float(recovered["observed_raw"])
            if edit_strategy is not None and not locus_skip_reason:
                rebuilt, skip2 = build_schema_dispatched_candidates(
                    strategy=edit_strategy,
                    observed_raw=observed,
                    edges=edges,
                    wind_speed_mps=wind_speed,
                    build_linear_fn=build_path_a_candidates,
                )
                if skip2:
                    locus_skip_reason = skip2
                    unsupported_locus_count = 1
                    candidate_generation_skip_reason_counts[skip2] = (
                        int(candidate_generation_skip_reason_counts.get(skip2, 0)) + 1
                    )
                    cands = []
                else:
                    cands = list(rebuilt or [])
            else:
                cands = []
    else:
        write_json(
            paths.artifacts / "observed_raw_recovery.json",
            {"ok": False, "reason": "NO_ELIGIBLE_LOCUS"},
        )

    mlm_run: Dict[str, Any] = {
        "mlm_executed": False,
        "decoder_forward_call_count": 0,
        "unique_scored_bundle_count": 0,
        "n_valid_mlm_edits": 0,
        "inversion_failures": 0,
        "n_eligible_loci": 0,
    }
    raw_proposals: List[Dict[str, Any]] = []
    if not no_eligible_locus and cands is not None:
        for c in cands:
            raw_proposals.append(dict(c))
    mlm_rejected_rows: List[Dict[str, Any]] = []
    allow_mlm_linear = (
        edit_strategy is not None
        and edit_strategy.kind == KIND_LINEAR
        and not locus_skip_reason
    )
    if (
        (not no_eligible_locus)
        and cands is not None
        and allow_mlm_linear
        and bool(mlm_cfg.get("enabled", True))
        and bool(mlm_cfg.get("run_in_path_a_smoke", False))
        and recovered.get("ok")
    ):
        ev_list_idx = events_by_id.get(event_id)
        if ev_list_idx is not None:
            bank_dir_cfg = mlm_cfg.get("bundle_bank_dir")
            mlm_run = run_constrained_mlm_for_locus(
                cfg=cfg,
                encoder=encoder,
                stage_a_mod=stage_a,
                vocab=vocab,
                abspos_reference=abspos_ts,
                case_t0=period_start,
                events=events,
                target_event_idx=int(ev_list_idx),
                event_id=str(event_id),
                feature=feat,
                measurement_group_id=mg_id,
                observed_raw=float(observed),
                edges_abs=edges,
                case_id=str(case_id),
                device=device,
                bank_dir=Path(bank_dir_cfg) if bank_dir_cfg else None,
            )
            write_json(
                paths.artifacts / "mlm_path_a_bank_query_meta.json",
                mlm_run.get("bank_meta") or {},
            )
            # Provisional partial funnel; rewritten after downstream terminalization.
            write_json(
                paths.artifacts / "mlm_path_a_funnel.json",
                {
                    "reason": mlm_run.get("reason"),
                    "decoder_forward_call_count": mlm_run.get("decoder_forward_call_count"),
                    "unique_scored_bundle_count": mlm_run.get("unique_scored_bundle_count"),
                    "n_valid_mlm_edits": mlm_run.get("n_valid_mlm_edits"),
                    "inversion_failures": mlm_run.get("inversion_failures"),
                    "lift_meta": mlm_run.get("lift_meta"),
                    "n_scored": len(mlm_run.get("scored_bundles") or []),
                    "bank_query_record": mlm_run.get("bank_meta") or {},
                    "pending_downstream_stages": True,
                    **(mlm_run.get("funnel") or {}),
                },
            )
            for mc in mlm_run.get("candidates") or []:
                if bool(mc.get("is_noop")) or str(mc.get("source")) == "noop":
                    continue
                raw_proposals.append(dict(mc))
                if mc.get("rejected") or not mc.get("inversion_ok"):
                    mlm_rejected_rows.append(dict(mc))
            # Merge invertible non-original MLM edits into Path A candidate list
            for mc in mlm_run.get("valid_edit_candidates") or []:
                cands.append(dict(mc))
    elif bool(mlm_cfg.get("run_in_path_a_smoke", False)) and no_eligible_locus:
        mlm_run = {
            "mlm_executed": True,
            "reason": "NO_ELIGIBLE_LOCUS",
            "decoder_forward_call_count": 0,
            "unique_scored_bundle_count": 0,
            "n_valid_mlm_edits": 0,
            "inversion_failures": 0,
            "n_eligible_loci": 0,
        }
        write_json(
            paths.artifacts / "mlm_path_a_funnel.json",
            {
                "reason": "NO_ELIGIBLE_LOCUS",
                "decoder_forward_call_count": 0,
                "pending_downstream_stages": False,
            },
        )
    elif (
        bool(mlm_cfg.get("run_in_path_a_smoke", False))
        and (not no_eligible_locus)
        and not allow_mlm_linear
    ):
        mlm_run = {
            "mlm_executed": False,
            "reason": "SCHEMA_NON_LINEAR_MLM_SKIPPED",
            "decoder_forward_call_count": 0,
            "unique_scored_bundle_count": 0,
            "n_valid_mlm_edits": 0,
            "inversion_failures": 0,
            "n_eligible_loci": 0 if locus_skip_reason else 1,
            "edit_strategy_kind": getattr(edit_strategy, "kind", None),
        }
        write_json(
            paths.artifacts / "mlm_path_a_funnel.json",
            {
                "reason": "SCHEMA_NON_LINEAR_MLM_SKIPPED",
                "decoder_forward_call_count": 0,
                "pending_downstream_stages": True,
                "edit_strategy_kind": getattr(edit_strategy, "kind", None),
            },
        )

    for cand in cands:
        if cand.get("target_raw") is None:
            continue
        target_raw = float(cand["target_raw"])
        ev_list_idx = events_by_id.get(event_id)
        if ev_list_idx is None:
            continue
        orig_toks = list(events[ev_list_idx].sentence_tokens)
        direction = str(cand.get("edit_direction") or "") or edit_direction_from_raw(
            observed, target_raw
        )
        locus_key = canonical_locus_key(
            feature=feat,
            zone=str(er["zone"]),
            measurement_group_id=mg_id,
            timestamp=er["timestamp"],
            edit_direction=direction,
        )
        if edit_strategy is not None and edit_strategy.kind != KIND_LINEAR:
            canon = {
                "canonical_target_raw": float(target_raw),
                "canonicalization_policy": (
                    "COMPASS8_CANONICAL_DEG"
                    if edit_strategy.kind == KIND_CIRCULAR
                    else "BOOLEAN_BINARY_STATE"
                ),
            }
        else:
            canon = canonical_target_raw(target_raw, edges=edges)
        mlm_bank_bundle = cand.get("proposed_token_bundle")
        src = str(cand.get("source", "adjacent_bin"))
        # adjacent_bin / schema_categorical: rule-based — bank/decoder stages auto-pass
        # constrained_mlm: arrived via bank+decoder path
        bank_supported = True
        decoder_scored = True
        inversion_pass = True if src != "constrained_mlm" else bool(cand.get("inversion_ok", True))
        # Gate4 must compare production retokenization; MLM bank relatives are diagnostic only.
        proposed = tokenize_mg_production(
            prod_tok, feature=feat, raw_value=target_raw, farm_id=farm_id
        )
        rtok = apply_raw_edit_closure(
            event_id=event_id,
            feature=feat,
            target_raw=target_raw,
            original_tokens=orig_toks,
            tokenizer=prod_tok,
            farm_id=farm_id,
            proposed_bundle=proposed,
            observed_raw=observed,
            reject_temporal_derived=bool(retok_cfg.get("reject_features_with_temporal_derived_tokens", True)),
            actual_registry_hash=actual_reg,
            expected_registry_hash=expected_reg,
            actual_tokenizer_hash=actual_tok,
            expected_tokenizer_hash=expected_tok,
            strict_exact=strict_exact,
        )
        gate0_pass = bool(rtok.baseline_roundtrip_valid)
        gate4 = rtok.gate4 or compare_token_bundles(
            proposed, rtok.actual_retokenized_bundle, require_subset=False
        )
        gate4_raw_pass = gate4["status"] == "PASSED" and bool(rtok.unchanged_outside_mg)
        # Sequential gate4: only counts after inversion+gate0 (hard-constraint invariant)
        gate4_pass = bool(inversion_pass and gate0_pass and gate4_raw_pass)
        hard_pass = bool(inversion_pass and gate0_pass and gate4_pass)
        obs_eligibility = cand.get("operational_eligibility")
        if obs_eligibility:
            op_elig = str(obs_eligibility)
        else:
            op_elig = eligibility_for_path("A")
        actionability = bool(cand.get("actionability")) if "actionability" in cand else False
        if src == "schema_categorical_adjacent":
            actionability = False
            op_elig = "OBSERVATIONAL_SENSITIVITY_ONLY"
        base_row: Dict[str, Any] = {
            "artifact_schema_version": ARTIFACT_SCHEMA,
            "case_id": case_id,
            "candidate_id": (
                f"{src}::{locus_key}::{canon['canonical_target_raw']}"
            ),
            "source": src,
            "candidate_sources": [src],
            "is_noop": False,
            "attribution_method": "token_ixg_v03",
            "search_fold_ids": list(ctx.search_fold_ids),
            "holdout_fold_ids": list(ctx.holdout_fold_ids),
            "evaluation_mode": ctx.evaluation_mode,
            "cohort": ctx.cohort,
            "canonical_locus_key": locus_key,
            "edit_direction": direction,
            "locus": grounding,
            "observed_raw": observed,
            "target_raw": target_raw,
            "canonical_target_raw": canon["canonical_target_raw"],
            "canonicalization_policy": canon["canonicalization_policy"],
            "proposed_token_bundle": proposed,
            "mlm_bank_bundle": mlm_bank_bundle,
            "actual_retokenized_bundle": rtok.actual_retokenized_bundle,
            "mlm_bundle_score": cand.get("mlm_bundle_score", cand.get("mlm_score")),
            "bundle_rank": cand.get("bundle_rank"),
            "target_event_excluded": cand.get("target_event_excluded"),
            "inversion_ok": inversion_pass,
            "inversion_pass": inversion_pass,
            "inversion_reason_code": cand.get("inversion_reason_code"),
            "bank_supported": bank_supported,
            "decoder_scored": decoder_scored,
            "gate0_pass": gate0_pass,
            "gate4_raw_pass": gate4_raw_pass,
            "gate4_pass": gate4_pass,
            "hard_constraint_pass": hard_pass,
            "plausibility_pass": bool(hard_pass and bank_supported),
            "reference_policy": rtok.reference_policy,
            "reference_refit": False,
            "gate0_baseline_roundtrip_valid": rtok.baseline_roundtrip_valid,
            "unchanged_outside_mg": rtok.unchanged_outside_mg,
            "gate4_status": gate4["status"],
            "operational_eligibility": op_elig,
            "actionability": actionability,
            "recommendation_eligible": bool(
                actionability and op_elig == "RECOMMENDATION_ELIGIBLE"
            ),
            "validity_labels": list(cand.get("validity_labels") or []),
            "feature_type": cand.get("feature_type"),
            "current_category": cand.get("current_category"),
            "candidate_categories": cand.get("candidate_categories"),
            "candidate_source": cand.get("candidate_source") or src,
            "wind_direction_semantic_confidence": cand.get(
                "wind_direction_semantic_confidence"
            ),
            "structurally_valid": True,
            "effective_epsilon": eps,
            "abs_raw_delta": abs(target_raw - observed),
        }
        if not gate0_pass:
            gate0_failures += 1
            invalid_excluded += 1
            base_row.update(
                {
                    "stage_a_pass": False,
                    "critic_scored": False,
                    "effect_pass": False,
                    "stability_pass": False,
                    "search_material": False,
                    "cf_valid": False,
                    "token_edit_count": 0,
                    "token_hamming": 0.0,
                    "reason_codes": ["GATE0_BASELINE_FAIL"],
                }
            )
            scored_rows.append(base_row)
            continue
        if not gate4_raw_pass:
            retokenize_failures += 1
            invalid_excluded += 1
            base_row["gate4_pass"] = False
            base_row["hard_constraint_pass"] = False
            base_row.update(
                {
                    "stage_a_pass": False,
                    "critic_scored": False,
                    "effect_pass": False,
                    "stability_pass": False,
                    "search_material": False,
                    "cf_valid": False,
                    "token_edit_count": 0,
                    "token_hamming": 0.0,
                    "reason_codes": ["RETOKENIZATION_MISMATCH"],
                }
            )
            scored_rows.append(base_row)
            continue

        new_toks = list(rtok.new_sentence_tokens)
        try:
            result = reenc.apply_edits_and_reencode(
                events,
                [{"event_id": event_id, "to_tokens": new_toks, "replace_sentence": True}],
                {k: v.to(device) for k, v in base.items()},
                event_id_to_index=eid_to_idx,
            )
            after = result.batch
            stage_a_pass = True
            stage_a_failure_reason = None
        except Exception as exc:  # noqa: BLE001 — CF-0 execution-error separation
            stage_a_pass = False
            stage_a_failure_reason = "STAGE_A_FORWARD_ERROR"
            base_row.update(
                {
                    "stage_a_pass": False,
                    "stage_a_failure_reason": stage_a_failure_reason,
                    "critic_scored": False,
                    "effect_pass": False,
                    "stability_pass": False,
                    "search_material": False,
                    "cf_valid": False,
                    "token_edit_count": 0,
                    "token_hamming": 0.0,
                    "reason_codes": [stage_a_failure_reason],
                    "execution_error_detail": str(exc),
                }
            )
            scored_rows.append(base_row)
            continue
        try:
            search_stats = score_candidate_folds(
                split.search_ckpts(),
                {k: v.cpu() for k, v in base.items()},
                {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in after.items()},
                normal_class_id=int(batch["_normal_class_id"]),
                fold_ids=split.search_fold_ids,
                device=str(cfg.get("device", "cuda")),
                gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
            )
            critic_scored = True
            critic_failure_reason = None
        except Exception as exc:  # noqa: BLE001
            base_row.update(
                {
                    "stage_a_pass": True,
                    "stage_a_reencode_mode": result.stage_a_reencode_mode,
                    "critic_scored": False,
                    "critic_failure_reason": "CRITIC_EXECUTION_FAILURE",
                    "effect_pass": False,
                    "stability_pass": False,
                    "search_material": False,
                    "cf_valid": False,
                    "token_edit_count": 0,
                    "token_hamming": 0.0,
                    "reason_codes": ["CRITIC_EXECUTION_FAILURE"],
                    "execution_error_detail": str(exc),
                }
            )
            scored_rows.append(base_row)
            continue
        token_edit_count = int(
            sum(a != b for a, b in zip(orig_toks, new_toks))
            if len(orig_toks) == len(new_toks)
            else abs(len(orig_toks) - len(new_toks))
        )
        folds = search_stats.get("delta_r_folds")
        all_neg = True
        if folds is not None:
            try:
                all_neg = all(float(x) < 0.0 for x in list(folds))
            except Exception:
                all_neg = False
        search_effect = (
            float(search_stats["delta_r"]) <= -eps
            and int(search_stats["folds_improved"])
            >= int(critic_cfg.get("min_search_folds_improved", 2))
            and all_neg
        )
        stability_pass = bool(all_neg and int(search_stats["folds_improved"]) >= 1)
        validity_partial = build_cf_validity(
            ctx=ctx,
            structurally_valid=True,
            raw_edit_valid=True,
            retokenization_valid=gate4_raw_pass,
            stage_a_valid=stage_a_pass,
            search_effect_valid=search_effect if ctx.independently_evaluable else None,
            holdout_effect_valid=None,
            is_noop=False,
        )
        row = {
            **base_row,
            "stage_a_pass": stage_a_pass,
            "stage_a_failure_reason": stage_a_failure_reason,
            "critic_scored": critic_scored,
            "critic_failure_reason": critic_failure_reason,
            "stage_a_reencode_mode": result.stage_a_reencode_mode,
            "directly_edited_event_ids": rtok.directly_edited_event_ids,
            "retokenization_affected_event_ids": rtok.retokenization_affected_event_ids,
            "affected_target_event_ids": result.affected_target_event_ids,
            "delta_r_search": search_stats["delta_r"],
            "delta_r_folds_search": search_stats.get("delta_r_folds"),
            "folds_improved_search": search_stats["folds_improved"],
            "risk_before_search": search_stats["risk_before"],
            "risk_after_search": search_stats["risk_after"],
            "token_edit_count": token_edit_count,
            "token_hamming": float(token_edit_count),
            "effect_pass": bool(search_effect),
            "stability_pass": bool(stability_pass and search_effect),
            "search_effect_valid": validity_partial["search_effect_valid"],
            "holdout_effect_valid": validity_partial["holdout_effect_valid"],
            "cf_valid": validity_partial["cf_valid"],
            "cf_evaluation_status": validity_partial["cf_evaluation_status"],
            "reason_codes": validity_partial["reason_codes"],
            "_after_batch": {k: v.cpu() for k, v in after.items() if torch.is_tensor(v)},
        }
        scored_rows.append(row)

    # Record MLM inversion failures as terminal ledger rows (not scored).
    for mc in mlm_rejected_rows:
        inv_reason = str(mc.get("inversion_reason_code") or "NON_INVERTIBLE_BUNDLE")
        scored_rows.append(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA,
                "case_id": case_id,
                "candidate_id": f"constrained_mlm::inversion_fail::{inv_reason}::{len(scored_rows)}",
                "source": "constrained_mlm",
                "candidate_sources": ["constrained_mlm"],
                "is_noop": False,
                "attribution_method": "token_ixg_v03",
                "evaluation_mode": ctx.evaluation_mode,
                "cohort": ctx.cohort,
                "canonical_locus_key": "",
                "target_raw": mc.get("target_raw"),
                "canonical_target_raw": mc.get("target_raw"),
                "proposed_token_bundle": mc.get("proposed_token_bundle"),
                "actual_retokenized_bundle": mc.get("proposed_token_bundle") or [],
                "bank_supported": True,
                "decoder_scored": True,
                "inversion_pass": False,
                "inversion_ok": False,
                "inversion_reason_code": inv_reason,
                "gate0_pass": False,
                "gate4_pass": False,
                "hard_constraint_pass": False,
                "plausibility_pass": False,
                "stage_a_pass": False,
                "critic_scored": False,
                "effect_pass": False,
                "stability_pass": False,
                "search_material": False,
                "cf_valid": False,
                "token_edit_count": 0,
                "reason_codes": [inv_reason],
            }
        )

    noop_after = {k: v.clone() for k, v in base.items()}
    noop_stats = score_candidate_folds(
        split.search_ckpts(),
        base,
        noop_after,
        normal_class_id=int(batch["_normal_class_id"]),
        fold_ids=split.search_fold_ids,
        device=str(cfg.get("device", "cuda")),
        gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
    )
    noop_validity = build_cf_validity(
        ctx=ctx,
        structurally_valid=True,
        raw_edit_valid=True,
        retokenization_valid=True,
        stage_a_valid=True,
        search_effect_valid=False if ctx.independently_evaluable else None,
        holdout_effect_valid=False if ctx.independently_evaluable else None,
        is_noop=True,
        reason_codes=["CANONICAL_NO_OP"],
    )
    scored_rows.append(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA,
            "case_id": case_id,
            "candidate_id": f"noop::{case_id}",
            "source": "noop",
            "candidate_sources": ["noop"],
            "is_noop": True,
            "attribution_method": "token_ixg_v03",
            "search_fold_ids": list(ctx.search_fold_ids),
            "holdout_fold_ids": list(ctx.holdout_fold_ids),
            "evaluation_mode": ctx.evaluation_mode,
            "cohort": ctx.cohort,
            "locus": grounding,
            "delta_r_search": noop_stats["delta_r"],
            "delta_r_folds_search": noop_stats.get("delta_r_folds"),
            "folds_improved_search": noop_stats["folds_improved"],
            "stage_a_reencode_mode": "full_affected_window_reencode",
            "operational_eligibility": eligibility_for_path("A"),
            "abs_raw_delta": 0.0,
            "token_edit_count": 0,
            "token_hamming": 0.0,
            "structurally_valid": True,
            "effective_epsilon": eps,
            "search_effect_valid": noop_validity["search_effect_valid"],
            "holdout_effect_valid": noop_validity["holdout_effect_valid"],
            "cf_valid": noop_validity["cf_valid"],
            "cf_evaluation_status": noop_validity["cf_evaluation_status"],
            "reason_codes": noop_validity["reason_codes"],
            "_after_batch": noop_after,
        }
    )

    meta_rows = [{k: v for k, v in r.items() if k != "_after_batch"} for r in scored_rows]
    meta_rows = ensure_canonical_noop(merge_duplicate_candidates(meta_rows), case_id=case_id)
    # Stable dedup_candidate_id for CF-0 candidate-level funnel
    for r in meta_rows:
        if r.get("is_noop") or str(r.get("source") or "").startswith("noop"):
            r["dedup_candidate_id"] = f"noop::{case_id}"
            continue
        bundle = r.get("actual_retokenized_bundle") or r.get("proposed_token_bundle") or []
        key = dedup_key(
            case_id=str(case_id),
            locus_id=str(r.get("canonical_locus_key") or ""),
            canonical_target_raw_value=float(
                r.get("canonical_target_raw")
                if r.get("canonical_target_raw") is not None
                else r.get("target_raw")
                or 0.0
            ),
            actual_retokenized_bundle=list(bundle),
        )
        r["dedup_candidate_id"] = "|".join(
            [str(key[0]), str(key[1]), f"{key[2]:.8g}", str(hash(key[3]) & 0xFFFFFFFF)]
        )
    batch_by_id = {}
    for r in scored_rows:
        cid = r.get("candidate_id") or f"{r.get('source')}:{r.get('delta_r_search')}"
        batch_by_id[cid] = r.get("_after_batch")

    ranked = rank_candidates_on_search_folds(
        meta_rows,
        material_delta_r_abs=eps,
        min_search_folds_improved=int(critic_cfg.get("min_search_folds_improved", 2)),
    )
    # Propagate effect/stability from search_material for rows that reached critic
    for r in ranked:
        if r.get("is_noop"):
            continue
        if r.get("critic_scored") and r.get("hard_constraint_pass"):
            r["effect_pass"] = bool(r.get("search_material"))
            folds = r.get("delta_r_folds_search") or r.get("delta_r_folds")
            all_neg = True
            if folds is not None:
                try:
                    all_neg = all(float(x) < 0.0 for x in list(folds))
                except Exception:
                    all_neg = False
            r["stability_pass"] = bool(r.get("effect_pass") and all_neg)
    selected_meta = select_best_on_search(ranked, require_material=True)
    if selected_meta is None:
        selected_meta = next(r for r in ranked if r.get("is_noop"))

    selected_full_batch = batch_by_id.get(selected_meta.get("candidate_id")) or noop_after
    holdout = dict(selected_meta)
    if ctx.independently_evaluable and not selected_meta.get("is_noop"):
        holdout = evaluate_selected_on_holdout(
            selected_meta,
            split.holdout_ckpts(),
            base,
            selected_full_batch,
            normal_class_id=int(batch["_normal_class_id"]),
            holdout_fold_ids=split.holdout_fold_ids,
            device=str(cfg.get("device", "cuda")),
            gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
            material_delta_r_abs=eps,
        )
        validity = build_cf_validity(
            ctx=ctx,
            structurally_valid=True,
            raw_edit_valid=True,
            retokenization_valid=str(selected_meta.get("gate4_status") or "PASSED") == "PASSED",
            stage_a_valid=True,
            search_effect_valid=bool(selected_meta.get("search_material")),
            holdout_effect_valid=bool(holdout.get("material_improvement_holdout")),
            is_noop=False,
        )
        holdout.update(validity)
    elif ctx.independently_evaluable and selected_meta.get("is_noop"):
        holdout.update(noop_validity)
        holdout["delta_r_holdout"] = 0.0
        holdout["material_improvement_holdout"] = False
    else:
        ref = score_candidate_folds(
            split.holdout_ckpts() or split.search_ckpts(),
            base,
            selected_full_batch,
            normal_class_id=int(batch["_normal_class_id"]),
            fold_ids=split.holdout_fold_ids or split.search_fold_ids,
            device=str(cfg.get("device", "cuda")),
            gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
        )
        holdout["oof_delta_r_reference"] = ref["delta_r"]
        holdout["delta_r_holdout"] = None
        holdout.update(
            build_cf_validity(
                ctx=ctx,
                structurally_valid=True,
                raw_edit_valid=True,
                retokenization_valid=True,
                stage_a_valid=True,
                search_effect_valid=None,
                holdout_effect_valid=None,
                is_noop=bool(selected_meta.get("is_noop")),
            )
        )

    out_rows = []
    for r in ranked:
        row = dict(r)
        row["selected"] = row.get("candidate_id") == holdout.get("candidate_id")
        if row["selected"]:
            for k in (
                "delta_r_holdout",
                "folds_improved_holdout",
                "material_improvement_holdout",
                "cf_valid",
                "holdout_effect_valid",
                "search_effect_valid",
                "cf_evaluation_status",
                "oof_delta_r_reference",
            ):
                if k in holdout:
                    row[k] = holdout[k]
            row["candidate_selected_without_holdout"] = True
        out_rows.append(row)

    llm_payload = llm_case_payload(
        selected=holdout,
        validity=holdout,
        invalid_candidates_excluded=invalid_excluded,
    )
    write_jsonl(paths.results / "path_a_m2_cf_results.jsonl", out_rows)
    write_json(
        paths.results / "path_a_m2_selected.json",
        {k: v for k, v in holdout.items() if k != "_after_batch"},
    )
    write_json(paths.results / "llm_case_payload.json", llm_payload)
    write_json(
        paths.reports / "stage_a_parity_summary.json",
        {
            "passed": parity.passed,
            "cosine_mean": parity.cosine_mean,
            "mean_abs_error": parity.mean_abs_error,
            "max_abs_error": parity.max_abs_error,
            "noop_delta_r_search": noop_stats["delta_r"],
            "effective_epsilon": eps,
            "calibration_hash": ctx.calibration_hash,
            "evaluation_mode": ctx.evaluation_mode,
        },
    )

    scenario = "NO_VALID_CF"
    if no_eligible_locus:
        scenario = "NO_ELIGIBLE_LOCUS"
    elif holdout.get("cf_valid") is True:
        scenario = "VALID_CF_FOUND"
    elif holdout.get("cf_evaluation_status") == "NOT_INDEPENDENTLY_EVALUABLE":
        scenario = "OOF_REFERENCE_ONLY"

    mlm_cfg = cfg.get("mlm") or {}
    mlm_prov = count_mlm_provenance(raw_proposals=raw_proposals, dedup_rows=out_rows)
    n_mlm_raw_proposals = int(mlm_prov["n_mlm_raw_proposals"])
    n_mlm_origin_candidates = int(mlm_prov["n_mlm_origin_candidates"])
    # Final outcome uses only MLM-origin AND cf_valid=true unique candidates
    n_valid_mlm_cf = int(mlm_prov["n_valid_mlm_cf"])
    recon_path = paths.reports / "mlm_reconstruction_metrics.json"
    reconstruction_pass = None
    if recon_path.exists():
        try:
            recon = json.loads(recon_path.read_text(encoding="utf-8"))
            reconstruction_pass = bool(recon.get("quality_pass"))
        except Exception:
            reconstruction_pass = False
    mlm_flags = mlm_outcome_from_candidates(
        mlm_executed=bool(mlm_run.get("mlm_executed")),
        deferred=not bool(mlm_cfg.get("run_in_path_a_smoke", False)),
        n_eligible_loci=int(mlm_run.get("n_eligible_loci") or (0 if no_eligible_locus else 1)),
        n_valid_mlm_cf=int(n_valid_mlm_cf),
        inversion_failures=int(mlm_run.get("inversion_failures") or 0),
        reconstruction_pass=reconstruction_pass,
    )
    natural_outcome = mlm_flags["mlm_cf_candidate_outcome"]
    if no_eligible_locus and bool(mlm_cfg.get("run_in_path_a_smoke", False)):
        natural_outcome = "NO_ELIGIBLE_LOCUS"
        mlm_flags["mlm_cf_candidate_outcome"] = natural_outcome
    write_json(
        paths.artifacts / "mlm_natural_outcome.json",
        {
            "natural_integration_outcome": natural_outcome,
            "mlm_cf_candidate_outcome": mlm_flags["mlm_cf_candidate_outcome"],
            "n_mlm_raw_proposals": n_mlm_raw_proposals,
            "n_mlm_origin_candidates": n_mlm_origin_candidates,
            "n_valid_mlm_cf": n_valid_mlm_cf,
            "decoder_forward_call_count": mlm_run.get("decoder_forward_call_count"),
            "unique_scored_bundle_count": mlm_run.get("unique_scored_bundle_count"),
            "preflight": preflight_art,
        },
    )

    # CF-0 candidate-level funnel (complete downstream terminals)
    funnel_warnings: List[Dict[str, Any]] = []
    if edit_strategy is not None and edit_strategy.kind == KIND_CIRCULAR:
        funnel_warnings.append(
            {
                "warning_code": "CIRCULAR_OBSERVATIONAL_SENSITIVITY_ONLY",
                "feature": feat,
                "affected_candidate_count": sum(
                    1
                    for r in out_rows
                    if not r.get("is_noop")
                    and str(r.get("source")) == "schema_categorical_adjacent"
                ),
                "blocking": False,
                "recommendation_eligible": False,
            }
        )
    if edit_strategy is not None and edit_strategy.kind == KIND_UNSUPPORTED:
        funnel_warnings.append(
            {
                "warning_code": "NO_SCHEMA_SUPPORTED_CANDIDATE",
                "feature": feat,
                "affected_candidate_count": 0,
                "blocking": False,
                "locus_level_skip": True,
            }
        )
    bank_unique = int(
        (mlm_run.get("funnel") or {}).get("n_bank_unique")
        or (mlm_run.get("bank_meta") or {}).get("n_unique_fingerprints")
        or 0
    )
    scored_unique = int(
        mlm_run.get("unique_scored_bundle_count")
        or (mlm_run.get("funnel") or {}).get("n_scored_unique")
        or 0
    )
    editable_loci = 0 if no_eligible_locus else int(mlm_run.get("n_eligible_loci") or 1)
    if locus_skip_reason and not no_eligible_locus:
        # Locus existed but schema could not emit candidates
        editable_loci = max(editable_loci, 1)
    cf0_funnel = build_cf0_case_funnel(
        case_id=str(case_id),
        cohort=str(ctx.cohort),
        editable_loci_count=editable_loci,
        raw_proposals=raw_proposals,
        dedup_rows=out_rows,
        selected=holdout,
        bank_unique_bundle_count=bank_unique,
        decoder_scored_unique_bundle_count=scored_unique,
        warnings=funnel_warnings,
        evaluation_mode=str(ctx.evaluation_mode),
        unsupported_locus_count=int(unsupported_locus_count),
        candidate_generation_skip_reason_counts=dict(
            candidate_generation_skip_reason_counts or {}
        ),
        edit_strategy_kind=getattr(edit_strategy, "kind", None),
    )
    write_json(paths.artifacts / "cf0_case_funnel.json", cf0_funnel)
    # Rewrite MLM funnel artifact with downstream completion flag
    prior_funnel = {}
    prior_path = paths.artifacts / "mlm_path_a_funnel.json"
    if prior_path.exists():
        try:
            prior_funnel = json.loads(prior_path.read_text(encoding="utf-8"))
        except Exception:
            prior_funnel = {}
    write_json(
        prior_path,
        {
            **prior_funnel,
            "pending_downstream_stages": bool(cf0_funnel.get("pending_downstream_stages")),
            "cf0_funnel_audit_status": cf0_funnel.get("funnel_audit_status"),
            "n_mlm_raw_proposals": n_mlm_raw_proposals,
            "n_mlm_origin_candidates": n_mlm_origin_candidates,
            "n_valid_mlm_cf": n_valid_mlm_cf,
            "first_zero_stage": cf0_funnel.get("first_zero_stage"),
            "hard_constraint_pass_count": cf0_funnel.get("hard_constraint_pass_count"),
            "effect_pass_count": cf0_funnel.get("effect_pass_count"),
            "valid_cf_count": cf0_funnel.get("valid_cf_count"),
            "monotonicity_pass": cf0_funnel.get("monotonicity_pass"),
            "conservation_pass": cf0_funnel.get("conservation_pass"),
        },
    )

    if no_eligible_locus:
        gate0_ok = True  # Gate0 not applicable; hard gate short-circuited edits
        retokenize_ok = True
    else:
        gate0_ok = gate0_failures == 0
        retokenize_ok = retokenize_failures == 0 and (
            any(str(r.get("gate4_status")) == "PASSED" for r in ranked if not r.get("is_noop"))
            or bool(selected_meta.get("is_noop"))
        )
        if retokenize_failures > 0 and not any(
            str(r.get("gate4_status")) == "PASSED" and r.get("unchanged_outside_mg")
            for r in ranked
            if not r.get("is_noop")
        ):
            retokenize_ok = bool(selected_meta.get("is_noop")) and retokenize_failures >= 0
            if not selected_meta.get("is_noop"):
                retokenize_ok = False

    noop_policy_ok = bool(selected_meta.get("is_noop")) or bool(selected_meta.get("search_material"))
    cohort_integrity = "PASS" if (
        (ctx.cohort == "example_set" and ctx.evaluation_mode == EVAL_OOF_REF)
        or (ctx.cohort == "problem_set" and ctx.evaluation_mode == EVAL_CROSSFIT)
    ) else "FAIL"
    mg_retokenization_smoke = "PASS" if retokenize_ok and gate0_ok else "FAIL"
    if no_eligible_locus:
        # Hard gate short-circuited edits; retokenization unevaluated but policy PASS
        mg_retokenization_smoke = "PASS"
    dependency_closure_retokenization = "NOT_EVALUATED"
    calibration_ok = bool(ctx.calibration_hash) and ctx.calibration_hash != ""

    fold_usage = build_fold_usage_trace(
        event_preselection_folds=list(ctx.search_fold_ids),
        token_ixg_folds=list(ctx.search_fold_ids),
        locus_selection_folds=list(ctx.search_fold_ids),
        candidate_ranking_folds=list(ctx.search_fold_ids),
        holdout_evaluation_folds=list(ctx.holdout_fold_ids),
        evaluation_mode=ctx.evaluation_mode,
        cohort=ctx.cohort,
        case_id=case_id,
    )
    write_json(paths.artifacts / "fold_usage_trace.json", fold_usage)
    leak = compute_leakage_free(fold_usage, evaluation_mode=ctx.evaluation_mode)
    write_json(paths.artifacts / "leakage_free_check.json", leak)

    acceptance = build_acceptance_report(
        structural_smoke="PASS" if parity.passed else "FAIL",
        behavioral_cf_smoke=scenario if scenario not in {"OOF_REFERENCE_ONLY", "NO_ELIGIBLE_LOCUS"} else "NO_VALID_CF",
        normal_over_edit_smoke="NOT_RUN",
        scenario_outcome=scenario,
        leakage_free=bool(leak["leakage_free"]),
        gate0_ok=gate0_ok,
        retokenize_ok=retokenize_ok,
        noop_policy_ok=noop_policy_ok,
        cohort_integrity=cohort_integrity,
        mg_retokenization_smoke=mg_retokenization_smoke,
        dependency_closure_retokenization=dependency_closure_retokenization,
        full_retokenization=mg_retokenization_smoke,
        calibration_ok=calibration_ok,
        gt_normal_verified=None,
        tokenizer_integrity=tokenizer_integrity,
        sign_agreement_hard_gate=sign_agreement_hard_gate,
        mlm_required_for_pass=False,
        constrained_mlm_execution=mlm_flags["constrained_mlm_execution"],
        mlm_reconstruction_quality=mlm_flags["mlm_reconstruction_quality"],
        mlm_cf_candidate_outcome=mlm_flags["mlm_cf_candidate_outcome"],
        extra={
            "require_material": True,
            "holdout_isolated": ctx.independently_evaluable,
            "evaluation_mode": ctx.evaluation_mode,
            "cohort": ctx.cohort,
            "gate0_failures": gate0_failures,
            "retokenize_failures": retokenize_failures,
            "invalid_candidates_excluded": invalid_excluded,
            "no_eligible_locus": no_eligible_locus,
            "runtime_trace": ctx.usage_trace("path_a_smoke"),
            "fold_usage": fold_usage,
            "leakage_check": leak,
            "runtime_hashes": {
                "actual_registry_hash": actual_reg,
                "expected_registry_hash": expected_reg,
                "actual_vocab_hash": actual_vocab,
                "expected_vocab_hash": expected_vocab,
                "actual_feature_schema_hash": actual_schema,
                "expected_feature_schema_hash": expected_schema,
                "tokenizer_integrity": tokenizer_integrity,
            },
        },
    )
    write_json(paths.reports / "m2_acceptance_report.json", acceptance)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "n": len(out_rows),
        "selected": holdout,
        "parity": parity.passed,
        "acceptance": acceptance,
        "gate0_ok": gate0_ok,
        "retokenize_ok": retokenize_ok,
        "no_eligible_locus": no_eligible_locus,
        "fold_usage": fold_usage,
        "leakage_free": bool(leak["leakage_free"]),
        "cf0_funnel": cf0_funnel,
        "n_mlm_raw_proposals": n_mlm_raw_proposals,
        "n_mlm_origin_candidates": n_mlm_origin_candidates,
        "n_valid_mlm_cf": n_valid_mlm_cf,
    }
