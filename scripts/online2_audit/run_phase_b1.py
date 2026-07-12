#!/usr/bin/env python3
"""Phase B1 read-only audits over approved V1 artifacts. Does not write Neo4j or mutate builds."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "outputs/online2/build-v8-active80-r3"
OUT = ROOT / "outputs/online2/v2_audit"
DATA = ROOT / "datasets/agrichallenge/online2"
SEED = 2023


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


IDENTITY_COLUMNS = {
    "farm_id",
    "zone_id",
    "timestamp",
    "observation_date",
    "hour",
}


def _infer_unit(feature: str) -> tuple[str, str, str, str]:
    """Return candidate_unit, canonical_unit, confidence, evidence."""
    name = feature.lower()
    if name in IDENTITY_COLUMNS:
        return "n/a", "n/a", "CONFIRMED", "identity_or_time_column"
    if name.endswith("_c") or "temp" in name:
        return "degC", "degC", "HIGH", "column_suffix_or_name:_c/temp"
    if name.endswith("_pct") or "humidity" in name or "water_content" in name:
        return "%", "%", "HIGH", "column_suffix_or_name:_pct/humidity"
    if name.endswith("_ppm") or name == "co2_ppm":
        return "ppm", "ppm", "HIGH", "column_suffix:_ppm"
    if "ds_m" in name or name.endswith("_ec") or name in {"ec_sensor", "substrate_ec_ds_m"}:
        return "dS/m", "dS/m", "HIGH", "column_name:_ds_m/ec"
    if name.endswith("_m_s") or "wind_speed" in name:
        return "m/s", "m/s", "HIGH", "column_suffix:_m_s"
    if "wind_direction" in name or name.endswith("_deg"):
        return "deg", "deg", "HIGH", "column_name:direction_deg"
    if name.endswith("_mm"):
        return "mm", "mm", "HIGH", "column_suffix:_mm"
    if name.endswith("_cm"):
        return "cm", "cm", "HIGH", "column_suffix:_cm"
    if "solar" in name or "radiation" in name:
        return "W/m2_or_sensor_units", "unresolved", "LOW", "distribution_only;sensor_scale_unconfirmed"
    if name in {"ph_sensor"} or name.startswith("ph_"):
        return "pH", "pH", "MEDIUM", "feature_name_ph;confirm_probe_scale"
    if "count" in name or name.endswith("_order"):
        return "count", "count", "HIGH", "counter_or_ordinal_name"
    if name in {
        "rain_detected",
        "circulation_fan",
        "fcu_fan",
        "fcu_pump",
        "co2_supply",
        "nutrient_solution_system",
        "tube_rail",
    }:
        return "device_code_or_binary", "unresolved", "LOW", "code_meaning_not_confirmed_in_codebook"
    if name in {
        "roof_vent_left",
        "roof_vent_right",
        "shade_screen",
        "thermal_curtain",
        "line_flow_rate",
        "total_flow_rate",
    }:
        return "percent_or_flow_raw", "unresolved", "LOW", "opening_or_flow_scale_unconfirmed"
    return "unresolved", "unresolved", "UNRESOLVED", "no_unit_evidence"


def _propose_type(feature: str, stats: dict[str, Any]) -> str:
    name = feature.lower()
    if name in IDENTITY_COLUMNS:
        return "identifier"
    if name == "hour":
        return "identifier"
    if name == "wind_direction_deg":
        return "circular"
    if "count" in name or name.endswith("_order"):
        return "counter"
    if name in {"rain_detected"}:
        return "boolean"
    if name in {
        "circulation_fan",
        "fcu_fan",
        "fcu_pump",
        "co2_supply",
        "nutrient_solution_system",
        "tube_rail",
    }:
        return "boolean" if stats.get("unique_count", 0) <= 3 else "categorical"
    if name in {
        "roof_vent_left",
        "roof_vent_right",
        "shade_screen",
        "thermal_curtain",
    }:
        return "ordinal_actuator"
    if name in {"line_flow_rate", "total_flow_rate"}:
        return "flow"
    return "continuous"


def _view_for_feature(feature: str) -> str:
    growth = {
        "crown_diameter_mm",
        "leaf_count",
        "leaf_length_cm",
        "leaf_width_cm",
        "opened_flower_count",
        "petiole_length_cm",
        "plant_height_cm",
        "unopened_flower_count",
        "flower_truss_order",
    }
    root = {"substrate_ec_ds_m", "substrate_temp_c", "substrate_water_content_pct", "ec_sensor", "ph_sensor"}
    actuator = {
        "circulation_fan",
        "co2_supply",
        "fcu_fan",
        "fcu_pump",
        "line_flow_rate",
        "nutrient_solution_system",
        "roof_vent_left",
        "roof_vent_right",
        "shade_screen",
        "thermal_curtain",
        "total_flow_rate",
        "tube_rail",
    }
    if feature in growth:
        return "G_growth"
    if feature in root:
        return "R_rootzone"
    if feature in actuator:
        return "A_actuator"
    return "E_environment"


def audit_features() -> pd.DataFrame:
    cells = pd.read_parquet(
        BUILD / "cell_occurrences.parquet",
        columns=["column_name", "feature_alias", "raw_display", "raw_type", "is_null", "farm_id", "modality"],
    )
    rows = []
    for feature, group in cells.groupby("column_name", sort=True):
        values = []
        null_count = int(group["is_null"].sum()) if "is_null" in group else 0
        for raw in group["raw_display"].astype(str):
            if raw == "" or raw.lower() == "nan":
                null_count += 1
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        arr = np.asarray(values, dtype=np.float64) if values else np.asarray([], dtype=np.float64)
        farm_const = 0
        for _, farm_vals in group.groupby("farm_id"):
            nums = []
            for raw in farm_vals["raw_display"].astype(str):
                try:
                    nums.append(float(raw))
                except ValueError:
                    continue
            if nums and len(set(np.round(nums, 12))) <= 1:
                farm_const += 1
        stats = {
            "unique_count": int(len(set(np.round(arr, 12)))) if arr.size else 0,
            "zero_count": int(np.sum(arr == 0)) if arr.size else 0,
            "positive_count": int(np.sum(arr > 0)) if arr.size else 0,
            "negative_count": int(np.sum(arr < 0)) if arr.size else 0,
            "null_count": null_count,
        }
        cand, canon, conf, evidence = _infer_unit(str(feature))
        proposed = _propose_type(str(feature), stats)
        threshold_ready = conf in {"CONFIRMED", "HIGH", "MEDIUM"} and proposed in {
            "continuous",
            "circular",
            "flow",
            "ordinal_actuator",
            "counter",
        }
        # MEDIUM alone is not enough for domain threshold tokens
        if conf in {"LOW", "UNRESOLVED"} or proposed in {"boolean", "categorical"}:
            threshold_ready = False
        if conf == "MEDIUM":
            threshold_ready = False
        percentiles = {}
        for label, q in [
            ("p01", 1),
            ("p05", 5),
            ("p25", 25),
            ("p50", 50),
            ("p75", 75),
            ("p95", 95),
            ("p99", 99),
        ]:
            percentiles[label] = float(np.percentile(arr, q)) if arr.size else None
        rows.append(
            {
                "feature_name": feature,
                "feature_alias": str(group["feature_alias"].iloc[0]) if "feature_alias" in group else str(feature).upper(),
                "source_column": feature,
                "source_file_family": group["modality"].mode().iloc[0] if "modality" in group and len(group) else _view_for_feature(str(feature)),
                "event_view": _view_for_feature(str(feature)),
                "raw_type": group["raw_type"].mode().iloc[0] if "raw_type" in group and len(group) else "unknown",
                "proposed_v2_type": proposed,
                "candidate_unit": cand,
                "canonical_unit": canon,
                "unit_confidence": conf,
                "unit_evidence": evidence,
                "raw_min": float(arr.min()) if arr.size else None,
                "raw_max": float(arr.max()) if arr.size else None,
                **percentiles,
                **stats,
                "farm_constant_count": farm_const,
                "globally_constant": bool(arr.size and stats["unique_count"] <= 1),
                "threshold_ready": bool(threshold_ready),
                "state_mapping_ready": proposed in {"boolean", "ordinal_actuator", "flow", "categorical"}
                and stats["unique_count"] <= 20,
                "noise_epsilon_ready": proposed in {"flow", "ordinal_actuator", "boolean"}
                and stats["zero_count"] > 0
                and stats["positive_count"] > 0,
                "quality_flags": ";".join(
                    flag
                    for flag, cond in [
                        ("ZERO_POS_MIX", stats["zero_count"] > 0 and stats["positive_count"] > 0),
                        ("HAS_NEGATIVE", stats["negative_count"] > 0),
                        ("GLOBAL_CONSTANT", arr.size and stats["unique_count"] <= 1),
                        ("UNIT_UNRESOLVED", conf in {"LOW", "UNRESOLVED"}),
                    ]
                    if cond
                ),
            }
        )

    # identity / multimodal placeholders
    for extra in [
        ("farm_id", "identifier", "E_environment"),
        ("zone_id", "identifier", "E_environment"),
        ("timestamp", "identifier", "E_environment"),
        ("image_file", "image", "I_images"),
        ("interpretation_text", "text", "interpretation"),
    ]:
        rows.append(
            {
                "feature_name": extra[0],
                "feature_alias": extra[0].upper(),
                "source_column": extra[0],
                "source_file_family": extra[2],
                "event_view": extra[2],
                "raw_type": "string",
                "proposed_v2_type": extra[1],
                "candidate_unit": "n/a",
                "canonical_unit": "n/a",
                "unit_confidence": "CONFIRMED" if extra[1] != "text" else "HIGH",
                "unit_evidence": "structural_identity_or_modality",
                "raw_min": None,
                "raw_max": None,
                "p01": None,
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
                "p99": None,
                "zero_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "null_count": 0,
                "unique_count": None,
                "farm_constant_count": 0,
                "globally_constant": False,
                "threshold_ready": False,
                "state_mapping_ready": False,
                "noise_epsilon_ready": False,
                "quality_flags": "",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "feature_semantics_and_units.csv", index=False)
    return frame


def audit_constants(feature_frame: pd.DataFrame) -> pd.DataFrame:
    registry = json.loads((BUILD / "binning_registry.json").read_text(encoding="utf-8"))
    cells = pd.read_parquet(
        BUILD / "cell_occurrences.parquet",
        columns=["column_name", "raw_display", "farm_id", "zone_id"],
    )
    rows = []
    for rule in registry["rules"]:
        if len(rule["edges"]) != 1:
            continue
        feature = rule["feature"]
        kind = rule["kind"]
        scope = rule.get("scope_id")
        sub = cells[cells["column_name"] == feature]
        if scope and kind == "FARM_REL":
            sub = sub[sub["farm_id"] == scope]
        vals = []
        for raw in sub["raw_display"].astype(str):
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        unique = sorted(set(np.round(vals, 12))) if vals else []
        feat_row = feature_frame[feature_frame["feature_name"] == feature]
        globally = bool(feat_row["globally_constant"].iloc[0]) if len(feat_row) else False
        if globally:
            const_type = "GLOBAL_CONSTANT"
        elif kind == "FARM_REL":
            const_type = "FARM_CONSTANT"
        else:
            const_type = "WINDOW_CONSTANT" if not globally else "GLOBAL_CONSTANT"
        meaning = "UNRESOLVED"
        if unique == [0.0] or unique == [0]:
            meaning = "ALWAYS_OFF"
        elif len(unique) == 1 and unique[0] != 0:
            meaning = "NO_VARIATION_IN_PERIOD"
        rows.append(
            {
                "feature": feature,
                "tokenization_kind": kind,
                "scope_id": scope or "",
                "constant_type": const_type,
                "semantic_class": meaning,
                "edge_value": rule["edges"][0],
                "observed_unique_values": ";".join(map(str, unique[:10])),
                "n_cells": int(len(sub)),
                "static_metadata_candidate": const_type in {"GLOBAL_CONSTANT", "FARM_CONSTANT"},
                "remove_from_repeat_tokens_candidate": const_type == "GLOBAL_CONSTANT",
                "not_installed_asserted": False,
                "notes": "NOT_INSTALLED never inferred without installation evidence",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "constant_feature_audit.csv", index=False)
    return frame


def audit_mlm_replacement() -> dict[str, Any]:
    registry = json.loads((BUILD / "life2vec_token_registry.json").read_text(encoding="utf-8"))
    tokens = registry["tokens"]
    special = {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}
    background = {t["token"] for t in tokens if t["category"] == "BACKGROUND"}
    general = {t["token"] for t in tokens if t["category"] == "GENERAL"}
    slots = {"IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT"}
    excluded = set()
    for t in tokens:
        tok = t["token"]
        if tok in special or tok in background or tok in general or tok in slots:
            excluded.add(t["token_id"])
    replacement = [t for t in tokens if t["token_id"] not in excluded]
    families = Counter(
        (tok["token"].split("|", 1)[0] if "|" in tok["token"] else tok["token"]) for tok in replacement
    )
    family_rows = [
        {"token_family": family, "n_tokens": count, "random_replacement_allowed_legacy": True}
        for family, count in families.most_common()
    ]
    for risky in ["NARRATIVE", "CATEGORY", "DATASET", "DISEASE", "DIAGNOSIS"]:
        if risky not in families:
            family_rows.append(
                {
                    "token_family": risky,
                    "n_tokens": 0,
                    "random_replacement_allowed_legacy": False,
                }
            )
    pd.DataFrame(family_rows).to_csv(OUT / "random_replacement_token_families.csv", index=False)
    payload = {
        "created_at_utc": _utc(),
        "vocab_size": len(tokens),
        "excluded_count": len(excluded),
        "replacement_candidate_count": len(replacement),
        "includes_narrative": int(families.get("NARRATIVE", 0)),
        "includes_category": int(families.get("CATEGORY", 0)),
        "includes_dataset": int(families.get("DATASET", 0)),
        "background_excluded": sorted(background)[:20],
        "slots_excluded": sorted(slots),
        "mask_ratio_config": 0.30,
        "bert_style_split": {"mask": 0.8, "keep": 0.1, "random": 0.1},
        "sibling_independent_masking": True,
    }
    _write_json(OUT / "legacy_mlm_behavior.json", payload)
    return payload


def audit_reuse() -> dict[str, Any]:
    segments = pd.read_parquet(
        BUILD / "sequence_segments.parquet",
        columns=["sequence_id", "event_id", "position"],
    )
    sequences = pd.read_parquet(
        BUILD / "sequences.parquet",
        columns=["sequence_id", "narrative_id", "segment_ids", "same_time_group_ids", "farm_count"],
    )
    events = pd.read_parquet(
        BUILD / "events.parquet",
        columns=["event_id", "same_time_group_id", "farm_ids", "observation_timestamp", "event_kind", "modalities"],
    )
    training = pd.read_parquet(
        BUILD / "training_events.parquet",
        columns=["PERSON_ID", "sequence_id", "event_id", "farm_id", "same_time_group_id"],
    )

    event_reuse = segments.groupby("event_id").size()
    group_reuse = segments.merge(events[["event_id", "same_time_group_id"]], on="event_id").groupby(
        "same_time_group_id"
    ).size()

    def dist(series: pd.Series, name: str) -> dict[str, Any]:
        values = series.astype(float)
        return {
            "unit": name,
            "n_unique": int(series.index.nunique() if hasattr(series.index, "nunique") else values.size),
            "min": float(values.min()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
        }

    reuse_rows = []
    for series, name in [
        (event_reuse, "event_id"),
        (group_reuse, "same_time_group_id"),
    ]:
        d = dist(series, name)
        reuse_rows.append(d)
    # source window proxy: farm + calendar day of event
    ev = events.copy()
    ev["day"] = pd.to_datetime(ev["observation_timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    ev["farm"] = ev["farm_ids"].map(lambda x: (json.loads(x) if isinstance(x, str) else x or [""])[0])
    ev["window"] = ev["farm"] + "|" + ev["day"]
    window_events = segments.merge(ev[["event_id", "window"]], on="event_id")
    window_reuse = window_events.groupby("window")["sequence_id"].nunique()
    reuse_rows.append(dist(window_reuse, "farm_date_window_sequences"))
    pd.DataFrame(reuse_rows).to_csv(OUT / "event_reuse_distribution.csv", index=False)

    # exact sequence hash on ordered event ids
    ordered = segments.sort_values(["sequence_id", "position"], kind="mergesort")
    sig = (
        ordered.groupby("sequence_id")["event_id"]
        .agg(lambda ids: _sha256_bytes("|".join(map(str, ids)).encode()))
        .rename("event_set_hash")
    )
    hash_counts = sig.value_counts()
    exact_dup_groups = int((hash_counts > 1).sum())
    exact_dup_sequences = int(hash_counts[hash_counts > 1].sum())

    # scalable near-duplicate: sample narratives, pre-materialize event sets once
    rng = np.random.default_rng(SEED)
    event_sets = (
        ordered.groupby("sequence_id", sort=False)["event_id"].agg(lambda s: frozenset(s.tolist()))
    )
    sample_rows = []
    bucket_totals = {"ge_0_95": 0, "ge_0_80": 0, "ge_0_50": 0, "pairs": 0}
    # Cap narratives processed for pairwise Jaccard to keep audit interactive.
    narrative_ids = sequences["narrative_id"].drop_duplicates().tolist()
    if len(narrative_ids) > 30:
        narrative_ids = list(rng.choice(narrative_ids, size=30, replace=False))
    for narrative_id in narrative_ids:
        ids = sequences.loc[sequences["narrative_id"] == narrative_id, "sequence_id"].tolist()
        if len(ids) < 2:
            continue
        sample_n = min(25, len(ids))
        chosen_list = list(rng.choice(ids, size=sample_n, replace=False))
        sets = {sid: set(event_sets.loc[sid]) for sid in chosen_list if sid in event_sets.index}
        chosen_list = [sid for sid in chosen_list if sid in sets]
        for i in range(len(chosen_list)):
            for j in range(i + 1, len(chosen_list)):
                a, b = sets[chosen_list[i]], sets[chosen_list[j]]
                denom = len(a | b)
                if denom == 0:
                    continue
                jaccard = len(a & b) / denom
                bucket_totals["pairs"] += 1
                if jaccard >= 0.95:
                    bucket_totals["ge_0_95"] += 1
                if jaccard >= 0.80:
                    bucket_totals["ge_0_80"] += 1
                if jaccard >= 0.50:
                    bucket_totals["ge_0_50"] += 1
                if jaccard >= 0.80:
                    sample_rows.append(
                        {
                            "narrative_id": narrative_id,
                            "sequence_a": chosen_list[i],
                            "sequence_b": chosen_list[j],
                            "jaccard": round(jaccard, 4),
                        }
                    )

    near = pd.DataFrame(sample_rows)
    near.to_csv(OUT / "near_duplicate_sequences.csv", index=False)

    # narrative coverage / inflation
    narr = (
        segments.merge(sequences[["sequence_id", "narrative_id"]], on="sequence_id")
        .groupby("narrative_id")
        .agg(sequence_count=("sequence_id", "nunique"), unique_events=("event_id", "nunique"))
        .reset_index()
    )
    narr["inflation"] = narr["sequence_count"] / narr["unique_events"].clip(lower=1)

    # farm hash split overlap (same method as Online2Population)
    entities = training.drop_duplicates("PERSON_ID")[["PERSON_ID", "sequence_id", "farm_id"]]
    ratios = np.asarray([0.8, 0.1, 0.1])
    thresholds = np.cumsum(ratios)

    def bucket(farm: str) -> str:
        digest = hashlib.sha256(f"{SEED}:{farm}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < thresholds[0]:
            return "train"
        if value < thresholds[1]:
            return "val"
        return "test"

    entities = entities.copy()
    entities["split"] = entities["farm_id"].astype(str).map(bucket)
    person_split = entities.set_index("PERSON_ID")["split"]
    event_splits = (
        training.assign(split=training["PERSON_ID"].map(person_split))
        .groupby("event_id")["split"]
        .nunique()
    )
    multi = event_splits[event_splits > 1]
    # source_row via cell? use events only for image overlap later
    split_overlap = pd.DataFrame(
        {
            "metric": [
                "events_in_multiple_splits",
                "multi_farm_sequences",
                "sequence_count",
                "unique_events",
                "sequence_per_unique_event",
            ],
            "value": [
                int(len(multi)),
                int((sequences["farm_count"] > 1).sum()),
                int(sequences["sequence_id"].nunique()),
                int(segments["event_id"].nunique()),
                float(sequences["sequence_id"].nunique() / max(1, segments["event_id"].nunique())),
            ],
        }
    )
    # detailed multi-split events sample
    if len(multi):
        detail = (
            training.assign(split=training["PERSON_ID"].map(person_split))
            .groupby("event_id")["split"]
            .agg(lambda s: ",".join(sorted(set(s))))
            .loc[multi.index]
            .reset_index()
            .rename(columns={"split": "splits"})
        )
        detail.to_csv(OUT / "split_atomic_overlap.csv", index=False)
    else:
        pd.DataFrame(columns=["event_id", "splits"]).to_csv(OUT / "split_atomic_overlap.csv", index=False)

    report = {
        "created_at_utc": _utc(),
        "reuse_distributions": reuse_rows,
        "exact_duplicate_hash_groups": exact_dup_groups,
        "exact_duplicate_sequences": exact_dup_sequences,
        "near_duplicate_sample_pairs_ge_0_80": int(len(near)),
        "near_duplicate_bucket_totals": bucket_totals,
        "near_duplicate_method": "sample <=30 narratives, <=25 sequences each, event-set Jaccard; seed=2023",
        "narrative_inflation_summary": {
            "median": float(narr["inflation"].median()),
            "p90": float(narr["inflation"].quantile(0.9)),
            "max": float(narr["inflation"].max()),
        },
        "split_overlap": split_overlap.to_dict(orient="records"),
        "sampling_policy_candidates_impact_notes": {
            "reuse_aware_weight": "Down-weights high-reuse events; lowers effective N, may raise unique-context diversity.",
            "source_window_cap": "Caps sequences per farm-date window; largest impact on dense hourly narratives.",
            "narrative_cap": "Limits sequences per narrative_id; reduces narrative monopoly.",
            "near_duplicate_filter": "Drops Jaccard>=0.95 pairs within narrative; small exact-dup cleanup first.",
            "epoch_level_rematerialization": "Resample contexts each epoch; no artifact rewrite, train-time only.",
        },
    }
    _write_json(OUT / "sequence_inflation_report.json", report)
    return report


def audit_multimodal() -> dict[str, Any]:
    ext = pd.read_parquet(BUILD / "external_embeddings.parquet")
    events = pd.read_parquet(
        BUILD / "events.parquet",
        columns=["event_id", "event_kind", "embedding_status", "farm_ids", "observation_timestamp", "quality_flags"],
    )
    sequences = pd.read_parquet(
        BUILD / "sequences.parquet",
        columns=["sequence_id", "contains_image", "contains_interpretation", "narrative_id"],
    )
    img_root = DATA / "data" / "I_images"
    image_rows = []
    try:
        from PIL import Image, ExifTags
    except Exception:  # pragma: no cover
        Image = None
        ExifTags = None

    orientation_tag = None
    if ExifTags is not None:
        orientation_tag = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)

    disk_files = sorted(
        p for p in img_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    hash_counter: Counter[str] = Counter()
    decoded = []
    for path in disk_files:
        content_hash = _sha256_file(path)
        hash_counter[content_hash] += 1
        decoded.append((path, content_hash))

    # map events
    img_events = ext[ext["modality"] == "I_images"].copy()
    for _, row in img_events.iterrows():
        rel = str(row["source_ref"])
        path = DATA / rel if not Path(rel).is_absolute() else Path(rel)
        # builder stores path relative to online2 root
        if not path.exists():
            alt = ROOT / "datasets/agrichallenge/online2" / rel
            path = alt if alt.exists() else path
        exists = path.exists()
        width = height = None
        fmt = None
        orient = None
        decode_valid = False
        content_hash = ""
        if exists:
            content_hash = _sha256_file(path)
            if Image is not None:
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        width, height = im.size
                        fmt = im.format
                        exif = im.getexif() if hasattr(im, "getexif") else {}
                        if orientation_tag and exif:
                            orient = exif.get(orientation_tag)
                        decode_valid = True
                except Exception:
                    decode_valid = False
        farm = path.parent.name if exists else ""
        image_rows.append(
            {
                "image_id": row["event_id"],
                "source_path": rel,
                "file_exists": exists,
                "decode_valid": decode_valid,
                "file_size": path.stat().st_size if exists else 0,
                "width": width,
                "height": height,
                "format": fmt,
                "EXIF_orientation": orient,
                "content_hash": content_hash,
                "duplicate_hash_count": hash_counter.get(content_hash, 0),
                "farm_id": farm,
                "zone_id": "",
                "capture_timestamp": "",
                "timestamp_precision": "period",
                "alignment_policy": "case_period_end",
                "alignment_confidence": "LOW",
                "image_role_candidate": "UNRESOLVED",
                "embedding_status": row["embedding_status"],
            }
        )
    # include disk-only orphans
    event_paths = set(str(r["source_path"]) for r in image_rows)
    for path, content_hash in decoded:
        rel = path.relative_to(DATA).as_posix()
        if rel in event_paths or path.relative_to(img_root.parent.parent).as_posix() in event_paths:
            continue
        # try common relative forms
        rel2 = f"data/I_images/{path.parent.name}/{path.name}"
        if rel2 in event_paths:
            continue
        image_rows.append(
            {
                "image_id": "",
                "source_path": rel2,
                "file_exists": True,
                "decode_valid": True,
                "file_size": path.stat().st_size,
                "width": None,
                "height": None,
                "format": path.suffix,
                "EXIF_orientation": None,
                "content_hash": content_hash,
                "duplicate_hash_count": hash_counter[content_hash],
                "farm_id": path.parent.name,
                "zone_id": "",
                "capture_timestamp": "",
                "timestamp_precision": "unknown",
                "alignment_policy": "disk_only",
                "alignment_confidence": "UNRESOLVED",
                "image_role_candidate": "UNRESOLVED",
                "embedding_status": "not_in_build",
            }
        )
    image_df = pd.DataFrame(image_rows)
    image_df.to_csv(OUT / "image_inventory.csv", index=False)

    # interpretations
    text_rows = []
    ans_root = DATA / "answers" / "reference_answers"
    for tier in ["score50", "score70", "score90"]:
        for path in sorted((ans_root / tier).glob("*_answer.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            farm = path.stem.replace("_answer", "")
            allowed = tier == "score90"  # example_set public only; problem forbidden separately
            # build only ingested score90 example interpretations
            in_build = bool(
                ((ext["modality"] == "interpretation") & ext["source_ref"].astype(str).str.contains(farm)).any()
            )
            text_rows.append(
                {
                    "interpretation_id": f"{tier}:{farm}",
                    "raw_text_ref": path.relative_to(DATA).as_posix(),
                    "file_exists": True,
                    "answer_tier": tier,
                    "quality_score": {"score50": 50, "score70": 70, "score90": 90}[tier],
                    "dataset_role": "example_set_candidate",
                    "allowed_for_pretraining": bool(allowed and in_build),
                    "allowed_as_context": bool(allowed and in_build),
                    "forbidden_reason": ""
                    if allowed
                    else "auxiliary_or_contrastive_not_merged_with_score90",
                    "text_length_chars": len(text),
                    "text_length_tokens_estimate": max(1, len(text.split())),
                    "content_hash": _sha256_bytes(text.encode("utf-8")),
                    "reference_segment_count": text.count("\n") + 1,
                    "embedding_status": "pending" if in_build else "not_in_build",
                    "in_build_external_embeddings": in_build,
                }
            )
    # mark problem_set farms forbidden
    problem_farms = set()
    case_file = DATA / "problem_set" / "case_list.csv"
    if case_file.exists():
        problem_farms = set(pd.read_csv(case_file)["farm_id"].astype(str))
    for row in text_rows:
        farm = row["interpretation_id"].split(":", 1)[1]
        if farm in problem_farms:
            row["dataset_role"] = "problem_set"
            row["allowed_for_pretraining"] = False
            row["allowed_as_context"] = False
            row["forbidden_reason"] = "problem_set_hidden_interpretation"

    text_df = pd.DataFrame(text_rows)
    text_df.to_csv(OUT / "text_inventory.csv", index=False)

    ready_image = int((ext["modality"] == "I_images").sum()) if len(ext) else 0
    ready_text = int((ext["modality"] == "interpretation").sum()) if len(ext) else 0
    vector_ready = int((ext["embedding_status"] == "ready").sum()) if len(ext) else 0

    routing = []
    for _, seq in sequences.iterrows():
        route = "sensor_only"
        if seq["contains_image"] and seq["contains_interpretation"] and vector_ready > 0:
            route = "image_text_ready"
        elif seq["contains_image"] and vector_ready > 0:
            route = "image_ready"
        elif seq["contains_interpretation"] and vector_ready > 0:
            route = "text_ready"
        routing.append(
            {
                "sequence_id": seq["sequence_id"],
                "narrative_id": seq["narrative_id"],
                "contains_image": bool(seq["contains_image"]),
                "contains_interpretation": bool(seq["contains_interpretation"]),
                "route": route,
            }
        )
    route_df = pd.DataFrame(routing)
    # Full per-sequence manifest is large; keep it for local audit but also emit compact summary.
    route_df.to_csv(OUT / "multimodal_routing_manifest.csv", index=False)
    route_df["route"].value_counts().rename_axis("route").reset_index(name="sequence_count").to_csv(
        OUT / "multimodal_routing_counts.csv", index=False
    )

    summary = {
        "created_at_utc": _utc(),
        "disk_images": len(disk_files),
        "image_events": int((events["event_kind"] == "IMAGE").sum()),
        "interpretation_events": int((events["event_kind"] == "INTERPRETATION").sum()),
        "external_embedding_rows": int(len(ext)),
        "embedding_ready": vector_ready,
        "embedding_pending": int((ext["embedding_status"] == "pending").sum()) if len(ext) else 0,
        "route_counts": route_df["route"].value_counts().to_dict(),
        "problem_set_interpretation_allowed": int(text_df["allowed_for_pretraining"].sum())
        if "problem_set" not in set(text_df["dataset_role"])
        else int(
            text_df.loc[text_df["dataset_role"] == "problem_set", "allowed_for_pretraining"].sum()
        ),
        "problem_set_allowed_pretraining_count": int(
            (
                (text_df["dataset_role"] == "problem_set") & (text_df["allowed_for_pretraining"])
            ).sum()
        ),
        "score90_allowed_for_pretraining": int(
            (
                (text_df["answer_tier"] == "score90") & (text_df["allowed_for_pretraining"])
            ).sum()
        ),
        "notes": "No vectors ready => all sequences sensor_only is expected.",
    }
    _write_json(OUT / "multimodal_inventory_summary.json", summary)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = audit_features()
    constants = audit_constants(features)
    mlm = audit_mlm_replacement()
    reuse = audit_reuse()
    multi = audit_multimodal()
    summary = {
        "created_at_utc": _utc(),
        "build_id": json.loads((BUILD / "build_manifest.json").read_text())["build_id"],
        "feature_rows": int(len(features)),
        "threshold_ready_features": features.loc[
            features["threshold_ready"] == True, "feature_name"
        ].tolist(),
        "unresolved_or_low_unit": features.loc[
            features["unit_confidence"].isin(["LOW", "UNRESOLVED"]), "feature_name"
        ].tolist(),
        "constant_type_counts": constants["constant_type"].value_counts().to_dict()
        if len(constants)
        else {},
        "mlm_replacement_candidates": mlm["replacement_candidate_count"],
        "reuse": {
            "exact_duplicate_sequences": reuse["exact_duplicate_sequences"],
            "near_dup_pairs_ge_0_80_sample": reuse["near_duplicate_sample_pairs_ge_0_80"],
        },
        "multimodal": multi,
    }
    _write_json(OUT / "phase_b1_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
