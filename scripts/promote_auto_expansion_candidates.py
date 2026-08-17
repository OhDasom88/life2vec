#!/usr/bin/env python3
"""검증된 자동생성 후보(narrative_auto_expansion_report.json)를 실제
online2_narrative_catalog.csv에 반영한다.

`generate_online2_narrative_catalog_auto_expansion.py`의 `build_auto_specs()`는
이미 `spec()` 호환 dict를 만들지만(occurrence() dry-run 검증만 하고 CSV는 안 씀),
두 가지 문제가 있어 그대로 승격하면 안 된다:

1. `req`가 항상 "farm_id;zone_id;timestamp;..."로 고정돼 있다 -- growth_any/
   crossfarm_growth/mixed_growth처럼 생육(G_growth, observation_date 기준)
   컬럼을 쓰는 서사도 timestamp로 잘못 표시된다. occurrence()는 이 필드를 안 써서
   dry-run 검증은 통과하지만, 실제 build_rows()는 이 req를 보고
   event_definition/sequence_definition 분기(mixed_growth 여부, "observation_date"
   in req 여부)를 결정하므로 고쳐야 한다.
2. `data_sources`가 테이블명이 아니라 원본 컬럼명으로 채워져 있다(문서화 버그,
   기능에는 영향 없지만 카탈로그를 읽는 사람에게 오해를 준다).

이 스크립트는 위 두 가지를 고친 뒤, 기존 88개 SPECS에 동일한 occurrence() 검증을
다시 통과한 후보만 이어붙이고 catalog 모듈의 build_rows()/validate()/write_csv()를
그대로 재사용해 online2_narrative_catalog.csv를 다시 쓴다(덮어씀).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_online2_narrative_catalog as catalog_mod  # noqa: E402
from generate_online2_narrative_catalog_auto_expansion import build_auto_specs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.online2.materializers import MaterializerRegistry  # noqa: E402

COLUMN_TABLE = {
    "outside_temp_c": "E_environment", "outside_humidity_pct": "E_environment",
    "outside_solar_radiation": "E_environment", "wind_direction_deg": "E_environment",
    "wind_speed_m_s": "E_environment", "rain_detected": "E_environment",
    "inside_temp_c": "E_environment", "inside_humidity_pct": "E_environment",
    "co2_ppm": "E_environment",
    "substrate_temp_c": "R_rootzone", "substrate_water_content_pct": "R_rootzone",
    "substrate_ec_ds_m": "R_rootzone",
    "fcu_fan": "A_actuator", "fcu_pump": "A_actuator", "circulation_fan": "A_actuator",
    "co2_supply": "A_actuator", "tube_rail": "A_actuator", "roof_vent_left": "A_actuator",
    "roof_vent_right": "A_actuator", "shade_screen": "A_actuator", "thermal_curtain": "A_actuator",
    "ec_sensor": "A_actuator", "ph_sensor": "A_actuator", "total_flow_rate": "A_actuator",
    "line_flow_rate": "A_actuator", "nutrient_solution_system": "A_actuator",
    "plant_height_cm": "G_growth", "leaf_length_cm": "G_growth", "leaf_width_cm": "G_growth",
    "petiole_length_cm": "G_growth", "leaf_count": "G_growth", "crown_diameter_mm": "G_growth",
    "flower_truss_order": "G_growth", "opened_flower_count": "G_growth",
    "unopened_flower_count": "G_growth",
}


def _sources_for(req_cols: list[str]) -> str:
    tables = []
    for c in req_cols:
        t = COLUMN_TABLE.get(c)
        if t and t not in tables:
            tables.append(t)
    return ";".join(tables) if tables else "E_environment"


def _fix_and_filter(specs: list[dict], hourly, growth, actual_columns) -> list[dict]:
    promoted = []
    n_growth_fixed = 0
    for s in specs:
        matcher = s["matcher"]
        prefix = matcher.split(":")[0]
        req = s["req"].split(";")

        if prefix in {"growth_any", "crossfarm_growth"}:
            req = [c if c != "timestamp" else "observation_date" for c in req]
            n_growth_fixed += 1
        elif prefix == "mixed_growth":
            base, growth_col, actuator_col = req[:3], req[3], req[4]
            req = base + [actuator_col, "observation_date", growth_col]
            n_growth_fixed += 1
        s["req"] = ";".join(req)
        s["sources"] = _sources_for([c for c in req if c not in catalog_mod.KEYS])

        if not MaterializerRegistry.supports(matcher):
            continue
        if any(c not in actual_columns for c in req):
            continue
        try:
            count, unique_count, farms = catalog_mod.occurrence(s, hourly, growth)
        except Exception:
            continue
        min_unique = (
            1 if matcher in {"crossfarm_growth", "crosszone_hourly", "crossfarm_hourly"}
            or matcher == "all_hourly" or matcher.startswith("window_summary:")
            else 3
        )
        if count < 1 or unique_count < min_unique:
            continue
        promoted.append(s)
    print(f"  growth req 수정: {n_growth_fixed}건")
    return promoted


def main() -> None:
    hourly, growth, actual_columns = catalog_mod.load_data()
    print(f"hourly={len(hourly)} rows, growth={len(growth)} rows")

    raw_auto_specs = build_auto_specs(hourly, growth)
    print(f"자동생성 원본 후보: {len(raw_auto_specs)}건")

    promoted = _fix_and_filter(raw_auto_specs, hourly, growth, actual_columns)
    print(f"실측 재검증 통과(승격 대상): {len(promoted)}건")

    existing_ids = {s["narrative_id"] for s in catalog_mod.SPECS}
    collisions = existing_ids & {s["narrative_id"] for s in promoted}
    assert not collisions, f"narrative_id 충돌: {collisions}"

    catalog_mod.SPECS = catalog_mod.SPECS + promoted
    rows = catalog_mod.build_rows(hourly, growth, actual_columns)
    catalog_mod.validate(rows, actual_columns)
    active_ids = {row["narrative_id"] for row in rows}

    dispositions = catalog_mod.disposition_rows(active_ids)
    assert len(dispositions) == 200
    assert all(
        not r["canonical_narrative_id"] or r["canonical_narrative_id"] in active_ids
        for r in dispositions
    )

    catalog_mod.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    catalog_mod.write_csv(catalog_mod.OUT, catalog_mod.HEADERS, rows)
    disposition_headers = [
        "source_catalog", "source_narrative_id", "disposition",
        "canonical_narrative_id", "reason",
    ]
    catalog_mod.write_csv(catalog_mod.DISPOSITION_OUT, disposition_headers, dispositions)
    generated_registries = catalog_mod.registry_rows(active_ids)
    for filename, registry in generated_registries.items():
        catalog_mod.write_csv(catalog_mod.REGISTRY_DIR / filename, catalog_mod.REGISTRY_HEADERS, registry)

    import pandas as pd
    parsed = pd.read_csv(catalog_mod.OUT, encoding="utf-8-sig")
    assert list(parsed.columns) == catalog_mod.HEADERS and len(parsed) == len(rows)
    assert catalog_mod.OUT.read_bytes().startswith(b"\xef\xbb\xbf")

    print(f"wrote={catalog_mod.OUT}")
    print(f"total narratives={len(rows)} (기존 {len(existing_ids)} + 승격 {len(promoted)})")
    from collections import Counter
    print("by_category=" + ";".join(f"{k}:{v}" for k, v in Counter(r["category"] for r in rows).items()))
    print("validation=PASS")


if __name__ == "__main__":
    main()
