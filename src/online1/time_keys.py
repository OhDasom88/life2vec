from __future__ import annotations

import re

import pandas as pd

TIME_RE = re.compile(r"DAT(\d+)\s+(\d{2}):(\d{2})")


def parse_dat_time(series: pd.Series) -> pd.DataFrame:
    parsed = series.astype(str).str.extract(TIME_RE)
    if parsed.isna().any().any():
        bad = series[parsed.isna().any(axis=1)].head(5).tolist()
        raise ValueError(f"시간 파싱 실패 예시: {bad}")
    out = pd.DataFrame(
        {
            "dat": parsed[0].astype(int),
            "hour": parsed[1].astype(int),
            "minute": parsed[2].astype(int),
        }
    )
    out["minutes_of_day"] = out["hour"] * 60 + out["minute"]
    out["t_ord"] = out["dat"] * 1440 + out["minutes_of_day"]
    return out


def format_dat_time(dat: int, minutes_of_day: int) -> str:
    h, m = divmod(int(minutes_of_day), 60)
    return f"DAT{int(dat)} {h:02d}:{m:02d}"


def assign_zone_ids(df: pd.DataFrame, cfg: dict, *, split: str) -> pd.DataFrame:
    """DAT blocks → zone_id.

    train: 6일×4구역 (마지막 구역 잔여 흡수)
    test : 3일×4구역
    """
    zcfg = cfg.get("zone", {})
    out = df.copy()
    if not zcfg.get("enabled", True):
        out["zone_id"] = 0
        return out
    days = int(
        zcfg["train_days_per_zone"] if split == "train" else zcfg["test_days_per_zone"]
    )
    max_z = int(zcfg.get("max_zones", 4))
    dat_min = int(out["dat"].min())
    z = ((out["dat"].astype(int) - dat_min) // days).clip(upper=max_z - 1)
    out["zone_id"] = z.astype(int)
    return out


def weather_day_index(dat: int, zone_id: int, days_per_zone: int) -> int:
    """Relative day within a zone block (shared weather stage index)."""
    return int(dat) - int(zone_id) * int(days_per_zone)
