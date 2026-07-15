"""Map consensus saliency → Structured Evidence JSON (finetune v0.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


def _period_from_row(r: pd.Series) -> Optional[str]:
    if r.get("period") not in (None, "", float("nan")) and not (isinstance(r.get("period"), float) and pd.isna(r.get("period"))):
        return str(r.get("period"))
    ts = r.get("timestamp")
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    return str(ts)


def _zones_from_row(r: pd.Series) -> Optional[List[str]]:
    z = r.get("zones")
    if isinstance(z, list):
        return [str(x) for x in z]
    if z is not None and not (isinstance(z, float) and pd.isna(z)) and str(z) not in {"", "nan", "None"}:
        return [str(z)]
    zone = r.get("zone")
    if zone is None or (isinstance(zone, float) and pd.isna(zone)):
        return None
    return [str(int(float(zone))) if str(zone).replace(".", "", 1).isdigit() else str(zone)]


def _modality_from_row(r: pd.Series) -> Optional[List[str]]:
    m = r.get("modalities")
    if isinstance(m, list):
        return [str(x) for x in m]
    view = r.get("view")
    if view is None or (isinstance(view, float) and pd.isna(view)):
        return None
    return [str(view)]


def evidence_rows(
    df: pd.DataFrame,
    *,
    direction: str,
    factor_fn=None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return out
    for _, r in df.iterrows():
        eid = str(r.get("event_id", ""))
        if factor_fn is not None:
            factor = factor_fn(r)
        else:
            factor = str(r.get("factor") or eid or "UNKNOWN")
        out.append(
            {
                "factor": factor,
                "direction": direction,
                "consensus_models": int(r.get("positive_agreement_count", 0) or 0),
                "median_saliency": float(r.get("median_saliency", 0.0) or 0.0),
                "period": _period_from_row(r),
                "zones": _zones_from_row(r),
                "event_ids": [eid] if eid else [],
                "modalities": _modality_from_row(r),
            }
        )
    return out


def build_structured_evidence(
    *,
    case_id: str,
    diagnosis: str,
    ensemble_probability: float,
    model_agreement: int,
    diagnosis_support: Optional[pd.DataFrame] = None,
    state_events: Optional[pd.DataFrame] = None,
    cause_events: Optional[pd.DataFrame] = None,
    image_events: Optional[pd.DataFrame] = None,
    disagreement: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Schema: diagnosis_support + state_evidence + cause_evidence + image_evidence."""

    def _factor(r: pd.Series) -> str:
        if r.get("factor"):
            return str(r["factor"])
        view = r.get("view", "?")
        zone = r.get("zone", "?")
        eid = r.get("event_id", "")
        return f"{view}|zone{zone}|{eid}"

    payload: Dict[str, Any] = {
        "case_id": case_id,
        "diagnosis": diagnosis,
        "ensemble_probability": float(ensemble_probability),
        "model_agreement": int(model_agreement),
        "diagnosis_support": evidence_rows(
            diagnosis_support if diagnosis_support is not None else pd.DataFrame(),
            direction="supports_diagnosis",
            factor_fn=_factor,
        ),
        "state_evidence": evidence_rows(
            state_events if state_events is not None else pd.DataFrame(),
            direction="supports_diagnosis",
            factor_fn=_factor,
        ),
        "cause_evidence": evidence_rows(
            cause_events if cause_events is not None else pd.DataFrame(),
            direction="possible_cause",
            factor_fn=_factor,
        ),
        "image_evidence": [],
        "disagreement": disagreement or [],
    }
    if image_events is not None and len(image_events):
        for _, r in image_events.iterrows():
            payload["image_evidence"].append(
                {
                    "image_id": str(r.get("image_id", "")),
                    "image_slot": int(r.get("image_slot", -1)),
                    "consensus_models": int(r.get("positive_agreement_count", 0) or 0),
                    "median_saliency": float(r.get("median_saliency", 0.0) or 0.0),
                    "role": r.get("role"),
                }
            )
    return payload


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
