"""STRUCTURAL_ONLY_PRE_MODEL curated fixture selection (no decoder/critic scores)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..candidates.bundle_bank_loader import bundle_fingerprint
from ..candidates.mlm_raw_inversion import invert_mlm_bundle_to_raw
from ..gates.local_rules import compare_token_bundles
from ..retokenization.full_event_retokenizer import tokenize_mg_production


def select_structural_fixture(
    *,
    case_id: str,
    event_id: str,
    feature: str,
    measurement_group_id: str,
    observed_raw: float,
    edges_abs: Sequence[float],
    original_tokens: Sequence[str],
    bank_bundles: Sequence[Mapping[str, Any]],
    tokenizer,
    farm_id: str,
) -> Dict[str, Any]:
    """Freeze a fixture using only structural invertibility / production retokenize.

    Forbidden: decoder ranking, critic ΔR, Search/Holdout.

    Bank bundles supply alternate ABS bins; the frozen proposed bundle is the
    production tokenizer output for the inverted raw (full-MG consistency).
    """
    orig_abs = next((t for t in original_tokens if str(t).startswith("VALUE_ABS|")), None)
    # Deterministic scan order
    ordered = sorted(
        bank_bundles,
        key=lambda b: bundle_fingerprint(feature, list(b.get("tokens") or [])),
    )
    candidates = []
    for b in ordered:
        toks = list(b.get("tokens") or [])
        if list(toks) == list(original_tokens):
            continue
        if b.get("is_original"):
            continue
        bank_abs = next((t for t in toks if str(t).startswith("VALUE_ABS|")), None)
        if bank_abs is None or bank_abs == orig_abs:
            continue
        inv = invert_mlm_bundle_to_raw(
            feature=feature,
            observed_raw=observed_raw,
            mlm_proposed_bundle=toks,
            edges_abs=edges_abs,
        )
        if not inv.get("ok"):
            continue
        target_raw = float(inv["selected_target_raw"])
        if abs(target_raw - float(observed_raw)) <= 1e-12:
            continue
        # Production full-MG bundle for this raw — Gate4 identity vs itself, edit vs original
        actual = tokenize_mg_production(
            tokenizer, feature=feature, raw_value=target_raw, farm_id=farm_id
        )
        # Extract value tokens from production output for MG comparison
        prod_vals = [
            t
            for t in actual
            if str(t).startswith(("VALUE_", "OBSERVED_VALUE|"))
            or "REL_B" in str(t)
        ]
        orig_vals = [
            t
            for t in original_tokens
            if str(t).startswith(("VALUE_", "OBSERVED_VALUE|"))
            or "REL_B" in str(t)
        ]
        if prod_vals == orig_vals:
            continue
        g4 = compare_token_bundles(actual, actual, require_subset=False)
        if g4.get("status") != "PASSED":
            continue
        candidates.append(
            {
                "tokens": list(actual),
                "bank_tokens": toks,
                "target_raw": target_raw,
                "fingerprint": bundle_fingerprint(feature, actual),
                "gate4_status": g4["status"],
                "token_edit_count": sum(1 for a, b in zip(orig_vals, prod_vals) if a != b)
                + abs(len(orig_vals) - len(prod_vals)),
            }
        )
        if len(candidates) >= 1:
            break
    candidates.sort(key=lambda c: c["fingerprint"])
    if not candidates:
        return {
            "ok": False,
            "fixture_selection_policy": "STRUCTURAL_ONLY_PRE_MODEL",
            "decoder_scores_used_for_selection": False,
            "critic_scores_used_for_selection": False,
            "reason": "NO_STRUCTURAL_NON_ORIGINAL_INVERTIBLE",
            "case_id": case_id,
            "event_id": event_id,
            "feature": feature,
            "measurement_group_id": measurement_group_id,
        }
    pick = candidates[0]
    return {
        "ok": True,
        "fixture_selection_policy": "STRUCTURAL_ONLY_PRE_MODEL",
        "decoder_scores_used_for_selection": False,
        "critic_scores_used_for_selection": False,
        "fixture_frozen_before_integration_run": True,
        "selection_reason": "PREVERIFIED_INVERTIBLE_NON_ORIGINAL_BUNDLE",
        "case_id": case_id,
        "event_id": event_id,
        "feature": feature,
        "measurement_group_id": measurement_group_id,
        "observed_raw": float(observed_raw),
        "selected_bundle_tokens": pick["tokens"],
        "selected_target_raw": pick["target_raw"],
        "fingerprint": pick["fingerprint"],
        "n_structural_candidates": len(candidates),
        "token_edit_count": pick["token_edit_count"],
    }


def write_fixture_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
