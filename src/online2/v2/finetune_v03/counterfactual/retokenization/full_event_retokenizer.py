"""Production TokenizerV2-based MG retokenization (frozen references, no refit).

Replaces manual ABS string assembly. Gate0/Gate4 compare full MG bundles
(ABS / GLOBAL_REL / FARM_REL / feature-specific / QUALITY) from TokenizerV2.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..gates.local_rules import compare_token_bundles
from ..io_utils import file_sha256


REFERENCE_POLICY = "FROZEN_ORIGINAL_ARTIFACT"


@dataclass
class RetokenizationResult:
    baseline_roundtrip_valid: bool
    baseline_bundle_match: str
    registry_hash_match: bool
    tokenizer_hash_match: bool
    reference_policy: str = REFERENCE_POLICY
    reference_refit: bool = False
    global_reference_hash: Optional[str] = None
    farm_reference_hash: Optional[str] = None
    directly_edited_event_ids: List[str] = field(default_factory=list)
    retokenization_affected_event_ids: List[str] = field(default_factory=list)
    stage_a_affected_target_event_ids: List[str] = field(default_factory=list)
    proposed_bundle: List[str] = field(default_factory=list)
    actual_retokenized_bundle: List[str] = field(default_factory=list)
    new_sentence_tokens: List[str] = field(default_factory=list)
    mg_span: Optional[Tuple[int, int]] = None
    unchanged_outside_mg: bool = False
    gate4: Optional[Dict[str, Any]] = None
    reason_codes: List[str] = field(default_factory=list)
    actual_registry_hash: Optional[str] = None
    expected_registry_hash: Optional[str] = None
    actual_tokenizer_hash: Optional[str] = None
    expected_tokenizer_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_roundtrip_valid": self.baseline_roundtrip_valid,
            "baseline_bundle_match": self.baseline_bundle_match,
            "registry_hash_match": self.registry_hash_match,
            "tokenizer_hash_match": self.tokenizer_hash_match,
            "reference_policy": self.reference_policy,
            "reference_refit": self.reference_refit,
            "global_reference_hash": self.global_reference_hash,
            "farm_reference_hash": self.farm_reference_hash,
            "directly_edited_event_ids": list(self.directly_edited_event_ids),
            "retokenization_affected_event_ids": list(self.retokenization_affected_event_ids),
            "stage_a_affected_target_event_ids": list(self.stage_a_affected_target_event_ids),
            "proposed_bundle": list(self.proposed_bundle),
            "actual_retokenized_bundle": list(self.actual_retokenized_bundle),
            "new_sentence_tokens": list(self.new_sentence_tokens),
            "mg_span": list(self.mg_span) if self.mg_span else None,
            "unchanged_outside_mg": self.unchanged_outside_mg,
            "gate4": self.gate4,
            "reason_codes": list(self.reason_codes),
            "actual_registry_hash": self.actual_registry_hash,
            "expected_registry_hash": self.expected_registry_hash,
            "actual_tokenizer_hash": self.actual_tokenizer_hash,
            "expected_tokenizer_hash": self.expected_tokenizer_hash,
        }


def _norm_feature(feature: str) -> str:
    f = str(feature).strip()
    if f.startswith("FEATURE|"):
        f = f.split("|", 1)[-1]
    return f.lower()


def _feature_token(feature: str) -> str:
    return f"FEATURE|{_norm_feature(feature).upper()}"


def find_mg_span(tokens: Sequence[str], feature: str) -> Optional[Tuple[int, int]]:
    """Return [start, end) span of FEATURE|{feat} ... until next MEAS_SEP/FEATURE/EVENT boundary."""
    want = _feature_token(feature)
    toks = [str(t) for t in tokens]
    start = None
    for i, t in enumerate(toks):
        if t == want or t.upper() == want:
            start = i
            break
    if start is None:
        # case-insensitive FEATURE match
        for i, t in enumerate(toks):
            if t.startswith("FEATURE|") and t.split("|", 1)[-1].lower() == _norm_feature(feature):
                start = i
                break
    if start is None:
        return None
    end = len(toks)
    for j in range(start + 1, len(toks)):
        tj = toks[j]
        if tj in {"[MEAS_SEP]", "[EVENT_SEP]"} or tj.startswith("FEATURE|") or tj.startswith("VIEW|"):
            end = j
            break
    return (start, end)


def extract_mg_tokens(tokens: Sequence[str], feature: str) -> List[str]:
    span = find_mg_span(tokens, feature)
    if span is None:
        return []
    return [str(t) for t in tokens[span[0] : span[1]]]


def splice_mg_tokens(
    original_tokens: Sequence[str],
    feature: str,
    new_mg_tokens: Sequence[str],
) -> Tuple[List[str], Tuple[int, int], bool]:
    """Replace only the target MG span; verify outside tokens unchanged."""
    toks = [str(t) for t in original_tokens]
    span = find_mg_span(toks, feature)
    if span is None:
        raise ValueError(f"FEATURE|{_norm_feature(feature)} not found in event sentence")
    s, e = span
    new_sentence = toks[:s] + [str(t) for t in new_mg_tokens] + toks[e:]
    # outside unchanged check on original indices (before length change)
    outside_ok = toks[:s] == new_sentence[:s] and toks[e:] == new_sentence[s + len(new_mg_tokens) :]
    return new_sentence, (s, s + len(new_mg_tokens)), outside_ok


def load_frozen_tokenizer_v2(
    *,
    feature_schema_path: Path,
    binning_registry_path: Path,
    vocab_path: Path,
    farm_relative_enabled: bool = True,
) -> Tuple[Any, Dict[str, str]]:
    """Load production TokenizerV2; return (tokenizer, runtime_hashes)."""
    from src.online2.v2.binning import BinningRegistryV2
    from src.online2.v2.feature_schema import FeatureSchema
    from src.online2.v2.tokenizer import TokenizerV2
    from src.online2.v2.vocab import VocabV2

    schema_p = Path(feature_schema_path)
    bin_p = Path(binning_registry_path)
    vocab_p = Path(vocab_path)
    schema = FeatureSchema.load(schema_p)
    binning = BinningRegistryV2.load(bin_p)
    vocab = VocabV2.load(vocab_p)
    tok = TokenizerV2(schema, binning, vocab, farm_relative_enabled=farm_relative_enabled)
    hashes = {
        "actual_feature_schema_hash": file_sha256(schema_p),
        "actual_binning_registry_hash": file_sha256(bin_p),
        "actual_vocab_hash": file_sha256(vocab_p),
        "feature_schema_path": str(schema_p),
        "binning_registry_path": str(bin_p),
        "vocab_path": str(vocab_p),
    }
    return tok, hashes


def tokenize_mg_production(
    tokenizer,
    *,
    feature: str,
    raw_value: float,
    farm_id: str = "",
) -> List[str]:
    """Call production TokenizerV2.tokenize_value (frozen binning/refs inside registry)."""
    feat = _norm_feature(feature)
    # schema keys are lowercase
    mt = tokenizer.tokenize_value(feat, str(raw_value), farm_id=str(farm_id or ""))
    return list(mt.tokens)


def gate0_baseline_roundtrip(
    *,
    original_tokens: Sequence[str],
    feature: str,
    observed_raw: float,
    tokenizer,
    farm_id: str = "",
    actual_registry_hash: Optional[str] = None,
    expected_registry_hash: Optional[str] = None,
    actual_tokenizer_hash: Optional[str] = None,
    expected_tokenizer_hash: Optional[str] = None,
    strict_exact: bool = True,
) -> Dict[str, Any]:
    """Compare original MG bundle vs production tokenizer on observed raw.

    Strict smoke: only EXACT is valid. EXACT_CORE (QUALITY-only drift) is
    diagnostic-only and fails under strict_exact=True.
    """
    original_bundle = extract_mg_tokens(original_tokens, feature)
    recomputed = tokenize_mg_production(
        tokenizer, feature=feature, raw_value=observed_raw, farm_id=farm_id
    )
    match = "EXACT" if original_bundle == recomputed else "MISMATCH"
    # QUALITY-only drift reported separately (never auto-pass in strict mode)
    if match != "EXACT":
        o_core = [t for t in original_bundle if not str(t).startswith("QUALITY|")]
        r_core = [t for t in recomputed if not str(t).startswith("QUALITY|")]
        if o_core == r_core:
            match = "EXACT_CORE"
    reg_ok = True
    tok_ok = True
    if actual_registry_hash is not None and expected_registry_hash is not None:
        reg_ok = str(actual_registry_hash) == str(expected_registry_hash)
    if actual_tokenizer_hash is not None and expected_tokenizer_hash is not None:
        tok_ok = str(actual_tokenizer_hash) == str(expected_tokenizer_hash)
    if strict_exact:
        match_ok = match == "EXACT"
    else:
        match_ok = match in {"EXACT", "EXACT_CORE"}
    valid = match_ok and reg_ok and tok_ok
    reasons = []
    if not match_ok:
        reasons.append("BASELINE_ROUNDTRIP_FAIL" if match == "MISMATCH" else "EXACT_CORE_REJECTED_STRICT")
    if not reg_ok:
        reasons.append("REGISTRY_HASH_MISMATCH")
    if not tok_ok:
        reasons.append("TOKENIZER_HASH_MISMATCH")
    return {
        "baseline_roundtrip_valid": valid,
        "baseline_bundle_match": match,
        "registry_hash_match": reg_ok,
        "tokenizer_hash_match": tok_ok,
        "strict_exact": bool(strict_exact),
        "original_bundle": original_bundle,
        "recomputed_bundle": recomputed,
        "reason_codes": reasons,
        "actual_registry_hash": actual_registry_hash,
        "expected_registry_hash": expected_registry_hash,
        "actual_tokenizer_hash": actual_tokenizer_hash,
        "expected_tokenizer_hash": expected_tokenizer_hash,
    }


def apply_raw_edit_closure(
    *,
    event_id: str,
    feature: str,
    target_raw: float,
    original_tokens: Sequence[str],
    tokenizer,
    farm_id: str = "",
    observed_raw: Optional[float] = None,
    proposed_bundle: Optional[Sequence[str]] = None,
    reject_temporal_derived: bool = True,
    neighbor_event_ids: Optional[Sequence[str]] = None,
    stage_a_affected_target_event_ids: Optional[Sequence[str]] = None,
    actual_registry_hash: Optional[str] = None,
    expected_registry_hash: Optional[str] = None,
    actual_tokenizer_hash: Optional[str] = None,
    expected_tokenizer_hash: Optional[str] = None,
    strict_exact: bool = True,
    # legacy unused kwargs kept for call-site compatibility
    edges_abs: Optional[Sequence[float]] = None,
    edges_global: Optional[Sequence[float]] = None,
    edges_farm: Optional[Sequence[float]] = None,
    global_ref_value: Optional[float] = None,
    farm_ref_value: Optional[float] = None,
    global_reference_hash: Optional[str] = None,
    farm_reference_hash: Optional[str] = None,
    registry_hash: Optional[str] = None,
    tokenizer_hash: Optional[str] = None,
) -> RetokenizationResult:
    """Gate0 → production tokenize target raw → MG splice → Gate4."""
    reasons: List[str] = []
    toks = [str(t) for t in original_tokens]
    if reject_temporal_derived and any(t.startswith(("TREND|", "DELTA|")) for t in toks):
        # only reject if the *target MG* has temporal derived tokens
        mg = extract_mg_tokens(toks, feature)
        if any(t.startswith(("TREND|", "DELTA|")) for t in mg):
            reasons.append("TEMPORAL_DERIVED_UNSUPPORTED")

    # Prefer explicit actual/expected; fall back to legacy identical kwargs as mismatch signal
    act_reg = actual_registry_hash or registry_hash
    exp_reg = expected_registry_hash
    act_tok = actual_tokenizer_hash or tokenizer_hash
    exp_tok = expected_tokenizer_hash
    # If caller passed only identical legacy kwargs without expected, do not claim hash verified
    if exp_reg is None and registry_hash is not None and actual_registry_hash is None:
        exp_reg = None  # leave unverified rather than tautology
        act_reg = None
    if exp_tok is None and tokenizer_hash is not None and actual_tokenizer_hash is None:
        exp_tok = None
        act_tok = None

    obs = float(observed_raw) if observed_raw is not None else float(target_raw)
    g0 = gate0_baseline_roundtrip(
        original_tokens=toks,
        feature=feature,
        observed_raw=obs,
        tokenizer=tokenizer,
        farm_id=farm_id,
        actual_registry_hash=act_reg,
        expected_registry_hash=exp_reg,
        actual_tokenizer_hash=act_tok,
        expected_tokenizer_hash=exp_tok,
        strict_exact=bool(strict_exact),
    )
    if not g0["baseline_roundtrip_valid"]:
        reasons.extend(g0.get("reason_codes") or ["BASELINE_ROUNDTRIP_FAIL"])

    actual = tokenize_mg_production(
        tokenizer, feature=feature, raw_value=float(target_raw), farm_id=farm_id
    )
    proposed = list(proposed_bundle) if proposed_bundle is not None else list(actual)
    gate4 = compare_token_bundles(proposed, actual, require_subset=False)
    if gate4["status"] != "PASSED":
        reasons.append("RETOKENIZATION_MISMATCH")

    try:
        new_sentence, mg_span, outside_ok = splice_mg_tokens(toks, feature, actual)
    except ValueError as e:
        reasons.append("MG_SPAN_NOT_FOUND")
        new_sentence, mg_span, outside_ok = list(toks), None, False
        reasons.append(str(e))

    if not outside_ok:
        reasons.append("OUTSIDE_MG_MUTATION")

    affected = [str(event_id)]
    if neighbor_event_ids:
        for n in neighbor_event_ids:
            if str(n) not in affected:
                affected.append(str(n))

    return RetokenizationResult(
        baseline_roundtrip_valid=bool(g0["baseline_roundtrip_valid"]),
        baseline_bundle_match=str(g0["baseline_bundle_match"]),
        registry_hash_match=bool(g0.get("registry_hash_match", True)),
        tokenizer_hash_match=bool(g0.get("tokenizer_hash_match", True)),
        reference_policy=REFERENCE_POLICY,
        reference_refit=False,
        global_reference_hash=global_reference_hash,
        farm_reference_hash=farm_reference_hash,
        directly_edited_event_ids=[str(event_id)],
        retokenization_affected_event_ids=affected,
        stage_a_affected_target_event_ids=list(stage_a_affected_target_event_ids or affected),
        proposed_bundle=proposed,
        actual_retokenized_bundle=actual,
        new_sentence_tokens=new_sentence,
        mg_span=mg_span,
        unchanged_outside_mg=outside_ok,
        gate4=gate4,
        reason_codes=reasons,
        actual_registry_hash=act_reg,
        expected_registry_hash=exp_reg,
        actual_tokenizer_hash=act_tok,
        expected_tokenizer_hash=exp_tok,
    )


# Back-compat alias used by older tests
def recover_observed_raw_for_mg(
    tokenizer,
    *,
    feature: str,
    farm_id: str,
    original_tokens: Sequence[str],
    grounded_raw: Optional[float] = None,
    edges_abs: Optional[Sequence[float]] = None,
    n_probe: int = 41,
    allow_abs_bin_probe: bool = False,
) -> Dict[str, Any]:
    """Find raw that EXACT-reproduces original MG bundle via production tokenizer.

    Strict CF path: only grounded_raw that EXACT-matches is acceptable.
    abs_bin_probe is diagnostic-only and never used as CF evaluation input when
    allow_abs_bin_probe=False (default).
    """
    from ..grounding.raw_target import decode_interval

    original_bundle = extract_mg_tokens(original_tokens, feature)
    if not original_bundle:
        return {"ok": False, "reason": "MG_NOT_FOUND", "observed_raw": grounded_raw}

    def _try(raw: float) -> bool:
        got = tokenize_mg_production(
            tokenizer, feature=feature, raw_value=float(raw), farm_id=farm_id
        )
        return got == original_bundle

    if grounded_raw is not None and _try(float(grounded_raw)):
        return {
            "ok": True,
            "observed_raw": float(grounded_raw),
            "source": "grounded_raw",
            "original_bundle": original_bundle,
        }

    # Diagnostic probe (never accepted as CF input unless explicitly allowed)
    probe_diag: Dict[str, Any] = {
        "probe_attempted": False,
        "probe_ok": False,
        "probe_raw": None,
    }
    abs_tok = next((t for t in original_bundle if str(t).startswith("VALUE_ABS|")), None)
    if abs_tok is not None and edges_abs is not None and len(edges_abs) >= 2:
        try:
            k = int(str(abs_tok).split("ABS_B")[-1])
            lo, hi = decode_interval(edges_abs, k)
            probe_diag["probe_attempted"] = True
            probe_diag["abs_bin"] = k
            probe_diag["interval"] = [lo, hi]
            for i in range(int(n_probe)):
                cand = lo + (hi - lo) * (i + 0.5) / float(n_probe)
                if _try(cand):
                    probe_diag["probe_ok"] = True
                    probe_diag["probe_raw"] = float(cand)
                    break
        except ValueError:
            probe_diag["probe_error"] = "ABS_PARSE_FAIL"

    if allow_abs_bin_probe and probe_diag.get("probe_ok"):
        return {
            "ok": True,
            "observed_raw": float(probe_diag["probe_raw"]),
            "source": "abs_bin_probe",
            "original_bundle": original_bundle,
            "grounded_raw_rejected": grounded_raw,
            "diagnostic": probe_diag,
        }

    return {
        "ok": False,
        "reason": "GROUNDED_RAW_DRIFT" if grounded_raw is not None else "NO_GROUNDED_RAW",
        "observed_raw": grounded_raw,
        "original_bundle": original_bundle,
        "diagnostic": probe_diag,
        "note": "abs_bin_probe reserved for diagnostics; not used as CF input in strict mode",
    }


def retokenize_mg_frozen(*args, **kwargs):
    raise RuntimeError(
        "retokenize_mg_frozen (manual bin assembly) removed; "
        "use tokenize_mg_production / apply_raw_edit_closure with TokenizerV2"
    )
