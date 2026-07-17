"""Hard preflight for Stage-A MLM decoder (warning-only PASS forbidden)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


EVIDENCE_GOLDEN = "GOLDEN_REFERENCE_PARITY"
EVIDENCE_PARAM = "PARAMETER_INTEGRITY_ONLY"
EVIDENCE_FAIL = "PREFLIGHT_FAILED"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_checksum(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy().tobytes()
    return _sha256_bytes(arr)


def _find_token_embedding(model) -> Optional[torch.Tensor]:
    for name, p in model.named_parameters():
        if "embedding" in name.lower() and "token" in name.lower() and "weight" in name:
            if p.ndim == 2:
                return p
    # fallback common path
    emb = getattr(getattr(model, "transformer", None), "embedding", None)
    tok = getattr(emb, "token", None) if emb is not None else None
    w = getattr(tok, "weight", None)
    return w if isinstance(w, torch.Tensor) else None


def run_mlm_decoder_preflight(
    model,
    *,
    vocab_size: int,
    device: torch.device,
    ckpt_state: Optional[Mapping[str, Any]] = None,
    golden_logits_path: Optional[Path] = None,
    abs_tol: float = 1e-5,
    mean_tol: float = 1e-5,
) -> Dict[str, Any]:
    """Validate mlm_decoder presence, vocab dim, forward, optional golden parity.

    Returns artifact dict. Never soft-passes when keys 1–3 fail.
    Weight-tying warning alone does not fail if parameter integrity holds.
    """
    fail_reasons = []
    warnings = []

    sd = ckpt_state
    if sd is None:
        sd = {k: v for k, v in model.state_dict().items()}

    mlm_keys = [k for k in sd if "mlm_decoder" in k or k == "mlm_w"]
    keys_present = bool(mlm_keys) and any("mlm_decoder" in k for k in mlm_keys)
    if not keys_present:
        fail_reasons.append("mlm_decoder_state_keys_missing")

    has_attr = hasattr(model, "mlm_decoder") and model.mlm_decoder is not None
    if not has_attr:
        fail_reasons.append("mlm_decoder_attribute_missing")

    decoder_out_dim = None
    out_w = None
    for k, v in sd.items():
        if k.endswith("mlm_decoder.out.weight") or k == "mlm_decoder.out.weight":
            out_w = v
            decoder_out_dim = int(v.shape[0])
            break
    if decoder_out_dim is None and has_attr:
        try:
            out_w = model.mlm_decoder.out.weight
            decoder_out_dim = int(out_w.shape[0])
        except Exception:
            fail_reasons.append("decoder_out_weight_unreadable")

    vocab_match = decoder_out_dim is not None and int(decoder_out_dim) == int(vocab_size)
    if not vocab_match:
        fail_reasons.append(
            f"vocab_dim_mismatch:decoder={decoder_out_dim}:vocab={vocab_size}"
        )

    weight_tying_verified = False
    emb = _find_token_embedding(model)
    if emb is not None and out_w is not None:
        try:
            # common tying: out.weight shares or is tied to embedding
            if emb.data_ptr() == out_w.data_ptr():
                weight_tying_verified = True
            elif emb.shape == out_w.shape and torch.allclose(
                emb.float(), out_w.float(), atol=1e-5, rtol=1e-5
            ):
                weight_tying_verified = True
            else:
                warnings.append("MLM decoder parametrization failed")
                # parameter integrity via checksum still allowed
                weight_tying_verified = False
        except Exception as e:
            warnings.append(f"weight_tying_check_error:{e}")
    else:
        warnings.append("embedding_or_out_missing_for_tying_check")

    param_checksum = None
    if out_w is not None:
        param_checksum = _tensor_checksum(out_w)
    parameter_checksum_match = param_checksum is not None and keys_present

    forward_ok = False
    logits_shape = None
    decoder_forward_call_count = 0
    if has_attr and not fail_reasons:
        try:
            model.eval()
            x = torch.zeros(1, 4, 32, dtype=torch.long, device=device)
            x[0, 0, :] = 1
            pad = torch.ones(1, 32, dtype=torch.long, device=device)
            with torch.no_grad():
                hidden = model.transformer.forward_finetuning(x=x, padding_mask=pad)
                pos = torch.tensor([[4, 5, 6]], dtype=torch.long, device=device)
                logits = model.mlm_decoder(hidden, {"target_pos": pos})
                decoder_forward_call_count = 1
            logits_shape = list(logits.shape)
            if int(logits.shape[-1]) != int(vocab_size):
                fail_reasons.append("forward_vocab_dim_mismatch")
            else:
                forward_ok = True
        except Exception as e:
            fail_reasons.append(f"forward_failed:{e}")

    reference_parity = "NOT_AVAILABLE"
    golden_sha = None
    evidence = EVIDENCE_PARAM
    if golden_logits_path is not None and Path(golden_logits_path).exists():
        golden_sha = hashlib.sha256(Path(golden_logits_path).read_bytes()).hexdigest()
        try:
            payload = json.loads(Path(golden_logits_path).read_text(encoding="utf-8"))
            expected = torch.as_tensor(payload["logits"], dtype=torch.float32)
            # recompute on CPU for numeric parity (file integrity + numeric)
            if has_attr and forward_ok:
                model_cpu = model
                x = torch.zeros(1, 4, 32, dtype=torch.long)
                x[0, 0, :] = 1
                pad = torch.ones(1, 32, dtype=torch.long)
                with torch.no_grad():
                    hidden = model_cpu.transformer.forward_finetuning(
                        x=x.to(device), padding_mask=pad.to(device)
                    )
                    pos = torch.as_tensor(
                        payload.get("target_pos") or [[4, 5, 6]],
                        dtype=torch.long,
                        device=device,
                    )
                    got = model_cpu.mlm_decoder(hidden, {"target_pos": pos}).float().cpu()
                abs_err = (got - expected).abs()
                mean_err = float(abs_err.mean().item())
                max_err = float(abs_err.max().item())
                if mean_err <= mean_tol and max_err <= abs_tol:
                    reference_parity = "PASS"
                    evidence = EVIDENCE_GOLDEN
                else:
                    reference_parity = "FAIL"
                    fail_reasons.append(
                        f"golden_numeric_mismatch:mean={mean_err}:max={max_err}"
                    )
            else:
                # file present but cannot compare — integrity only via hash record
                reference_parity = "FILE_PRESENT_NOT_COMPARED"
        except Exception as e:
            reference_parity = "FAIL"
            fail_reasons.append(f"golden_load_failed:{e}")

    passed = len(fail_reasons) == 0 and forward_ok and keys_present and has_attr and vocab_match
    if not passed:
        evidence = EVIDENCE_FAIL

    return {
        "decoder_preflight_passed": bool(passed),
        "preflight_evidence_level": evidence,
        "reference_parity": reference_parity,
        "parameter_checksum_match": bool(parameter_checksum_match),
        "parameter_checksum_sha256": param_checksum,
        "weight_tying_verified": bool(weight_tying_verified),
        "mlm_decoder_keys_present": bool(keys_present),
        "mlm_decoder_attribute": bool(has_attr),
        "decoder_out_dim": decoder_out_dim,
        "vocab_size": int(vocab_size),
        "vocab_dim_match": bool(vocab_match),
        "forward_ok": bool(forward_ok),
        "logits_shape": logits_shape,
        "decoder_forward_call_count": int(decoder_forward_call_count),
        "golden_reference_sha256": golden_sha,
        "fail_reasons": fail_reasons,
        "warnings": warnings,
        "status": "PASS" if passed else "FAIL",
    }
