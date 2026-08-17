"""Assemble one case's LLM-evidence bundle: saliency-ranked events, risk
scores, image-fusion presence, and (via `nearest_examples.py`) similar
labeled training cases.

Saliency reuses `risk_saliency.py::event_ixg_abnormal_margin` as-is (no new
attribution method) -- the same function `counterfactual/attribution/` and
`concept_tcav/` already build on. Event descriptions are looked up from
`events_tokenized_v2.parquet` by `event_id` (only the needed ids, via a
pyarrow filter -- the full parquet is too large to load, see
`concept_tcav/narrative_concept_probe.py::narrative_event_ids` for the same
pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.online2.v2.finetune_v03.risk_saliency import event_ixg_abnormal_margin


def lookup_event_sentences(events_path: Path, event_ids: Sequence[str]) -> Dict[str, str]:
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(events_path), format="parquet")
    table = dataset.to_table(
        columns=["event_id", "SENTENCE"], filter=ds.field("event_id").isin(list(set(event_ids)))
    )
    return dict(zip(table.column("event_id").to_pylist(), table.column("SENTENCE").to_pylist()))


def build_case_evidence(
    model: torch.nn.Module,
    sample: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    *,
    normal_class_id: int,
    label_names: Sequence[str],
    events_path: Optional[Path] = None,
    k_salient: int = 5,
    use_binary: bool = True,
) -> Dict[str, Any]:
    """`batch` must be `collate_fn([sample])` (batch_size=1) already moved to device."""
    saliency = event_ixg_abnormal_margin(
        model, batch, normal_class_id=normal_class_id, use_binary=use_binary
    )
    saliency = saliency.detach().cpu().numpy()
    event_ids = list(sample.get("event_ids", []))[: saliency.shape[0]]

    order = np.argsort(-saliency)  # most abnormal-risk-increasing first
    top_idx = [i for i in order[:k_salient]]
    bottom_idx = [i for i in order[-k_salient:]] if len(order) > k_salient else []

    sentences: Dict[str, str] = {}
    if events_path is not None and event_ids:
        wanted = [event_ids[i] for i in top_idx + bottom_idx if i < len(event_ids)]
        sentences = lookup_event_sentences(events_path, wanted)

    def _row(i: int) -> Dict[str, Any]:
        eid = event_ids[i] if i < len(event_ids) else None
        return {
            "event_id": eid,
            "case_age_hours": float(sample["case_age_hours"][i]) if i < len(sample["case_age_hours"]) else None,
            "saliency_score": float(saliency[i]),
            "sentence": sentences.get(eid) if eid else None,
        }

    with torch.no_grad():
        model.eval()
        out = model(
            event_mean=batch["event_mean"],
            event_max=batch["event_max"],
            case_age_hours=batch["case_age_hours"],
            view_id=batch["view_id"],
            zone_id=batch["zone_id"],
            local_hour=batch["local_hour"],
            padding_mask=batch["padding_mask"],
            dino_vec=batch.get("dino_vec"),
            dino_mask=batch.get("dino_mask"),
        )
        p_abnormal = float(torch.sigmoid(out["abnormal_logit"])[0].item())
        fine_probs = torch.softmax(out["logits"], dim=-1)[0].detach().cpu().numpy()

    dino_vec = batch.get("dino_vec")
    has_image_evidence = bool(
        dino_vec is not None and torch.any(dino_vec.abs() > 1e-6).item()
    )

    return {
        "case_id": sample.get("case_id"),
        "n_events": int(saliency.shape[0]),
        "p_abnormal": p_abnormal,
        "fine_probs": {str(label_names[i]): float(fine_probs[i]) for i in range(len(label_names))},
        "top_abnormal_events": [_row(i) for i in top_idx],
        "top_normal_events": [_row(i) for i in bottom_idx],
        "has_image_evidence": has_image_evidence,
        "image_evidence_note": (
            "DINO 이미지 임베딩이 이 케이스의 이벤트 표현에 융합되어 있음 (원시 이미지 자체는 "
            "이 프롬프트에 포함되지 않음 -- 현재 파이프라인에 vision 모델 연동이 없음)"
            if has_image_evidence
            else "이 케이스에서 유효한 DINO 이미지 임베딩을 찾지 못함"
        ),
    }
