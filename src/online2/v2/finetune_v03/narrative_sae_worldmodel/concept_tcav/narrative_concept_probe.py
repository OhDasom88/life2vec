"""Glue: define a TCAV concept from narrative catalog matches, run it against
the finetune diagnosis model's abnormal-risk sensitivity.

Concept membership is defined by the EXISTING narrative catalog rather than a
new hand-labeled concept set: "concept" = events matched by a given
`narrative_id` in `training_events_v2.parquet`, "random" = a same-size sample
of events from the same cases NOT matched by that narrative. This reuses the
matcher-defined ground truth the whole online2 narrative pipeline already
relies on instead of inventing a parallel concept-labeling scheme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from .cav import ConceptActivationVectors, TCAVScore, tcav_score, train_cavs
from .event_gradients import event_activation_and_gradient


def narrative_event_ids(
    training_events_path: Path, narrative_id: str, farm_ids: Set[str]
) -> Set[str]:
    """Event ids matched by `narrative_id`, restricted to `farm_ids` (avoids
    loading the full corpus -- flat training_events_v2.parquet can be tens of GB).

    2026-07-26: `training_events_path`는 이제 보통 `sequences_v2.parquet`(시퀀스당
    1행, event_ids/farm_ids가 JSON 리스트)를 가리킨다 -- 260배 중복 펼친
    flat 파일을 더 이상 만들지 않기 때문. 스키마를 보고 자동으로 분기해서
    옛 flat 파일이 남아있는 경우도 그대로 지원한다."""
    import json as _json

    import pyarrow.dataset as ds

    dataset = ds.dataset(str(training_events_path), format="parquet")
    schema_names = set(dataset.schema.names)
    if "event_ids" in schema_names and "farm_ids" in schema_names:
        table = dataset.to_table(
            columns=["narrative_id", "event_ids", "farm_ids"],
            filter=ds.field("narrative_id") == narrative_id,
        )
        out: Set[str] = set()
        for eids_json, fids_json in zip(
            table.column("event_ids").to_pylist(), table.column("farm_ids").to_pylist()
        ):
            fids = set(_json.loads(fids_json)) if fids_json else set()
            if fids & farm_ids:
                out.update(_json.loads(eids_json) if eids_json else [])
        return out
    table = dataset.to_table(
        columns=["event_id", "narrative_id", "farm_id"],
        filter=(ds.field("narrative_id") == narrative_id) & ds.field("farm_id").isin(list(farm_ids)),
    )
    return set(table.column("event_id").to_pylist())


def collect_case_activation_gradients(
    model: torch.nn.Module,
    dataset,
    *,
    collate_fn,
    normal_class_id: int,
    use_binary: bool = False,
    device: torch.device,
) -> Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]:
    """For every case in `dataset`, return (event_ids, activations[T,D], gradients[T,D]).

    `dataset` must yield samples with an `event_ids` field per case aligned to
    `event_mean`/`event_max` rows (i.e. `DiagnosisEventDatasetV03`-shaped).
    Requires `event_ids` to be threaded through the dataset/collate path --
    the base `DiagnosisEventDatasetV03` does not currently carry them, so
    callers must pass a dataset variant that does (see module README).
    """
    out: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]] = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        batch = collate_fn([sample])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        act, grad = event_activation_and_gradient(
            model, batch, normal_class_id=normal_class_id, use_binary=use_binary
        )
        event_ids = list(sample.get("event_ids", []))[: act.shape[0]]
        out[str(sample["case_id"])] = (event_ids, act, grad)
    return out


def split_concept_vs_random(
    case_activation_gradients: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    concept_event_ids: Set[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool (concept_activations, concept_gradients, random_activations, random_gradients)
    across all cases, splitting each case's events by narrative-concept membership."""
    concept_act, concept_grad, random_act, random_grad = [], [], [], []
    for event_ids, act, grad in case_activation_gradients.values():
        is_concept = np.array([eid in concept_event_ids for eid in event_ids], dtype=bool)
        if is_concept.any():
            concept_act.append(act[is_concept])
            concept_grad.append(grad[is_concept])
        if (~is_concept).any():
            random_act.append(act[~is_concept])
            random_grad.append(grad[~is_concept])
    if not concept_act or not random_act:
        raise ValueError("no concept-matched events or no random events found across cases")
    return (
        np.concatenate(concept_act),
        np.concatenate(concept_grad),
        np.concatenate(random_act),
        np.concatenate(random_grad),
    )


def run_narrative_tcav(
    model: torch.nn.Module,
    dataset,
    *,
    collate_fn,
    training_events_path: Path,
    narrative_id: str,
    farm_ids: Set[str],
    normal_class_id: int,
    device: torch.device,
    use_binary: bool = False,
    n_bootstraps: int = 200,
    seed: int = 0,
) -> Tuple[ConceptActivationVectors, TCAVScore]:
    """End-to-end: narrative_id -> CAV -> TCAV score against abnormal-risk gradient."""
    concept_ids = narrative_event_ids(training_events_path, narrative_id, farm_ids)
    case_ag = collect_case_activation_gradients(
        model,
        dataset,
        collate_fn=collate_fn,
        normal_class_id=normal_class_id,
        use_binary=use_binary,
        device=device,
    )
    c_act, c_grad, r_act, r_grad = split_concept_vs_random(case_ag, concept_ids)
    cavs = train_cavs(
        c_act,
        r_act,
        concept_name=narrative_id,
        position="z_binary" if use_binary else "z_fine",
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    # score against the CONCEPT events' own gradients (does the concept direction
    # align with how those specific events push the abnormal-risk objective)
    score = tcav_score(c_grad, cavs, target="abnormal_risk")
    return cavs, score
