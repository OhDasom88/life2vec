import json

import pandas as pd
import pytest
import torch

from src.data_new.datamodule import timedelta_to_position
from src.data_new.sources.online2 import Online2ParquetTokenSource
from src.data_new.vocabulary import RegistryVocabulary
from src.tasks.base import Task
from src.transformer.external_embeddings import ExternalEmbeddingProjection
from src.transformer.models import masked_sop_loss


def test_task_passes_optional_dataframe_metadata():
    frame = pd.DataFrame(
        {
            "SENTENCE": ["A", "B"],
            "START_DATE": [0, 1],
            "AGE": [1.0, 2.0],
            "AFTER_THRESHOLD": [False, False],
            "BIRTHDAY": pd.to_datetime(["2000-01-01", "2000-01-01"]),
            "RES_ORIGIN": ["X", "X"],
            "GENDER": ["U", "U"],
            "event_id": ["E1", "E2"],
            "same_time_group_id": ["G1", "G2"],
            "event_kind": ["sensor", "derived"],
            "modality_ref": [None, "row:2"],
            "segment_id": [2, 3],
            "order_semantics": ["STRICT_CHRONOLOGICAL"] * 2,
            "op_eligible": [True, True],
        }
    )
    frame.name = 99
    doc = Task(name="metadata", max_length=32).get_document(frame)
    assert doc.event_ids == ["E1", "E2"]
    assert doc.same_time_group_ids == ["G1", "G2"]
    assert doc.event_kinds == ["sensor", "derived"]
    assert doc.modality_refs == [None, "row:2"]
    assert doc.segment == [2, 3]
    assert doc.order_semantics == "STRICT_CHRONOLOGICAL"
    assert bool(doc.op_eligible)


def test_registry_vocabulary_is_stable_and_oov_maps_to_unk(tmp_path):
    registry = {
        "tokens": [
            {"token_id": 6, "token": "FARM_A", "category": "FARM", "registry_version": "v1"},
            {"token_id": 0, "token": "[PAD]", "category": "GENERAL", "registry_version": "v1"},
            {"token_id": 4, "token": "[UNK]", "category": "GENERAL", "registry_version": "v1"},
            {"token_id": 3, "token": "[MASK]", "category": "GENERAL", "registry_version": "v1"},
            {"token_id": 2, "token": "[SEP]", "category": "GENERAL", "registry_version": "v1"},
            {"token_id": 1, "token": "[CLS]", "category": "GENERAL", "registry_version": "v1"},
            {"token_id": 5, "token": "IMAGE_EMBED_SLOT", "category": "MODALITY", "registry_version": "v1"},
        ]
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    vocab = RegistryVocabulary(registry_path=str(path), registry_version="v1")

    assert vocab.tokens() == [
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[UNK]",
        "IMAGE_EMBED_SLOT",
        "FARM_A",
    ]
    assert vocab.token2index.get("NOT_IN_REGISTRY", vocab.token2index["[UNK]"]) == 4
    assert RegistryVocabulary(
        registry_path=str(path), registry_version="v1"
    ).token2index == vocab.token2index


def test_online2_parquet_source_has_deterministic_event_order(tmp_path):
    path = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "PERSON_ID": [2, 1, 1, 3],
            "START_DATE": [
                "2025-01-01 00:00",
                "2025-01-01 02:00",
                "2025-01-01 01:00",
                "2025-01-01 01:00",
            ],
            "SENTENCE": ["C", "B", "A", "A"],
            "event_id": ["E3", "E2", "E1", "E1"],
            "same_time_group_id": ["G3", "G2", "G1", "G1"],
            "event_kind": ["sensor"] * 4,
            "event_position": [0, 1, 0, 0],
            "time_group_rank": [0, 1, 0, 0],
            "sequence_id": ["S2", "S1", "S1", "S3"],
            "build_id": ["build-1"] * 4,
            "registry_version": ["v1"] * 4,
        }
    ).to_parquet(path)
    source = Online2ParquetTokenSource(
        path=str(path), build_id="build-1", registry_version="v1"
    )
    result = source.tokenized().compute().reset_index()
    assert result["event_id"].tolist() == ["E1", "E2", "E3", "E1"]
    assert all(
        group["START_DATE"].is_monotonic_increasing
        for _, group in result.groupby("PERSON_ID")
    )


def test_hour_time_unit_preserves_intraday_resolution():
    dates = pd.Series(pd.to_datetime(["2025-01-01 00:00", "2025-01-01 03:00"]))
    delta = dates - pd.Timestamp("2025-01-01")
    assert timedelta_to_position(delta, "hour").tolist() == [0, 3]
    assert timedelta_to_position(delta, "day").tolist() == [0, 0]
    with pytest.raises(ValueError):
        timedelta_to_position(delta, "fortnight")


def test_external_projection_detaches_inputs_and_enforces_missing_policy():
    module = ExternalEmbeddingProjection(3, 4, missing_policy="zero")
    source = torch.randn(2, 3, requires_grad=True)
    output = module(source, torch.tensor([True, False]))
    output.sum().backward()
    assert source.grad is None
    assert module.projection.weight.grad is not None

    strict = ExternalEmbeddingProjection(3, 4, missing_policy="error")
    with pytest.raises(ValueError, match="Missing external embedding"):
        strict(source.detach(), torch.tensor([True, False]))
    with pytest.raises(ValueError, match="dim"):
        strict(torch.randn(2, 5))


def test_sop_loss_excludes_ineligible_rows_and_supports_legacy_batch():
    losses = torch.tensor([2.0, 100.0, 4.0], requires_grad=True)
    masked = masked_sop_loss(losses, torch.tensor([1.0, 0.0, 1.0]))
    assert masked.item() == 3.0
    assert masked_sop_loss(losses).item() == pytest.approx(106.0 / 3.0)
    empty = masked_sop_loss(losses, torch.zeros(3))
    assert empty.item() == 0.0
    empty.backward()
    assert losses.grad is not None
