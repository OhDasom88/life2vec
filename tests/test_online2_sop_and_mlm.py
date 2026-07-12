from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.data_new.types import PersonDocument
from src.tasks.mlm import MLM


SEMANTICS = "CHRONOLOGICAL_WITH_TIES"


def document(groups=("G1", "G1", "G1", "G2")):
    size = len(groups)
    return PersonDocument(
        person_id=17,
        sentences=[[f"token_{i}"] for i in range(size)],
        abspos=[10 if group == "G1" else 20 if group == "G2" else 30 for group in groups],
        age=[float(i) for i in range(size)],
        timecut_pos=size,
        segment=list(range(2, size + 2)),
        event_ids=[f"E{i}" for i in range(size)],
        same_time_group_ids=list(groups),
        event_kinds=[f"K{i}" for i in range(size)],
        modality_refs=[f"M{i}" for i in range(size)],
        order_semantics=SEMANTICS,
        op_eligible=True,
    )


def aligned_signature(doc):
    return [
        (
            sentence[0],
            doc.abspos[i],
            doc.age[i],
            doc.segment[i],
            doc.event_ids[i],
            doc.same_time_group_ids[i],
            doc.event_kinds[i],
            doc.modality_refs[i],
        )
        for i, sentence in enumerate(doc.sentences)
    ]


@pytest.mark.parametrize(
    ("groups", "inside_order", "force_label", "expected_label", "expected_groups"),
    [
        (("G1", "G1", "G1", "G2"), (0, 1, 2, 3), 0, 0, ["G1", "G1", "G1", "G2"]),
        (("G1", "G1", "G1", "G2"), (2, 0, 1, 3), 0, 0, ["G1", "G1", "G1", "G2"]),
        (("G1", "G1", "G1", "G2"), (2, 1, 0, 3), 0, 0, ["G1", "G1", "G1", "G2"]),
        (("G1", "G1", "G1", "G2"), (0, 1, 2, 3), 1, 1, ["G2", "G1", "G1", "G1"]),
        (("G1", "G2", "G3"), (0, 1, 2), 2, 2, None),
        (("G1", "G2", "G3"), (0, 1, 2), 1, 1, ["G3", "G2", "G1"]),
        (("G1", "G1", "G1"), (0, 1, 2), 0, 0, ["G1", "G1", "G1"]),
    ],
)
def test_required_sop_cases(
    groups, inside_order, force_label, expected_label, expected_groups
):
    task = MLM(name="test", max_length=64)
    source = document(groups)
    source.select_events(inside_order)
    before = deepcopy(source)

    transformed, label, mask = task.cls_task(source, force_label=force_label)

    assert int(label) == expected_label
    assert aligned_signature(source) == aligned_signature(before)  # no in-place change
    if expected_groups is not None:
        assert transformed.same_time_group_ids == expected_groups
    else:
        assert transformed.same_time_group_ids not in [
            list(groups),
            list(reversed(groups)),
        ]
    if len(set(groups)) == 1:
        assert float(mask) == 0.0
    else:
        assert float(mask) == 1.0

    original_by_event = {row[4]: row for row in aligned_signature(before)}
    for row in aligned_signature(transformed):
        assert row == original_by_event[row[4]]


def test_two_groups_never_emit_shuffled():
    transformed, label, mask = MLM(name="test", max_length=64).cls_task(
        document(), force_label=2
    )
    assert int(label) == 1
    assert float(mask) == 1.0
    assert transformed.same_time_group_ids[0] == "G2"


def test_non_chronological_semantics_are_ineligible_even_if_declared_true():
    source = document()
    source.order_semantics = "RELATION_ORDERED"
    _, label, mask = MLM(name="test", max_length=64).cls_task(source, force_label=1)
    assert int(label) == 0
    assert float(mask) == 0.0


def test_evaluation_sop_is_deterministic():
    task = MLM(
        name="test",
        max_length=64,
        sop_reverse_probability=0.5,
        sop_shuffle_probability=0.5,
        evaluation_seed=7,
    )
    first = task.cls_task(document(("G1", "G2", "G3")), is_train=False)
    second = task.cls_task(document(("G1", "G2", "G3")), is_train=False)
    assert int(first[1]) == int(second[1])
    assert first[0].event_ids == second[0].event_ids


class FakeVocabulary:
    general_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    background_tokens = ["BG"]
    token2index = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[MASK]": 3,
        "[UNK]": 4,
        "BG": 5,
        "A": 6,
    }


def test_mlm_caps_legal_mask_and_protects_special_background():
    task = MLM(name="test", max_length=20, mask_ratio=1.0)
    task.datamodule = SimpleNamespace(vocabulary=FakeVocabulary())
    tokens = np.array([1, 2, 4, 5, 6], dtype=np.int64)

    masked, positions, targets = task.mlm_mask(tokens.copy())

    selected = positions[targets != 0]
    assert selected.tolist() == [4]
    assert targets[targets != 0].tolist() == [6]
    assert masked[:4].tolist() == tokens[:4].tolist()
    assert positions.dtype == np.int64
    assert targets.dtype == np.int64


def test_mlm_mask_never_exceeds_max_masked_slots():
    task = MLM(name="test", max_length=16, mask_ratio=0.3)
    task.datamodule = SimpleNamespace(vocabulary=FakeVocabulary())
    # Many legal tokens after the CLS position.
    tokens = np.array([1] + [6] * 40, dtype=np.int64)
    _, positions, targets = task.mlm_mask(tokens.copy())
    max_masked = int(np.floor(task.mask_ratio * task.max_length))
    assert len(positions) == max_masked
    assert int((targets != 0).sum()) <= max_masked
