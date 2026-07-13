import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import List, Tuple, TypeVar, cast
from xml.dom.minidom import Document

import numpy as np
import torch

from src.data_new.types import Background, PersonDocument, EncodedDocument
from src.tasks.base import Task

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MLM(Task):
    """
    Task for used with Masked language modelling.

    .. todo::
        Describe MLM

    :param mask_ratio: Fraction of tokens to mask.
    :param smart_masking: Whether to apply smart masking (use tokens from the same group when choosing randoms).
    """

    # MLM Specific params
    mask_ratio: float = 0.30
    smart_masking: bool = False
    mask_special_tokens: bool = False
    mask_background_tokens: bool = False
    non_maskable_tokens: Tuple[str, ...] = ("IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT")
    sop_reverse_probability: float = 0.05
    sop_shuffle_probability: float = 0.05
    evaluation_seed: int = 2023

    def encode_preprocessed_document(
        self, document: PersonDocument, is_train: bool
    ) -> "MLMEncodedDocument":
        return self.encode_document(document, is_train=is_train)

    def encode_document(
        self, document: PersonDocument, is_train: bool = True
    ) -> "MLMEncodedDocument":

        prefix_sentence = (
            ["[CLS]"] + Background.get_sentence(document.background) + ["[SEP]"]
        )

        ############################################
        ### CLS TASK
        document, targ_cls, target_cls_mask = self.cls_task(
            document, is_train=is_train
        )
        ############################################

        sentences = [prefix_sentence] + [s + ["[SEP]"] for s in document.sentences]
        sentence_lengths = [len(x) for x in sentences]

        def expand(x: List[T]) -> List[T]:
            assert len(x) == len(sentence_lengths)
            return list(
                chain.from_iterable(
                    length * [i] for length, i in zip(sentence_lengths, x)
                )
            )

        abspos_expanded = expand([0] + document.abspos)
        age_expanded = expand([0.0] + document.age)
        assert document.segment is not None
        segment_expanded = expand([1] + document.segment)

        flat_sentences = np.concatenate(sentences)

        token2index = self.datamodule.vocabulary.token2index
        unk_id = token2index["[UNK]"]

        #print(flat_sentences[500:550])
        token_ids = np.array([token2index.get(x, unk_id) for x in flat_sentences])
        masked_sentences, masked_indx, masked_tokens = self.mlm_mask(token_ids.copy())

        length = len(token_ids)

        input_ids = np.zeros((4, self.max_length), dtype=np.float32)
        input_ids[0, :length] = masked_sentences
        input_ids[1, :length] = abspos_expanded
        input_ids[2, :length] = age_expanded
        input_ids[3, :length] = segment_expanded

        padding_mask = np.repeat(False, self.max_length)
        padding_mask[:length] = True

        # TODO: Consider renaming, to document/sentences instead of sequence...
        # would require refactoring of the modelling also though

        original_sequence = np.zeros(self.max_length, dtype=np.int64)
        original_sequence[:length] = token_ids

        sequence_id = np.array(document.person_id)

        return MLMEncodedDocument(
            sequence_id=sequence_id,
            input_ids=input_ids,
            padding_mask=padding_mask,
            target_tokens=masked_tokens,
            target_pos=masked_indx,
            target_cls=targ_cls,
            target_cls_mask=target_cls_mask,
            original_sequence=original_sequence,
        )

    # These could (maybe should?) also be calculated in the __post_init__.
    # Accessing the serialized methods in a parallel context may give problems down
    # the line.
    @cached_property
    def token_groups(self) -> List[Tuple[int, int]]:
        """Return pairs of first and last index for each token category in the
        vocabulary excluding GENERAL.
        """

        vocab = self.datamodule.vocabulary.vocab()
        no_general = vocab.CATEGORY != "GENERAL"
        token_groups = (
            vocab.loc[no_general]
            .groupby("CATEGORY")
            .ID.agg(["first", "last"])
            .sort_values("first")
            .to_records(index=False)
            .tolist()
        )
        return cast(List[Tuple[int, int]], token_groups)

    @staticmethod
    def _group_blocks(document: PersonDocument) -> List[List[int]]:
        """Return event-index blocks; legacy documents treat each sentence as a group."""
        group_ids = document.same_time_group_ids
        if group_ids is None:
            return [[i] for i in range(len(document.sentences))]
        blocks: List[List[int]] = []
        positions = {}
        for index, group_id in enumerate(group_ids):
            # Keep a group together even if malformed input made it non-contiguous.
            if group_id not in positions:
                positions[group_id] = len(blocks)
                blocks.append([])
            blocks[positions[group_id]].append(index)
        return blocks

    def _sop_is_eligible(
        self, document: PersonDocument, distinct_group_count: int
    ) -> bool:
        if document.same_time_group_ids is None:
            # Preserve the old SOP behavior for legacy documents.
            return True
        semantics_ok = document.order_semantics in {
            "STRICT_CHRONOLOGICAL",
            "SPARSE_CHRONOLOGICAL",
            "CHRONOLOGICAL_WITH_TIES",
        }
        declared = (
            bool(document.op_eligible)
            if document.op_eligible is not None
            else semantics_ok
        )
        return declared and semantics_ok and distinct_group_count >= 2

    def cls_task(
        self,
        document: PersonDocument,
        is_train: bool = True,
        force_label: int | None = None,
    ):
        """Apply a 3-class SOP permutation to whole SameTimeGroup blocks."""
        result = deepcopy(document)
        result.validate_event_alignment()
        blocks = self._group_blocks(result)
        eligible = self._sop_is_eligible(result, len(blocks))
        if not eligible:
            return result, np.int64(0), np.float32(0.0)

        if is_train:
            rng = np.random
        else:
            # Stable per-person seed; unlike hash(), this is process-independent.
            person_bytes = str(result.person_id).encode("utf-8")
            person_seed = int.from_bytes(person_bytes[:8].ljust(8, b"\0"), "little")
            rng = np.random.RandomState((self.evaluation_seed ^ person_seed) & 0xFFFFFFFF)

        if force_label is None:
            draw = float(rng.random())
            if draw < self.sop_reverse_probability:
                label = 1
            elif (
                draw < self.sop_reverse_probability + self.sop_shuffle_probability
                and len(blocks) >= 3
            ):
                label = 2
            else:
                label = 0
        else:
            if force_label not in (0, 1, 2):
                raise ValueError("SOP label must be 0, 1, or 2")
            label = force_label

        block_order = list(range(len(blocks)))
        if label == 1:
            block_order.reverse()
        elif label == 2:
            if len(blocks) < 3:
                # With two groups, the only non-arranged order is REVERSED.
                label = 1
                block_order.reverse()
            else:
                identity = tuple(block_order)
                reversed_order = tuple(reversed(block_order))
                for _ in range(32):
                    rng.shuffle(block_order)
                    if tuple(block_order) not in (identity, reversed_order):
                        break
                else:
                    block_order = block_order[1:2] + block_order[:1] + block_order[2:]

        event_order = [index for block in block_order for index in blocks[block]]
        result.select_events(event_order)
        result.shuffled = label != 0
        return result, np.int64(label), np.float32(1.0)

    def mlm_mask(
        self, token_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mask out the tokens for mlm training"""

        vocabulary = self.datamodule.vocabulary
        token2index = vocabulary.token2index

        unk_id = token2index["[UNK]"]
        mask_id = token2index["[MASK]"]
        sep_id = token2index["[SEP]"]

        # limit is length of an actual sequence
        n_tokens = len(token_ids)

        requested = int(np.floor(n_tokens * self.mask_ratio))

        excluded_ids = {sep_id, unk_id, token2index.get("[PAD]", 0)}
        if not self.mask_special_tokens:
            excluded_ids.update(
                token2index[token]
                for token in vocabulary.general_tokens
                if token in token2index
            )
            try:
                general = vocabulary.vocab()
                excluded_ids.update(
                    general.loc[general.CATEGORY == "GENERAL", "ID"].astype(int).tolist()
                )
            except (AttributeError, KeyError):
                # Minimal/legacy vocabulary implementations may only expose lists.
                pass
        if not self.mask_background_tokens:
            excluded_ids.update(
                token2index[token]
                for token in vocabulary.background_tokens
                if token in token2index
            )
        excluded_ids.update(
            token2index[token]
            for token in self.non_maskable_tokens
            if token in token2index
        )
        legal_mask = ~np.isin(token_ids[1:], list(excluded_ids))
        legal_indx = np.arange(start=1, stop=n_tokens)[legal_mask]
        max_masked_num = int(np.floor(self.mask_ratio * self.max_length))
        num_tokens_to_mask = min(requested, len(legal_indx), max_masked_num)
        # 80% [MASK], 10% unchanged, 10% random.
        pos_unchange = int(np.floor(num_tokens_to_mask * 0.1))
        pos_random = num_tokens_to_mask - pos_unchange

        indx_to_mask = (
            np.random.choice(a=legal_indx, size=num_tokens_to_mask, replace=False)
            if num_tokens_to_mask
            else np.asarray([], dtype=np.int64)
        ).astype(np.int64)

        # positions of the masked tokens
        y_indx = np.full(
            shape=max_masked_num,
            fill_value=int(self.max_length - 1),
            dtype=np.int64,
        )
        y_indx[: len(indx_to_mask)] = indx_to_mask.copy()

        # remember the actual tokens on positions
        y_token = np.zeros(shape=max_masked_num, dtype=np.int64)
        y_token[: len(indx_to_mask)] = token_ids[indx_to_mask].copy()

        # masked token_ids #rather change the sampling domain for accurate masking
        # ratio?
        token_ids[indx_to_mask[pos_unchange:pos_random]] = mask_id

        replacement_ids = np.array(
            sorted(set(token2index.values()) - excluded_ids), dtype=np.int64
        )

        if self.smart_masking:

            smart_edge = int(pos_random + int(pos_unchange * 0.3))

            # Random 7% of all random cases
            if len(replacement_ids):
                token_ids[indx_to_mask[smart_edge:]] = np.random.choice(
                    replacement_ids, size=len(indx_to_mask[smart_edge:])
                )

            # Smart Random 3% of all the cases
            smart_values = token_ids[indx_to_mask[pos_random:smart_edge]]

            for i, j in self.token_groups:
                smart_values = self.smart_masked(smart_values, i, j + 1)  # background

            token_ids[indx_to_mask[pos_random:smart_edge]] = smart_values

        else:
            if len(replacement_ids) and len(indx_to_mask[pos_random:]):
                token_ids[indx_to_mask[pos_random:]] = np.random.choice(
                    replacement_ids, size=len(indx_to_mask[pos_random:])
                )
        return token_ids, y_indx, y_token

    @staticmethod
    def smart_masked(x: np.ndarray, min_i: int, max_i: int) -> np.ndarray:
        """Applies the smart_masking scheme"""
        ix = np.argwhere((x >= min_i) & (x < max_i))
        if len(ix) > 0:
            x[ix] = np.random.randint(low=min_i, high=max_i, size=(len(ix), 1))
        return x


@dataclass
class MLMEncodedDocument(EncodedDocument[MLM]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    padding_mask: np.ndarray
    target_tokens: np.ndarray
    target_pos: np.ndarray
    target_cls: np.ndarray
    target_cls_mask: np.ndarray
    original_sequence: np.ndarray
