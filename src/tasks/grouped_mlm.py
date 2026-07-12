"""Grouped MLM for Online2 V2 measurement groups."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import chain
from typing import List, Optional, Sequence, TypeVar, cast

import numpy as np

from src.data_new.types import Background, PersonDocument, EncodedDocument
from src.online2.v2.masking import GroupedMLMMasker
from src.online2.v2.vocab import VocabV2
from src.tasks.mlm import MLM, MLMEncodedDocument

T = TypeVar("T")


@dataclass
class GroupedMLM(MLM):
    """V2 MLM that masks measurement groups together with family-aware replacement."""

    mask_feature_identity: bool = False
    vocab_v2_path: str = ""
    measurement_group_field: str = "measurement_group_ids"
    token_role_field: str = "token_roles"

    def __post_init__(self) -> None:
        self._vocab_v2: Optional[VocabV2] = None
        self._masker: Optional[GroupedMLMMasker] = None

    def _ensure_masker(self) -> GroupedMLMMasker:
        if self._masker is None:
            if not self.vocab_v2_path:
                raise ValueError("vocab_v2_path is required for GroupedMLM")
            self._vocab_v2 = VocabV2.load(__import__("pathlib").Path(self.vocab_v2_path))
            self._masker = GroupedMLMMasker(
                self._vocab_v2,
                mask_ratio=self.mask_ratio,
                mask_feature_identity=self.mask_feature_identity,
            )
        return self._masker

    def encode_document(
        self, document: PersonDocument, is_train: bool = True
    ) -> MLMEncodedDocument:
        # Prefer sequence-level sentence (V2 export stores full sequence in first/only sentence)
        prefix_sentence = ["[CLS]"] + Background.get_sentence(document.background) + ["[SEP]"]
        document, targ_cls, target_cls_mask = self.cls_task(document, is_train=is_train)

        if len(document.sentences) == 1:
            sentences = [prefix_sentence] + [document.sentences[0] + ["[SEP]"]]
        else:
            sentences = [prefix_sentence] + [s + ["[SEP]"] for s in document.sentences]
        sentence_lengths = [len(x) for x in sentences]

        def expand(x: List[T]) -> List[T]:
            assert len(x) == len(sentence_lengths)
            return list(
                chain.from_iterable(length * [i] for length, i in zip(sentence_lengths, x))
            )

        abspos_expanded = expand([0] + (document.abspos or [0] * (len(sentences) - 1)))
        age_expanded = expand([0.0] + (document.age or [0.0] * (len(sentences) - 1)))
        assert document.segment is not None
        segment_expanded = expand([1] + document.segment)

        flat_sentences = np.concatenate(sentences)
        token2index = self.datamodule.vocabulary.token2index
        unk_id = token2index["[UNK]"]
        token_ids = np.array([token2index.get(x, unk_id) for x in flat_sentences])

        # Measurement groups / roles from task_info if present
        group_ids = ["NONE"] * len(token_ids)
        roles = ["meta"] * len(token_ids)
        info = document.task_info if isinstance(document.task_info, dict) else {}
        mg = info.get(self.measurement_group_field)
        tr = info.get(self.token_role_field)
        if mg and tr:
            mg_list = json.loads(mg) if isinstance(mg, str) else list(mg)
            role_list = json.loads(tr) if isinstance(tr, str) else list(tr)
            # Align to non-prefix tokens: prefix is CLS+background+SEP
            prefix_len = len(prefix_sentence)
            body = flat_sentences[prefix_len:]
            # body may include trailing SEP; roles cover body without final SEP maybe
            for i, _tok in enumerate(body):
                pos = prefix_len + i
                if i < len(mg_list):
                    group_ids[pos] = str(mg_list[i])
                if i < len(role_list):
                    roles[pos] = str(role_list[i])

        masker = self._ensure_masker()
        masked_sentences, masked_indx, masked_tokens, report = masker.mask(
            token_ids, group_ids, roles
        )
        # Pad targets to max_length style used by MLM
        max_masked_num = int(np.floor(self.mask_ratio * self.max_length))
        y_indx = np.full(max_masked_num, int(self.max_length - 1), dtype=np.int64)
        y_token = np.zeros(max_masked_num, dtype=np.int64)
        n = min(len(masked_indx), max_masked_num)
        y_indx[:n] = masked_indx[:n]
        y_token[:n] = masked_tokens[:n]

        length = len(token_ids)
        input_ids = np.zeros((4, self.max_length), dtype=np.float32)
        input_ids[0, :length] = masked_sentences[: self.max_length]
        input_ids[1, :length] = abspos_expanded[: self.max_length]
        input_ids[2, :length] = age_expanded[: self.max_length]
        input_ids[3, :length] = segment_expanded[: self.max_length]
        padding_mask = np.repeat(False, self.max_length)
        padding_mask[: min(length, self.max_length)] = True
        original_sequence = np.zeros(self.max_length, dtype=np.int64)
        original_sequence[: min(length, self.max_length)] = token_ids[: self.max_length]

        # stash report on document for logging
        if document.task_info is None:
            document.task_info = {}
        if isinstance(document.task_info, dict):
            document.task_info["mask_report"] = report.__dict__

        return MLMEncodedDocument(
            sequence_id=np.array(document.person_id),
            input_ids=input_ids,
            padding_mask=padding_mask,
            target_tokens=y_token,
            target_pos=y_indx,
            target_cls=targ_cls,
            target_cls_mask=target_cls_mask,
            original_sequence=original_sequence,
        )
