"""Grouped MLM for Online2 V2 measurement groups."""

from __future__ import annotations

import json
from copy import deepcopy
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

    def get_document(self, person_sentences) -> PersonDocument:
        document = super().get_document(person_sentences)
        info = document.task_info if isinstance(document.task_info, dict) else {}
        # Event-grain: one measurement_group_ids / token_roles JSON string per event row.
        # Flat legacy: a single row with sequence-aligned arrays (expand path only).
        for field in (self.measurement_group_field, self.token_role_field):
            if field not in person_sentences.columns:
                continue
            values = person_sentences[field].tolist()
            if len(person_sentences) > 1:
                info[f"_per_event_{field}"] = values
            elif values:
                info[field] = values[0]
        document.task_info = info or None
        return document

    @staticmethod
    def _split_tokens_on_sep(tokens: Sequence[str], sep: str = "[SEQ_SEP]") -> list[list[str]]:
        chunks: list[list[str]] = []
        cur: list[str] = []
        for tok in tokens:
            if tok == sep:
                if cur:
                    chunks.append(cur)
                    cur = []
                continue
            cur.append(tok)
        if cur:
            chunks.append(cur)
        return chunks

    def _expand_flat_v2_for_sop(self, document: PersonDocument) -> PersonDocument:
        """LEGACY ONLY — recover SOP blocks from a flat one-row-per-sequence export.

        Not the formal grain. Prefer event-level ``training_events_v2`` (1 row =
        sequence×event) so SameTimeGroup SOP runs natively via ``same_time_group_ids``.
        This helper clears STG ids and treats each ``[SEQ_SEP]`` chunk as its own
        block — it does **not** reproduce V1 SameTimeGroup block SOP.
        """
        if len(document.sentences) != 1:
            return document
        tokens = document.sentences[0]
        if "[SEQ_SEP]" not in tokens:
            return document
        chunks = self._split_tokens_on_sep(tokens, "[SEQ_SEP]")
        if len(chunks) < 2:
            return document

        n = len(chunks)
        info = document.task_info if isinstance(document.task_info, dict) else {}
        mg_raw = info.get(self.measurement_group_field)
        role_raw = info.get(self.token_role_field)
        mg_list = json.loads(mg_raw) if isinstance(mg_raw, str) else (list(mg_raw) if mg_raw else [])
        role_list = (
            json.loads(role_raw) if isinstance(role_raw, str) else (list(role_raw) if role_raw else [])
        )
        event_mgs: list[list[str]] = []
        event_roles: list[list[str]] = []
        if mg_list or role_list:
            cur_mg: list[str] = []
            cur_role: list[str] = []
            for i, tok in enumerate(tokens):
                if tok == "[SEQ_SEP]":
                    event_mgs.append(cur_mg)
                    event_roles.append(cur_role)
                    cur_mg, cur_role = [], []
                    continue
                cur_mg.append(str(mg_list[i]) if i < len(mg_list) else "NONE")
                cur_role.append(str(role_list[i]) if i < len(role_list) else "meta")
            if cur_mg or cur_role or chunks:
                if len(event_mgs) < n:
                    event_mgs.append(cur_mg)
                    event_roles.append(cur_role)
            while len(event_mgs) < n:
                event_mgs.append(["NONE"] * len(chunks[len(event_mgs)]))
                event_roles.append(["meta"] * len(chunks[len(event_roles)]))
            event_mgs = event_mgs[:n]
            event_roles = event_roles[:n]
            info["_event_measurement_group_ids"] = event_mgs
            info["_event_token_roles"] = event_roles

        document.sentences = [list(c) for c in chunks]
        # Flat recovery cannot restore real SameTimeGroup ids.
        document.same_time_group_ids = None
        if document.abspos is not None and len(document.abspos) == 1:
            base = int(document.abspos[0])
            document.abspos = [base + i for i in range(n)]
        elif document.abspos is None or len(document.abspos) != n:
            document.abspos = list(range(1, n + 1))
        if document.age is not None and len(document.age) == 1:
            document.age = [float(document.age[0])] * n
        elif document.age is None or len(document.age) != n:
            document.age = [0.0] * n
        if document.segment is not None and len(document.segment) == 1:
            document.segment = [int(document.segment[0])] * n
        elif document.segment is None or len(document.segment) != n:
            document.segment = [1] * n
        if document.event_ids is not None and len(document.event_ids) == 1:
            document.event_ids = [f"{document.event_ids[0]}#{i}" for i in range(n)]
        elif document.event_ids is not None and len(document.event_ids) != n:
            document.event_ids = [f"evt#{i}" for i in range(n)]
        if document.event_kinds is not None and len(document.event_kinds) != n:
            base_kind = document.event_kinds[0] if document.event_kinds else "SEQUENCE"
            document.event_kinds = [base_kind] * n
        if document.modality_refs is not None and len(document.modality_refs) != n:
            base_mod = document.modality_refs[0] if document.modality_refs else "[]"
            document.modality_refs = [base_mod] * n
        if document.order_semantics is None:
            document.order_semantics = "STRICT_CHRONOLOGICAL"
        if document.op_eligible is None:
            document.op_eligible = True
        info["v2_sop_expanded"] = True
        info["v2_sop_n_events"] = n
        document.task_info = info
        return document

    def _rejoin_v2_after_sop(self, document: PersonDocument) -> PersonDocument:
        """LEGACY ONLY — re-flatten after flat-export SOP expand for grouped masking."""
        info = document.task_info if isinstance(document.task_info, dict) else {}
        if not info.get("v2_sop_expanded"):
            return document
        if len(document.sentences) <= 1:
            return document
        flat: list[str] = []
        flat_mg: list[str] = []
        flat_roles: list[str] = []
        event_mgs = info.get("_event_measurement_group_ids")
        event_roles = info.get("_event_token_roles")
        for i, sent in enumerate(document.sentences):
            flat.extend(sent)
            if isinstance(event_mgs, list) and i < len(event_mgs):
                flat_mg.extend(list(event_mgs[i]))
            else:
                flat_mg.extend(["NONE"] * len(sent))
            if isinstance(event_roles, list) and i < len(event_roles):
                flat_roles.extend(list(event_roles[i]))
            else:
                flat_roles.extend(["meta"] * len(sent))
            if i < len(document.sentences) - 1:
                flat.append("[SEQ_SEP]")
                flat_mg.append("NONE")
                flat_roles.append("sep")
        document.sentences = [flat]
        document.same_time_group_ids = ["flattened"]
        document.abspos = [document.abspos[0]] if document.abspos else [1]
        document.age = [document.age[0]] if document.age else [0.0]
        document.segment = [document.segment[0]] if document.segment else [1]
        if document.event_ids is not None:
            document.event_ids = [document.event_ids[0]]
        if document.event_kinds is not None:
            document.event_kinds = [document.event_kinds[0]]
        if document.modality_refs is not None:
            document.modality_refs = [document.modality_refs[0]]
        info[self.measurement_group_field] = json.dumps(flat_mg)
        info[self.token_role_field] = json.dumps(flat_roles)
        document.task_info = info
        return document

    @staticmethod
    def _parse_event_side_array(raw: object, n_tokens: int, default: str) -> list[str]:
        if raw is None:
            return [default] * n_tokens
        if isinstance(raw, str):
            try:
                values = json.loads(raw)
            except json.JSONDecodeError:
                return [default] * n_tokens
        else:
            values = list(raw)
        out = [str(values[i]) if i < len(values) else default for i in range(n_tokens)]
        return out

    def _flatten_event_grain_side_channels(
        self, document: PersonDocument, sentences: list[list[str]], prefix_len: int
    ) -> tuple[list[str], list[str]]:
        """Build token-aligned mg/roles for multi-event documents (native grain)."""
        info = document.task_info if isinstance(document.task_info, dict) else {}
        per_mg = info.get(f"_per_event_{self.measurement_group_field}")
        per_role = info.get(f"_per_event_{self.token_role_field}")
        # sentences[0] is prefix; body events are sentences[1:]
        body = sentences[1:]
        flat_mg: list[str] = ["NONE"] * prefix_len
        flat_roles: list[str] = ["meta"] * prefix_len
        n_events = len(document.sentences)
        for i, sent_with_sep in enumerate(body):
            # each body sentence is event_tokens + [SEP]
            event_len = max(len(sent_with_sep) - 1, 0)
            mg_raw = per_mg[i] if isinstance(per_mg, list) and i < len(per_mg) else None
            role_raw = (
                per_role[i] if isinstance(per_role, list) and i < len(per_role) else None
            )
            flat_mg.extend(self._parse_event_side_array(mg_raw, event_len, "NONE"))
            flat_roles.extend(self._parse_event_side_array(role_raw, event_len, "meta"))
            flat_mg.append("NONE")
            flat_roles.append("sep")
            if i >= n_events:
                break
        return flat_mg, flat_roles

    def clip_document(self, document: PersonDocument) -> PersonDocument:
        """Truncate long V2 sequence sentences instead of dropping all events."""
        sep_size = 0 if self.no_sep else 1
        prefix_length = len(Background.get_sentence(document.background)) + 1 + sep_size
        max_sequence_length = self.max_length - prefix_length
        if max_sequence_length <= 1:
            return document

        # Legacy flat single-sentence export.
        if len(document.sentences) == 1 and len(document.sentences[0]) > max_sequence_length:
            keep = max_sequence_length - sep_size
            keep = max(keep, 1)
            document.sentences = [document.sentences[0][:keep]]
            if isinstance(document.task_info, dict):
                for field in (self.measurement_group_field, self.token_role_field):
                    raw = document.task_info.get(field)
                    if isinstance(raw, str):
                        try:
                            values = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        document.task_info[field] = json.dumps(values[:keep])
                    elif isinstance(raw, list):
                        document.task_info[field] = raw[:keep]
            if document.abspos:
                document.abspos = document.abspos[:1]
            if document.age:
                document.age = document.age[:1]
            if document.segment:
                document.segment = document.segment[:1]
            if document.event_ids:
                document.event_ids = document.event_ids[:1]
            return document

        # Event grain: drop whole events from the left until the document fits
        # (SameTimeGroup SOP needs ≥2 groups when possible — prefer keeping recent tail).
        n_before = len(document.sentences)
        document = super().clip_document(document)
        n_after = len(document.sentences)
        if n_after < n_before and isinstance(document.task_info, dict):
            drop = n_before - n_after
            for field in (self.measurement_group_field, self.token_role_field):
                key = f"_per_event_{field}"
                values = document.task_info.get(key)
                if isinstance(values, list) and len(values) == n_before:
                    document.task_info[key] = values[drop:]
        return document

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

    def cls_task(
        self,
        document: PersonDocument,
        is_train: bool = True,
        force_label: int | None = None,
    ):
        document = self._expand_flat_v2_for_sop(document)
        info = document.task_info if isinstance(document.task_info, dict) else {}
        event_mgs = info.get("_event_measurement_group_ids")
        event_roles = info.get("_event_token_roles")
        per_mg = info.get(f"_per_event_{self.measurement_group_field}")
        per_role = info.get(f"_per_event_{self.token_role_field}")

        # Shared SOP permutation; side channels differ by path.
        result = deepcopy(document)
        result.validate_event_alignment()
        blocks = self._group_blocks(result)
        eligible = self._sop_is_eligible(result, len(blocks))
        if not eligible:
            return result, np.int64(0), np.float32(0.0)

        if is_train:
            rng = np.random
        else:
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
        rinfo = result.task_info if isinstance(result.task_info, dict) else {}
        if info.get("v2_sop_expanded"):
            # Legacy flat-expand: event-as-block (not real SameTimeGroup).
            if isinstance(event_mgs, list) and len(event_mgs) == len(event_order):
                rinfo["_event_measurement_group_ids"] = [event_mgs[i] for i in event_order]
            if isinstance(event_roles, list) and len(event_roles) == len(event_order):
                rinfo["_event_token_roles"] = [event_roles[i] for i in event_order]
        else:
            # Native event grain: SameTimeGroup blocks + per-event side channels.
            if isinstance(per_mg, list) and len(per_mg) == len(event_order):
                rinfo[f"_per_event_{self.measurement_group_field}"] = [
                    per_mg[i] for i in event_order
                ]
            if isinstance(per_role, list) and len(per_role) == len(event_order):
                rinfo[f"_per_event_{self.token_role_field}"] = [
                    per_role[i] for i in event_order
                ]
        result.task_info = rinfo
        result.shuffled = label != 0
        return result, np.int64(label), np.float32(1.0)

    def encode_document(
        self, document: PersonDocument, is_train: bool = True
    ) -> MLMEncodedDocument:
        prefix_sentence = ["[CLS]"] + Background.get_sentence(document.background) + ["[SEP]"]
        document, targ_cls, target_cls_mask = self.cls_task(document, is_train=is_train)
        document = self._rejoin_v2_after_sop(document)

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

        group_ids = ["NONE"] * len(token_ids)
        roles = ["meta"] * len(token_ids)
        info = document.task_info if isinstance(document.task_info, dict) else {}

        if len(document.sentences) > 1 or info.get(f"_per_event_{self.measurement_group_field}"):
            # Native event grain (or multi-sentence without rejoin).
            flat_mg, flat_roles = self._flatten_event_grain_side_channels(
                document, sentences, len(prefix_sentence)
            )
            for i in range(min(len(token_ids), len(flat_mg))):
                group_ids[i] = flat_mg[i]
                roles[i] = flat_roles[i] if i < len(flat_roles) else "meta"
        else:
            mg = info.get(self.measurement_group_field)
            tr = info.get(self.token_role_field)
            if mg and tr:
                mg_list = json.loads(mg) if isinstance(mg, str) else list(mg)
                role_list = json.loads(tr) if isinstance(tr, str) else list(tr)
                prefix_len = len(prefix_sentence)
                body = flat_sentences[prefix_len:]
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

        if document.task_info is None:
            document.task_info = {}
        if isinstance(document.task_info, dict):
            document.task_info["mask_report"] = report.__dict__
            document.task_info["sop_path"] = (
                "legacy_flat_expand" if info.get("v2_sop_expanded") else "native_sametimegroup"
            )

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
