"""서사→시계열 검색 (계획서 §5.1).

967,012개 sequence 인스턴스를 전부 임베딩하는 대신, 80개 narrative 템플릿
텍스트(narrative_name_ko/purpose/window_definition/threshold_or_rule/
agronomic_interpretation 등, ``normalized_catalog.csv``)를 1회만 임베딩해
"어떤 템플릿을 말하는 질의인가"를 랭킹하고(Time-Text 임베딩 단계), farm_id/zone
등 질의에서 뽑아낸 구조적 힌트로 해당 템플릿의 실제 인스턴스(``sequences.parquet``
행)를 좁힌다(ontology 규칙 단계). 두 단계를 결합해 §5.1이 요구하는
``AUTO_ACCEPT_CANDIDATE|REVIEW|QUARANTINE|REJECT`` 분기를 만든다.

텍스트 임베딩 모델은 새로 고르지 않는다. ``scripts/online2_v2/generate_external_embeddings.py``가
이미 ``Qwen/Qwen3-Embedding-0.6B``를 online2 텍스트 임베딩 모델로 채택했으므로
``Qwen3EmbeddingProvider``가 동일 모델·동일 pooling 규칙(pooler_output 우선,
없으면 last_hidden_state mean)을 재사용한다. torch/transformers는 실제 임베딩을
계산할 때만 지연 import한다 — 이 모듈을 그냥 읽거나 테스트(가짜 provider 사용)할
때는 GPU/모델 다운로드가 필요 없다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from .schemas import DataWindow, Narrative
from .from_online2_corpus import narrative_from_sequence_row

GROUNDING_DECISIONS = frozenset(
    {"AUTO_ACCEPT_CANDIDATE", "REVIEW", "QUARANTINE", "REJECT"}
)

_FARM_ID_RE = re.compile(r"F\d{5,6}")
_ZONE_RE = re.compile(r"(?:zone|구역)\s*[:#]?\s*(\d+)", re.IGNORECASE)

_TEMPLATE_TEXT_FIELDS = (
    "narrative_name_ko",
    "purpose",
    "window_definition",
    "start_condition",
    "end_condition",
    "threshold_or_rule",
    "expected_pattern",
    "agronomic_interpretation",
)

# 1차 임계값 — 실측 라벨(§5.4 평가) 없이 정한 튜닝 전 수치. review_queue/evaluation
# 단계에서 실제 데이터로 재보정하기 전까지는 placeholder로 취급한다.
_ACCEPT_THRESHOLD = 0.75
_REVIEW_THRESHOLD = 0.5
_QUARANTINE_THRESHOLD = 0.3
_TEMPLATE_SIMILARITY_WEIGHT = 0.7
_STRUCTURED_SCORE_WEIGHT = 0.3


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """텍스트 배치를 (n, dim) float32 벡터로 인코딩한다."""
        ...


class Qwen3EmbeddingProvider:
    """``scripts/online2_v2/generate_external_embeddings.py``와 동일한 모델·pooling.

    torch/transformers는 ``embed`` 최초 호출 시에만 로드한다.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(device).eval()
        self._device = device

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._ensure_loaded()
        vectors = []
        with torch.no_grad():
            for text in texts:
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                out = self._model(**inputs)
                if getattr(out, "pooler_output", None) is not None:
                    vec = out.pooler_output[0].detach().float().cpu().numpy()
                else:
                    vec = out.last_hidden_state.mean(dim=1)[0].detach().float().cpu().numpy()
                vectors.append(vec)
        return np.stack(vectors).astype(np.float32)


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-12)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return matrix_norm @ query_norm


def _template_text(row: Mapping[str, str]) -> str:
    parts = [row.get(field, "") for field in _TEMPLATE_TEXT_FIELDS]
    return " / ".join(part for part in parts if part)


class TemplateEmbeddingIndex:
    """80개 narrative 템플릿(``normalized_catalog.csv``)의 1회성 텍스트 임베딩 인덱스."""

    def __init__(self, catalog: Mapping[str, Mapping[str, str]], provider: EmbeddingProvider):
        self._narrative_ids = list(catalog.keys())
        texts = [_template_text(catalog[nid]) for nid in self._narrative_ids]
        self._vectors = provider.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)

    def rank(self, query_vector: np.ndarray) -> list[tuple[str, float]]:
        """유사도 내림차순 (narrative_id, cosine_similarity) 목록."""
        if not self._narrative_ids:
            return []
        similarities = _cosine_similarity(query_vector, self._vectors)
        order = np.argsort(-similarities)
        return [(self._narrative_ids[i], float(similarities[i])) for i in order]


@dataclass(frozen=True)
class StructuredHints:
    """질의 텍스트에서 뽑아낸 ontology 규칙 신호(farm/zone)."""

    farm_ids: tuple[str, ...]
    zone_ids: tuple[str, ...]


def extract_structured_hints(query: str) -> StructuredHints:
    farm_ids = tuple(sorted(set(_FARM_ID_RE.findall(query))))
    zone_ids = tuple(sorted({m.group(1) for m in _ZONE_RE.finditer(query)}))
    return StructuredHints(farm_ids=farm_ids, zone_ids=zone_ids)


def _structured_score(window: DataWindow, hints: StructuredHints) -> float:
    """질의가 명시한 farm/zone과 window의 실제 farm/zone이 겹치는 비율.

    질의가 farm/zone을 전혀 언급하지 않았으면(순수 의미 질의) 0으로 벌점을 주지
    않고 중립값 0.5를 준다 — structured 신호가 없다는 것과 틀렸다는 것은 다르다.
    """
    dims = 0
    matched = 0
    if hints.farm_ids:
        dims += 1
        if set(hints.farm_ids) & set(window.farm_ids):
            matched += 1
    if hints.zone_ids:
        dims += 1
        if set(hints.zone_ids) & set(window.zone_ids):
            matched += 1
    return (matched / dims) if dims else 0.5


def _classify(combined_score: float, structured_score: float) -> tuple[str, str]:
    if combined_score >= _ACCEPT_THRESHOLD and structured_score >= 0.5:
        return "AUTO_ACCEPT_CANDIDATE", f"combined={combined_score:.3f}, structured={structured_score:.3f}"
    if combined_score >= _REVIEW_THRESHOLD:
        return "REVIEW", f"combined={combined_score:.3f} in [{_REVIEW_THRESHOLD}, {_ACCEPT_THRESHOLD})"
    if combined_score >= _QUARANTINE_THRESHOLD:
        return "QUARANTINE", f"combined={combined_score:.3f} in [{_QUARANTINE_THRESHOLD}, {_REVIEW_THRESHOLD})"
    return "REJECT", f"combined={combined_score:.3f} < {_QUARANTINE_THRESHOLD}"


@dataclass(frozen=True)
class GroundingCandidate:
    narrative: Narrative
    template_similarity: float
    structured_score: float
    combined_score: float
    decision: str
    decision_reason: str

    def __post_init__(self) -> None:
        if self.decision not in GROUNDING_DECISIONS:
            raise ValueError(f"invalid grounding decision: {self.decision!r}")


def search_text_to_window(
    query: str,
    *,
    catalog: Mapping[str, Mapping[str, str]],
    template_index: TemplateEmbeddingIndex,
    provider: EmbeddingProvider,
    corpus_rows: Iterable[Mapping[str, object]],
    top_k_templates: int = 3,
    max_windows_per_template: int = 20,
) -> list[GroundingCandidate]:
    """임의 서사 질의 -> 근거 window 후보 목록(결합 점수 내림차순).

    ``corpus_rows``는 ``sequences.parquet``에서 미리 걸러낸(예: 특정 farm으로
    필터링한) 행 iterable을 받는다 — 967K행 전체를 매 질의마다 스캔하는 설계가
    아니다. 호출자가 인덱싱/파티셔닝 전략을 정한다(예: farm_id 기준 사전 분할).
    """
    hints = extract_structured_hints(query)
    query_vector = provider.embed([query])[0]
    ranked_templates = template_index.rank(query_vector)[:top_k_templates]
    top_ids = {narrative_id for narrative_id, _ in ranked_templates}
    similarity_by_id = dict(ranked_templates)

    per_template_count: dict[str, int] = {}
    candidates: list[GroundingCandidate] = []
    for row in corpus_rows:
        narrative_id = str(row["narrative_id"])
        if narrative_id not in top_ids:
            continue
        if per_template_count.get(narrative_id, 0) >= max_windows_per_template:
            continue
        per_template_count[narrative_id] = per_template_count.get(narrative_id, 0) + 1

        narrative = narrative_from_sequence_row(row, catalog)
        window = narrative.supporting_windows[0]
        # cosine 유사도를 [-1, 1] -> [0, 1]로 정규화해 structured_score와 같은 척도로 결합한다.
        similarity_unit = (similarity_by_id[narrative_id] + 1.0) / 2.0
        structured = _structured_score(window, hints)
        combined = (
            _TEMPLATE_SIMILARITY_WEIGHT * similarity_unit + _STRUCTURED_SCORE_WEIGHT * structured
        )
        decision, reason = _classify(combined, structured)
        candidates.append(
            GroundingCandidate(
                narrative=narrative,
                template_similarity=similarity_by_id[narrative_id],
                structured_score=structured,
                combined_score=combined,
                decision=decision,
                decision_reason=reason,
            )
        )

    candidates.sort(key=lambda candidate: candidate.combined_score, reverse=True)
    return candidates
