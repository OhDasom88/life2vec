"""시계열–서사 대조학습 pair 품질 등급과 hard negative (계획서 §7.3).

새 신호를 만들지 않는다 — 이미 이번 세션에서 구현한 ``narrative_grounding``의
결과물을 그대로 pair 품질로 승격한다:

- §5.2 어댑터(``from_online2_corpus.narrative_from_sequence_row``)로 나온
  narrative는 규칙 기반(rule-grounded) — 기본 ``SILVER_RULE_GROUNDED``.
- §5.1 검색(``text_to_window.search_text_to_window``)의 4분기 판정이 있으면
  그걸로 승격/강등한다.
- §5.3 사람 결정(``decisions.jsonl``)이 있으면 그게 최종 등급을 덮어쓴다
  (사람 판단이 항상 최우선).
"""

from __future__ import annotations

import re
from typing import Sequence

from ..narrative_grounding.schemas import Narrative

PAIR_QUALITY_TIERS = (
    "GOLD_EXPERT",
    "SILVER_RULE_GROUNDED",
    "SILVER_MODEL_GROUNDED",
    "WEAK_TEMPORAL_MATCH",
    "UNVERIFIED",
    "CONTRADICTED",
)

# 등급이 낮을수록(신뢰 낮음) 대조학습 배치에서 가중치를 줄인다. CONTRADICTED는
# 0 — 양성 pair로 아예 쓰지 않는다(호출자가 별도로 음성/제외 처리해야 함).
TIER_WEIGHTS: dict[str, float] = {
    "GOLD_EXPERT": 1.0,
    "SILVER_RULE_GROUNDED": 0.8,
    "SILVER_MODEL_GROUNDED": 0.6,
    "WEAK_TEMPORAL_MATCH": 0.3,
    "UNVERIFIED": 0.1,
    "CONTRADICTED": 0.0,
}


def pair_quality_tier(
    *,
    human_decision: str | None = None,
    grounding_search_decision: str | None = None,
    is_rule_grounded: bool = True,
) -> str:
    """세 신호 중 있는 것을 우선순위(사람 > §5.1 검색 > §5.2 규칙 기반)로 결합해 등급을 낸다.

    - ``human_decision``: decisions.jsonl의 ``ACCEPT``/``REJECT``/``SKIP``/``None``.
    - ``grounding_search_decision``: §5.1의 ``AUTO_ACCEPT_CANDIDATE``/``REVIEW``/
      ``QUARANTINE``/``REJECT``/``None``(§5.1 검색을 거치지 않은 narrative).
    - ``is_rule_grounded``: §5.2 어댑터로 만들어진 narrative는 기본 True.
    """
    if human_decision == "ACCEPT":
        return "GOLD_EXPERT"
    if human_decision == "REJECT":
        return "CONTRADICTED"

    if grounding_search_decision == "REJECT":
        return "CONTRADICTED"
    if grounding_search_decision == "AUTO_ACCEPT_CANDIDATE":
        return "SILVER_MODEL_GROUNDED"
    if grounding_search_decision in {"REVIEW", "QUARANTINE"}:
        return "WEAK_TEMPORAL_MATCH"

    if is_rule_grounded:
        return "SILVER_RULE_GROUNDED"
    return "UNVERIFIED"


def find_hard_negatives(
    anchor: Narrative, candidates: Sequence[Narrative], *, max_negatives: int = 5
) -> list[Narrative]:
    """앵커와 farm이 겹치지만 서사 템플릿(의미)이 다른 후보를 어려운 음성으로 고른다.

    무작위 negative는 farm_id 토큰만 봐도 구분되는 shortcut을 만든다(§7.3).
    같은 farm/zone 맥락에서 다른 걸 말하는 서사라야 "farm이 아니라 내용을
    보고 구분해야 하는" 진짜 어려운 음성이 된다. 같은 템플릿을 쓰는 후보는
    사실상 같은 의미일 수 있으므로 negative에서 제외한다.
    """
    if not anchor.supporting_windows:
        return []
    anchor_window = anchor.supporting_windows[0]
    anchor_farms = set(anchor_window.farm_ids)

    negatives: list[Narrative] = []
    for candidate in candidates:
        if candidate.narrative_instance_id == anchor.narrative_instance_id:
            continue
        if not candidate.supporting_windows:
            continue
        candidate_window = candidate.supporting_windows[0]
        if candidate_window.narrative_template_id == anchor_window.narrative_template_id:
            continue
        if anchor_farms & set(candidate_window.farm_ids):
            negatives.append(candidate)
        if len(negatives) >= max_negatives:
            break
    return negatives


_FARM_ID_RE = re.compile(r"F\d{5,6}")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def strip_identifying_tokens(text: str) -> str:
    """farm ID·ISO 날짜를 제거한 텍스트 — shortcut 검사(§7.3)의 전처리 단계.

    "farm ID나 날짜만으로 정답을 맞히는 shortcut을 검사"하려면 이렇게 지운
    텍스트로도 검색이 여전히 맞는 후보를 찾는지 실제 학습된 time-text
    인코더로 비교해야 한다. 그 인코더가 아직 없어(이 모듈은 손실 함수까지만
    구현했고 실제 학습은 하지 않았다) 여기서는 비교에 쓸 입력을 만드는
    전처리만 제공한다 — README "아직 없는 것" 참조.
    """
    text = _FARM_ID_RE.sub("[FARM]", text)
    text = _ISO_DATE_RE.sub("[DATE]", text)
    return text
