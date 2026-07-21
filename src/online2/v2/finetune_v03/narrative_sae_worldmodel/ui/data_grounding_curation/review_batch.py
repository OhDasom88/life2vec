"""§5.3 검토 큐 UI의 순수 로직 계층 (Streamlit 의존성 없음, 단독 테스트 가능).

``app.py``는 이 모듈을 얇게 감싸는 렌더링 레이어일 뿐이다. 배치 샘플링,
review queue 구성, 결정(decision) 영속화는 전부 여기 있다 — UI 프레임워크를
바꾸더라도(Streamlit -> 다른 것) 이 파일은 그대로 재사용한다.

결정은 append-only JSONL로 기록한다(§15 "UI 화면 상태는 정본이 아니다, 모든
변경은 versioned artifact와 immutable transaction으로 저장한다"). 같은
narrative를 다시 검토해 새 결정을 남기면 기존 줄을 덮어쓰지 않고 새 줄을
추가한다 — ``load_decisions``는 그중 마지막 줄을 "현재 상태"로 취급한다.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.contradictions import (
    augment_with_contradictions,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.from_online2_corpus import (
    load_catalog,
    narrative_from_sequence_row,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.review_queue import (
    ReviewContext,
    ReviewQueueEntry,
    build_review_queue,
)

_FARM_TOKEN_RE = re.compile(r"FARM\|(F\d+)")

DECISIONS = frozenset({"ACCEPT", "REJECT", "SKIP"})


def load_corpus(build_dir: Path) -> tuple[dict[str, dict[str, str]], "pq.Table"]:
    catalog = load_catalog(build_dir / "normalized_catalog.csv")
    table = pq.read_table(build_dir / "sequences.parquet")
    return catalog, table


def distinct_farms(table: "pq.Table") -> list[str]:
    farms = set()
    for token_json in table.column("background_tokens").to_pylist():
        match = _FARM_TOKEN_RE.search(token_json)
        if match:
            farms.add(match.group(1))
    return sorted(farms)


def rows_for_farm(table: "pq.Table", farm_id: str) -> list[dict[str, Any]]:
    mask = pc.match_substring(table.column("background_tokens"), f"FARM|{farm_id}")
    return table.filter(mask).to_pylist()


def sample_per_template(
    rows: list[dict[str, Any]], per_template: int, seed: int
) -> list[dict[str, Any]]:
    """narrative_id(템플릿)별로 최대 ``per_template``개만 무작위 추출.

    16,573개(예: farm F130230)를 전부 큐에 올리면 검토가 불가능하므로, 배치
    크기를 템플릿 단위로 제한한다.
    """
    by_template: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_template.setdefault(str(row["narrative_id"]), []).append(row)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for group in by_template.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:per_template])
    return sampled


def build_batch_queue(
    farm_rows: list[dict[str, Any]],
    catalog: dict[str, dict[str, str]],
    *,
    per_template: int,
    seed: int,
    low_frequency_threshold: int = 50,
) -> tuple[list[ReviewQueueEntry], dict[str, int]]:
    """farm 단위 원시 행 -> (검토 큐, 배치 통계).

    반례 탐지(``augment_with_contradictions``)는 같은 farm의 전체 행을
    candidate pool로 쓴다 — 샘플링된 소수만 후보로 쓰면 실제로 존재하는
    반례를 놓칠 수 있기 때문에, 샘플링은 "무엇을 검토 큐에 올릴지"에만
    적용하고 "반례가 있는지 찾을 때"는 farm 전체를 본다.
    """
    template_frequency: dict[str, int] = {}
    for row in farm_rows:
        narrative_id = str(row["narrative_id"])
        template_frequency[narrative_id] = template_frequency.get(narrative_id, 0) + 1

    sampled_rows = sample_per_template(farm_rows, per_template, seed)
    items = []
    for row in sampled_rows:
        narrative = narrative_from_sequence_row(row, catalog)
        narrative = augment_with_contradictions(narrative, catalog, farm_rows)
        items.append((narrative, ReviewContext()))

    queue = build_review_queue(
        items, template_frequency=template_frequency, low_frequency_threshold=low_frequency_threshold
    )
    stats = {
        "farm_row_count": len(farm_rows),
        "sampled_count": len(sampled_rows),
        "queue_length": len(queue),
    }
    return queue, stats


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    """narrative_instance_id -> 마지막(가장 최근) 결정 레코드."""
    if not path.exists():
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            decisions[record["narrative_instance_id"]] = record
    return decisions


def make_decision_record(
    entry: ReviewQueueEntry, decision: str, reviewer: str
) -> dict[str, Any]:
    if decision not in DECISIONS:
        raise ValueError(f"invalid decision: {decision!r}")
    return {
        "narrative_instance_id": entry.narrative.narrative_instance_id,
        "narrative_template_id": (
            entry.narrative.supporting_windows[0].narrative_template_id
            if entry.narrative.supporting_windows
            else None
        ),
        "decision": decision,
        "reviewer": reviewer or "anonymous",
        "reasons_at_review_time": [asdict(reason) for reason in entry.reasons],
        "priority_score": entry.priority_score,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def append_decision(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
