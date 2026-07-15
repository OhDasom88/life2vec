"""Gemma multimodal report scaffolding (finetune v0.2).

One shared Gemma — not per-fold. Generation is intentionally separate from
the diagnosis ensemble so semantic/report failures cannot alter F1 gates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


REPORT_SCHEMA = """
[상태진단]
- 예측 진단 / 확률 / 모델 합의
- 현재 상태에 대한 핵심 설명 (consensus state evidence만)

[핵심 관찰 근거]
- 환경·근권·제어·생육
- 이미지
- 시점 / zone

[원인분석]
- 직접 관찰 요인
- 시간적 연결
- 가능한 원인 vs 보조 원인

[반대 근거 및 불확실성]
- saliency 불일치
- 근거 부족
- 이미지 OOD
""".strip()


GEMMA_CONSTRAINTS = [
    "consensus evidence에 없는 사실을 만들지 말 것",
    "saliency를 인과 확정 증거로 단정하지 말 것",
    "5개 모델 중 4개 미만 합의 요인을 핵심 근거로 쓰지 말 것",
    "원인과 단순 동시 발생을 구분할 것",
    "이미지만으로 진단을 확정하지 말 것",
    "확률·합의·반대 근거를 함께 표시할 것",
]


@dataclass
class GemmaReportInput:
    case_id: str
    diagnosis: str
    probabilities: Dict[str, float]
    model_agreement: int
    structured_evidence: Dict[str, Any]
    retrieved_state_text: Optional[str] = None
    retrieved_cause_text: Optional[str] = None
    image_paths: Optional[List[str]] = None


def build_gemma_prompt(inp: GemmaReportInput) -> str:
    return (
        "당신은 스마트팜 진단 보고서 작성기입니다.\n"
        f"제약:\n- " + "\n- ".join(GEMMA_CONSTRAINTS) + "\n\n"
        f"출력 스키마:\n{REPORT_SCHEMA}\n\n"
        f"케이스: {inp.case_id}\n"
        f"앙상블 진단: {inp.diagnosis}\n"
        f"모델 합의(top-1): {inp.model_agreement}/5\n"
        f"클래스 확률: {json.dumps(inp.probabilities, ensure_ascii=False)}\n"
        f"구조화 근거 JSON:\n{json.dumps(inp.structured_evidence, ensure_ascii=False, indent=2)}\n"
        f"고품질 상태진단 예시:\n{inp.retrieved_state_text or '(없음)'}\n"
        f"고품질 원인분석 예시:\n{inp.retrieved_cause_text or '(없음)'}\n"
    )


def write_gemma_inputs(path: Path, inputs: List[GemmaReportInput]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for inp in inputs:
            rec = {
                "case_id": inp.case_id,
                "prompt": build_gemma_prompt(inp),
                "image_paths": inp.image_paths or [],
                "structured_evidence": inp.structured_evidence,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate_report_stub(inp: GemmaReportInput) -> str:
    """Offline placeholder until Gemma multimodal runtime is wired."""
    return (
        f"[상태진단]\n- 예측 진단: {inp.diagnosis}\n"
        f"- 모델 합의: {inp.model_agreement}/5\n"
        f"- (stub) consensus state evidence {len(inp.structured_evidence.get('state_evidence', []))}개\n\n"
        f"[원인분석]\n- (stub) consensus cause evidence "
        f"{len(inp.structured_evidence.get('cause_evidence', []))}개\n\n"
        f"[반대 근거 및 불확실성]\n- (stub) disagreement "
        f"{len(inp.structured_evidence.get('disagreement', []))}개\n"
    )
