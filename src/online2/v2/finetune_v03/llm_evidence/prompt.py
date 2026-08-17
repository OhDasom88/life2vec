"""Build the final-diagnosis prompt from a case's evidence bundle + nearest
labeled examples. Same anti-fabrication discipline as
`scripts/online2_v2/v03/generate_cf1s_case_analysis_v03.py::SYSTEM_PROMPT`
(the only other LLM prompt in this repo) -- ground strictly in the given
numbers/text, never invent unobserved sensor/actuator/growth content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

SYSTEM_PROMPT = (
    "당신은 온실 스마트팜 시계열 이상탐지 모델의 예측 결과를 검토하는 애널리스트입니다. "
    "아래 제공된 실측 근거(모델 확률, saliency로 뽑힌 이벤트, 유사 학습 사례)만 근거로 "
    "최종 진단을 판단하세요. 주어지지 않은 센서/구동기/생육/이미지 내용은 절대 지어내지 마세요. "
    "이미지 증거는 원시 이미지가 아니라 '이미지 임베딩이 융합되었는지 여부'만 제공됩니다 — "
    "이미지 내용 자체를 서술하지 마세요.\n\n"
    "출력은 반드시 아래 형식을 따르세요.\n\n"
    "[최종 진단]\n"
    "(주어진 진단 후보 목록 중 하나를 정확히 그대로 선택해서 쓰세요)\n\n"
    "[근거 요약]\n"
    "(모델 확률, saliency 이벤트, 유사 사례 중 무엇이 이 판단을 뒷받침하는지 3~5문장)\n\n"
    "[불확실성]\n"
    "(모델 확률이 애매하거나 saliency/유사사례가 서로 다른 방향을 가리키면 명시, 1~3문장)"
)


def format_evidence_block(evidence: Dict[str, Any]) -> str:
    lines: List[str] = [
        f"case_id: {evidence['case_id']}",
        f"n_events: {evidence['n_events']}",
        f"model_p_abnormal: {evidence['p_abnormal']:.3f}",
        "model_fine_probs: "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(evidence["fine_probs"].items(), key=lambda kv: -kv[1])),
        f"image_evidence: {evidence['image_evidence_note']}",
        "",
        "가장 비정상 위험을 높인 saliency 상위 이벤트:",
    ]
    for row in evidence["top_abnormal_events"]:
        sent = (row["sentence"] or "")[:200]
        lines.append(
            f"  - t={row['case_age_hours']:.1f}h score={row['saliency_score']:.4f} sentence={sent}"
        )
    if evidence["top_normal_events"]:
        lines.append("")
        lines.append("가장 정상 방향으로 기여한 이벤트:")
        for row in evidence["top_normal_events"]:
            sent = (row["sentence"] or "")[:200]
            lines.append(
                f"  - t={row['case_age_hours']:.1f}h score={row['saliency_score']:.4f} sentence={sent}"
            )
    return "\n".join(lines)


def format_nearest_examples_block(nearest: Sequence[Dict[str, Any]]) -> str:
    if not nearest:
        return "유사 학습 사례: 없음"
    lines = ["유사도 순 학습 사례 (코사인 유사도, 표현 공간):"]
    for ex in nearest:
        lines.append(f"  - case_id={ex['case_id']} diagnosis={ex['diagnosis']} sim={ex['cosine_similarity']:.3f}")
    return "\n".join(lines)


def build_diagnosis_prompt(
    evidence: Dict[str, Any],
    nearest_examples: Sequence[Dict[str, Any]],
    label_names: Sequence[str],
) -> Tuple[str, str]:
    user = "\n\n".join(
        [
            "진단 후보 목록: " + ", ".join(label_names),
            format_evidence_block(evidence),
            format_nearest_examples_block(nearest_examples),
        ]
    )
    return SYSTEM_PROMPT, user
