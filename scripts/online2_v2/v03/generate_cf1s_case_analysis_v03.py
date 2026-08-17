#!/usr/bin/env python3
"""Generate a Gemma4 analysis document per CF1S case from real governance-
verified counterfactual findings (outputs/cf1s_core/ALL55_CASE_FINDINGS.json).
Each output has a 400-character summary plus a 상태진단/원인분석/관리제안 body,
grounded strictly in the CF1S multi-event-edit test result for that case —
no fabricated environment/actuator/growth/image content."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT = (
    "당신은 온실 스마트팜 시계열 이상탐지 모델에 대한 반사실적(counterfactual) "
    "다중 이벤트 편집 인과검증 결과를 설명하는 애널리스트입니다. "
    "주어진 실측 데이터(disposition, 임계값, fold별 delta risk 등)만 근거로 사용하고, "
    "주어지지 않은 환경/구동기/생육/이미지 데이터는 절대 지어내지 마세요. "
    "출력은 반드시 아래 형식을 따르고, 반드시 [관리 제안]까지 끝까지 작성하세요. "
    "각 섹션은 3~5문장 이내로 간결하게 쓰세요.\n\n"
    "[요약]\n"
    "(400자 이내의 한국어 요약문 — 최대한 400자에 가깝게 채우되 넘기지 마세요)\n\n"
    "[상태 진단]\n"
    "(이 케이스에서 어떤 다중 이벤트 편집이 테스트되었고 결과가 무엇이었는지, 3~5문장)\n\n"
    "[원인 분석]\n"
    "(왜 그런 결과가 나왔는지 — 임계값 대비 delta, fold 간 일치/불일치, 구성 가능 여부 등 실측 근거로 설명, 3~5문장)\n\n"
    "[관리 제안]\n"
    "(이 결과를 바탕으로 다음에 무엇을 점검/시도할지 제안, 2~4문장. 이 섹션을 반드시 포함하세요.)"
)


def _describe_finding(f: Dict[str, Any]) -> str:
    lines = [
        f"cohort: {f['cohort']}",
        f"case_id: {f['case_id']}",
        f"disposition: {f['disposition']}",
        f"construction_status: {f['construction_status']}",
        f"scientific_status: {f['scientific_status']}",
    ]
    if f.get("candidate_id"):
        lines.append(f"tested_candidate: {f['candidate_id']}")
    if f.get("event_ids"):
        lines.append(f"edited_event_ids: {f['event_ids']}")
    if f.get("candidate_disposition"):
        lines.append(f"candidate_disposition: {f['candidate_disposition']}")
    if f.get("search_delta_by_fold"):
        lines.append(f"search_delta_by_fold: {f['search_delta_by_fold']}")
    if f.get("reeval_delta_by_fold"):
        lines.append(f"reeval_delta_by_fold(holdout fold2): {f['reeval_delta_by_fold']}")
    if f.get("control_result"):
        lines.append(f"control_result: {f['control_result']}")
    if f.get("locked_threshold") is not None:
        lines.append(f"locked_min_effect_abs_delta: {f['locked_threshold']}")
    return "\n".join(lines)


def _http_chat(api_base: str, model: str, system: str, user: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 2200,
        "stream": False,
    }
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--findings",
        type=Path,
        default=ROOT / "outputs/cf1s_core/ALL55_CASE_FINDINGS.json",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/cf1s_core/reports/gemma4_case_analysis",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    findings = json.loads(args.findings.read_text(encoding="utf-8"))
    if args.limit:
        findings = findings[: args.limit]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    failures = []
    for i, f in enumerate(findings):
        case_id = f["case_id"]
        out_path = out_dir / f"{case_id}_analysis.txt"
        if out_path.exists():
            print(f"[{i+1}/{len(findings)}] SKIP existing {case_id}")
            n_ok += 1
            continue
        user_prompt = _describe_finding(f)
        text = None
        last_err = None
        for attempt in range(args.retries):
            try:
                text = _http_chat(
                    args.api_base, args.model, SYSTEM_PROMPT, user_prompt, args.timeout
                )
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = str(exc)
                time.sleep(2)
        if text is None:
            print(f"[{i+1}/{len(findings)}] FAIL {case_id}: {last_err}")
            failures.append({"case_id": case_id, "error": last_err})
            n_fail += 1
            continue
        out_path.write_text(text, encoding="utf-8")
        print(f"[{i+1}/{len(findings)}] OK {case_id} ({len(text)} chars)")
        n_ok += 1

    summary = {
        "status": "DONE",
        "n_requested": len(findings),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "failures": failures,
        "out_dir": str(out_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
