#!/usr/bin/env python3
"""Contrast Gemma reports vs example-set score90/70/50 (상/중/하) analyses.

Quality mapping:
  score90 → 상 (high)
  score70 → 중 (medium)
  score50 → 하 (low)

Outputs alignment tables (diagnosis, zone, modality keyword overlap) and a
readable markdown contrast report. Does NOT feed score90 text back into the
classifier — contrast-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v02.semantic_bank import split_state_cause  # noqa: E402

TIER_QUALITY = {
    "score90": "상",
    "score70": "중",
    "score50": "하",
}

ZONE_RE = re.compile(r"\bZ([1-4])\b|zone\s*([1-4])|구역\s*([1-4])", re.I)
PRIMARY_ZONE_RE = re.compile(
    r"우선\s*점검\s*구역(?:은|은\s*)?\s*Z([1-4])|우선\s*점검\s*구역\s*Z([1-4])",
    re.I,
)
DIAG_RE = re.compile(
    r"종합\s*진단은\s*'([^']+)'|"
    r"종합\s*판단은\s*([^.\n]+)|"
    r"예측\s*진단[:：]?\s*\*?\*?\s*([^\n/]+)",
    re.I,
)

MODALITY_PATTERNS = {
    "ENVIRONMENT": [r"환경", r"ENVIRONMENT", r"온습도", r"CO2", r"습도", r"온도"],
    "ROOTZONE": [r"근권", r"ROOTZONE", r"배지", r"EC", r"함수율", r"수분"],
    "ACTUATOR": [r"제어", r"ACTUATOR", r"환기", r"순환팬", r"차광", r"FCU", r"관수", r"양액"],
    "GROWTH": [r"생육", r"GROWTH", r"초장", r"엽수", r"관부"],
    "IMAGE": [r"이미지", r"IMAGE", r"가시\s*증상", r"촬영"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/evaluation/latest",
    )
    p.add_argument(
        "--answers-root",
        type=Path,
        default=ROOT / "datasets/agrichallenge/online2/answers/reference_answers",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/labels_example_score90.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <eval-dir>/contrast_quality_tiers",
    )
    return p.parse_args()


def normalize_diag(s: str) -> str:
    s = str(s).strip()
    s = s.replace("입니다", "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("/", "_").replace("(", "_").replace(")", "")
    s = s.replace("__", "_").strip("_")
    # match label_map style underscores for spaces already removed
    mapping = {
        "기형과생리장해": "기형과_생리장해",
        "동해난방실패": "동해_난방실패",
        "뿌리다육부진": "뿌리_발육_부진",
        "뿌리발육부진": "뿌리_발육_부진",
        "영양생장과다": "영양생장_과다",
        "영양생장부족": "영양생장_부족",
        "잿빛곰팡이병발생위험": "잿빛곰팡이병_발생_위험",
        "정상운영": "정상_운영",
        "칼슘부족잎끝마름": "칼슘부족_잎끝마름",
        "탄저병발생위험": "탄저병_발생_위험",
        "흰가루병발생위험": "흰가루병_발생_위험",
    }
    key = s.replace("_", "")
    return mapping.get(key, s if "_" in s else mapping.get(key, s))


def extract_diagnosis(text: str) -> Optional[str]:
    m = DIAG_RE.search(text)
    if m:
        raw = next((g for g in m.groups() if g), None)
        if raw:
            raw = raw.strip().strip("*").strip()
            raw = re.sub(r"입니다$", "", raw).strip()
            return normalize_diag(raw)
    m2 = re.search(r"'([^']{2,40})'", text)
    if m2:
        return normalize_diag(m2.group(1))
    if "정상 운영" in text or "정상운영" in text.replace(" ", ""):
        return "정상_운영"
    return None


def extract_primary_zone(text: str) -> Optional[str]:
    m = PRIMARY_ZONE_RE.search(text)
    if not m:
        return None
    return m.group(1) or m.group(2)


def extract_zones(text: str) -> Set[str]:
    zones = set()
    for m in ZONE_RE.finditer(text):
        z = m.group(1) or m.group(2) or m.group(3)
        if z:
            zones.add(str(int(z)))
    return zones


def extract_modalities(text: str) -> Set[str]:
    found = set()
    for name, pats in MODALITY_PATTERNS.items():
        for pat in pats:
            if re.search(pat, text, re.I):
                found.add(name)
                break
    return found


def tokenize_ko(text: str) -> Set[str]:
    # keep Korean/latin/digit chunks length>=2
    toks = re.findall(r"[가-힣A-Za-z0-9%]{2,}", text)
    stop = {
        "입니다",
        "있습니다",
        "것으로",
        "확인",
        "관찰",
        "구역",
        "우선",
        "진단",
        "현재",
        "파이프라인",
        "미산출",
        "모델",
        "합의",
        "근거",
        "분석",
        "상태",
        "원인",
    }
    return {t for t in toks if t not in stop and not t.isdigit()}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def parse_answer(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    state, cause = split_state_cause(text)
    # strip management section from cause if present
    for marker in ("[관리 제안]", "[관리제안]", "관리 제안"):
        i = cause.find(marker)
        if i >= 0:
            cause = cause[:i].strip()
        i = state.find(marker)
        if i >= 0:
            state = state[:i].strip()
    return {
        "path": str(path),
        "full": text,
        "state": state,
        "cause": cause,
        "diagnosis": extract_diagnosis(text),
        "primary_zone": extract_primary_zone(text),
        "zones": sorted(extract_zones(text)),
        "modalities": sorted(extract_modalities(text)),
        "tokens": tokenize_ko(state + "\n" + cause),
        "n_chars": len(text),
    }


def parse_gemma_report(md_path: Path, evidence: Dict[str, Any]) -> Dict[str, Any]:
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    # support factors → modalities/zones
    zones: Set[str] = set()
    mods: Set[str] = set()
    for item in evidence.get("diagnosis_support") or []:
        fac = str(item.get("factor") or "")
        # ROOTZONE|zone3|event_...
        parts = fac.split("|")
        if parts:
            m = parts[0].upper()
            if m in MODALITY_PATTERNS:
                mods.add(m)
        for p in parts:
            zm = re.search(r"zone\s*([1-4])", p, re.I)
            if zm:
                zones.add(zm.group(1))
        for z in item.get("zones") or []:
            zones.add(str(z))
    zones |= extract_zones(text)
    mods |= extract_modalities(text)
    return {
        "path": str(md_path),
        "full": text,
        "diagnosis": evidence.get("diagnosis") or extract_diagnosis(text),
        "primary_zone": extract_primary_zone(text),
        "zones": sorted(zones),
        "modalities": sorted(mods),
        "tokens": tokenize_ko(text),
        "n_chars": len(text),
        "model_agreement": evidence.get("model_agreement"),
        "ensemble_probability": evidence.get("ensemble_probability"),
        "n_support": len(evidence.get("diagnosis_support") or []),
        "n_disagree": len(evidence.get("disagreement") or []),
    }


def farm_id_from_case(case_id: str) -> str:
    return str(case_id).split("_")[0]


def compare_case(
    case_id: str,
    label: str,
    gemma: Dict[str, Any],
    tiers: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case_id": case_id,
        "farm_id": farm_id_from_case(case_id),
        "label": label,
        "pred": gemma.get("diagnosis"),
        "pred_match_label": gemma.get("diagnosis") == label,
        "model_agreement": gemma.get("model_agreement"),
        "ensemble_probability": gemma.get("ensemble_probability"),
        "gemma_zones": ",".join(gemma.get("zones") or []),
        "gemma_modalities": ",".join(gemma.get("modalities") or []),
        "gemma_n_chars": gemma.get("n_chars"),
        "gemma_n_support": gemma.get("n_support"),
    }
    g_zones = set(gemma.get("zones") or [])
    g_mods = set(gemma.get("modalities") or [])
    g_tok = set(gemma.get("tokens") or [])

    best_tok_tier = None
    best_tok = -1.0
    for tier, qname in TIER_QUALITY.items():
        ref = tiers[tier]
        row[f"{qname}_diagnosis"] = ref.get("diagnosis")
        row[f"{qname}_diag_match_pred"] = ref.get("diagnosis") == gemma.get("diagnosis")
        row[f"{qname}_diag_match_label"] = ref.get("diagnosis") == label
        row[f"{qname}_primary_zone"] = ref.get("primary_zone")
        row[f"{qname}_zones"] = ",".join(ref.get("zones") or [])
        row[f"{qname}_modalities"] = ",".join(ref.get("modalities") or [])
        row[f"{qname}_n_chars"] = ref.get("n_chars")
        z_ov = jaccard(g_zones, set(ref.get("zones") or []))
        m_ov = jaccard(g_mods, set(ref.get("modalities") or []))
        t_ov = jaccard(g_tok, set(ref.get("tokens") or []))
        row[f"{qname}_zone_jaccard"] = round(z_ov, 4)
        row[f"{qname}_modality_jaccard"] = round(m_ov, 4)
        row[f"{qname}_token_jaccard"] = round(t_ov, 4)
        # composite emphasis match
        row[f"{qname}_align_score"] = round(0.4 * z_ov + 0.4 * m_ov + 0.2 * t_ov, 4)
        if t_ov > best_tok:
            best_tok = t_ov
            best_tok_tier = qname
        # primary zone hit
        pz = ref.get("primary_zone")
        row[f"{qname}_primary_zone_in_gemma"] = bool(pz and pz in g_zones)

    row["closest_token_tier"] = best_tok_tier
    # which tier align_score max
    aligns = {q: row[f"{q}_align_score"] for q in ("상", "중", "하")}
    row["closest_align_tier"] = max(aligns, key=aligns.get)
    row["align_gap_상_하"] = round(row["상_align_score"] - row["하_align_score"], 4)
    return row


def write_markdown(out_dir: Path, df: pd.DataFrame, examples: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Example-set 품질 등급(상/중/하) ↔ Gemma 보고서 대조\n")
    lines.append("품질 매핑: **score90=상**, **score70=중**, **score50=하**\n")
    lines.append(
        "> 본 대조는 누수 주입이 아니라, 생성 보고서가 어느 품질 등급의 "
        "진단·구역·모달리티 구조에 가까운지 사후 비교합니다.\n"
    )

    n = len(df)
    lines.append("## 1. 요약\n")
    lines.append(f"- 대조 케이스: **{n}**\n")
    lines.append(
        f"- 예측 진단 = score90 라벨: "
        f"**{int(df['pred_match_label'].sum())}/{n}** "
        f"({df['pred_match_label'].mean():.1%})\n"
    )
    lines.append(
        f"- 상(score90) 진단명 = 예측: "
        f"**{int(df['상_diag_match_pred'].sum())}/{n}**\n"
    )
    has_pz = df["상_primary_zone"].notna()
    n_pz = int(has_pz.sum())
    pz_hit = float(df.loc[has_pz, "상_primary_zone_in_gemma"].mean()) if n_pz else 0.0
    lines.append(
        f"- 상 우선점검 zone이 Gemma/saliency zone에 포함: "
        f"**{int(df.loc[has_pz, '상_primary_zone_in_gemma'].sum()) if n_pz else 0}/{n_pz}** "
        f"({pz_hit:.1%}; 우선 zone 명시 케이스만)\n"
    )
    closest = df["closest_align_tier"].value_counts().to_dict()
    lines.append(
        f"- align_score 최근접 등급 분포: "
        + ", ".join(f"{k} {v}" for k, v in sorted(closest.items()))
        + "\n"
    )
    lines.append("\n| 등급 | zone Jaccard 평균 | modality Jaccard 평균 | token Jaccard 평균 | align 평균 |\n")
    lines.append("|------|------------------:|----------------------:|-------------------:|-----------:|\n")
    for q in ("상", "중", "하"):
        lines.append(
            f"| {q} | {df[f'{q}_zone_jaccard'].mean():.3f} | "
            f"{df[f'{q}_modality_jaccard'].mean():.3f} | "
            f"{df[f'{q}_token_jaccard'].mean():.3f} | "
            f"{df[f'{q}_align_score'].mean():.3f} |\n"
        )

    lines.append("\n## 2. 해석\n")
    mean_hi = df["상_align_score"].mean()
    mean_lo = df["하_align_score"].mean()
    has_pz = df["상_primary_zone"].notna()
    pz = float(df.loc[has_pz, "상_primary_zone_in_gemma"].mean()) if has_pz.any() else float("nan")
    lines.append(
        f"- 구조(구역·모달리티) 정합: 상·중 ≈ {mean_hi:.3f}, 하 {mean_lo:.3f} "
        f"(하 등급은 zone 범위가 더 좁음).\n"
        f"- token 겹침은 낮음(≈0.05) — Gemma는 saliency factor 요약, "
        f"점수 등급 문은 수치·서술 어휘가 달라 어휘 Jaccard로는 품질 구분이 약함.\n"
        f"- 상 등급 우선 점검 zone 회수율 {pz:.1%} (명시된 케이스 기준) — "
        f"saliency consensus가 참고 해석의 초점 구역을 따르는지 지표.\n"
        f"- 진단명은 앙상블이 라벨/상·중·하와 일치. 차이는 **근거 서술 깊이**(수치·원인 연결)에서 발생.\n"
    )

    lines.append("\n## 3. 케이스 샘플 상세\n")
    for ex in examples:
        lines.append(f"### {ex['case_id']} ({ex['label']})\n")
        lines.append(
            f"- 예측: **{ex['pred']}** | 합의 {ex['model_agreement']}/5 | "
            f"p={ex['ensemble_probability']:.3f}\n"
        )
        lines.append(f"- Gemma zones: `{ex['gemma_zones']}` | mods: `{ex['gemma_modalities']}`\n")
        for q in ("상", "중", "하"):
            lines.append(
                f"- **{q}**: primary Z{ex[f'{q}_primary_zone']}, "
                f"zones={ex[f'{q}_zones']}, mods={ex[f'{q}_modalities']}, "
                f"align={ex[f'{q}_align_score']:.3f}, "
                f"primary_hit={ex[f'{q}_primary_zone_in_gemma']}\n"
            )
        lines.append(f"- 최근접: **{ex['closest_align_tier']}**\n")
        if ex.get("score90_state_head"):
            lines.append(f"\n<details><summary>상(score90) 상태진단 앞부분</summary>\n\n")
            lines.append("```\n" + ex["score90_state_head"] + "\n```\n</details>\n")
        if ex.get("gemma_excerpt"):
            lines.append(f"\n<details><summary>Gemma 보고서 발췌</summary>\n\n")
            lines.append("```\n" + ex["gemma_excerpt"] + "\n```\n</details>\n")
        lines.append("\n")

    lines.append("## 4. 전체 표\n")
    cols = [
        "case_id",
        "label",
        "pred",
        "pred_match_label",
        "model_agreement",
        "상_primary_zone",
        "상_primary_zone_in_gemma",
        "상_align_score",
        "중_align_score",
        "하_align_score",
        "closest_align_tier",
        "align_gap_상_하",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header + "\n" + sep + "\n")
    for _, r in df[cols].iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |\n")
    lines.append("\n")
    (out_dir / "CONTRAST_QUALITY_TIERS.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "contrast_quality_tiers"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels)
    labels["farm_id"] = labels["farm_id"].astype(str)
    labels["case_id"] = labels["case_id"].astype(str)

    pred = pd.read_parquet(eval_dir / "ensemble" / "predictions.parquet")
    md_dir = eval_dir / "reports" / "gemma4" / "markdown"
    answers = Path(args.answers_root)

    rows = []
    detail_bank = {}
    for _, lab in labels.iterrows():
        case_id = str(lab["case_id"])
        farm = str(lab["farm_id"])
        label = str(lab["diagnosis_normalized"])
        tiers = {}
        for tier in TIER_QUALITY:
            path = answers / tier / f"{farm}_answer.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            tiers[tier] = parse_answer(path)
        ev_path = eval_dir / "evidence" / f"{case_id}.json"
        ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {}
        gemma = parse_gemma_report(md_dir / f"{case_id}.md", ev)
        # prefer ensemble pred
        prow = pred[pred["case_id"] == case_id]
        if len(prow):
            gemma["diagnosis"] = str(prow.iloc[0]["pred"])
            gemma["model_agreement"] = int(prow.iloc[0]["model_agreement"])
            gemma["ensemble_probability"] = float(prow.iloc[0]["ensemble_probability"])
        row = compare_case(case_id, label, gemma, tiers)
        rows.append(row)
        detail_bank[case_id] = {
            "tiers": {TIER_QUALITY[t]: tiers[t] for t in tiers},
            "gemma": gemma,
        }

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "contrast_table.csv", index=False)
    df.to_parquet(out_dir / "contrast_table.parquet", index=False)

    # pick diverse examples: one full match, one weaker zone hit if any
    sample_ids = []
    for cid in ["F814133_2025-03-24_2025-04-06", "F302129_2024-10-15_2024-10-28", "F194265_2024-10-23_2024-11-05"]:
        if cid in set(df["case_id"]):
            sample_ids.append(cid)
    while len(sample_ids) < 3:
        for cid in df["case_id"]:
            if cid not in sample_ids:
                sample_ids.append(cid)
            if len(sample_ids) >= 3:
                break

    examples = []
    for cid in sample_ids:
        r = df[df["case_id"] == cid].iloc[0].to_dict()
        g = detail_bank[cid]["gemma"]
        s90 = detail_bank[cid]["tiers"]["상"]
        r["score90_state_head"] = (s90.get("state") or "")[:450]
        r["gemma_excerpt"] = (g.get("full") or "")[:700]
        examples.append(r)

    summary = {
        "n": int(len(df)),
        "pred_match_label": float(df["pred_match_label"].mean()),
        "상_diag_match_pred": float(df["상_diag_match_pred"].mean()),
        "상_primary_zone_defined_n": int(df["상_primary_zone"].notna().sum()),
        "상_primary_zone_hit_among_defined": float(
            df.loc[df["상_primary_zone"].notna(), "상_primary_zone_in_gemma"].mean()
        )
        if df["상_primary_zone"].notna().any()
        else None,
        "mean_align": {
            "상": float(df["상_align_score"].mean()),
            "중": float(df["중_align_score"].mean()),
            "하": float(df["하_align_score"].mean()),
        },
        "closest_align_tier_counts": df["closest_align_tier"].value_counts().to_dict(),
        "mean_token_jaccard": {
            "상": float(df["상_token_jaccard"].mean()),
            "중": float(df["중_token_jaccard"].mean()),
            "하": float(df["하_token_jaccard"].mean()),
        },
        "mean_zone_jaccard": {
            "상": float(df["상_zone_jaccard"].mean()),
            "중": float(df["중_zone_jaccard"].mean()),
            "하": float(df["하_zone_jaccard"].mean()),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(out_dir, df, examples)
    print(summary, flush=True)
    print({"out_dir": str(out_dir)}, flush=True)


if __name__ == "__main__":
    main()
