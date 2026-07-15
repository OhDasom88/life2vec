#!/usr/bin/env python3
"""Build High/Medium/Low × state/cause interpretation embedding banks (v0.2).

Sources:
  score90 → high
  score70 → medium
  score50 → low

Embeddings: character/word TF-IDF + TruncatedSVD → L2-normalized vectors
(no transformers dependency; suitable for semantic cosine alignment heads).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v02.semantic_bank import (  # noqa: E402
    mask_diagnosis_in_cause,
    split_state_cause,
)
from src.online2.v2.finetune_v02.version import DEFAULT_OUTPUT_ROOT  # noqa: E402

TIER_TO_QUALITY = {"score90": "high", "score70": "medium", "score50": "low"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
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
        "--out",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_OUTPUT_ROOT}/interpretation_banks/state_cause_bank.parquet",
    )
    p.add_argument("--dim", type=int, default=192, help="Used only for --embedder tfidf")
    p.add_argument(
        "--embedder",
        choices=["tfidf", "none"],
        default="tfidf",
        help="tfidf=local placeholder; then run embed_interpretation_bank_qwen.py for Qwen targets",
    )
    p.add_argument("--mask-diagnosis", action="store_true", default=True)
    p.add_argument("--no-mask-diagnosis", dest="mask_diagnosis", action="store_false")
    return p.parse_args()


def _strip_management(text: str) -> str:
    out = text
    for marker in ("[관리 제안]", "[관리제안]", "관리 제안"):
        i = out.find(marker)
        if i >= 0:
            out = out[:i].strip()
    return out


def load_tier_text(answers_root: Path, farm_id: str, tier: str) -> Optional[str]:
    path = answers_root / tier / f"{farm_id}_answer.txt"
    if not path.exists():
        matches = list((answers_root / tier).glob(f"{farm_id}*answer*.txt"))
        if not matches:
            return None
        path = matches[0]
    return path.read_text(encoding="utf-8", errors="replace")


def collect_rows(labels: pd.DataFrame, answers_root: Path, *, mask: bool) -> Tuple[List[dict], List[str]]:
    diag_names = set(labels["diagnosis_normalized"].astype(str).tolist())
    if "diagnosis_raw" in labels.columns:
        diag_names.update(labels["diagnosis_raw"].astype(str).tolist())
    # also space-collapsed
    diag_names.update({re.sub(r"\s+", "", d) for d in list(diag_names)})

    rows: List[dict] = []
    missing: List[str] = []
    for _, row in labels.iterrows():
        case_id = str(row["case_id"])
        farm = str(row["farm_id"]) if "farm_id" in row else case_id.split("_")[0]
        for tier, quality in TIER_TO_QUALITY.items():
            text = load_tier_text(answers_root, farm, tier)
            if text is None:
                missing.append(f"{case_id}:{tier}")
                continue
            state, cause = split_state_cause(text)
            state = _strip_management(state)
            cause = _strip_management(cause)
            if mask:
                cause = mask_diagnosis_in_cause(cause, diag_names)
            if state.strip():
                rows.append(
                    {
                        "case_id": case_id,
                        "farm_id": farm,
                        "quality": quality,
                        "tier": tier,
                        "axis": "state",
                        "text": state.strip(),
                    }
                )
            if cause.strip():
                rows.append(
                    {
                        "case_id": case_id,
                        "farm_id": farm,
                        "quality": quality,
                        "tier": tier,
                        "axis": "cause",
                        "text": cause.strip(),
                    }
                )
    return rows, missing


def fit_embed(texts: Sequence[str], dim: int) -> Tuple[np.ndarray, dict]:
    # char n-grams + word tokens work without Korean morphological analyzer
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
    )
    x = vectorizer.fit_transform(list(texts))
    n_comp = min(int(dim), max(2, x.shape[0] - 1), x.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=2023)
    z = svd.fit_transform(x).astype(np.float32)
    # pad if SVD rank < requested dim
    if z.shape[1] < dim:
        pad = np.zeros((z.shape[0], dim - z.shape[1]), dtype=np.float32)
        z = np.concatenate([z, pad], axis=1)
    # L2 normalize
    norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    z = z / norms
    meta = {
        "method": "tfidf_charwb_2_4_truncated",
        "tfidf_features": int(len(vectorizer.vocabulary_)),
        "svd_components": int(n_comp),
        "explained_variance_sum": float(svd.explained_variance_ratio_.sum()),
        "dim": int(dim),
    }
    return z, meta


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels)
    rows, missing = collect_rows(labels, args.answers_root, mask=bool(args.mask_diagnosis))
    if not rows:
        raise SystemExit("no interpretation rows collected")

    texts = [r["text"] for r in rows]
    if args.embedder == "tfidf":
        embs, emb_meta = fit_embed(texts, int(args.dim))
        for i, r in enumerate(rows):
            r["embedding"] = embs[i].tolist()
    else:
        emb_meta = {"method": "none", "note": "run embed_interpretation_bank_qwen.py next"}
        for r in rows:
            r["embedding"] = [0.0] * int(args.dim)

    out = args.out or (
        args.root / DEFAULT_OUTPUT_ROOT / "interpretation_banks" / "state_cause_bank.parquet"
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)

    meta = {
        "out": str(out),
        "n_rows": int(len(df)),
        "n_cases": int(df["case_id"].nunique()),
        "counts": df.groupby(["quality", "axis"]).size().astype(int).to_dict(),
        "missing": missing,
        "mask_diagnosis_in_cause": bool(args.mask_diagnosis),
        "embedding": emb_meta,
    }
    # json-serialize tuple keys from groupby
    meta["counts"] = {
        f"{q}/{a}": int(n)
        for (q, a), n in df.groupby(["quality", "axis"]).size().items()
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(meta, flush=True)


if __name__ == "__main__":
    main()
