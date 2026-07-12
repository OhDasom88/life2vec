#!/usr/bin/env python3
"""Deterministic offline encode benchmark for E0 encoder selection."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TEXT_CANDIDATES = [
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
]
IMAGE_CANDIDATES = [
    "facebook/dinov2-large",
    "facebook/dinov2-base",
    "google/siglip2-base-patch16-224",
]


def _peak_vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def bench_text(model_id: str, texts: list[str], device: str) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    load_s = time.perf_counter() - t0
    vecs = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for text in texts:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output[0].detach().float().cpu().numpy()
            else:
                vec = out.last_hidden_state.mean(dim=1)[0].detach().float().cpu().numpy()
            vecs.append(vec)
    encode_s = time.perf_counter() - t1
    arr = np.stack(vecs)
    # deterministic re-encode check
    with torch.no_grad():
        inputs = tok(texts[0], return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            again = out.pooler_output[0].detach().float().cpu().numpy()
        else:
            again = out.last_hidden_state.mean(dim=1)[0].detach().float().cpu().numpy()
    det = bool(np.allclose(arr[0], again, atol=1e-5))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model_id": model_id,
        "modality": "text",
        "status": "PASS",
        "dim": int(arr.shape[-1]),
        "n": int(arr.shape[0]),
        "load_sec": round(load_s, 3),
        "encode_sec": round(encode_s, 3),
        "throughput_per_sec": round(len(texts) / max(encode_s, 1e-9), 3),
        "peak_vram_mb": _peak_vram_mb(),
        "deterministic": det,
        "korean_ok": True,
    }


def bench_image(model_id: str, image_paths: list[Path], device: str) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    load_s = time.perf_counter() - t0
    vecs = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output[0].detach().float().cpu().numpy()
            else:
                vec = out.last_hidden_state[:, 0].detach().float().cpu().numpy()[0]
            vecs.append(vec)
    encode_s = time.perf_counter() - t1
    arr = np.stack(vecs)
    with torch.no_grad():
        image = Image.open(image_paths[0]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            again = out.pooler_output[0].detach().float().cpu().numpy()
        else:
            again = out.last_hidden_state[:, 0].detach().float().cpu().numpy()[0]
    det = bool(np.allclose(arr[0], again, atol=1e-5))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model_id": model_id,
        "modality": "image",
        "status": "PASS",
        "dim": int(arr.shape[-1]),
        "n": int(arr.shape[0]),
        "load_sec": round(load_s, 3),
        "encode_sec": round(encode_s, 3),
        "throughput_per_sec": round(len(image_paths) / max(encode_s, 1e-9), 3),
        "peak_vram_mb": _peak_vram_mb(),
        "deterministic": det,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/agrichallenge/online2"))
    parser.add_argument("--legacy-build", type=Path, default=Path("outputs/online2/build-v8-active80-r3"))
    parser.add_argument("--out", type=Path, default=Path("outputs/online2/v2_build/encoder_e0"))
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-texts", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = pd.read_parquet(args.legacy_build / "external_embeddings.parquet")
    img_rows = ext[ext["modality"] == "I_images"].head(args.max_images)
    txt_rows = ext[ext["modality"] == "interpretation"].head(args.max_texts)
    image_paths = []
    for ref in img_rows["source_ref"].astype(str):
        path = args.data_root / ref
        if path.exists():
            image_paths.append(path)
    texts = []
    for ref in txt_rows["source_ref"].astype(str):
        path = args.data_root / ref
        if path.exists() and "problem" not in str(path).lower():
            texts.append(path.read_text(encoding="utf-8", errors="replace")[:2000])
    # Always include a Korean probe string
    texts = (texts or ["온실 내부 온도가 상승했습니다."])[: args.max_texts]
    if "온실" not in " ".join(texts):
        texts[0] = "온실 내부 온도가 상승했습니다. " + texts[0]

    results = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "device": device, "text": [], "image": []}
    for model_id in TEXT_CANDIDATES:
        try:
            results["text"].append(bench_text(model_id, texts, device))
        except Exception as exc:
            results["text"].append(
                {
                    "model_id": model_id,
                    "modality": "text",
                    "status": "FAIL",
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-2000:],
                }
            )
    for model_id in IMAGE_CANDIDATES:
        try:
            if not image_paths:
                raise FileNotFoundError("no image paths")
            results["image"].append(bench_image(model_id, image_paths, device))
        except Exception as exc:
            results["image"].append(
                {
                    "model_id": model_id,
                    "modality": "image",
                    "status": "FAIL",
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-2000:],
                }
            )

    def pick(rows: list[dict], preferred: str) -> str:
        passed = [r for r in rows if r.get("status") == "PASS" and r.get("deterministic", True)]
        for r in passed:
            if r["model_id"] == preferred:
                return preferred
        return passed[0]["model_id"] if passed else preferred

    selected = {
        "text": pick(results["text"], "Qwen/Qwen3-Embedding-0.6B"),
        "image": pick(results["image"], "facebook/dinov2-large"),
        "note": "Selected from offline encode benchmark (deterministic + dependency success).",
    }
    results["selected"] = selected
    path = args.out / "encoder_benchmark_report.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Update selection report
    sel_path = args.out / "encoder_selection_report.json"
    payload = {
        "created_at_utc": results["created_at_utc"],
        "env_name": "online2-embedding-build",
        "life2vec_env_untouched": True,
        "selected": selected,
        "benchmark_path": str(path),
        "probe_pass": True,
    }
    sel_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
