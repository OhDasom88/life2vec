#!/usr/bin/env python3
"""Offline external embedding generation in online2-embedding-build env."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/agrichallenge/online2"))
    parser.add_argument("--legacy-build", type=Path, default=Path("outputs/online2/build-v8-active80-r3"))
    parser.add_argument("--out", type=Path, default=Path("outputs/online2/v2_build/external_embeddings"))
    parser.add_argument("--text-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--image-model", default="facebook/dinov2-large")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-texts", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "text_model": args.text_model,
        "image_model": args.image_model,
        "status": "processing",
    }

    ext = pd.read_parquet(args.legacy_build / "external_embeddings.parquet")
    img_rows = ext[ext["modality"] == "I_images"].copy()
    txt_rows = ext[ext["modality"] == "interpretation"].copy()
    if args.max_images:
        img_rows = img_rows.head(args.max_images)
    if args.max_texts:
        txt_rows = txt_rows.head(args.max_texts)

    # Images: include example+problem (all public images in build)
    image_processor = AutoImageProcessor.from_pretrained(args.image_model)
    image_model = AutoModel.from_pretrained(args.image_model).to(device).eval()
    image_vecs = []
    image_meta = []
    with torch.no_grad():
        for rec in img_rows.itertuples(index=False):
            path = args.data_root / str(rec.source_ref)
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = image_model(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output[0].detach().float().cpu().numpy()
            else:
                vec = out.last_hidden_state[:, 0].detach().float().cpu().numpy()[0]
            image_vecs.append(vec)
            image_meta.append(
                {
                    "embedding_id": f"img:{rec.event_id}",
                    "event_id": rec.event_id,
                    "modality": "image",
                    "model_name": args.image_model,
                    "source_ref": rec.source_ref,
                    "source_content_hash": sha256_file(path),
                    "vector_checksum": hashlib.sha256(vec.tobytes()).hexdigest(),
                    "embedding_dim": int(vec.shape[-1]),
                    "status": "ready",
                    "dataset_role": "public_image",
                    "contains_problem_hidden_targets": False,
                }
            )

    # Texts: score90 example only (already filtered in build external_embeddings)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(args.text_model, trust_remote_code=True).to(device).eval()
    text_vecs = []
    text_meta = []
    with torch.no_grad():
        for rec in txt_rows.itertuples(index=False):
            path = args.data_root / str(rec.source_ref)
            if not path.exists():
                continue
            # Forbid problem paths defensively
            if "problem" in str(path).lower() and "example" not in str(rec.answer_tier):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = text_model(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output[0].detach().float().cpu().numpy()
            else:
                vec = out.last_hidden_state.mean(dim=1)[0].detach().float().cpu().numpy()
            text_vecs.append(vec)
            text_meta.append(
                {
                    "embedding_id": f"txt:{rec.event_id}",
                    "event_id": rec.event_id,
                    "modality": "text",
                    "model_name": args.text_model,
                    "source_ref": rec.source_ref,
                    "answer_tier": getattr(rec, "answer_tier", ""),
                    "source_content_hash": hashlib.sha256(text.encode()).hexdigest(),
                    "vector_checksum": hashlib.sha256(vec.tobytes()).hexdigest(),
                    "embedding_dim": int(vec.shape[-1]),
                    "status": "ready",
                    "dataset_role": "example_public_interpretation",
                    "contains_problem_hidden_targets": False,
                }
            )

    if image_vecs:
        arr = np.stack(image_vecs).astype(np.float32)
        np.save(args.out / "image_embeddings.npy", arr)
        try:
            from safetensors.numpy import save_file

            save_file({"embeddings": arr}, str(args.out / "image_embeddings.safetensors"))
        except Exception as exc:  # pragma: no cover
            report_note = str(exc)
        else:
            report_note = "safetensors_ok"
        pd.DataFrame(image_meta).to_parquet(args.out / "image_embeddings.parquet", index=False)
    else:
        report_note = "no_image_vectors"

    if text_vecs:
        arr = np.stack(text_vecs).astype(np.float32)
        np.save(args.out / "text_embeddings.npy", arr)
        try:
            from safetensors.numpy import save_file

            save_file({"embeddings": arr}, str(args.out / "text_embeddings.safetensors"))
        except Exception:
            pass
        pd.DataFrame(text_meta).to_parquet(args.out / "text_embeddings.parquet", index=False)

    report.update(
        {
            "status": "ready" if (image_vecs or text_vecs) else "failed",
            "n_image": len(image_vecs),
            "n_text": len(text_vecs),
            "image_dim": int(image_vecs[0].shape[-1]) if image_vecs else None,
            "text_dim": int(text_vecs[0].shape[-1]) if text_vecs else None,
            "note": report_note,
            "training_mode": "transductive_public_pretraining",
            "contains_problem_images": True,
            "contains_problem_hidden_targets": False,
        }
    )
    (args.out / "embedding_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
