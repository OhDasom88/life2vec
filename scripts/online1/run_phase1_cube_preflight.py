#!/usr/bin/env python3
"""Phase 1: Cube band image conversion + frozen encoder smoke + cost estimate."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/dasom/life2vec")
sys.path.insert(0, str(ROOT))

from src.online1.paths import Online1Paths, load_pipeline_config, load_yaml, CONFIG_DIR, write_json
from src.online1.cube_io import load_envi_cube, parse_cube_sample, validate_cube_array
from src.online1.cube_bands import fit_global_percentiles, cube_to_band_images
from src.online1.cube_embeddings import FrozenImageEncoder, pool_band_embeddings, save_embedding_npz
from src.online1.report import write_phase_report, write_markdown_summary
from src.online1.inventory import inventory_cubes


def main(n_fit: int = 20, n_embed: int = 30):
    cfg = load_pipeline_config()
    paths = Online1Paths.from_cfg(cfg)
    out = paths.output_root / "phase1"
    emb_dir = out / "embeddings_sample"
    out.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    contract = load_yaml(CONFIG_DIR / "band_input_contract.yaml")["band_input_contract"]
    inv = inventory_cubes(paths.train_ms)
    records = inv["records"]
    if not records:
        raise SystemExit("No cubes found")

    # Validate first cube thoroughly
    first = Path(records[0]["path"])
    cube, meta = load_envi_cube(first / "cube.hdr")
    sample = parse_cube_sample(first)
    val = validate_cube_array(cube, contract)
    assert val["pass"], val

    # Fit percentiles on n_fit cubes (scope=train inductive sample)
    def cube_iter():
        for r in records[:n_fit]:
            c, _ = load_envi_cube(Path(r["path"]) / "cube.hdr")
            yield c

    t_fit0 = time.time()
    percentiles = fit_global_percentiles(
        cube_iter(),
        low=float(contract["percentile_low"]),
        high=float(contract["percentile_high"]),
        dead_value=int(contract.get("dead_pixel_value", 0)),
        n_bands=10,
    )
    t_fit = time.time() - t_fit0
    write_json(out / "percentiles_train_sample.json", percentiles)

    encoder = FrozenImageEncoder()
    sha_before = encoder.weight_sha

    # Process n_embed cubes, strided across the full record list so the sample spans the
    # whole DAT range instead of only the first few DATs in file order (records[:n_embed]
    # previously only covered DAT109-111, leaving Phase2's later-DAT C1 sequences with no
    # matching cube embeddings at all — a silent coverage gap, not a real preflight sample).
    stride = max(1, len(records) // n_embed)
    sample_records = records[::stride][:n_embed]
    cosines = []
    invalid_max = 0.0
    t0 = time.time()
    for r in sample_records:
        p = Path(r["path"])
        c, _ = load_envi_cube(p / "cube.hdr")
        images, stats, meta_img = cube_to_band_images(c, percentiles, contract)
        invalid_max = max(invalid_max, meta_img["max_invalid_ratio"])
        embs = encoder.encode_bands(images)
        pooled = pool_band_embeddings(embs, mode="mean")
        # reproducibility: re-encode first band
        embs2 = encoder.encode_rgb01(images[0])
        cos = float(
            np.dot(embs[0], embs2)
            / (np.linalg.norm(embs[0]) * np.linalg.norm(embs2) + 1e-8)
        )
        cosines.append(cos)
        save_embedding_npz(
            emb_dir / f"{r['cube_id']}.npz",
            r["cube_id"],
            embs,
            pooled,
            {"stats": stats, "meta": meta_img, "sample": parse_cube_sample(p)},
        )
    dt = time.time() - t0
    sha_after = encoder.verify_frozen()

    n_total = inv["n_valid"] + inventory_cubes(paths.test_ms)["n_valid"]
    sec_per = dt / max(n_embed, 1)
    est_hours = (n_total * sec_per) / 3600.0
    # storage estimate: 10*512 float32 + pooled 512 ≈ 11*512*4 bytes
    bytes_per = 11 * 512 * 4
    est = {
        "n_cubes_total": n_total,
        "n_band_images": n_total * 10,
        "sample_processed": n_embed,
        "sec_per_cube": sec_per,
        "est_image_forward_gpu_hours": est_hours,
        "est_embedding_store_bytes": n_total * bytes_per,
        "est_embedding_store_gib": (n_total * bytes_per) / (1024**3),
        "percentile_fit_seconds": t_fit,
        "note": "Full store not created; awaiting Phase1 approval gate",
    }
    write_json(out / "cost_estimate.json", est)

    min_cos = float(np.min(cosines)) if cosines else 0.0
    phase1_pass = (
        val["pass"]
        and not sha_after["frozen_weight_sha_changed"]
        and min_cos >= 0.9999
        and inv["n_malformed"] == 0
    )

    report = write_phase_report(
        out,
        "phase1",
        {
            "pass": phase1_pass,
            "first_cube": {"sample": sample, "validate": val, "wavelength": meta.get("wavelength")},
            "encoder": encoder.meta,
            "sha_before": sha_before,
            "sha_after": sha_after,
            "embedding_reproducibility_min_cosine": min_cos,
            "invalid_pixel_ratio_max_observed": invalid_max,
            "cost_estimate": est,
            "n_embeddings_written": len(sample_records),
        },
    )
    write_markdown_summary(
        out / "PHASE1_SUMMARY.md",
        "Online1 Phase 1 Cube Preflight",
        [
            ("Verdict", f"PASS={phase1_pass}"),
            ("Encoder", f"{encoder.meta['name']} frozen sha={sha_before[:12]}..."),
            ("Reproducibility", f"min_cosine={min_cos}"),
            ("Cost", f"est_gpu_hours={est_hours:.2f}, emb_store_GiB={est['est_embedding_store_gib']:.3f}"),
        ],
    )
    print("PHASE1", phase1_pass, report)


if __name__ == "__main__":
    main()
