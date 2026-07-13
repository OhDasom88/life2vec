#!/usr/bin/env python3
"""Memory-conscious V2 retokenization + validation after adaptive binning.

Phases:
  1) smoke  — small max_events retokenize into *_smoke artifacts
  2) validate — OOV / narrative leak / threshold token presence
  3) full   — full retokenize into main sequences/training parquet (streaming)
  4) sanity — optional short pretrain on smoke parquet

Avoids loading old multi-GB sequence parquet into RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/online2/v2_build"
DEFAULT_LEGACY = ROOT / "outputs/online2/build-v8-active80-r3"


def rss_gb() -> float:
    # Linux: ru_maxrss is KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return -1.0
    return -1.0


def log(msg: str) -> None:
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] {msg} "
        f"(rss_max={rss_gb():.2f}Gi avail={mem_available_gb():.1f}Gi)",
        flush=True,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def ensure_freeze(out: Path) -> dict[str, Any]:
    vocab = json.loads((out / "vocab_v2.json").read_text())
    binning = json.loads((out / "binning_registry_v2_transductive.json").read_text())
    meta = {
        "vocab_size": len(vocab["tokens"]),
        "binning_policy_version": binning["meta"].get("policy_version"),
        "binning_edge_policy": binning["meta"].get("edge_policy"),
        "binning_hash": binning["meta"].get("registry_hash"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if meta["vocab_size"] < 1000:
        raise SystemExit(f"vocab looks stale/small: {meta['vocab_size']}")
    if meta["binning_policy_version"] != "v2.1_adaptive":
        raise SystemExit(f"unexpected binning policy: {meta['binning_policy_version']}")
    write_json(out / "retokenize_freeze.json", meta)
    return meta


def run_build(out: Path, legacy: Path, *, max_events: int, skip_registries: bool) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/online2_v2/build_v2.py"),
        "--out",
        str(out),
        "--legacy-build",
        str(legacy),
        "--max-events",
        str(max_events),
    ]
    if skip_registries:
        cmd.append("--skip-registries")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    log(f"exec: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def validate_events(events_path: Path, vocab_path: Path, out_report: Path, sample_sentences: int = 2000) -> dict[str, Any]:
    import pandas as pd

    vocab = {t["token"] for t in json.loads(vocab_path.read_text())["tokens"]}
    unk = "[UNK]"
    pf = pq.ParquetFile(events_path)
    n = 0
    oov = Counter()
    narrative_hits = 0
    category_hits = 0
    disease_hits = 0
    thr_tokens = Counter()
    anchors = {
        "INSIDE_HUMIDITY_PCT|ABS_": 0,
        "INSIDE_TEMP_C|ABS_": 0,
        "SUBSTRATE_EC_DS_M|ABS_": 0,
        "LINE_FLOW_RATE|": 0,
    }
    scanned = 0
    for rg in range(pf.metadata.num_row_groups):
        cols = ["SENTENCE", "token_ids"] if "token_ids" in pf.schema_arrow.names else ["SENTENCE"]
        # only needed columns
        names = [c for c in cols if c in {f.name for f in pf.schema_arrow}]
        part = pf.read_row_group(rg, columns=names).to_pandas()
        for sent in part["SENTENCE"].astype(str).tolist():
            if scanned >= sample_sentences and n > 0:
                break
            toks = sent.split()
            n += len(toks)
            scanned += 1
            for t in toks:
                if t not in vocab:
                    oov[t] += 1
                if t.startswith("NARRATIVE|"):
                    narrative_hits += 1
                if t.startswith("CATEGORY|"):
                    category_hits += 1
                if t.startswith("DISEASE|"):
                    disease_hits += 1
                for prefix in list(anchors):
                    if t.startswith(prefix):
                        anchors[prefix] += 1
                if "ABS_B" in t and any(x in t for x in ("HUMIDITY", "TEMP_C", "EC_DS")):
                    thr_tokens[t] += 1
        if scanned >= sample_sentences:
            break
        del part
        gc.collect()

    top_high = [t for t, _ in thr_tokens.most_common(30) if any(x in t for x in ("B3", "B4", "B2"))]
    report = {
        "events_path": str(events_path),
        "sentences_scanned": scanned,
        "tokens_scanned": n,
        "oov_unique": len(oov),
        "oov_top20": oov.most_common(20),
        "narrative_token_hits": narrative_hits,
        "category_token_hits": category_hits,
        "disease_token_hits": disease_hits,
        "feature_prefix_counts": anchors,
        "high_bin_samples": top_high[:20],
        "pass": len(oov) == 0 and narrative_hits == 0 and category_hits == 0 and disease_hits == 0,
        "rss_max_gb": rss_gb(),
    }
    write_json(out_report, report)
    return report


def make_smoke_training(out: Path, sequences_path: Path, limit: int = 200) -> Path:
    import pandas as pd

    pf = pq.ParquetFile(sequences_path)
    frames = []
    need = limit
    for rg in range(pf.metadata.num_row_groups):
        if need <= 0:
            break
        part = pf.read_row_group(rg).to_pandas().head(need)
        frames.append(part)
        need -= len(part)
        del part
    seq = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Convert to training_events-like rows (1 row / sequence)
    rows = []
    for rec in seq.itertuples(index=False):
        farm_ids = json.loads(rec.farm_ids) if getattr(rec, "farm_ids", None) else []
        stgs = json.loads(rec.same_time_group_ids) if getattr(rec, "same_time_group_ids", None) else []
        event_ids = json.loads(rec.event_ids) if getattr(rec, "event_ids", None) else [""]
        rows.append(
            {
                "PERSON_ID": rec.PERSON_ID,
                "sequence_id": rec.sequence_id,
                "event_id": event_ids[0] if event_ids else "",
                "event_position": 0,
                "time_group_rank": 0,
                "same_time_group_id": stgs[0] if stgs else "",
                "START_DATE": pd.Timestamp("2024-01-01"),
                "AGE": 0.0,
                "SENTENCE": rec.SENTENCE,
                "event_kind": "SEQUENCE",
                "narrative_id": rec.narrative_id,
                "order_semantics": "STRICT_CHRONOLOGICAL",
                "op_eligible": True,
                "modality_ref": "[]",
                "SEGMENT": 1,
                "farm_id": farm_ids[0] if farm_ids else "",
                "zone_id": "",
                "BACKGROUND_TOKENS": "[]",
                "build_id": "v2_transductive",
                "registry_version": "v2",
                "embedding_status": "pending",
                "sampling_weight": getattr(rec, "sampling_weight", 1.0),
                "shadow_split": getattr(rec, "shadow_split", "train"),
                "canonical_context_id": getattr(rec, "canonical_context_id", ""),
                "split_group_id": getattr(rec, "split_group_id", ""),
                "measurement_group_ids": getattr(rec, "measurement_group_ids", "[]"),
                "token_roles": getattr(rec, "token_roles", "[]"),
                "training_mode": "transductive_public_pretraining",
                "contains_problem_observations": True,
                "contains_problem_images": True,
                "contains_problem_hidden_targets": False,
            }
        )
    smoke = out / "training_events_v2_smoke.parquet"
    pd.DataFrame(rows).to_parquet(smoke, index=False)
    log(f"wrote smoke training rows={len(rows)} -> {smoke}")
    return smoke


def neo4j_dry_run(out: Path) -> dict[str, Any]:
    """Validate Neo4j connectivity and current V2 node counts without writing sequences."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    import os
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI") or f"bolt://{os.environ.get('NEO4J_HOST')}:{os.environ.get('NEO4J_BOLT', '7687')}"
    db = os.environ.get("NEO4J_DATABASE") or os.environ.get("NEO4j_DB") or os.environ.get("NEO4J_DB")
    user = os.environ["NEO4J_USER"]
    pwd = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    report: dict[str, Any] = {"uri": uri, "database": db, "writes": False}
    with driver.session(database=db) as s:
        report["sequence_schema"] = s.run(
            "MATCH (x:Sequence) RETURN x.schema_version AS sv, count(*) AS n ORDER BY n DESC"
        ).data()
        for label in [
            "VocabularyV2",
            "BinningRegistryV2",
            "TokenizationResultV2",
            "MaterializationRunV2",
            "ExternalEmbeddingV2",
        ]:
            report[f"count_{label}"] = s.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"]
        report["v2_like_tokens"] = s.run(
            """
            MATCH (t:CategoricalToken)
            WHERE t.token_string STARTS WITH 'FEATURE|'
               OR t.token_string STARTS WITH 'VALUE_ABS|'
               OR t.token_string CONTAINS 'GLOBAL_REL'
            RETURN count(t) AS n
            """
        ).single()["n"]
    driver.close()
    report["recommendation"] = (
        "Keep V1 Sequence nodes; add TokenizationResultV2 linked to Sequence after full retokenize. "
        "Do not overwrite V1 HAS_TOKEN until acceptance on parquet passes."
    )
    write_json(out / "neo4j_pre_load_dry_run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--legacy-build", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--smoke-events", type=int, default=8000)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--skip-neo4j-dry-run", action="store_true")
    parser.add_argument("--min-available-gb", type=float, default=12.0)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    avail = mem_available_gb()
    if 0 < avail < args.min_available_gb:
        raise SystemExit(f"MemAvailable {avail:.1f}Gi < {args.min_available_gb}; abort")

    log("freeze check")
    freeze = ensure_freeze(out)
    log(f"freeze ok: {freeze}")

    # Ensure registry_build_report exists for --skip-registries
    if not (out / "registry_build_report.json").exists():
        write_json(
            out / "registry_build_report.json",
            {
                "schema_hash": "from_existing_artifacts",
                "binning_hash": freeze["binning_hash"],
                "vocab_size": freeze["vocab_size"],
                "note": "created by retokenize_v2_memory_safe for skip-registries",
            },
        )

    summary: dict[str, Any] = {"freeze": freeze, "started_at": datetime.now(timezone.utc).isoformat()}

    if not args.skip_smoke:
        log("phase smoke retokenize")
        # Backup existing smoke training if present
        smoke_out_events = out / "events_tokenized_v2.parquet"
        # Use dedicated smoke dir to avoid clobbering full artifacts mid-way
        smoke_dir = out / "_smoke_retok"
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
        smoke_dir.mkdir(parents=True)
        # Copy frozen registries into smoke dir (small files only)
        for name in [
            "vocab_v2.json",
            "binning_registry_v2_transductive.json",
            "feature_schema_v2.yaml",
            "token_usage_policy_v2.json",
            "binning_policy_v2.yaml",
            "binning_policy_v2.json",
            "registry_build_report.json",
            "measurement_group_policy_v2.json",
        ]:
            src = out / name
            if src.exists():
                shutil.copy2(src, smoke_dir / name)
        run_build(smoke_dir, args.legacy_build, max_events=args.smoke_events, skip_registries=True)
        # Move key smoke artifacts to canonical smoke names under out/
        for src_name, dst_name in [
            ("events_tokenized_v2.parquet", "events_tokenized_v2_smoke.parquet"),
            ("sequences_v2.parquet", "sequences_v2_smoke.parquet"),
            ("training_events_v2.parquet", "training_events_v2_smoke.parquet"),
        ]:
            src = smoke_dir / src_name
            if src.exists():
                shutil.copy2(src, out / dst_name)
                log(f"copied {src_name} -> {dst_name} ({(out/dst_name).stat().st_size/1e6:.1f}MB)")
        val = validate_events(
            out / "events_tokenized_v2_smoke.parquet",
            out / "vocab_v2.json",
            out / "retokenize_smoke_validation.json",
            sample_sentences=3000,
        )
        summary["smoke_validation"] = val
        log(f"smoke validation pass={val['pass']} oov={val['oov_unique']}")
        if not val["pass"]:
            write_json(out / "retokenize_summary.json", summary)
            raise SystemExit("smoke validation failed; abort full retokenize")
        gc.collect()

    if not args.skip_full:
        avail = mem_available_gb()
        if 0 < avail < max(args.min_available_gb, 20.0):
            raise SystemExit(f"MemAvailable {avail:.1f}Gi too low for full retokenize")
        log("phase full retokenize (may take long)")
        # Backup previous full artifacts by rename (no read)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = out / f"_bak_pre_adaptive_{ts}"
        bak.mkdir(parents=True, exist_ok=True)
        for name in [
            "events_tokenized_v2.parquet",
            "sequences_v2.parquet",
            "training_events_v2.parquet",
            "sequences_v2_pre_dedup.parquet",
        ]:
            p = out / name
            if p.exists():
                p.rename(bak / name)
                log(f"backed up {name}")
        run_build(out, args.legacy_build, max_events=0, skip_registries=True)
        # Sample validation on full events (first row groups only)
        val_full = validate_events(
            out / "events_tokenized_v2.parquet",
            out / "vocab_v2.json",
            out / "retokenize_full_validation.json",
            sample_sentences=5000,
        )
        summary["full_validation"] = val_full
        log(f"full validation pass={val_full['pass']} oov={val_full['oov_unique']}")
        # refresh smoke training from new full sequences without loading all
        if (out / "sequences_v2.parquet").exists():
            make_smoke_training(out, out / "sequences_v2.parquet", limit=200)
        gc.collect()

    if not args.skip_sanity and (out / "training_events_v2_smoke.parquet").exists():
        log("phase sanity pretrain")
        run_dir = ROOT / "outputs/online2/v2_runs/sanity_adaptive"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/online2_v2/run_v2_pretrain_loop.py"),
            "--mode",
            "sanity",
            "--build-dir",
            str(out),
            "--parquet",
            str(out / "training_events_v2_smoke.parquet"),
            "--max-rows",
            "200",
            "--steps",
            "20",
            "--batch-size",
            "2",
            "--max-length",
            "512",
            "--ckpt-every",
            "10",
            "--run-dir",
            str(run_dir),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        env["ONLINE2_V2_VOCAB_SIZE"] = str(freeze["vocab_size"])
        log_path = out / "sanity_adaptive_train.log"
        with log_path.open("w") as fh:
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        summary["sanity_rc"] = rc
        summary["sanity_log"] = str(log_path)
        log(f"sanity rc={rc}")

    if not args.skip_neo4j_dry_run:
        log("phase neo4j dry-run")
        summary["neo4j_dry_run"] = neo4j_dry_run(out)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["rss_max_gb"] = rss_gb()
    write_json(out / "retokenize_summary.json", summary)
    log(f"DONE summary -> {out / 'retokenize_summary.json'}")


if __name__ == "__main__":
    main()
