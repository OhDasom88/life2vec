#!/usr/bin/env python3
"""Assemble saliency + image-fusion + nearest-example evidence per case and
get a final diagnosis judgment from an LLM.

No live LLM endpoint is configured in this environment (verified: no local
server on 127.0.0.1:8080 -- the only other LLM caller in this repo,
`generate_cf1s_case_analysis_v03.py`, targets that same local server; no
cloud API key is set either). Use --dry-run to build and save every prompt
without calling anything -- this exercises 100% of the pipeline except the
network call, so it stays fully testable. Once a local model server (or a
cloud endpoint + --api-key) is available, drop --dry-run.

Usage:
    python scripts/online2_v2/v03/run_llm_evidence_diagnosis_v03.py \\
        --ckpt outputs/online2/v2_finetune_v03/runs/probe_oof_3fold/r0_fold0_best.pt \\
        --oof-dir outputs/online2/v2_finetune_v03/runs/probe_oof_3fold \\
        --case-id F551358_2024-09-21_2024-10-04 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.checkpoint import load_fold_model_v03  # noqa: E402
from src.online2.v2.finetune_v03.dataset import (  # noqa: E402
    DiagnosisEventDatasetV03,
    collate_diagnosis_batch_v03,
    load_label_map,
    normal_class_id_from_map,
)
from src.online2.v2.finetune_v03.llm_evidence.evidence_bundle import build_case_evidence  # noqa: E402
from src.online2.v2.finetune_v03.llm_evidence.llm_client import LLMCallError, chat_completion  # noqa: E402
from src.online2.v2.finetune_v03.llm_evidence.nearest_examples import nearest_labeled_examples  # noqa: E402
from src.online2.v2.finetune_v03.llm_evidence.prompt import build_diagnosis_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument(
        "--oof-dir",
        type=Path,
        required=True,
        help="dir with *_oof_representations.npz, used as the nearest-example pool",
    )
    p.add_argument(
        "--emb-dir", type=Path, default=ROOT / "outputs/online2/v2_finetune_v02/event_embeddings"
    )
    p.add_argument("--labels", type=Path, default=ROOT / "outputs/online2/v2_finetune/labels_example_score90.csv")
    p.add_argument("--label-map", type=Path, default=ROOT / "outputs/online2/v2_finetune/label_map.json")
    p.add_argument(
        "--events-path",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet",
        help="for looking up salient-event SENTENCE text",
    )
    p.add_argument("--case-id", type=str, default=None, help="default: all cases in --labels")
    p.add_argument("--k-salient", type=int, default=5)
    p.add_argument("--k-nearest", type=int, default=3)
    p.add_argument("--nearest-rep", choices=["h_binary", "z_proj"], default="h_binary")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/online2/v2_finetune_v03/llm_evidence")
    p.add_argument("--dry-run", action="store_true", help="build+save prompts, do not call the LLM")
    p.add_argument("--api-base", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="gemma-4-31b-it")
    p.add_argument("--api-key", default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_oof_pool(oof_dir: Path, rep_name: str) -> tuple[np.ndarray, list[str], list[float]]:
    case_ids: list[str] = []
    reps: list[np.ndarray] = []
    y_abn: list[float] = []
    seen: set[str] = set()
    for path in sorted(oof_dir.glob("*_oof_representations.npz")):
        data = np.load(path, allow_pickle=True)
        for cid, y, rep in zip(data["case_id"], data["y_abnormal"], data[rep_name]):
            cid = str(cid)
            if cid in seen:
                continue
            seen.add(cid)
            case_ids.append(cid)
            reps.append(rep)
            y_abn.append(float(y))
    return np.stack(reps), case_ids, y_abn


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, meta = load_fold_model_v03(args.ckpt, device)
    normal_id = int(meta["normal_class_id"])

    labels_df = pd.read_csv(args.labels)
    label_map = load_label_map(args.label_map)
    label_names = list(label_map["labels"])
    diag_by_case = labels_df.set_index(labels_df["case_id"].astype(str))["diagnosis_normalized"].to_dict()

    pool_reps, pool_case_ids, _ = load_oof_pool(args.oof_dir, args.nearest_rep)
    pool_labels = [diag_by_case.get(cid, "UNKNOWN") for cid in pool_case_ids]

    case_ids = [args.case_id] if args.case_id else labels_df["case_id"].astype(str).tolist()

    ds = DiagnosisEventDatasetV03(
        case_ids,
        labels_df=labels_df,
        label_map=label_map,
        emb_dir=args.emb_dir,
        interpretation_bank=None,
        semantic_dim=192,
        max_events=4096,
        require_complete=True,
        normal_class_id=normal_id,
    )
    print(f"assembling evidence for {len(ds)} case(s), dry_run={args.dry_run}")

    results = []
    for i in range(len(ds)):
        sample = ds[i]
        cid = sample["case_id"]
        batch = collate_diagnosis_batch_v03([sample], max_events=4096)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        evidence = build_case_evidence(
            model,
            sample,
            batch,
            normal_class_id=normal_id,
            label_names=label_names,
            events_path=args.events_path,
            k_salient=args.k_salient,
        )
        query_idx = pool_case_ids.index(cid) if cid in pool_case_ids else None
        query_rep = pool_reps[query_idx] if query_idx is not None else None
        nearest = (
            nearest_labeled_examples(
                query_rep, pool_reps, pool_case_ids, pool_labels, k=args.k_nearest, exclude_case_id=cid
            )
            if query_rep is not None
            else []
        )
        system, user = build_diagnosis_prompt(evidence, nearest, label_names)

        record = {
            "case_id": cid,
            "ground_truth": diag_by_case.get(cid),
            "evidence": evidence,
            "nearest_examples": nearest,
            "prompt_system": system,
            "prompt_user": user,
        }

        if args.dry_run:
            record["llm_stage"] = "DRY_RUN_NOT_CALLED"
        else:
            try:
                record["llm_response"] = chat_completion(
                    api_base=args.api_base,
                    model=args.model,
                    system=system,
                    user=user,
                    api_key=args.api_key,
                )
                record["llm_stage"] = "CALLED"
            except LLMCallError as exc:
                record["llm_stage"] = "FAILED"
                record["llm_error"] = str(exc)

        out_path = args.out_dir / f"{cid}_evidence.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
        print(f"[{i+1}/{len(ds)}] {cid}: {record['llm_stage']} -> {out_path}")
        results.append({"case_id": cid, "llm_stage": record["llm_stage"]})

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
