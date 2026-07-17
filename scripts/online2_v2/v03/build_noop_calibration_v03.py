#!/usr/bin/env python3
"""Build frozen NO_OP end-to-end round-trip calibration (example OOF cohort).

Measurement path (strict):
  grounded raw → production TokenizerV2 (same raw) → MG splice
  → Stage A affected-window reencode → OOF critic before/after

Rejects identical-tensor rescoring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.counterfactual.evaluation.case_batch import (
    load_case_batch,
    tensor_batch_only,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.case_fold_manifest import (
    load_split_manifest,
    oof_fold_for_case,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.crossfit_critic import (
    score_candidate_folds,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.noop_calibration import (
    build_noop_noise_artifact,
    write_noop_noise_artifact,
)
from src.online2.v2.finetune_v03.counterfactual.grounding.locus_raw import (
    ground_locus_raw,
    is_exact,
)
from src.online2.v2.finetune_v03.counterfactual.manifest import discover_fold_ckpts
from src.online2.v2.finetune_v03.counterfactual.pipeline_m2 import (
    _bin_edges_for_feature,
    _load_case_events,
    _normalize_feature_name,
    _resolve_vocab_and_encoder,
)
from src.online2.v2.finetune_v03.counterfactual.retokenization.full_event_retokenizer import (
    apply_raw_edit_closure,
    load_frozen_tokenizer_v2,
    recover_observed_raw_for_mg,
)
from src.online2.v2.finetune_v03.counterfactual.stage_a_reencoder import StageAReencoder


def _pick_event_feature(side: pd.DataFrame, events, feature: str):
    feat = _normalize_feature_name(feature)
    events_by_id = {str(ev.event_id): ev for ev in events}
    for row in side.itertuples():
        eid = str(row.event_id)
        ev = events_by_id.get(eid)
        if ev is None:
            continue
        toks = [str(t) for t in ev.sentence_tokens]
        want = f"FEATURE|{feat.upper()}"
        if any(t == want or (t.startswith("FEATURE|") and t.split("|", 1)[-1].lower() == feat) for t in toks):
            return eid, feat, row
    return None, feat, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/cf_calibration/noop_noise.json",
    )
    ap.add_argument("--feature", type=str, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    feature = args.feature or cfg.get("path_a_probe_feature") or "inside_temp_c"

    labels = pd.read_csv(cfg["labels_path"])
    if "set" in labels.columns:
        examples = labels[labels["set"].astype(str) == "example_set"]["case_id"].astype(str).tolist()
    else:
        examples = labels["case_id"].astype(str).tolist()

    split_path = Path(cfg["run_dir"]) / "split_manifest.json"
    splits = load_split_manifest(split_path, repeat=int(cfg.get("repeat", 0)))
    ckpts = discover_fold_ckpts(Path(cfg["run_dir"]), repeat=int(cfg.get("repeat", 0)))
    fold_ids = []
    for p in ckpts:
        stem = p.stem
        fold_ids.append(int(stem.split("_fold")[-1].split("_")[0]))

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    stage_a, encoder, vocab, abspos_ts, _hp = _resolve_vocab_and_encoder(cfg, device)
    feature_schema_path = Path(cfg.get("feature_schema_path") or ROOT / "outputs/online2/v2_build/feature_schema_v2.yaml")
    vocab_path = Path(cfg.get("vocabulary_path") or cfg["tokenizer_path"])
    prod_tok, _hashes = load_frozen_tokenizer_v2(
        feature_schema_path=feature_schema_path,
        binning_registry_path=Path(cfg["binning_registry_path"]),
        vocab_path=vocab_path,
        farm_relative_enabled=True,
    )
    cells = pd.read_parquet(cfg["cells_path"])
    sa_cfg = cfg.get("stage_a") or {}

    deltas = []
    used = []
    for case_id in examples:
        if len(used) >= int(args.n_samples):
            break
        oof = oof_fold_for_case(case_id, splits)
        if oof is None:
            continue
        emb = Path(cfg["embeddings_dir"]) / f"{case_id}.parquet"
        if not emb.exists():
            continue
        try:
            batch, side = load_case_batch(
                case_id=case_id,
                embeddings_dir=Path(cfg["embeddings_dir"]),
                labels_path=Path(cfg["labels_path"]),
                label_map_path=Path(
                    cfg.get("label_map_path") or Path(cfg["labels_path"]).parent / "label_map.json"
                ),
            )
            _sa, events, farm_id, period_start = _load_case_events(cfg, case_id)
            eid, feat, er = _pick_event_feature(side, events, feature)
            if eid is None or er is None:
                print(f"skip {case_id}: no feature={feature} in events", flush=True)
                continue
            grounding = ground_locus_raw(
                cells=cells,
                farm_id=str(farm_id),
                feature=feat,
                zone_id=str(er.zone),
                timestamp=er.timestamp,
                event_id=eid,
                measurement_group_id="",
                tolerance_seconds=float((cfg.get("raw_grounding") or {}).get("tolerance_seconds", 0)),
                require_exact=True,
            )
            if not is_exact(grounding):
                print(f"skip {case_id}: grounding not EXACT", flush=True)
                continue
            observed = float(grounding["observed_raw"])
            edges = _bin_edges_for_feature(cfg, feat)
            ev0 = next(ev for ev in events if str(ev.event_id) == eid)
            recovered = recover_observed_raw_for_mg(
                prod_tok,
                feature=feat,
                farm_id=str(farm_id),
                original_tokens=list(ev0.sentence_tokens),
                grounded_raw=observed,
                edges_abs=edges,
                allow_abs_bin_probe=False,
            )
            if not recovered.get("ok"):
                print(f"skip {case_id}: Gate0 grounded drift {recovered.get('reason')}", flush=True)
                continue
            observed = float(recovered["observed_raw"])
            # True NO_OP: same raw through production tokenizer + MG splice
            rtok = apply_raw_edit_closure(
                event_id=eid,
                feature=feat,
                target_raw=observed,
                original_tokens=list(ev0.sentence_tokens),
                tokenizer=prod_tok,
                farm_id=str(farm_id),
                observed_raw=observed,
                strict_exact=True,
            )
            if not rtok.baseline_roundtrip_valid or not rtok.unchanged_outside_mg:
                print(f"skip {case_id}: Gate0/4 NO_OP retokenize fail", flush=True)
                continue
            # Confirm token bundle identical (true NO_OP)
            if list(rtok.actual_retokenized_bundle) != list(recovered["original_bundle"]):
                print(f"skip {case_id}: NO_OP bundle changed", flush=True)
                continue

            reenc = StageAReencoder(
                encoder=encoder,
                stage_a_mod=stage_a,
                vocab=vocab,
                abspos_reference=abspos_ts,
                case_t0=period_start,
                device=device,
                parity_cosine_min=float(sa_cfg.get("parity_cosine_min", 0.9999)),
                parity_mean_max_abs_error=float(sa_cfg.get("parity_mean_max_abs_error", 1e-4)),
                parity_max_max_abs_error=float(sa_cfg.get("parity_max_max_abs_error", 5e-4)),
            )
            base = tensor_batch_only(batch)
            eid_to_idx = {str(r.event_id): int(r.event_index) for r in side.itertuples()}
            result = reenc.apply_edits_and_reencode(
                events,
                [{"event_id": eid, "to_tokens": list(rtok.new_sentence_tokens), "replace_sentence": True}],
                {k: v.to(device) for k, v in base.items()},
                event_id_to_index=eid_to_idx,
            )
            ckpt = [ckpts[fold_ids.index(int(oof))]]
            stats = score_candidate_folds(
                ckpt,
                {k: v.cpu() for k, v in base.items()},
                {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in result.batch.items()},
                normal_class_id=int(batch["_normal_class_id"]),
                fold_ids=[int(oof)],
                device=str(cfg.get("device", "cuda")),
                gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
            )
            deltas.append(float(stats["delta_r"]))
            used.append(case_id)
            print(
                f"ok {case_id} oof={oof} delta_r={stats['delta_r']} "
                f"method=tokenizer_stage_a_critic_e2e",
                flush=True,
            )
        except Exception as e:
            print(f"skip {case_id}: {e}", flush=True)

    if len(deltas) < int(args.n_samples):
        raise SystemExit(
            f"need >= {args.n_samples} e2e calibration samples, got {len(deltas)}"
        )

    art = build_noop_noise_artifact(
        noop_delta_rs=deltas,
        calibration_cohort="example_oof",
        configured_epsilon=float(
            (cfg.get("critic") or {}).get("configured_material_epsilon") or 0.01
        ),
        noop_noise_multiplier=float(
            (cfg.get("critic") or {}).get("noop_noise_multiplier") or 10.0
        ),
        case_ids=used,
        measurement_method="tokenizer_stage_a_critic_e2e",
    )
    write_noop_noise_artifact(args.out, art)
    log_path = args.out.with_name("build_log.txt")
    log_path.write_text(
        f"n={art['n_samples']} method={art['measurement_method']} "
        f"p99={art['noop_delta_p99']} eps={art['effective_epsilon']}\n"
        + "\n".join(used)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.out} n={art['n_samples']} method={art['measurement_method']} "
        f"p99={art['noop_delta_p99']} eps={art['effective_epsilon']}"
    )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
