#!/usr/bin/env python3
"""Online2 V2 grouped-MLM sanity / resumable pretrain without HDF5 prepare_data.

Reads training_events parquet directly, encodes with GroupedMLM, trains TransformerEncoder.
Supports checkpoint resume, OOM batch-size fallback, global abspos, and wandb logging.

Time channels (life2vec contract):
  - abspos: hours since corpus-global reference (abspos_reference.json)
  - age:    sequence-relative hours from parquet AGE (first event in sequence)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchmetrics


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_abspos_reference(build: Path) -> dict[str, Any]:
    path = build / "abspos_reference.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Compute with min(START_DATE) over training_events_v2 first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ref = pd.Timestamp(payload["reference_timestamp"])
    if ref.tzinfo is not None:
        ref = ref.tz_convert("UTC").tz_localize(None)
    payload["_reference_ts"] = ref
    return payload


def person_hash_unit(person_id: int, seed: int = 2023) -> float:
    digest = hashlib.sha256(f"{seed}:{int(person_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def make_split_predicate(
    *,
    use_shadow_split: bool,
    want: str,
    val_frac: float,
    seed: int,
) -> Callable[[pd.DataFrame, list[int]], list[int]]:
    """Return a filter that keeps PERSON_IDs for train or val.

    Prefer parquet ``shadow_split`` when it has a real val set; otherwise fall back to
    a deterministic PERSON_ID hash split (current V2 export is all-train).
    """

    def _filter(frame: pd.DataFrame, ids: list[int]) -> list[int]:
        if use_shadow_split and "shadow_split" in frame.columns:
            split = frame.drop_duplicates("PERSON_ID").set_index("PERSON_ID")["shadow_split"]
            keep = []
            for pid in ids:
                label = str(split.get(pid, "train")).lower()
                if want == "train" and label == "train":
                    keep.append(pid)
                elif want == "val" and label in {"val", "valid", "validation"}:
                    keep.append(pid)
            return keep
        keep = []
        for pid in ids:
            is_val = person_hash_unit(int(pid), seed=seed) < val_frac
            if want == "val" and is_val:
                keep.append(pid)
            elif want == "train" and not is_val:
                keep.append(pid)
        return keep

    return _filter


def detect_shadow_val(parquet: Path, max_row_groups: int = 8) -> bool:
    pf = pq.ParquetFile(parquet)
    for i in range(min(max_row_groups, pf.metadata.num_row_groups)):
        part = pf.read_row_group(i, columns=["shadow_split"]).to_pandas()
        labels = {str(x).lower() for x in part["shadow_split"].unique()}
        if labels & {"val", "valid", "validation"}:
            return True
    return False


def _iter_person_batches_normalized(
    sequences_parquet: Path,
    events_parquet: Path,
    batch_size: int,
    max_rows: int | None,
    *,
    id_filter: Callable[[pd.DataFrame, list[int]], list[int]] | None = None,
):
    """2026-07-26: `training_events_v2.parquet`(사전에 260배 중복 펼쳐 놓은 flat
    파일)를 미리 만들어두지 않고, `sequences_v2.parquet`(시퀀스당 1행, event_ids만
    참조) + `events_tokenized_v2.parquet`(고유 이벤트당 1행, 222k행, 전체를
    메모리에 캐시)를 로딩 시점에 조인해서 그때그때 펼친다. build 단계의
    `export_training_events_v2_event_grain.py`가 disk에 쓰던 것과 정확히 같은
    행 스키마를 만들어야 하므로, 그 스크립트의 `load_event_lookup`/
    `expand_sequence_rows`를 그대로 재사용한다(로직 중복 없음) — 그래서
    이 함수가 만드는 프레임은 flat 경로가 만들던 것과 컬럼이 100% 동일하고,
    아래 `encode_batch`/`task.get_document`는 어느 경로든 손댈 필요가 없다.
    """
    from export_training_events_v2_event_grain import (  # noqa: E402
        expand_sequence_rows,
        load_event_lookup,
    )

    event_lookup = load_event_lookup(events_parquet)
    light_cols = [
        "sequence_id",
        "narrative_id",
        "PERSON_ID",
        "event_ids",
        "farm_ids",
        "canonical_context_id",
        "split_group_id",
        "training_mode",
        "sampling_weight",
        "shadow_split",
    ]
    pf = pq.ParquetFile(sequences_parquet)
    available = set(pf.schema_arrow.names)
    cols = [c for c in light_cols if c in available]
    seen = 0
    for i in range(pf.metadata.num_row_groups):
        part = pf.read_row_group(i, columns=cols).to_pandas()
        if max_rows is not None:
            remain = max_rows - seen
            if remain <= 0:
                break
            part = part.head(remain)
        seen += len(part)
        ids = part["PERSON_ID"].astype(int).unique().tolist()
        if id_filter is not None:
            ids = id_filter(part, ids)
        if not ids:
            continue
        for start in range(0, len(ids), batch_size):
            chosen = ids[start : start + batch_size]
            sub = part[part["PERSON_ID"].isin(chosen)]
            rows: list[dict[str, Any]] = []
            for rec in sub.itertuples(index=False):
                expanded, _stats = expand_sequence_rows(rec, event_lookup, build_id="v2_event_grain_lazy")
                rows.extend(expanded)
            if not rows:
                continue
            yield pd.DataFrame(rows), chosen
        if max_rows is not None and seen >= max_rows:
            break


def iter_person_batches(
    parquet: Path,
    batch_size: int,
    max_rows: int | None,
    *,
    id_filter: Callable[[pd.DataFrame, list[int]], list[int]] | None = None,
    events_parquet: Path | None = None,
):
    """Yield dataframe chunks without requiring the full corpus in RAM.

    events_parquet가 주어지면 `parquet`는 flat training_events_v2가 아니라
    `sequences_v2.parquet`로 취급하고, 로딩 시점 조인으로 정규화 경로를 탄다
    (`_iter_person_batches_normalized`). None(기본값)이면 기존처럼 이미 펼쳐진
    flat parquet를 그대로 스트리밍한다 -- 기존 호출부·동작은 완전히 그대로다.
    """
    if events_parquet is not None:
        yield from _iter_person_batches_normalized(
            parquet, events_parquet, batch_size, max_rows, id_filter=id_filter
        )
        return
    pf = pq.ParquetFile(parquet)
    seen = 0
    for i in range(pf.metadata.num_row_groups):
        part = pf.read_row_group(i).to_pandas()
        if max_rows is not None:
            remain = max_rows - seen
            if remain <= 0:
                break
            part = part.head(remain)
        seen += len(part)
        ids = part["PERSON_ID"].astype(int).unique().tolist()
        if id_filter is not None:
            ids = id_filter(part, ids)
        if not ids:
            continue
        for start in range(0, len(ids), batch_size):
            chosen = ids[start : start + batch_size]
            yield part[part["PERSON_ID"].isin(chosen)].copy(), chosen
        if max_rows is not None and seen >= max_rows:
            break


def encode_batch(
    task,
    frame: pd.DataFrame,
    person_ids: list[int],
    device: torch.device,
    abspos_reference: pd.Timestamp,
    *,
    is_train: bool = True,
):
    """Encode persons with global abspos and sequence-relative AGE.

    START_DATE → integer hours since ``abspos_reference`` (get_document adds +1 → abspos).
    AGE is left as parquet sequence-relative hours — never rebase START_DATE to person.min().
    """
    from src.tasks.base import collate_encoded_documents

    docs = []
    for person_id in person_ids:
        person = frame[frame["PERSON_ID"] == person_id].copy()
        if person.empty:
            continue
        person.name = int(person_id)
        person["BIRTHDAY"] = np.datetime64("2000-01-01")
        person["GENDER"] = "U"
        person["RES_ORIGIN"] = "UNK"
        person["AFTER_THRESHOLD"] = False
        # AGE: sequence-relative hours (already exported)
        person["AGE"] = person["AGE"].astype(float)
        # abspos path: global hour offset; get_document uses abspos = START_DATE + 1
        sd = pd.to_datetime(person["START_DATE"])
        if getattr(sd.dt, "tz", None) is not None:
            sd = sd.dt.tz_convert("UTC").dt.tz_localize(None)
        person["START_DATE"] = (
            (sd - abspos_reference) / np.timedelta64(1, "h")
        ).astype(int)
        if "SEGMENT" not in person.columns:
            person["SEGMENT"] = 1
        doc = task.get_document(person)
        encoded = task.get_preprocessor(is_train=is_train)(doc)
        docs.append(encoded)
    if not docs:
        raise RuntimeError("empty batch")
    batch = collate_encoded_documents(docs)
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


SOP_CLASS_NAMES = ("normal_order", "reversed", "shuffled")


def make_sop_metrics(device: torch.device) -> dict[str, dict[str, torchmetrics.Metric]]:
    """SOP(3-class: 0=정상순서, 1=역순, 2=셔플, src/tasks/grouped_mlm.py:321-331 라벨
    규칙과 동일)의 마스크(target_cls_mask==1, 즉 SOP 적용 가능했던 문서만) 반영
    accuracy/F1 + 클래스별 F1을 별도로 관리한다.

    model.train_cls_acc/train_cls_f1(model.py)은 마스크를 안 봐서 SOP 미적용
    문서(다수인 정상순서 placeholder)까지 다 섞여 점수가 부풀려진다 -- 여기서는
    실제로 SOP 판정이 걸린 문서만 골라 계산. model 자체에 새 서브모듈을 추가하면
    기존 체크포인트 state_dict 로드가 깨질 수 있어(strict load) model 밖의 별도
    dict로 관리한다.
    """
    def _make() -> dict[str, torchmetrics.Metric]:
        return {
            "acc": torchmetrics.Accuracy(num_classes=3, average="macro").to(device),
            "f1_macro": torchmetrics.F1Score(num_classes=3, average="macro").to(device),
            "f1_per_class": torchmetrics.F1Score(num_classes=3, average=None).to(device),
        }

    return {"train": _make(), "val": _make()}


def compute_mlm_sop_metrics(
    model: torch.nn.Module,
    sop_metrics: dict[str, dict[str, torchmetrics.Metric]],
    mlm_preds: torch.Tensor,
    mlm_targs: torch.Tensor,
    cls_preds: torch.Tensor,
    cls_targs: torch.Tensor,
    cls_mask: Optional[torch.Tensor],
    stage: str,
) -> dict[str, float]:
    """MLM(top-5, vocab 전체)과 SOP(3-class, 마스크 적용) 상세 지표 한 스텝/배치분.

    model.{train,val}_{accuracy,precision,recall,f1}는 model.py에 이미 정의돼
    있었지만 원래 Lightning의 training_step/validation_step 안에서만 호출되게
    짜여 있어서(self.log 경유) 이 커스텀 루프에서는 한 번도 안 불렸다 -- 여기서
    직접 호출해 값만 꺼내 쓴다(Lightning self.log는 self.trainer가 필요해서
    이 루프에서는 못 씀).
    """
    out: dict[str, float] = {}
    mlm_preds_sm = F.softmax(mlm_preds.detach(), dim=-1).permute(0, 2, 1)
    mlm_acc = model.train_accuracy if stage == "train" else model.val_accuracy
    mlm_prec = model.train_precision if stage == "train" else model.val_precision
    mlm_rec = model.train_recall if stage == "train" else model.val_recall
    mlm_f1 = model.train_f1 if stage == "train" else model.val_f1
    out[f"{stage}/mlm_top5_accuracy"] = float(mlm_acc(mlm_preds_sm, mlm_targs))
    out[f"{stage}/mlm_top5_precision"] = float(mlm_prec(mlm_preds_sm, mlm_targs))
    out[f"{stage}/mlm_top5_recall"] = float(mlm_rec(mlm_preds_sm, mlm_targs))
    out[f"{stage}/mlm_top5_f1"] = float(mlm_f1(mlm_preds_sm, mlm_targs))

    if torch.is_tensor(cls_mask):
        mask_flat = cls_mask.detach().reshape(-1).bool()
        if mask_flat.any():
            cls_preds_flat = F.softmax(cls_preds.detach(), dim=-1).reshape(-1, cls_preds.shape[-1])
            cls_targs_flat = cls_targs.detach().reshape(-1)
            preds_m = cls_preds_flat[mask_flat]
            targs_m = cls_targs_flat[mask_flat]
            m = sop_metrics[stage]
            out[f"{stage}/sop_masked_accuracy"] = float(m["acc"](preds_m, targs_m))
            out[f"{stage}/sop_masked_f1"] = float(m["f1_macro"](preds_m, targs_m))
            per_class = m["f1_per_class"](preds_m, targs_m)
            for i, name in enumerate(SOP_CLASS_NAMES):
                out[f"{stage}/sop_f1_{name}"] = float(per_class[i])
            out[f"{stage}/sop_eligible_n"] = float(mask_flat.sum())
    return out


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    encode_fn,
    data_iter,
    device: torch.device,
    *,
    max_batches: int,
    sop_metrics: Optional[dict[str, dict[str, torchmetrics.Metric]]] = None,
) -> dict[str, float]:
    """Average val losses over up to ``max_batches`` person-batches.

    sop_metrics(옵션)를 넘기면 MLM top-5/SOP masked 상세 지표도 val window
    전체에 대해 누적(update)한 뒤 마지막에 한 번만 .compute()해서 반환한다
    (배치별 F1을 단순 평균하는 것보다 통계적으로 올바름 -- 배치별 metric은
    버려지고 window 전체 pooled 값만 씀).
    """
    from src.transformer.models import masked_sop_loss

    model.eval()
    totals = {
        "loss": 0.0,
        "mlm_loss": 0.0,
        "sop_loss": 0.0,
        "sop_mask_mean": 0.0,
        "n": 0.0,
    }
    sop_eligible_seen = 0.0
    for _ in range(max_batches):
        try:
            frame, chosen = next(data_iter)
        except StopIteration:
            break
        if not chosen:
            continue
        batch = encode_fn(frame, chosen, device)
        mlm_preds, cls_preds = model(batch)
        mlm_loss = model.mlm_loss(
            mlm_preds.permute(0, 2, 1), target=batch["target_tokens"].long()
        )
        sop_loss = masked_sop_loss(
            model.cls_loss(cls_preds, target=batch["target_cls"].long()),
            batch.get("target_cls_mask"),
        )
        loss = model.cls_w * sop_loss + model.mlm_w * mlm_loss
        totals["loss"] += float(loss.detach().cpu())
        totals["mlm_loss"] += float(mlm_loss.detach().cpu())
        totals["sop_loss"] += float(sop_loss.detach().cpu())
        if torch.is_tensor(batch.get("target_cls_mask")):
            totals["sop_mask_mean"] += float(batch["target_cls_mask"].float().mean().cpu())
        totals["n"] += 1.0
        if sop_metrics is not None:
            compute_mlm_sop_metrics(
                model, sop_metrics,
                mlm_preds, batch["target_tokens"].long(),
                cls_preds, batch["target_cls"].long(),
                batch.get("target_cls_mask"),
                stage="val",
            )
            mask = batch.get("target_cls_mask")
            if torch.is_tensor(mask):
                sop_eligible_seen += float(mask.detach().sum().cpu())
        del batch, mlm_preds, cls_preds, loss, mlm_loss, sop_loss
    model.train()
    n = max(totals["n"], 1.0)
    detailed: dict[str, float] = {}
    if sop_metrics is not None:
        m = sop_metrics["val"]
        detailed["val/mlm_top5_accuracy"] = float(model.val_accuracy.compute())
        detailed["val/mlm_top5_precision"] = float(model.val_precision.compute())
        detailed["val/mlm_top5_recall"] = float(model.val_recall.compute())
        detailed["val/mlm_top5_f1"] = float(model.val_f1.compute())
        model.val_accuracy.reset()
        model.val_precision.reset()
        model.val_recall.reset()
        model.val_f1.reset()
        if sop_eligible_seen > 0:
            detailed["val/sop_masked_accuracy"] = float(m["acc"].compute())
            detailed["val/sop_masked_f1"] = float(m["f1_macro"].compute())
            per_class = m["f1_per_class"].compute()
            for i, name in enumerate(SOP_CLASS_NAMES):
                detailed[f"val/sop_f1_{name}"] = float(per_class[i])
        m["acc"].reset()
        m["f1_macro"].reset()
        m["f1_per_class"].reset()
    return {
        **detailed,
        "val/loss": totals["loss"] / n,
        "val/mlm_loss": totals["mlm_loss"] / n,
        "val/sop_loss": totals["sop_loss"] / n,
        "val/sop_mask_mean": totals["sop_mask_mean"] / n,
        "val/batches": totals["n"],
    }


def calibrate_batch_size(
    model: torch.nn.Module,
    encode_fn,
    data_iter,
    device: torch.device,
    start_bs: int,
    target_frac: float = 0.70,
    max_bs: int = 64,
) -> int:
    """Grow batch size until reserved VRAM reaches target_frac of device memory."""
    if device.type != "cuda":
        return start_bs
    total = torch.cuda.get_device_properties(device).total_memory
    target = int(total * target_frac)
    bs = max(1, start_bs)
    best = bs
    while bs <= max_bs:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            frame, chosen = next(data_iter)
            people = list(chosen)
            while len(people) < bs:
                people.extend(chosen)
            people = people[:bs]
            batch = encode_fn(frame, people, device)
            opt_probe = torch.optim.AdamW(model.parameters(), lr=1e-4)
            opt_probe.zero_grad(set_to_none=True)
            mlm_preds, cls_preds = model(batch)
            mlm_targs = batch["target_tokens"].long()
            cls_targs = batch["target_cls"].long()
            from src.transformer.models import masked_sop_loss

            mlm_loss = model.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
            sop_loss = masked_sop_loss(
                model.cls_loss(cls_preds, target=cls_targs),
                batch.get("target_cls_mask"),
            )
            loss = model.cls_w * sop_loss + model.mlm_w * mlm_loss
            loss.backward()
            opt_probe.step()
            del batch, mlm_preds, cls_preds, loss, mlm_loss, sop_loss, opt_probe
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            used = peak
            print(
                {
                    "vram_calibrate_bs": bs,
                    "peak_alloc_gb": round(peak / 1e9, 2),
                    "reserved_gb": round(reserved / 1e9, 2),
                    "target_gb": round(target / 1e9, 2),
                },
                flush=True,
            )
            best = bs
            if used >= target:
                break
            if used >= int(total * 0.85):
                break
            if used >= int(total * 0.40):
                next_bs = min(max_bs, bs + 8)
            else:
                next_bs = min(max_bs, bs * 2 if bs >= 4 else bs + 2)
            if (
                next_bs > bs
                and used > 0
                and used * (next_bs / max(bs, 1)) > int(total * 0.88)
            ):
                next_bs = min(max_bs, bs + 8)
            if next_bs <= bs:
                break
            if used > 0 and used * (next_bs / max(bs, 1)) > int(total * 0.88):
                break
            bs = next_bs
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(f"VRAM_CALIBRATE_OOM at bs={bs}, using {best}", flush=True)
                break
            raise
    return max(1, best)


def init_wandb(
    args,
    manifest: dict,
    abspos_ref: dict,
    model: Optional[torch.nn.Module] = None,
) -> Optional[Any]:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed. `pip install wandb` or pass --no-wandb."
        ) from exc
    if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        raise SystemExit(
            "wandb login missing (WANDB_API_KEY / ~/.netrc). "
            "Run `wandb login` or pass --no-wandb."
        )
    tags = [t for t in (args.wandb_tags or "").split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name
        or args.run_dir.name
        or "online2_v2_event_grain_global_abspos",
        tags=tags
        or ["online2_v2", "event_grain", "global_abspos", args.mode],
        config={
            **{k: v for k, v in manifest.items() if k not in {"attempts", "history_tail"}},
            "abspos_reference": {
                k: abspos_ref[k]
                for k in (
                    "reference_timestamp",
                    "unit",
                    "method",
                    "source_parquet",
                    "start_date_max",
                )
                if k in abspos_ref
            },
            "sop_reverse": args.sop_reverse,
            "sop_shuffle": args.sop_shuffle,
            "wandb_watch": getattr(args, "wandb_watch", False),
            "wandb_watch_log": getattr(args, "wandb_watch_log", "all"),
            "wandb_watch_freq": getattr(args, "wandb_watch_freq", 100),
        },
        reinit=True,
    )
    if model is not None and getattr(args, "wandb_watch", False):
        # Histograms of parameters + gradients (wandb UI: Model → Gradients / Parameters)
        wandb.watch(
            model,
            log=args.wandb_watch_log,
            log_freq=max(1, int(args.wandb_watch_freq)),
            log_graph=False,
        )
        print(
            {
                "wandb_watch": True,
                "log": args.wandb_watch_log,
                "log_freq": args.wandb_watch_freq,
            },
            flush=True,
        )
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "outputs/online2/v2_build")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/online2/v2_runs/sanity")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument(
        "--corpus-format",
        choices=["auto", "flat", "normalized"],
        default="auto",
        help=(
            "flat=training_events_v2.parquet(사전에 260배 중복 펼쳐 저장). "
            "normalized=sequences_v2.parquet+events_tokenized_v2.parquet를 로딩 "
            "시점에 조인(중복 저장 없음, 2026-07-26 권장). "
            "auto=flat 파일이 있으면 flat, 없으면 normalized."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max training steps (sanity default 30; full default 100000 with early stop).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="0=disabled (use --steps). If >0, train until this many full passes over the "
        "training split (or early-stop patience) instead of a fixed step count -- avoids "
        "confounding corpus-size differences with training-amount differences across arms.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity")
    parser.add_argument("--target-vram-frac", type=float, default=0.70)
    parser.add_argument("--lr", type=float, default=None, help="Override default learning rate.")
    parser.add_argument("--sop-reverse", type=float, default=0.20)
    parser.add_argument("--sop-shuffle", type=float, default=0.20)
    parser.add_argument("--skip-vram-calibrate", action="store_true")
    parser.add_argument("--abspos-reference", type=Path, default=None)
    parser.add_argument("--val-frac", type=float, default=0.05, help="Hash val frac if no shadow val")
    parser.add_argument("--val-seed", type=int, default=2023)
    parser.add_argument("--val-every", type=int, default=100, help="Validate every N train steps")
    parser.add_argument("--val-batches", type=int, default=32, help="Val batches per evaluation")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Val checks without improve")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="Berry2Vec")
    parser.add_argument("--wandb-entity", type=str, default="dasom-oh")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="online2_v2,event_grain,global_abspos",
    )
    parser.add_argument(
        "--wandb-watch",
        dest="wandb_watch",
        action="store_true",
        default=None,
        help="Track parameter/gradient histograms via wandb.watch (default: on when --wandb).",
    )
    parser.add_argument("--no-wandb-watch", dest="wandb_watch", action="store_false")
    parser.add_argument(
        "--wandb-watch-log",
        choices=["gradients", "parameters", "all"],
        default="all",
        help="What wandb.watch logs (default: all = params + grads).",
    )
    parser.add_argument(
        "--wandb-watch-freq",
        type=int,
        default=100,
        help="Log param/grad histograms every N backward steps.",
    )
    args = parser.parse_args()
    if args.wandb is None:
        args.wandb = args.mode == "full"
    if args.wandb_watch is None:
        args.wandb_watch = bool(args.wandb)
    if args.steps is None:
        args.steps = 30 if args.mode == "sanity" else 100_000
    if args.mode == "sanity":
        args.val_every = min(args.val_every, 10)
        args.val_batches = min(args.val_batches, 8)
        args.early_stop_patience = min(args.early_stop_patience, 5)
        args.wandb_watch_freq = min(args.wandb_watch_freq, 10)

    build = args.build_dir
    run_dir = args.run_dir
    sweep_run_id = os.environ.get("WANDB_RUN_ID") or os.environ.get("WANDB_SWEEP_ID")
    if sweep_run_id and os.environ.get("WANDB_SWEEP_ID"):
        # Under `wandb agent`, multiple trials share one --run-dir from the sweep
        # command template -- disambiguate so concurrent/sequential trials don't
        # clobber each other's checkpoints.
        run_dir = run_dir.parent / f"{run_dir.name}_{sweep_run_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # 2026-07-26: normalized 경로 -- sequences_v2.parquet(시퀀스당 1행) +
    # events_tokenized_v2.parquet(고유 이벤트당 1행)를 로딩 시점에 조인한다.
    # training_events_v2.parquet(260배 중복 펼친 flat 파일)를 미리 만들 필요가
    # 없어져서, 그 export 단계(수시간)를 통째로 건너뛴다. auto는 flat 파일이
    # 있으면 flat, 없으면 normalized로 자동 선택.
    flat_parquet = args.parquet or (
        build / "training_events_v2_smoke.parquet"
        if args.mode == "sanity" and (build / "training_events_v2_smoke.parquet").exists()
        else build / "training_events_v2.parquet"
    )
    sequences_parquet = build / "sequences_v2.parquet"
    events_tokenized_parquet = build / "events_tokenized_v2.parquet"
    corpus_format = args.corpus_format
    if corpus_format == "auto":
        corpus_format = "flat" if flat_parquet.exists() else "normalized"
    if corpus_format == "normalized":
        if not (sequences_parquet.exists() and events_tokenized_parquet.exists()):
            raise FileNotFoundError(
                f"--corpus-format normalized 이려면 {sequences_parquet}와 "
                f"{events_tokenized_parquet}가 둘 다 있어야 합니다."
            )
        parquet = sequences_parquet
        events_parquet = events_tokenized_parquet
    else:
        parquet = flat_parquet
        events_parquet = None
    print({"corpus_format": corpus_format, "parquet": str(parquet)}, flush=True)
    registry = build / "life2vec_token_registry_v2.json"
    vocab_v2 = build / "vocab_v2.json"

    if args.abspos_reference is not None:
        payload = json.loads(args.abspos_reference.read_text(encoding="utf-8"))
        ref_ts = pd.Timestamp(payload["reference_timestamp"])
        if ref_ts.tzinfo is not None:
            ref_ts = ref_ts.tz_convert("UTC").tz_localize(None)
        payload["_reference_ts"] = ref_ts
        abspos_ref = payload
    else:
        abspos_ref = load_abspos_reference(build)
    global_ref: pd.Timestamp = abspos_ref["_reference_ts"]
    print(
        {
            "abspos_reference": abspos_ref["reference_timestamp"],
            "unit": abspos_ref.get("unit", "hour"),
        },
        flush=True,
    )

    from src.data_new.vocabulary import RegistryVocabulary
    from src.tasks.grouped_mlm import GroupedMLM
    from src.transformer.models import TransformerEncoder

    max_rows = None if args.max_rows <= 0 else args.max_rows
    print(f"streaming parquet={parquet} max_rows={max_rows}", flush=True)

    vocab = RegistryVocabulary(registry_path=str(registry), registry_version="v2")
    if args.mode == "full" and args.max_length == 512:
        max_length = 1024
    else:
        max_length = args.max_length

    task = GroupedMLM(
        name="online2_v2_grouped_mlm",
        max_length=max_length,
        mask_ratio=0.3,
        vocab_v2_path=str(vocab_v2),
        mask_feature_identity=False,
        non_maskable_tokens=["[IMAGE_SLOT]", "[TEXT_SLOT]", "IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT"],
        sop_reverse_probability=args.sop_reverse,
        sop_shuffle_probability=args.sop_shuffle,
        evaluation_seed=2023,
        shuffle_within_sentences=False,
    )
    task.datamodule = SimpleNamespace(vocabulary=vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = args.mode == "full"
    hparams = {
        "vocabulary": vocab,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "hidden_size": 128 if args.mode == "sanity" else 384,
        "hidden_ff": 512 if args.mode == "sanity" else 1536,
        "hidden_act": "swish",
        "n_encoders": 2 if args.mode == "sanity" else 6,
        "n_heads": 4 if args.mode == "sanity" else 8,
        "n_local": 2 if args.mode == "sanity" else 4,
        "local_window_size": 16 if args.mode == "sanity" else 64,
        "norm_type": "rezero",
        "att_dropout": 0.1,
        "fw_dropout": 0.1,
        "dc_dropout": 0.1,
        "emb_dropout": 0.1,
        "parametrize_emb": False,
        "norm_input_emb": False,
        "norm_output_emb": True,
        "weight_tying": "wt",
        "training_task": "mlm",
        "experiment_name": f"online2_v2_{args.mode}",
        "experiment_version": "2.2",
        "attention_type": "performer",
        "multihead_dc": False,
        "num_random_features": 64 if args.mode == "sanity" else 128,
        "learning_rate": args.lr if args.lr is not None else (1e-3 if args.mode == "sanity" else 5e-4),
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "cls_num_targs": 3,
        "epsilon": 1e-6,
        "stage": "pre_training",
        "implementation": "online2_v2",
        "version": "2.2",
        "vocab_size": vocab.size(),
        "abspos_reference": abspos_ref["reference_timestamp"],
        "abspos_unit": "hour",
    }
    model = TransformerEncoder(hparams).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=hparams["learning_rate"], weight_decay=0.01)
    sop_metrics = make_sop_metrics(device)

    use_shadow_val = detect_shadow_val(parquet)
    train_filter = make_split_predicate(
        use_shadow_split=use_shadow_val,
        want="train",
        val_frac=args.val_frac,
        seed=args.val_seed,
    )
    val_filter = make_split_predicate(
        use_shadow_split=use_shadow_val,
        want="val",
        val_frac=args.val_frac,
        seed=args.val_seed,
    )
    print(
        {
            "val_split": "shadow_split" if use_shadow_val else f"hash_person_frac={args.val_frac}",
            "val_every": args.val_every,
            "val_batches": args.val_batches,
            "early_stop_patience": None if args.no_early_stop else args.early_stop_patience,
            "max_steps": args.steps,
        },
        flush=True,
    )

    def _encode_train(frame, people, dev):
        return encode_batch(task, frame, people, dev, global_ref, is_train=True)

    def _encode_val(frame, people, dev):
        return encode_batch(task, frame, people, dev, global_ref, is_train=False)

    start_step = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} step={start_step}", flush=True)

    batch_size = args.batch_size if args.mode == "sanity" else max(args.batch_size, 8)
    if full and not args.skip_vram_calibrate and device.type == "cuda":
        cal_iter = iter_person_batches(
            parquet, max(2, batch_size), max_rows, id_filter=train_filter, events_parquet=events_parquet
        )
        batch_size = calibrate_batch_size(
            model,
            _encode_train,
            cal_iter,
            device,
            start_bs=batch_size,
            target_frac=args.target_vram_frac,
            max_bs=64,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=hparams["learning_rate"], weight_decay=0.01)
        torch.cuda.empty_cache()
        print({"batch_size_final": batch_size, "max_length": max_length}, flush=True)

    steps = args.steps if args.mode == "sanity" else max(args.steps, 50)
    history = []
    val_history = []
    sop_nonzero = 0
    best_val = float("inf")
    best_step = 0
    patience_left = args.early_stop_patience
    stopped_early = False
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "training_mode": "transductive_public_pretraining",
        "contains_problem_observations": True,
        "contains_problem_images": True,
        "contains_problem_hidden_targets": False,
        "parquet": str(parquet),
        "vocab_size": vocab.size(),
        "device": str(device),
        "status": "running",
        "attempts": [],
        "max_rows": max_rows,
        "max_length": max_length,
        "batch_size": batch_size,
        "sop_reverse_probability": args.sop_reverse,
        "sop_shuffle_probability": args.sop_shuffle,
        "target_vram_frac": args.target_vram_frac,
        "hidden_size": hparams["hidden_size"],
        "n_encoders": hparams["n_encoders"],
        "abspos_reference": abspos_ref["reference_timestamp"],
        "abspos_unit": "hour",
        "age_semantics": "sequence_relative_hours",
        "wandb_enabled": bool(args.wandb),
        "val_split": "shadow_split" if use_shadow_val else "hash_person",
        "val_frac": args.val_frac if not use_shadow_val else None,
        "val_every": args.val_every,
        "val_batches": args.val_batches,
        "early_stop_patience": None if args.no_early_stop else args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "max_steps": steps if args.epochs <= 0 else None,
        "target_epochs": args.epochs if args.epochs > 0 else None,
        "wandb_watch": bool(args.wandb_watch),
        "wandb_watch_log": args.wandb_watch_log,
        "wandb_watch_freq": args.wandb_watch_freq,
    }
    write_json(run_dir / "run_manifest_v2.json", manifest)

    wb_run = None
    try:
        wb_run = init_wandb(args, manifest, abspos_ref, model=model)
        if wb_run is not None:
            manifest["wandb_run_id"] = wb_run.id
            manifest["wandb_url"] = wb_run.url
            write_json(run_dir / "run_manifest_v2.json", manifest)
            print({"wandb_url": wb_run.url, "wandb_run_id": wb_run.id}, flush=True)
    except SystemExit:
        raise
    except Exception as exc:
        if args.wandb:
            raise SystemExit(f"wandb.init failed: {exc}. Use --no-wandb to skip.") from exc

    from src.transformer.models import masked_sop_loss

    step = start_step
    epoch = 0
    epoch_boundary_val = False
    data_iter = iter_person_batches(
        parquet, batch_size, max_rows, id_filter=train_filter, events_parquet=events_parquet
    )
    val_iter = iter_person_batches(
        parquet, batch_size, max_rows, id_filter=val_filter, events_parquet=events_parquet
    )

    def _loop_active() -> bool:
        if args.epochs > 0:
            return epoch < args.epochs
        return step < start_step + steps

    try:
        while _loop_active():
            try:
                try:
                    frame, chosen = next(data_iter)
                except StopIteration:
                    epoch += 1
                    epoch_boundary_val = True
                    # Don't break here even if the epoch target is now reached --
                    # let this one extra step run through so the normal end-of-step
                    # val/checkpoint code below actually executes (checkpointing on
                    # a bare `break` here would skip it and leave no last.ckpt/best.ckpt).
                    # The outer while-condition catches the epoch limit next iteration.
                    data_iter = iter_person_batches(
                        parquet, batch_size, max_rows, id_filter=train_filter, events_parquet=events_parquet
                    )
                    frame, chosen = next(data_iter)
                people = list(chosen)
                if len(people) < batch_size:
                    while len(people) < batch_size:
                        people.extend(chosen)
                    people = people[:batch_size]
                batch = _encode_train(frame, people, device)
                opt.zero_grad(set_to_none=True)
                mlm_preds, cls_preds = model(batch)
                mlm_targs = batch["target_tokens"].long()
                cls_targs = batch["target_cls"].long()
                mlm_loss = model.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
                sop_loss = masked_sop_loss(
                    model.cls_loss(cls_preds, target=cls_targs),
                    batch.get("target_cls_mask"),
                )
                loss = model.cls_w * sop_loss + model.mlm_w * mlm_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss {loss}")
                detailed_train_metrics = compute_mlm_sop_metrics(
                    model, sop_metrics, mlm_preds, mlm_targs, cls_preds, cls_targs,
                    batch.get("target_cls_mask"), stage="train",
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                sop_v = float(sop_loss.detach().cpu())
                if sop_v > 0:
                    sop_nonzero += 1
                rec = {
                    "step": step,
                    "train/loss": float(loss.detach().cpu()),
                    "train/mlm_loss": float(mlm_loss.detach().cpu()),
                    "train/mlm_perplexity": float(torch.exp(mlm_loss.detach()).cpu()),
                    "train/sop_loss": sop_v,
                    "train/sop_mask_mean": float(batch["target_cls_mask"].float().mean().cpu())
                    if torch.is_tensor(batch.get("target_cls_mask"))
                    else None,
                    "train/grad_norm": float(grad_norm.detach().cpu())
                    if torch.is_tensor(grad_norm)
                    else float(grad_norm),
                    "batch_size": batch_size,
                    **detailed_train_metrics,
                }
                if device.type == "cuda" and step % 20 == 0:
                    rec["train/vram_alloc_gb"] = round(
                        torch.cuda.memory_allocated(device) / 1e9, 2
                    )
                    rec["train/vram_reserved_gb"] = round(
                        torch.cuda.memory_reserved(device) / 1e9, 2
                    )
                    rec["train/vram_frac"] = round(
                        torch.cuda.memory_reserved(device)
                        / torch.cuda.get_device_properties(device).total_memory,
                        3,
                    )
                history.append(rec)
                print(rec, flush=True)
                if wb_run is not None:
                    import wandb

                    wandb.log({k: v for k, v in rec.items() if k != "step"}, step=step)

                do_val = (
                    step % args.val_every == 0
                    or (args.epochs <= 0 and step == start_step + steps)
                    or epoch_boundary_val
                )
                epoch_boundary_val = False
                if do_val:
                    val_metrics = evaluate_validation(
                        model,
                        _encode_val,
                        val_iter,
                        device,
                        max_batches=args.val_batches,
                        sop_metrics=sop_metrics,
                    )
                    if val_metrics["val/batches"] == 0:
                        # Restart val iterator if exhausted mid-run.
                        val_iter = iter_person_batches(
                            parquet, batch_size, max_rows, id_filter=val_filter, events_parquet=events_parquet
                        )
                        val_metrics = evaluate_validation(
                            model,
                            _encode_val,
                            val_iter,
                            device,
                            max_batches=args.val_batches,
                            sop_metrics=sop_metrics,
                        )
                    val_metrics["val/mlm_perplexity"] = float(np.exp(val_metrics["val/mlm_loss"]))
                    val_rec = {"step": step, **val_metrics}
                    val_history.append(val_rec)
                    print(val_rec, flush=True)
                    if wb_run is not None:
                        import wandb

                        wandb.log(
                            {k: v for k, v in val_rec.items() if k != "step"},
                            step=step,
                        )

                    val_loss = float(val_metrics["val/loss"])
                    improved = val_loss < (best_val - args.early_stop_min_delta)
                    if improved:
                        best_val = val_loss
                        best_step = step
                        patience_left = args.early_stop_patience
                        best_path = run_dir / "best.ckpt"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "opt": opt.state_dict(),
                                "step": step,
                                "best_val_loss": best_val,
                                "abspos_reference": abspos_ref["reference_timestamp"],
                                "hparams": {
                                    k: v for k, v in hparams.items() if k != "vocabulary"
                                },
                            },
                            best_path,
                        )
                        print(
                            {
                                "best_ckpt": str(best_path),
                                "best_val_loss": best_val,
                                "step": step,
                            },
                            flush=True,
                        )
                    elif not args.no_early_stop:
                        patience_left -= 1
                        print(
                            {
                                "early_stop_patience_left": patience_left,
                                "best_val_loss": best_val,
                                "val_loss": val_loss,
                            },
                            flush=True,
                        )
                        if patience_left <= 0:
                            stopped_early = True
                            print(
                                {
                                    "EARLY_STOP": True,
                                    "best_step": best_step,
                                    "best_val_loss": best_val,
                                    "step": step,
                                },
                                flush=True,
                            )

                if (
                    step % args.ckpt_every == 0
                    or (args.epochs <= 0 and step == start_step + steps)
                    or (args.epochs > 0 and not _loop_active())
                    or stopped_early
                ):
                    ckpt_path = run_dir / f"checkpoint_step_{step}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "step": step,
                            "hparams": {
                                k: v for k, v in hparams.items() if k != "vocabulary"
                            },
                            "abspos_reference": abspos_ref["reference_timestamp"],
                            "best_val_loss": best_val if best_val < float("inf") else None,
                        },
                        ckpt_path,
                    )
                    last = run_dir / "last.ckpt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "step": step,
                            "abspos_reference": abspos_ref["reference_timestamp"],
                            "best_val_loss": best_val if best_val < float("inf") else None,
                        },
                        last,
                    )
                    print(f"saved {ckpt_path}", flush=True)
                if stopped_early:
                    break
            except RuntimeError as exc:
                msg = str(exc).lower()
                manifest["attempts"].append(
                    {"step": step, "error": str(exc), "batch_size": batch_size}
                )
                write_json(run_dir / "run_manifest_v2.json", manifest)
                if "out of memory" in msg:
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        data_iter = iter_person_batches(
                            parquet, batch_size, max_rows, id_filter=train_filter, events_parquet=events_parquet
                        )
                        print(f"OOM_RECOVERY batch_size->{batch_size}", flush=True)
                        torch.cuda.empty_cache()
                        continue
                raise

        # Prefer best.ckpt for resume verify when early-stopped / validated.
        verify_path = run_dir / "best.ckpt"
        if not verify_path.exists():
            verify_path = run_dir / "last.ckpt"
        verify = torch.load(verify_path, map_location=device)
        model.load_state_dict(verify["model"])
        print(
            {
                "resume_verify_step": verify["step"],
                "verify_ckpt": str(verify_path),
                "pass": True,
            },
            flush=True,
        )

        manifest.update(
            {
                "status": "PASS",
                "final_step": step,
                "final_epoch": epoch,
                "stopped_early": stopped_early,
                "best_val_loss": best_val if best_val < float("inf") else None,
                "best_step": best_step if best_step else None,
                "history_tail": history[-5:],
                "val_history_tail": val_history[-5:],
                "batch_size_final": batch_size,
                "sop_nonzero_steps": sop_nonzero,
                "sop_nonzero_rate": round(sop_nonzero / max(len(history), 1), 4),
                "checkpoint": str(run_dir / "last.ckpt"),
                "best_checkpoint": str(run_dir / "best.ckpt")
                if (run_dir / "best.ckpt").exists()
                else None,
                "resume_command": (
                    f"python scripts/online2_v2/run_v2_pretrain_loop.py --mode {args.mode} "
                    f"--build-dir {build} --run-dir {run_dir} --resume {run_dir/'last.ckpt'} "
                    f"--steps {steps} --batch-size {batch_size} --max-rows {args.max_rows} "
                    f"--max-length {max_length}"
                ),
            }
        )
        write_json(run_dir / "run_manifest_v2.json", manifest)
        write_json(
            run_dir / "train_history.json",
            {"history": history, "val_history": val_history},
        )
        print("DONE", manifest["status"], manifest["final_step"], flush=True)
    finally:
        if wb_run is not None:
            import wandb

            try:
                wandb.unwatch()
            except Exception:
                pass
            wandb.finish()


if __name__ == "__main__":
    main()
