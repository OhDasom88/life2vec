#!/usr/bin/env bash
# Continues NEW_LINEAGE after pretrain probe completes:
# champion_full → embeddings → finetune sweep → noop calibration.
set -euo pipefail
ROOT=/home/dasom/life2vec
cd "$ROOT"
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=/home/dasom/miniconda3/envs/life2vec/bin/python
REP="$ROOT/outputs/online2/new_lineage_20260718/reports"
SWEEP="$ROOT/outputs/online2/new_lineage_20260718/sweeps"
LOG="$REP/pipeline_continue.log"
mkdir -p "$REP"

exec >>"$LOG" 2>&1
echo "=== pipeline continue start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

PROBE_PID_FILE="$REP/pretrain_probe_sweep.pid"
if [[ -f "$PROBE_PID_FILE" ]]; then
  PROBE_PID=$(cat "$PROBE_PID_FILE")
  echo "waiting for pretrain probe pid=$PROBE_PID"
  while kill -0 "$PROBE_PID" 2>/dev/null; do
    sleep 60
  done
fi

PROBE_JSON="$SWEEP/pretrain_sweep_results.json"
if [[ ! -f "$PROBE_JSON" ]]; then
  echo "ERROR: missing $PROBE_JSON"
  exit 1
fi
echo "probe complete; starting champion_full"

"$PY" scripts/online2_v2/run_new_lineage_pretrain_sweep_v03.py \
  --phase champion_full \
  --champion-from "$PROBE_JSON"

CKPT="$ROOT/outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing champion ckpt $CKPT"
  exit 1
fi
echo "champion ckpt ready; caching event embeddings"

"$PY" scripts/online2_v2/cache_stage_a_event_embeddings.py \
  --ckpt "$CKPT" \
  --out-dir "$ROOT/outputs/online2/v2_finetune_v02/event_embeddings" \
  --manifest "$ROOT/outputs/online2/v2_finetune/case_manifest.csv" \
  --skip-existing \
  --batch-targets 8 \
  --device cuda

echo "embeddings done; starting finetune sweep"

"$PY" scripts/online2_v2/run_new_lineage_finetune_sweep_v03.py

echo "finetune champion locked; building noop calibration"

"$PY" scripts/online2_v2/v03/build_noop_calibration_v03.py \
  --config conf/m1/cf_m2_prereq_smoke.yaml \
  --n-samples 20 \
  --out outputs/cf_calibration/noop_noise.json

"$PY" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path("/home/dasom/life2vec")
dec = ROOT / "reports/p1_rejection_recovery/recovery_decision.json"
d = json.loads(dec.read_text())
for k in ["stage_a_ckpt", "embeddings_dir", "run_dir", "noop_noise"]:
    if k in d.get("missing", []):
        d["missing"].remove(k)
d["restored"] = sorted(set(d.get("restored", []) + ["stage_a_ckpt", "embeddings_dir", "run_dir", "noop_noise", "pretrain_champion", "finetune_champion"]))
d["next_gate"] = "LOCKED_PREFLIGHT"
d["progress_updates"] = d.get("progress_updates", [])
d["progress_updates"].append({
    "at": datetime.now(timezone.utc).isoformat(),
    "note": "pretrain+finetune champions locked; embeddings+noop calibration ready",
    "still_missing": d.get("missing", []),
})
dec.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
print("recovery_decision updated; missing=", d.get("missing"))
PY

echo "=== pipeline continue done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
