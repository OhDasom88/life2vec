#!/usr/bin/env bash
# Create (optional) + run wandb agent for diagnosis finetune v0.2.
#
# Usage:
#   # create new sweep then run N trials
#   bash scripts/online2_v2/v02/run_diagnosis_finetune_v02_sweep.sh --create --count 20
#
#   # attach to existing sweep
#   SWEEP_ID=xxxx bash scripts/online2_v2/v02/run_diagnosis_finetune_v02_sweep.sh --count 20

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

ENTITY="${WANDB_ENTITY:-dasom-oh}"
PROJECT="${WANDB_PROJECT:-Berry2Vec}"
SWEEP_YAML="${SWEEP_YAML:-conf/sweep/diagnosis_finetune_v02.yaml}"
COUNT="${COUNT:-20}"
CREATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --create) CREATE=1; shift ;;
    --count) COUNT="$2"; shift 2 ;;
    --sweep-id) SWEEP_ID="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

export PATH="/home/dasom/miniconda3/envs/life2vec/bin:${PATH}"

if [[ "$CREATE" -eq 1 || -z "${SWEEP_ID:-}" ]]; then
  echo "Creating sweep from ${SWEEP_YAML} ..."
  OUT="$(wandb sweep --project "$PROJECT" --entity "$ENTITY" "$SWEEP_YAML" 2>&1)"
  echo "$OUT"
  SWEEP_ID="$(echo "$OUT" | sed -n 's/.*wandb agent [^ ]*\/[^ ]*\/\([a-z0-9]*\).*/\1/p' | tail -1)"
  if [[ -z "$SWEEP_ID" ]]; then
    SWEEP_ID="$(echo "$OUT" | sed -n 's/.*sweeps\/\([a-z0-9]*\).*/\1/p' | tail -1)"
  fi
  [[ -n "$SWEEP_ID" ]] || { echo "Failed to parse SWEEP_ID"; exit 1; }
fi

echo "Agent: ${ENTITY}/${PROJECT}/${SWEEP_ID}  count=${COUNT}"
exec wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}" --count "$COUNT"
