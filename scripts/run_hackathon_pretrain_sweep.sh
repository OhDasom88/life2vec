#!/usr/bin/env bash
# Launch W&B sweep for Hackathon greenhouse MLM pretrain.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f setup.sh ]]; then
  # shellcheck source=/dev/null
  source setup.sh
fi

echo "Preparing hackathon data (if needed)..."
conda run -n life2vec python -m src.prepare_data 'target=${data_new.sources.hackathon_environment}' single_threaded=true
conda run -n life2vec python -m src.prepare_data +data_new/population=hackathon_set 'target=${data_new.population}' single_threaded=true
conda run -n life2vec python -m src.prepare_data +data_new/corpus=hackathon_set 'target=${data_new.corpus}' single_threaded=true
conda run -n life2vec python -m src.prepare_data +data_new/vocabulary=hackathon_set 'target=${data_new.vocabulary}' single_threaded=true

echo ""
echo "Creating hackathon pretrain GRID sweep (project: Berry2Vec)..."
conda run -n life2vec wandb sweep conf/sweep/hackathon_pretrain.yaml

echo ""
echo "Then run the agent, e.g.:"
echo "  bash scripts/run_hackathon_pretrain_sweep_agent.sh <sweep_id> 20"
echo ""
echo "After pretrain finishes, update HACKATHON_PRETRAIN_SWEEP_ID in src/wandb_pretrain.py"
echo "and refresh finetune sweep run ids:"
echo "  python scripts/generate_hackathon_sweep_yaml.py --write --sweep-id <pretrain_sweep_id>"
