#!/usr/bin/env bash
# Launch W&B sweep for Hackathon binary CLS finetune.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f setup.sh ]]; then
  # shellcheck source=/dev/null
  source setup.sh
fi

echo "Preparing hackathon CLS data (if needed)..."
conda run -n life2vec python -m src.prepare_data '+data_new/population=hackathon_cls_set' 'target=${data_new.population}' single_threaded=true
conda run -n life2vec python -m src.prepare_data '+data_new/corpus=hackathon_cls_set' 'target=${data_new.corpus}' single_threaded=true
conda run -n life2vec python -m src.prepare_data +datamodule=hackathon_cls_set 'target=${datamodule}' single_threaded=true

echo ""
echo "Optional: refresh pretrained_run_id list from finished pretrain sweep"
conda run -n life2vec python scripts/generate_hackathon_sweep_yaml.py --list --top-k 8 || true

echo ""
echo "Creating hackathon CLS finetune GRID sweep (project: Berry2Vec)..."
conda run -n life2vec wandb sweep conf/sweep/hackathon_finetune.yaml

echo ""
echo "Then run the agent, e.g.:"
echo "  bash scripts/run_hackathon_finetune_sweep_agent.sh <sweep_id> 50"
