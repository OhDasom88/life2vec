#!/usr/bin/env bash
# Launch a W&B sweep for Agri fruiting finetune hyperparameter search.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f setup.sh ]]; then
  # shellcheck source=/dev/null
  source setup.sh
fi

echo "Creating sweep (project: Berry2Vec)..."
wandb sweep conf/sweep/agri_fruiting.yaml

echo ""
echo "Then run the agent, e.g.:"
echo "  wandb agent dasom-oh/Berry2Vec/<sweep_id>"
