#!/usr/bin/env bash
# Launch a W&B sweep for Agri fruiting finetune (downstream hyperparameter search).
# Uses pretrained encoders from W&B pretrain sweep ugf1f2ld.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f setup.sh ]]; then
  # shellcheck source=/dev/null
  source setup.sh
fi

echo "Checking pretrain weights and overfit cohort (40 samples, train=val=test)..."
conda run -n life2vec python scripts/generate_fruiting_sweep_yaml.py --list --top-k 12
conda run -n life2vec python -c "
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path
conf_dir = str((Path('.')/'conf').resolve())
with initialize_config_dir(config_dir=conf_dir, version_base='1.3'):
    cfg = compose(config_name='config', overrides=['experiment=sweep_agri_fruiting'])
dm = instantiate(cfg.datamodule, _convert_='all')
dm.setup()
s = dm.corpus.population.data_split()
print(f'  cohort: train={len(s.train)} val={len(s.val)} test={len(s.test)} weighted_sampling={dm.weighted_sampling}')
assert len(s.train)==len(s.val)==40
"

echo ""
echo "Creating fruiting finetune GRID sweep — overfit sanity (project: Berry2Vec)..."
COMBOS=$((12 * 2 * 2 * 3 * 4 * 3 * 3 * 2 * 2 * 2 * 3))
echo "  total combinations (max): ${COMBOS}"
conda run -n life2vec wandb sweep conf/sweep/agri_fruiting.yaml

echo ""
echo "Then run the agent, e.g.:"
echo "  bash scripts/run_fruiting_sweep_agent.sh <sweep_id> 50"
echo ""
echo "W&B dashboard: parallel coordinates with pretrain/* and finetune params."
