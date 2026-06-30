#!/usr/bin/env bash
# Project environment setup for life2vec / Berry2Vec experiments.
#
# Usage:
#   source setup.sh          # configure env vars in current shell
#   bash setup.sh            # create conda env and install dependencies
#
# W&B credentials: copy setup.local.sh.example -> setup.local.sh and set WANDB_API_KEY.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_NAME="${ENV_NAME:-life2vec}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# --- W&B defaults (match wandb.init entity/project in your Berry2Vec workspace) ---
export WANDB_ENTITY="${WANDB_ENTITY:-dasom-oh}"
export WANDB_PROJECT="${WANDB_PROJECT:-Berry2Vec}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Load local secrets (API key etc.) — never commit setup.local.sh
if [[ -f "${SCRIPT_DIR}/setup.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/setup.local.sh"
fi

configure_wandb() {
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[setup] WANDB_API_KEY is not set."
    echo "        Create setup.local.sh from setup.local.sh.example, or run: wandb login"
  else
    echo "[setup] W&B configured: entity=${WANDB_ENTITY}, project=${WANDB_PROJECT}"
  fi
}

install_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] conda env '${ENV_NAME}' already exists"
  else
    echo "[setup] creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"

  echo "[setup] installing Python dependencies"
  pip install -r requirements.txt
  pip install -e .

  configure_wandb
  echo "[setup] done. Run: conda activate ${ENV_NAME}"
}

# Sourced: only export env vars. Executed: install dependencies.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  configure_wandb
else
  install_env
fi
