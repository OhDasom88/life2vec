#!/usr/bin/env python3
"""Create/use online2-embedding-build env notes and run lightweight E0 checks.

Does not install into the Life2Vec env. Downloads happen only inside the embedding env.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/online2/v2_build/encoder_e0"
ENV_NAME = "online2-embedding-build"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_name": ENV_NAME,
        "life2vec_env_untouched": True,
        "candidates": {
            "text": ["Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-m3", "intfloat/multilingual-e5-base"],
            "image": ["facebook/dinov2-large", "facebook/dinov2-base", "google/siglip2-base-patch16-224"],
        },
        "preferred": {"text": "Qwen/Qwen3-Embedding-0.6B", "image": "facebook/dinov2-large"},
        "steps": [],
    }

    # Check if embedding env exists
    envs = run(["conda", "env", "list"])
    report["steps"].append({"conda_env_list_rc": envs.returncode})
    exists = ENV_NAME in (envs.stdout or "")
    report["embedding_env_exists"] = exists

    if not exists:
        # Create env with torch matching host CUDA without touching life2vec
        create = run(
            [
                "conda",
                "create",
                "-y",
                "-n",
                ENV_NAME,
                "python=3.10",
            ]
        )
        report["steps"].append(
            {
                "create_env_rc": create.returncode,
                "stderr_tail": (create.stderr or "")[-500:],
            }
        )
        exists = create.returncode == 0

    if exists:
        # Install deps inside embedding env only
        pip = run(
            [
                "conda",
                "run",
                "-n",
                ENV_NAME,
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ]
        )
        report["steps"].append({"pip_upgrade_rc": pip.returncode})
        pkgs = [
            "torch==2.6.0+cu124",
            "torchvision==0.21.0+cu124",
            "transformers>=4.51,<5",
            "safetensors>=0.4.5",
            "timm>=1.0.9",
            "pillow>=10",
            "numpy",
            "pandas",
            "pyarrow",
            "tqdm",
            "accelerate>=0.33",
        ]
        install = run(
            [
                "conda",
                "run",
                "-n",
                ENV_NAME,
                "python",
                "-m",
                "pip",
                "install",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu124",
                *pkgs,
            ]
        )
        report["steps"].append(
            {
                "pip_install_rc": install.returncode,
                "stderr_tail": (install.stderr or "")[-1000:],
                "stdout_tail": (install.stdout or "")[-1000:],
            }
        )
        # Probe imports + tiny encode if models available offline
        probe = run(
            [
                "conda",
                "run",
                "-n",
                ENV_NAME,
                "python",
                "-c",
                (
                    "import torch,transformers,safetensors,timm,PIL;"
                    "print('ok', torch.__version__, torch.cuda.is_available(), transformers.__version__)"
                ),
            ]
        )
        report["steps"].append(
            {
                "probe_rc": probe.returncode,
                "probe_out": (probe.stdout or "").strip(),
                "probe_err": (probe.stderr or "")[-500:],
            }
        )
        report["selected"] = {
            "text": "Qwen/Qwen3-Embedding-0.6B",
            "image": "facebook/dinov2-large",
            "note": "Final selection deferred to successful offline encode benchmark; preferred defaults retained if probe PASS.",
            "probe_pass": probe.returncode == 0,
        }
    else:
        report["selected"] = {
            "text": "Qwen/Qwen3-Embedding-0.6B",
            "image": "facebook/dinov2-large",
            "note": "Embedding env creation failed; selection remains planned default.",
            "probe_pass": False,
        }

    (OUT / "encoder_selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
