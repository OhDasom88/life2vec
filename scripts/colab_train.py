#!/usr/bin/env python3
"""Colab VM에서 life2vec 학습 실행 런처.

사용 예:
  colab exec -s life2vec -f scripts/colab_train.py -- pretrain_agri
  colab exec -s life2vec -f scripts/colab_train.py -- finetune_agri_fruiting
"""
from __future__ import annotations

import os
import subprocess
import sys

REMOTE_PROJECT = "/content/life2vec"
DEFAULT_EXPERIMENT = "pretrain_agri"


def main() -> None:
    experiment = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPERIMENT
    project = os.environ.get("LIFE2VEC_PROJECT", REMOTE_PROJECT)

    if not os.path.isdir(project):
        raise SystemExit(
            f"프로젝트 디렉터리가 없습니다: {project}\n"
            "먼저 scripts/train_on_colab.sh sync 를 실행하세요."
        )

    os.chdir(project)
    if project not in sys.path:
        sys.path.insert(0, project)

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"experiment={experiment}",
        "trainer.devices=[0]",
    ]
    env = {**os.environ, "HYDRA_FULL_ERROR": "1"}
    print(f"[colab_train] cwd={os.getcwd()}")
    print(f"[colab_train] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
