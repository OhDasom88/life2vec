#!/usr/bin/env python3
"""Step 3 entrypoint: MLM decoder preflight (hard fail; warning-only PASS forbidden)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_new.vocabulary import RegistryVocabulary
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    dump_runtime_imports_from_env,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.mlm_preflight import (
    run_mlm_decoder_preflight,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths
from src.online2.v2.finetune_v03.counterfactual.stage_a_loader import load_stage_a_module


def main() -> int:
    exit_code = 1
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
        ap.add_argument("--golden-logits", type=Path, default=None)
        args = ap.parse_args()

        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        out_root = Path(cfg["output_root"])
        paths = M1Paths(out_root)
        device = torch.device(str(cfg.get("device", "cpu")))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        stage_a = load_stage_a_module()
        vocab = RegistryVocabulary(
            registry_path=str(cfg.get("vocabulary_path") or cfg["tokenizer_path"]),
            registry_version="v2",
        )
        encoder, _hp, _abspos = stage_a.load_frozen_encoder(
            Path(cfg["stage_a_ckpt"]), vocab, device
        )
        encoder = encoder.to(device)
        encoder.eval()

        result = run_mlm_decoder_preflight(
            encoder,
            vocab_size=int(vocab.size()),
            device=device,
            golden_logits_path=args.golden_logits,
        )
        write_json(paths.artifacts / "mlm_decoder_preflight.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        exit_code = 0 if result.get("decoder_preflight_passed") else 1
        return exit_code
    finally:
        dump_runtime_imports_from_env(ROOT, entrypoint=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
