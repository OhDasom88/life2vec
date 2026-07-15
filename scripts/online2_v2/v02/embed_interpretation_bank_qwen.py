#!/usr/bin/env python3
"""Embed interpretation bank texts with Qwen3-Embedding (alignment targets only).

Uses the offline env that built external embeddings:
  /home/dasom/miniconda3/envs/online2-embedding-build

Does NOT feed texts into Stage A. Outputs overwrite/augment:
  outputs/online2/v2_finetune_v02/interpretation_banks/state_cause_bank.parquet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BANK = ROOT / "outputs/online2/v2_finetune_v02/interpretation_banks/state_cause_bank.parquet"
EMBED_PY = "/home/dasom/miniconda3/envs/online2-embedding-build/bin/python"


WORKER = r'''
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

bank_path = Path(sys.argv[1])
model_id = sys.argv[2]
device = sys.argv[3]
df = pd.read_parquet(bank_path)
texts = df["text"].astype(str).tolist()
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
vecs = []
with torch.no_grad():
    for t in texts:
        inputs = tok(t, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            v = out.pooler_output[0].detach().float().cpu().numpy()
        else:
            v = out.last_hidden_state.mean(dim=1)[0].detach().float().cpu().numpy()
        n = float(np.linalg.norm(v) + 1e-8)
        vecs.append((v / n).astype(np.float32).tolist())
df["embedding"] = vecs
df.to_parquet(bank_path, index=False)
meta = {
    "embedder": model_id,
    "dim": int(len(vecs[0])),
    "n_rows": int(len(df)),
    "role": "alignment_target_only",
}
print(json.dumps(meta))
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--python", type=Path, default=Path(EMBED_PY))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bank.exists():
        raise SystemExit(
            f"bank missing: {args.bank}\n"
            "First run: python scripts/online2_v2/v02/build_interpretation_banks.py"
        )
    if not args.python.exists():
        raise SystemExit(f"embedding python missing: {args.python}")

    cmd = [str(args.python), "-c", WORKER, str(args.bank), args.model, args.device]
    print({"cmd": cmd[:3], "bank": str(args.bank)}, flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    # last json line
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    meta = json.loads(lines[-1]) if lines else {"stdout": proc.stdout[-500:]}
    meta_path = args.bank.with_suffix(".qwen.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(meta, flush=True)


if __name__ == "__main__":
    main()
