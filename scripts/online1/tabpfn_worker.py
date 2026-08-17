#!/usr/bin/env python3
"""Standalone TabPFN worker — runs under the `tabpfn_online1` conda env (which already has
tabpfn + a matching torch build installed, unlike the `life2vec` env), invoked via subprocess
from src.online1.regression_eval.try_tabpfn() so the shared life2vec env's torch version
doesn't have to change.

Uses TabPFN-v2 (Apache-licensed, ships via TabPFNRegressor.create_default_for_version) per
/home/dasom/agrichallenge/online1_tabpfn/train.py's own convention — its docstring notes v2
needs no PriorLabs license/token and uses the local cache (~/.cache/tabpfn/), unlike the
default TabPFNRegressor() constructor which now points at the gated v3 model and requires a
TABPFN_TOKEN we don't have. The v2 checkpoint (tabpfn-v2-regressor.ckpt) was already present
in that cache from tabpfn_online1's own past runs.

Usage: tabpfn_worker.py <input_npz> <output_npz>
  input_npz: X_train, y_train (n,3), mask_train (n,3), X_valid
  output_npz on success: pred_valid (n_valid,3)
  output_npz on failure: error (str), feasible=False
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: tabpfn_worker.py <input_npz> <output_npz>", file=sys.stderr)
        return 2
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    Xtr, ytr, mtr, Xva = data["X_train"], data["y_train"], data["mask_train"], data["X_valid"]

    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:  # pragma: no cover
        np.savez(out_path, feasible=False, error=f"import_failed: {e}")
        return 0

    n_targets = ytr.shape[1]
    preds = []
    try:
        for j in range(n_targets):
            m = mtr[:, j].astype(bool)
            if m.sum() < 10:
                preds.append(np.full(len(Xva), float(np.nanmean(ytr[:, j]))))
                continue
            reg = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2,
                device=device,
                n_estimators=8,
                random_state=42,
                ignore_pretraining_limits=True,
            )
            reg.fit(Xtr[m], ytr[m, j])
            preds.append(reg.predict(Xva))
        pred = np.stack(preds, axis=1)
        np.savez(out_path, feasible=True, pred_valid=pred, model_version="v2")
    except Exception as e:
        np.savez(out_path, feasible=False, error=str(e)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
