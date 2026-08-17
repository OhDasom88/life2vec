from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.transformer.transformer import Flat_Decoder, AttentionDecoderL
from src.online1.pretrain_smoke import pad_batch


TARGETS = ["soil_moisture", "soil_ec", "soil_temp"]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    out = {}
    rmses = []
    maes = []
    r2s = []
    nrms = []
    for j, name in enumerate(TARGETS):
        m = mask[:, j].astype(bool)
        if m.sum() == 0:
            out[name] = {"rmse": None, "mae": None, "r2": None, "nrmse": None}
            continue
        yt = y_true[m, j]
        yp = y_pred[m, j]
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp)) if len(yt) > 1 else None
        scale = float(np.std(yt) + 1e-8)
        nrmse = rmse / scale
        out[name] = {"rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse}
        rmses.append(rmse)
        maes.append(mae)
        if r2 is not None:
            r2s.append(r2)
        nrms.append(nrmse)
    out["macro"] = {
        "rmse": float(np.mean(rmses)) if rmses else None,
        "mae": float(np.mean(maes)) if maes else None,
        "r2": float(np.mean(r2s)) if r2s else None,
        "nrmse": float(np.mean(nrms)) if nrms else None,
    }
    return out


def _xy(df: pd.DataFrame, X: np.ndarray):
    y = df[TARGETS].to_numpy(dtype=float)
    mask = np.stack(df["target_mask"].to_numpy()).astype(float)
    # fill nan targets with 0 for estimators that need dense y; mask used in metrics
    y_fill = np.nan_to_num(y, nan=0.0)
    return X, y_fill, mask


def mean_baseline(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, Any]:
    pred = np.zeros((len(valid_df), 3), dtype=float)
    for j, t in enumerate(TARGETS):
        pred[:, j] = float(train_df[t].mean(skipna=True))
    y = valid_df[TARGETS].to_numpy(dtype=float)
    mask = np.stack(valid_df["target_mask"].to_numpy())
    return {"model": "train_global_mean", "metrics": _metrics(y, pred, mask)}


def zone_mean_baseline(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, Any]:
    means = train_df.groupby("zone_id")[TARGETS].mean()
    global_mean = train_df[TARGETS].mean()
    pred = []
    for _, r in valid_df.iterrows():
        z = int(r["zone_id"])
        if z in means.index:
            pred.append(means.loc[z].to_numpy())
        else:
            pred.append(global_mean.to_numpy())
    pred = np.asarray(pred, dtype=float)
    y = valid_df[TARGETS].to_numpy(dtype=float)
    mask = np.stack(valid_df["target_mask"].to_numpy())
    return {"model": "fold_local_zone_mean", "metrics": _metrics(y, pred, mask)}


def fit_predict_sklearn(name: str, model, Xtr, ytr, mask_tr, Xva, yva, mask_va) -> dict[str, Any]:
    # sample weights per target via repeating fit; use rows with any target
    keep = mask_tr.sum(axis=1) > 0
    model.fit(Xtr[keep], ytr[keep])
    pred = model.predict(Xva)
    return {"model": name, "metrics": _metrics(yva, pred, mask_va)}


def life2vec_flat_decoder(Xtr, ytr, mask_tr, Xva, yva, mask_va, epochs: int = 200) -> dict[str, Any]:
    """Real src.transformer.transformer.Flat_Decoder — the same decoder head life2vec
    downstream models (e.g. hexaco_model.py's Transformer_PSY) use on top of frozen CLS
    representations — trained by gradient descent, not a sklearn MLP stand-in."""
    hidden = Xtr.shape[1]
    hparams = SimpleNamespace(
        hidden_size=hidden,
        hidden_ff=max(32, hidden * 2),
        dc_dropout=0.1,
        num_targets=1,
    )
    model = Flat_Decoder(hparams, num_outputs=len(TARGETS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    mtr_t = torch.tensor(mask_tr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).unsqueeze(1)

    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(Xtr_t).squeeze(1)
        denom = mtr_t.sum().clamp_min(1.0)
        loss = ((pred - ytr_t) ** 2 * mtr_t).sum() / denom
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_va = model(Xva_t).squeeze(1).numpy()
    return {"model": "life2vec_flat_decoder", "metrics": _metrics(yva, pred_va, mask_va)}


def attention_pooled_regression(
    model,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    max_length: int,
    device: str,
    epochs: int = 30,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Third summary-vector design: instead of pre-pooling to a fixed vector (CLS or masked
    mean) before regression, train life2vec's own AttentionDecoderL — the learned attention
    pooling + output head used by downstream models like hexaco_model.py's Transformer_PSY
    when hparams.pooled=True — end to end on top of the frozen encoder's per-token hidden
    states. The encoder stays frozen (no_grad); only the attention decoder's weights train."""
    hparams = SimpleNamespace(hidden_size=model.hidden, num_targets=1, dc_dropout=0.1, epsilon=1e-6)
    head = AttentionDecoderL(hparams, num_outputs=len(TARGETS), num_heads=4, init=True).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    model.eval()

    def _batches(df: pd.DataFrame):
        for i in range(0, len(df), batch_size):
            yield df.iloc[i : i + batch_size]

    head.train()
    for _ in range(epochs):
        for bdf in _batches(train_df.sample(frac=1.0, random_state=None)):
            x, m = pad_batch(bdf["token_ids"].tolist(), max_length)
            x, m = x.to(device), m.to(device)
            with torch.no_grad():
                h = model.encode(x, m)
            y = torch.tensor(np.nan_to_num(bdf[TARGETS].to_numpy(dtype=float)), dtype=torch.float32, device=device)
            mt = torch.tensor(np.stack(bdf["target_mask"].to_numpy()), dtype=torch.float32, device=device)
            pred = head(h, m.float()).squeeze(1)
            denom = mt.sum().clamp_min(1.0)
            loss = ((pred - y) ** 2 * mt).sum() / denom
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    head.eval()
    preds, ys, masks = [], [], []
    with torch.no_grad():
        for bdf in _batches(valid_df):
            x, m = pad_batch(bdf["token_ids"].tolist(), max_length)
            x, m = x.to(device), m.to(device)
            h = model.encode(x, m)
            pred = head(h, m.float()).squeeze(1)
            preds.append(pred.cpu().numpy())
            ys.append(bdf[TARGETS].to_numpy(dtype=float))
            masks.append(np.stack(bdf["target_mask"].to_numpy()))
    pred_all = np.concatenate(preds)
    y_all = np.concatenate(ys)
    mask_all = np.concatenate(masks)
    return {"model": "life2vec_attention_decoder_pooled", "metrics": _metrics(y_all, pred_all, mask_all)}


TABPFN_ENV_PYTHON = "/home/dasom/miniconda3/envs/tabpfn_online1/bin/python"
TABPFN_WORKER = str(Path(__file__).resolve().parents[2] / "scripts" / "online1" / "tabpfn_worker.py")


def try_tabpfn(Xtr, ytr, mask_tr, Xva, yva, mask_va) -> dict[str, Any]:
    """The life2vec conda env has no tabpfn installed (a different torch build than
    tabpfn needs). A separate `tabpfn_online1` env (from agrichallenge/online1_tabpfn)
    already has tabpfn + its own torch — this shells out to that env's interpreter via
    tabpfn_worker.py instead of installing tabpfn into the shared life2vec env.

    Uses TabPFN-v2 (Apache-licensed, TabPFNRegressor.create_default_for_version) per
    online1_tabpfn/train.py's own documented convention. The bare TabPFNRegressor()
    constructor now defaults to the gated v3 model and requires a PriorLabs TABPFN_TOKEN
    (https://ux.priorlabs.ai) that was never obtained here — v2 needs no token and its
    weights are already in the shared ~/.cache/tabpfn/ from tabpfn_online1's past runs."""
    import os
    import subprocess
    import tempfile

    if Xtr.shape[0] > 10000 or Xtr.shape[1] > 500:
        return {
            "model": "tabpfn",
            "feasible": False,
            "error": f"shape_gate_failed: {Xtr.shape}",
            "metrics": None,
        }
    if not Path(TABPFN_ENV_PYTHON).exists():
        return {
            "model": "tabpfn",
            "feasible": False,
            "error": f"tabpfn_online1 env interpreter not found at {TABPFN_ENV_PYTHON}",
            "metrics": None,
        }
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.npz"
        out_path = Path(td) / "out.npz"
        np.savez(in_path, X_train=Xtr, y_train=ytr, mask_train=mask_tr, X_valid=Xva)
        try:
            proc = subprocess.run(
                [TABPFN_ENV_PYTHON, TABPFN_WORKER, str(in_path), str(out_path)],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            return {"model": "tabpfn", "feasible": False, "error": "worker_timeout", "metrics": None}
        if proc.returncode != 0 or not out_path.exists():
            return {
                "model": "tabpfn",
                "feasible": False,
                "error": f"worker_failed rc={proc.returncode}: {proc.stderr[-1500:]}",
                "metrics": None,
            }
        result = np.load(out_path, allow_pickle=True)
        if not bool(result["feasible"]):
            return {
                "model": "tabpfn",
                "feasible": False,
                "error": str(result["error"]),
                "metrics": None,
            }
        pred = result["pred_valid"]
        return {"model": "tabpfn", "feasible": True, "metrics": _metrics(yva, pred, mask_va)}


def run_regression_suite(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    X_train: np.ndarray,
    X_valid: np.ndarray,
) -> dict[str, Any]:
    Xtr, ytr, mtr = _xy(train_df, X_train)
    Xva, yva, mva = _xy(valid_df, X_valid)
    results = [
        mean_baseline(train_df, valid_df),
        zone_mean_baseline(train_df, valid_df),
        fit_predict_sklearn(
            "ridge_frozen",
            MultiOutputRegressor(Ridge(alpha=1.0)),
            Xtr, ytr, mtr, Xva, yva, mva,
        ),
        fit_predict_sklearn(
            "elasticnet_frozen",
            MultiOutputRegressor(ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000)),
            Xtr, ytr, mtr, Xva, yva, mva,
        ),
        fit_predict_sklearn(
            "mlp_frozen",
            MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(128,), max_iter=200, random_state=0)),
            Xtr, ytr, mtr, Xva, yva, mva,
        ),
        life2vec_flat_decoder(Xtr, ytr, mtr, Xva, yva, mva),
        try_tabpfn(Xtr, ytr, mtr, Xva, yva, mva),
    ]
    # optional lightgbm
    try:
        from lightgbm import LGBMRegressor

        results.append(
            fit_predict_sklearn(
                "lightgbm_frozen",
                MultiOutputRegressor(LGBMRegressor(n_estimators=100, verbose=-1)),
                Xtr, ytr, mtr, Xva, yva, mva,
            )
        )
    except Exception as e:
        results.append({"model": "lightgbm_frozen", "feasible": False, "error": str(e)[:200]})

    # note: last-known target baseline intentionally excluded
    return {
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "excluded_baselines": ["previous_soil_target_last_known"],
        "results": results,
    }
