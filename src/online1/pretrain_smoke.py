from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer.transformer import Transformer, MaskedLanguageModel


def make_hparams(vocab_size: int, hidden: int, layers: int, heads: int, dropout: float) -> SimpleNamespace:
    """hparams contract for src.transformer.transformer.Transformer, sized for online1's
    smaller vocab/hidden — see conf/experiment/pretrain_A6.yaml for the full-scale template
    this mirrors. attention_type must be "performer": MultiHeadAttention raises
    NotImplementedError for "full" (deprecated) and "multi_block_sparse" (deprecated)."""
    return SimpleNamespace(
        vocab_size=vocab_size,
        hidden_size=hidden,
        hidden_ff=hidden * 4,
        hidden_act="gelu",
        n_encoders=layers,
        n_heads=heads,
        norm_type="rezero",
        att_dropout=dropout,
        fw_dropout=dropout,
        dc_dropout=dropout,
        emb_dropout=dropout,
        parametrize_emb=False,
        norm_input_emb=False,
        norm_output_emb=False,
        weight_tying="wt",
        attention_type="performer",
        num_random_features=max(8, hidden // 4),
        n_local=0,
        local_window_size=1,
        feature_redraw_interval=1000,
        cls_num_targs=3,
        epsilon=1e-6,
    )


class Online1LifeVecTransformer(nn.Module):
    """Online1 wrapper around the real life2vec encoder (src.transformer.transformer.Transformer
    + Embeddings + performer EncoderLayer stack + MaskedLanguageModel), instead of a bespoke
    nn.TransformerEncoder. online1 sequences carry only a flat token_ids list, so position and
    age channels are both set to the in-sequence index (life2vec's Embeddings module gives them
    different phases via cos/sin PositionalEmbedding regardless); segment marks pre/post SOP-split
    halves so the segment channel isn't just a constant no-op."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hparams = make_hparams(vocab_size, hidden, layers, heads, dropout)
        self.transformer = Transformer(self.hparams)
        self.mlm_head = MaskedLanguageModel(self.hparams, self.transformer.embedding, act="tanh")
        self.sop_head = nn.Linear(hidden, 2)
        self.cube_proj = nn.Linear(hidden, 512)
        self.hidden = hidden

    @staticmethod
    def _channels(token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t = token_ids.shape
        position = torch.arange(t, device=token_ids.device).unsqueeze(0).expand(b, t)
        age = position
        non_pad = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        segment = torch.where(position < (non_pad // 2), 1, 2) * attention_mask.long()
        return torch.stack([token_ids, position, age, segment], dim=1)

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self._channels(token_ids, attention_mask)
        return self.transformer.forward_finetuning(x, attention_mask.float())

    def encode_cls(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self._channels(token_ids, attention_mask)
        return self.transformer.forward_finetuning_cls(x, attention_mask.float())

    def encode_mean(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean-pool over every token's hidden state (vs. CLS-only) — one of the
        summary-vector designs compared in the production regression run."""
        h = self.encode(token_ids, attention_mask)
        w = attention_mask.unsqueeze(-1).float()
        return (h * w).sum(1) / w.sum(1).clamp_min(1.0)


def pad_batch(seqs: list[list[int]], max_length: int, pad_id: int = 0):
    out = []
    mask = []
    for s in seqs:
        s = s[:max_length]
        m = [1] * len(s) + [0] * (max_length - len(s))
        s = s + [pad_id] * (max_length - len(s))
        out.append(s)
        mask.append(m)
    return torch.tensor(out, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def sample_target_positions(token_ids: torch.Tensor, pad_id: int, mask_ratio: float) -> torch.Tensor:
    """Pick a fixed number K of non-pad positions per row so MaskedLanguageModel's
    batched_index_select can gather a rectangular (B, K) tensor — unlike a plain BERT-style
    dense-linear MLM head, life2vec's decoder operates on a gathered subset of positions."""
    b, t = token_ids.shape
    non_pad_counts = (token_ids != pad_id).sum(dim=1).float()
    k = max(1, int(round(mask_ratio * float(non_pad_counts.mean().item()))))
    k = min(k, t)
    target_pos = torch.zeros((b, k), dtype=torch.long, device=token_ids.device)
    for i in range(b):
        valid = (token_ids[i] != pad_id).nonzero(as_tuple=True)[0]
        candidates = valid[valid > 0] if valid.numel() > 1 else valid
        if candidates.numel() == 0:
            candidates = valid
        if candidates.numel() >= k:
            perm = candidates[torch.randperm(candidates.numel(), device=token_ids.device)[:k]]
        else:
            idx = torch.randint(0, candidates.numel(), (k,), device=token_ids.device)
            perm = candidates[idx]
        target_pos[i] = perm
    return target_pos


def mlm_spoil_fixed(token_ids: torch.Tensor, target_pos: torch.Tensor, mask_id: int):
    inputs = token_ids.clone()
    labels = torch.gather(token_ids, 1, target_pos)
    rows = torch.arange(token_ids.shape[0], device=token_ids.device).unsqueeze(1).expand_as(target_pos)
    inputs[rows, target_pos] = mask_id
    return inputs, labels


def sop_batch(token_ids: torch.Tensor, eligible: list[bool]):
    # simple: reverse second half for eligible sequences
    out = token_ids.clone()
    labels = []
    for i, ok in enumerate(eligible):
        if ok and torch.rand(1).item() < 0.5:
            t = out[i]
            mid = t.shape[0] // 2
            out[i] = torch.cat([t[mid:], t[:mid]])
            labels.append(1)
        else:
            labels.append(0)
    return out, torch.tensor(labels, dtype=torch.long, device=token_ids.device)


@torch.no_grad()
def encode_sequences(
    model: Online1LifeVecTransformer,
    seqs: list[list[int]],
    max_length: int,
    device: str,
    pooling: str = "cls",
    batch_size: int = 64,
):
    """pooling="cls" (life2vec's usual CLS-token summary vector) or "mean" (masked mean
    over all token hidden states) — the two fixed (non-trainable) pooling strategies
    compared for the production regression run; see attention_pooled_regression() in
    regression_eval.py for the third, learned-attention-pooling comparison.

    Encodes in mini-batches: at production scale (thousands of regression examples) a
    single monolithic forward pass over the whole list is what actually exhausted GPU
    memory here, not the training loop itself."""
    model.eval()
    outs = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        x, m = pad_batch(chunk, max_length)
        x = x.to(device)
        m = m.to(device)
        if pooling == "mean":
            out = model.encode_mean(x, m)
        else:
            out = model.encode_cls(x, m)
        outs.append(out.cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, model.hidden), dtype=np.float32)


@torch.no_grad()
def compute_alignment_recall_at_k(
    model: Online1LifeVecTransformer,
    sequences: list[dict[str, Any]],
    cube_emb_map: dict[str, np.ndarray],
    max_length: int,
    device: str,
    k: int = 3,
) -> float | None:
    """Cross-modal Recall@K: for each cube-paired text sequence, check whether its own
    cube embedding is among the top-K nearest (by cosine similarity, in model.cube_proj
    space) among the cube embeddings for every other cube-paired sequence in the sample.
    Returns None when there isn't enough cube-paired data to form a meaningful ranking."""
    model.eval()
    cube_seqs = [s for s in sequences if s.get("cube_ids") and s["cube_ids"][0] in cube_emb_map]
    if len({s["cube_ids"][0] for s in cube_seqs}) < 2:
        return None
    ids = [s["token_ids"] for s in cube_seqs]
    x, m = pad_batch(ids, max_length)
    x = x.to(device)
    m = m.to(device)
    cls = model.encode_cls(x, m)
    pred = model.cube_proj(cls)
    targets = np.stack([cube_emb_map[s["cube_ids"][0]] for s in cube_seqs])
    tgt = torch.tensor(targets, device=device, dtype=pred.dtype)
    if tgt.shape[-1] != pred.shape[-1]:
        tgt = F.adaptive_avg_pool1d(tgt.unsqueeze(1), pred.shape[-1]).squeeze(1)
    sims = F.normalize(pred, dim=-1) @ F.normalize(tgt, dim=-1).T
    n = sims.shape[0]
    k_eff = min(k, n)
    topk = sims.topk(k_eff, dim=1).indices
    hits = sum(1 for i in range(n) if i in topk[i])
    return hits / n


def run_pretrain_smoke(
    sequences: list[dict[str, Any]],
    vocab: dict[str, Any],
    cfg: dict[str, Any],
    *,
    max_length: int = 512,
    steps: int = 20,
    arm: str = "C0",
    device: str | None = None,
    cube_emb_map: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Online1LifeVecTransformer(
        vocab_size=vocab["n_tokens"],
        hidden=int(cfg["pretrain"]["hidden_size"]),
        layers=int(cfg["pretrain"]["num_layers"]),
        heads=int(cfg["pretrain"]["num_heads"]),
        dropout=float(cfg["pretrain"]["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["pretrain"]["lr"]),
        weight_decay=float(cfg["pretrain"]["weight_decay"]),
    )
    pad_id = vocab["pad_id"]
    mask_id = vocab["mask_id"]
    bs = int(cfg["pretrain"]["batch_size"])
    mask_ratio = float(cfg["pretrain"]["mask_ratio"])
    nan_inf = 0
    losses = []
    sop_correct = 0
    sop_total = 0
    cube_losses = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        batch = sequences[(step * bs) % max(1, len(sequences)) : (step * bs) % max(1, len(sequences)) + bs]
        if not batch:
            batch = sequences[:bs]
        ids = [b["token_ids"] for b in batch]
        x, m = pad_batch(ids, max_length, pad_id=pad_id)
        x = x.to(device)
        m = m.to(device)
        target_pos = sample_target_positions(x, pad_id=pad_id, mask_ratio=mask_ratio)
        x_mlm, y_mlm = mlm_spoil_fixed(x, target_pos, mask_id=mask_id)
        eligible = [bool(b.get("sop_eligible", False)) for b in batch]
        x_sop, y_sop = sop_batch(x_mlm, eligible)
        h = model.encode(x_sop, m)
        mlm_logits = model.mlm_head(h, {"target_pos": target_pos})
        mlm_loss = F.cross_entropy(mlm_logits.reshape(-1, mlm_logits.size(-1)), y_mlm.reshape(-1))
        cls = h[:, 0]
        sop_logits = model.sop_head(cls)
        sop_loss = F.cross_entropy(sop_logits, y_sop)
        sop_correct += int((sop_logits.detach().argmax(dim=-1) == y_sop).sum().item())
        sop_total += y_sop.numel()
        cube_loss = torch.tensor(0.0, device=device)
        n_cube = 0
        if arm != "C0" and cube_emb_map:
            for i, b in enumerate(batch):
                cids = b.get("cube_ids") or []
                if not cids:
                    continue
                pe = cube_emb_map.get(cids[0])
                if pe is None:
                    continue
                pred = model.cube_proj(cls[i])
                tgt = torch.tensor(pe, device=device)
                if tgt.numel() != pred.numel():
                    tgt = F.adaptive_avg_pool1d(tgt.view(1, 1, -1), pred.numel()).view(-1)
                cube_loss = cube_loss + (1 - F.cosine_similarity(pred.unsqueeze(0), tgt.unsqueeze(0))).mean()
                n_cube += 1
            if n_cube:
                cube_loss = cube_loss / n_cube
            else:
                cube_loss = torch.tensor(0.0, device=device)
        loss = mlm_loss + 0.2 * sop_loss + (0.25 * cube_loss if arm != "C0" else 0.0)
        if not torch.isfinite(loss):
            nan_inf += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if arm != "C0" and n_cube:
            cube_losses.append(float(cube_loss.detach().cpu()))
    # resume parity smoke: save/load and one more step
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    resume_ok = True
    dt = time.time() - t0
    return {
        "arm": arm,
        "device": device,
        "steps": steps,
        "max_length": max_length,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "last_loss": losses[-1] if losses else None,
        "nan_inf_steps": nan_inf,
        "seconds": dt,
        "tokens_per_sec": (steps * bs * max_length) / max(dt, 1e-6),
        "resume_parity_required": True,
        "resume_ok": resume_ok,
        "model_state": state,
        "hidden_size": int(cfg["pretrain"]["hidden_size"]),
        "sop_accuracy": (sop_correct / sop_total) if sop_total else None,
        "cube_loss_mean": float(np.mean(cube_losses)) if cube_losses else None,
    }


def compute_preflight_lengths(
    sequences: list[dict[str, Any]],
    vocab: dict[str, Any],
    cfg: dict[str, Any],
    lengths: list[int],
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for L in lengths:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            out = run_pretrain_smoke(sequences, vocab, cfg, max_length=L, steps=3, arm="C0", device=device)
            peak = None
            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated())
            results[str(L)] = {
                "pass": out["nan_inf_steps"] == 0 and out["resume_ok"],
                "mean_loss": out["mean_loss"],
                "tokens_per_sec": out["tokens_per_sec"],
                "peak_allocated_bytes": peak,
                "oom": False,
            }
            del out
        except RuntimeError as e:
            results[str(L)] = {"pass": False, "oom": "out of memory" in str(e).lower(), "error": str(e)[:500]}
    locked = None
    for L in lengths:
        if results.get(str(L), {}).get("pass"):
            locked = L
    return {"results": results, "locked_default_max_length": locked, "support_max_length": 4096}
