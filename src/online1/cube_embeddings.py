from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _weights_sha_placeholder(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


class FrozenImageEncoder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)
        self.weight_sha = _weights_sha_placeholder(self.model.state_dict())
        self.transform = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.meta = {
            "name": "torchvision.models.resnet18",
            "weights": "IMAGENET1K_V1",
            "frozen": True,
            "weight_sha": self.weight_sha,
            "device": self.device,
        }

    @torch.no_grad()
    def encode_rgb01(self, rgb01: np.ndarray) -> np.ndarray:
        """rgb01: HxWx3 float in [0,1] → embedding vector."""
        x = torch.from_numpy(rgb01).permute(2, 0, 1).float().unsqueeze(0)
        x = self.transform(x).to(self.device)
        emb = self.model(x)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def encode_bands(self, band_rgbs: list[np.ndarray]) -> np.ndarray:
        embs = [self.encode_rgb01(img) for img in band_rgbs]
        return np.stack(embs, axis=0)

    def verify_frozen(self) -> dict[str, Any]:
        sha2 = _weights_sha_placeholder(self.model.state_dict())
        grads = [float(p.grad.norm().item()) for p in self.model.parameters() if p.grad is not None]
        return {
            "weight_sha": self.weight_sha,
            "weight_sha_now": sha2,
            "frozen_weight_sha_changed": sha2 != self.weight_sha,
            "grad_norm_max": max(grads) if grads else 0.0,
        }


def pool_band_embeddings(embs: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        return embs.mean(axis=0)
    if mode == "max":
        return embs.max(axis=0)
    raise ValueError(mode)


def save_embedding_npz(path: Path, cube_id: str, band_embs: np.ndarray, pooled: np.ndarray, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cube_id=cube_id,
        band_embeddings=band_embs.astype(np.float32),
        pooled_embedding=pooled.astype(np.float32),
        meta_json=np.array([str(meta)]),
    )
