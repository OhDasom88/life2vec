"""Trainable adapter for already-materialized frozen external embeddings.

This module intentionally does not load or execute a VLM/LLM encoder.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class ExternalEmbeddingSpec:
    dimension: int
    model_id: str
    preprocessing_version: str
    checksum: str

    def validate(self, metadata: Mapping[str, object]) -> None:
        expected = {
            "dimension": self.dimension,
            "model_id": self.model_id,
            "preprocessing_version": self.preprocessing_version,
            "checksum": self.checksum,
        }
        mismatches = {
            key: (value, metadata.get(key))
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"External embedding metadata mismatch: {mismatches}")

    def validate_file(self, path: str) -> None:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != self.checksum:
            raise ValueError(
                f"External embedding checksum mismatch: expected {self.checksum}, "
                f"found {digest}"
            )


class ExternalEmbeddingProjection(nn.Module):
    """Detach frozen inputs, then apply trainable projection, norm and gate."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        missing_policy: Literal["error", "zero", "learned"] = "error",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if missing_policy not in {"error", "zero", "learned"}:
            raise ValueError(f"Unsupported missing_policy={missing_policy!r}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.missing_policy = missing_policy
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.missing = (
            nn.Parameter(torch.zeros(input_dim))
            if missing_policy == "learned"
            else None
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        present_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected external embedding dim {self.input_dim}, "
                f"found {embeddings.shape[-1]}"
            )
        values = embeddings.detach()
        if present_mask is not None:
            mask = present_mask.to(device=values.device, dtype=torch.bool)
            if mask.shape != values.shape[:-1]:
                raise ValueError(
                    "present_mask shape must match embeddings without the final dimension"
                )
            if self.missing_policy == "error" and not bool(mask.all()):
                raise ValueError("Missing external embedding under 'error' policy")
            replacement = (
                self.missing.to(device=values.device, dtype=values.dtype)
                if self.missing is not None
                else torch.zeros(self.input_dim, device=values.device, dtype=values.dtype)
            )
            values = torch.where(mask.unsqueeze(-1), values, replacement)

        projected = self.norm(self.projection(values))
        return projected * torch.sigmoid(self.gate(projected))
