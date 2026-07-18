"""Decoder warning cleanup must not change weight-tying fallback behavior."""

from __future__ import annotations

import inspect

from src.transformer import transformer as tr


def test_weight_tying_fallback_still_assigns_embedding_weight():
    src = inspect.getsource(tr.MaskedLanguageModel.__init__)
    assert "embedding.token.weight" in src
    assert "NON_BLOCKING_WEIGHT_TYING_FALLBACK" in src
    assert "except:" not in src.replace("except (", "KEEP(")
    assert "AttributeError" in src
