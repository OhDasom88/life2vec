"""Optional W&B logging for M1 metrics."""

from __future__ import annotations
from typing import Any, Dict, Optional


def maybe_log(metrics: Dict[str, Any], *, enabled: bool, project: str, run_name: str, config: Optional[dict] = None) -> None:
    if not enabled:
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        wandb.init(project=project, name=run_name, config=config or {}, reinit=True)
    wandb.log(metrics)
