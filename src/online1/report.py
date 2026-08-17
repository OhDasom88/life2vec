from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import write_json


def write_phase_report(out_dir: Path, phase: str, payload: dict[str, Any]) -> Path:
    doc = {
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    path = out_dir / f"{phase}_report.json"
    write_json(path, doc)
    return path


def write_markdown_summary(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for h, body in sections:
        lines.append(f"## {h}")
        lines.append("")
        lines.append(body)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
