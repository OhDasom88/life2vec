"""Source lock for reproducibility: git + config + archive + dirty snapshot."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_SOURCE_GLOBS = (
    "src/online2/v2/finetune_v03/counterfactual/**/*.py",
    "tests/v2/counterfactual_m1/**/*.py",
    "conf/m1/cf_m2_prereq_smoke.yaml",
    "scripts/online2_v2/v03/run_cf_m2_prereq_smoke_v03.py",
    "scripts/online2_v2/v03/build_noop_calibration_v03.py",
)


def _git(cwd: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expand_globs(root: Path, globs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for g in globs:
        out.extend(sorted(root.glob(g)))
    # unique preserve order
    seen = set()
    uniq = []
    for p in out:
        rp = p.resolve()
        if rp in seen or not p.is_file():
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def hash_source_tree(
    root: Path,
    *,
    globs: Sequence[str] = DEFAULT_SOURCE_GLOBS,
) -> Dict[str, Any]:
    files = _expand_globs(root, globs)
    rows = []
    digest = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(root))
        sha = file_sha256(p)
        rows.append({"path": rel, "sha256": sha})
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(sha.encode())
        digest.update(b"\n")
    return {
        "n_files": len(rows),
        "source_tree_sha256": digest.hexdigest(),
        "files": rows,
    }


def build_source_lock(
    *,
    root: Path,
    config_path: Path,
    config_sha256: str,
    source_archive_path: Optional[Path] = None,
    globs: Sequence[str] = DEFAULT_SOURCE_GLOBS,
) -> Dict[str, Any]:
    root = Path(root)
    commit = _git(root, "rev-parse", "HEAD") or "UNKNOWN"
    porcelain = _git(root, "status", "--porcelain")
    clean = porcelain == ""
    tree = hash_source_tree(root, globs=globs)
    archive_sha = None
    if source_archive_path and Path(source_archive_path).exists():
        archive_sha = file_sha256(Path(source_archive_path))

    tracked_diff = _git(root, "diff", "HEAD", "--", "src/online2/v2/finetune_v03/counterfactual")
    tracked_diff_sha = (
        hashlib.sha256(tracked_diff.encode()).hexdigest() if tracked_diff else None
    )

    lock: Dict[str, Any] = {
        "git_commit": commit,
        "working_tree_clean": clean,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256,
        "source_tree_sha256": tree["source_tree_sha256"],
        "source_tree_n_files": tree["n_files"],
        "source_archive_path": str(source_archive_path) if source_archive_path else None,
        "source_archive_sha256": archive_sha,
        "tracked_cf_diff_sha256": tracked_diff_sha,
        "source_of_truth": [
            "git_commit" if clean else "source_tree_sha256",
            "execution_yaml",
            "artifact_config_hash",
            "source_archive_sha256" if archive_sha else "archive_reference_only",
        ],
    }
    if not clean:
        lock["git_status_porcelain"] = porcelain.splitlines()[:200]
        lock["dirty_source_files"] = tree["files"]
        lock["reproducibility"] = "PARTIAL"
        lock["reproducibility_note"] = (
            "working_tree dirty: commit hash alone is insufficient; "
            "use source_tree_sha256 / source_archive_sha256 / dirty_source_files"
        )
    else:
        lock["reproducibility"] = "PASS" if archive_sha or commit != "UNKNOWN" else "PARTIAL"
    return lock
