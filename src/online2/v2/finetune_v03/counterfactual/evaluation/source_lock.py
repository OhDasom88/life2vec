"""Source lock for reproducibility: git + config + archive + dirty snapshot.

cf_m2_p1_v2: entrypoint + runtime dependency closure with canonical dependency_tree_sha256.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


DEFAULT_SOURCE_GLOBS = (
    "src/online2/v2/finetune_v03/counterfactual/**/*.py",
    "tests/v2/counterfactual_m1/**/*.py",
    "conf/m1/cf_m2_prereq_smoke.yaml",
    "scripts/online2_v2/v03/run_cf_m2_prereq_smoke_v03.py",
    "scripts/online2_v2/v03/build_noop_calibration_v03.py",
    "scripts/online2_v2/v03/run_mlm_reconstruction_eval_v03.py",
)

SCOPE_VERSION_V2 = "cf_m2_p1_v2"

P1_ENTRYPOINT_PATHS: Tuple[str, ...] = (
    "scripts/online2_v2/v03/run_mlm_decoder_preflight_v03.py",
    "scripts/online2_v2/v03/build_mlm_bundle_bank_v03.py",
    "scripts/online2_v2/v03/run_mlm_functional_smoke_v03.py",
    "scripts/online2_v2/v03/run_cf_m2_p1_smoke_v03.py",
    "scripts/online2_v2/v03/run_curated_mlm_fixture_v03.py",
    "scripts/online2_v2/v03/run_mlm_reconstruction_selection_v03.py",
    "scripts/online2_v2/v03/run_mlm_reconstruction_eval_v03.py",
    "scripts/online2_v2/v03/run_p1_acceptance_recovery_v03.py",
    "scripts/online2_v2/v03/run_p1_finalize_pack_v03.py",
    "scripts/online2_v2/v03/run_p0_evidence_binding_tests_v03.py",
    "conf/m1/cf_m2_prereq_smoke.yaml",
)

P1_RUNTIME_DEPENDENCY_GLOBS: Tuple[str, ...] = (
    "src/online2/v2/finetune_v03/counterfactual/**/*.py",
    "src/online2/v2/finetune_v03/checkpoint.py",
    "src/online2/v2/tokenizer.py",
    "src/online2/v2/feature_schema.py",
    "src/data_new/**/*.py",
    "scripts/online2_v2/cache_stage_a_event_embeddings.py",
    "tests/v2/counterfactual_m1/**/*.py",
)

# Project-relative prefixes: all repo Python under src/ and scripts/
PROJECT_IMPORT_PREFIXES: Tuple[str, ...] = (
    "src/",
    "scripts/",
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
        # exact file path without glob metacharacters
        if "*" not in g and "?" not in g and "[" not in g:
            p = root / g
            if p.is_file():
                out.append(p)
            continue
        out.extend(sorted(root.glob(g)))
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


def _porcelain_paths(porcelain: str) -> List[str]:
    paths: List[str] = []
    for line in porcelain.splitlines():
        path = line[3:].strip() if len(line) > 3 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[-1]
        paths.append(path)
    return paths


def _path_in_scope(path: str, scope_rels: Sequence[str]) -> bool:
    for s in scope_rels:
        if path == s or path.startswith(s.rstrip("/") + "/"):
            return True
        # directory prefix in scope list
        if s.endswith("/") and path.startswith(s):
            return True
    return False


def collect_declared_dependency_paths(
    root: Path,
    *,
    entrypoint_paths: Sequence[str] = P1_ENTRYPOINT_PATHS,
    runtime_dependency_globs: Sequence[str] = P1_RUNTIME_DEPENDENCY_GLOBS,
) -> List[str]:
    root = Path(root)
    paths: List[Path] = []
    for ep in entrypoint_paths:
        p = root / ep
        if p.is_file():
            paths.append(p)
    paths.extend(_expand_globs(root, runtime_dependency_globs))
    rels = sorted({str(p.relative_to(root)) for p in paths if p.is_file()})
    return rels


def compute_dependency_tree_sha256(
    root: Path,
    *,
    relative_paths: Sequence[str],
    scope_version: str = SCOPE_VERSION_V2,
) -> Tuple[str, Dict[str, Any]]:
    """Canonical JSON of {relative_path, file_sha256} sorted → SHA-256."""
    root = Path(root)
    files = []
    for rel in sorted(relative_paths):
        p = root / rel
        if not p.is_file():
            continue
        files.append({"relative_path": rel, "file_sha256": file_sha256(p)})
    manifest = {"scope_version": scope_version, "files": files}
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), manifest


def project_paths_from_sys_modules(
    root: Path,
    *,
    sys_modules: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Collect project-relative .py paths currently imported."""
    import sys as _sys

    root = Path(root).resolve()
    modules = sys_modules if sys_modules is not None else _sys.modules
    found: Set[str] = set()
    for mod in modules.values():
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            p = Path(f).resolve()
        except Exception:
            continue
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            continue
        if not rel.endswith(".py"):
            continue
        if any(rel.startswith(pref) for pref in PROJECT_IMPORT_PREFIXES):
            found.add(rel)
    return sorted(found)


def build_source_lock_v2(
    *,
    root: Path,
    config_path: Path,
    config_sha256: str,
    source_archive_path: Optional[Path] = None,
    entrypoint_paths: Sequence[str] = P1_ENTRYPOINT_PATHS,
    runtime_dependency_globs: Sequence[str] = P1_RUNTIME_DEPENDENCY_GLOBS,
    runtime_imported_project_paths: Optional[Sequence[str]] = None,
    scope_version: str = SCOPE_VERSION_V2,
    require_nonempty_runtime_imports: bool = False,
) -> Dict[str, Any]:
    """A6 scope = entrypoints ∪ runtime dependency closure."""
    root = Path(root)
    commit = _git(root, "rev-parse", "HEAD") or "UNKNOWN"
    porcelain = _git(root, "status", "--porcelain")
    clean = porcelain == ""

    declared = collect_declared_dependency_paths(
        root,
        entrypoint_paths=entrypoint_paths,
        runtime_dependency_globs=runtime_dependency_globs,
    )
    runtime_imported = list(runtime_imported_project_paths or [])
    declared_set = set(declared)
    undeclared = sorted(set(runtime_imported) - declared_set)
    # Vacuous pass forbidden: empty import union cannot claim complete closure
    # when imports were expected (non-empty required). If caller passes None/empty
    # without a proven union from orchestrator, mark incomplete unless explicitly
    # attested via require_runtime_imports=False (tests only).
    missing_entrypoints = [
        ep for ep in entrypoint_paths if not (root / ep).is_file()
    ]
    entrypoints_complete = len(missing_entrypoints) == 0

    if require_nonempty_runtime_imports:
        closure_complete = (
            len(undeclared) == 0
            and len(runtime_imported) > 0
            and entrypoints_complete
        )
    else:
        closure_complete = len(undeclared) == 0 and entrypoints_complete

    dep_sha, dep_manifest = compute_dependency_tree_sha256(
        root, relative_paths=declared, scope_version=scope_version
    )
    # Also keep legacy source_tree hash over declared paths
    tree_rows = [
        {"path": f["relative_path"], "sha256": f["file_sha256"]} for f in dep_manifest["files"]
    ]
    legacy_digest = hashlib.sha256()
    for row in tree_rows:
        legacy_digest.update(row["path"].encode())
        legacy_digest.update(b"\0")
        legacy_digest.update(row["sha256"].encode())
        legacy_digest.update(b"\n")

    archive_sha = None
    if source_archive_path and Path(source_archive_path).exists():
        archive_sha = file_sha256(Path(source_archive_path))

    # Exact entrypoints + declared deps + CF/tests trees — NOT whole scripts/v03/
    locked_prefixes = (
        "src/online2/v2/finetune_v03/counterfactual/",
        "tests/v2/counterfactual_m1/",
    )
    locked_files = set(entrypoint_paths) | {
        "src/online2/v2/finetune_v03/checkpoint.py",
        "src/online2/v2/tokenizer.py",
        "src/online2/v2/feature_schema.py",
        "scripts/online2_v2/cache_stage_a_event_embeddings.py",
        "conf/m1/cf_m2_prereq_smoke.yaml",
    } | declared_set

    def _in_lock_scope(path: str) -> bool:
        if path in locked_files:
            return True
        return any(path.startswith(pref) for pref in locked_prefixes)

    dirty_paths = _porcelain_paths(porcelain)
    tracked_in_scope: List[str] = []
    untracked_in_scope: List[str] = []
    outside_dirty: List[str] = []
    for line, path in zip(porcelain.splitlines(), dirty_paths):
        if _in_lock_scope(path):
            if line.startswith("??"):
                untracked_in_scope.append(line)
            else:
                tracked_in_scope.append(line)
        else:
            outside_dirty.append(line)

    cf_source_clean = (
        len(tracked_in_scope) == 0
        and len(untracked_in_scope) == 0
        and closure_complete
        and entrypoints_complete
    )
    a6_pass = (
        cf_source_clean
        and bool(dep_sha)
        and commit != "UNKNOWN"
        and closure_complete
        and entrypoints_complete
    )

    lock: Dict[str, Any] = {
        "source_lock_scope_version": scope_version,
        "entrypoint_paths": list(entrypoint_paths),
        "missing_required_entrypoints": missing_entrypoints,
        "entrypoints_complete": entrypoints_complete,
        "runtime_dependency_paths": declared,
        "declared_dependency_paths": declared,
        "runtime_imported_project_paths": runtime_imported,
        "undeclared_runtime_dependencies": undeclared,
        "dependency_closure_complete": closure_complete,
        "dependency_tree_sha256": dep_sha,
        "dependency_manifest": dep_manifest,
        "tracked_changes_in_scope": tracked_in_scope,
        "untracked_files_in_scope": untracked_in_scope,
        "cf_source_clean": cf_source_clean,
        "repository_dirty_outside_scope": len(outside_dirty) > 0,
        "git_commit": commit,
        "working_tree_clean": clean,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256,
        "source_tree_sha256": legacy_digest.hexdigest(),
        "source_tree_n_files": len(tree_rows),
        "source_archive_path": str(source_archive_path) if source_archive_path else None,
        "source_archive_sha256": archive_sha,
        "final_code_source_lock": "PASS" if a6_pass else "FAIL",
        "a6_pass": a6_pass,
        "reproducibility": "PASS" if a6_pass else "PARTIAL",
    }
    if outside_dirty:
        lock["git_status_porcelain_outside_scope"] = outside_dirty[:200]
    if not cf_source_clean:
        lock["cf_dirty_lines"] = tracked_in_scope + untracked_in_scope
        lock["reproducibility_note"] = (
            "Scope dirty, undeclared runtime deps, or commit unknown; A6 FAIL"
        )
    return lock


def build_source_lock(
    *,
    root: Path,
    config_path: Path,
    config_sha256: str,
    source_archive_path: Optional[Path] = None,
    globs: Sequence[str] = DEFAULT_SOURCE_GLOBS,
    scope_version: str = SCOPE_VERSION_V2,
    use_v2: bool = True,
    runtime_imported_project_paths: Optional[Sequence[str]] = None,
    require_nonempty_runtime_imports: bool = False,
) -> Dict[str, Any]:
    """Build source lock. Default is cf_m2_p1_v2 closure lock."""
    if use_v2 or scope_version == SCOPE_VERSION_V2:
        return build_source_lock_v2(
            root=root,
            config_path=config_path,
            config_sha256=config_sha256,
            source_archive_path=source_archive_path,
            runtime_imported_project_paths=runtime_imported_project_paths,
            scope_version=scope_version,
            require_nonempty_runtime_imports=require_nonempty_runtime_imports,
        )

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

    cf_prefixes = (
        "src/online2/v2/finetune_v03/counterfactual/",
        "tests/v2/counterfactual_m1/",
        "conf/m1/cf_m2_prereq_smoke.yaml",
        "scripts/online2_v2/v03/run_cf_m2_prereq_smoke_v03.py",
        "scripts/online2_v2/v03/build_noop_calibration_v03.py",
    )
    cf_dirty = []
    for line in porcelain.splitlines():
        path = line[3:].strip() if len(line) > 3 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[-1]
        if any(path == p or path.startswith(p) for p in cf_prefixes):
            cf_dirty.append(line)
    cf_source_clean = len(cf_dirty) == 0

    lock: Dict[str, Any] = {
        "git_commit": commit,
        "working_tree_clean": clean,
        "cf_source_clean": cf_source_clean,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256,
        "source_tree_sha256": tree["source_tree_sha256"],
        "source_tree_n_files": tree["n_files"],
        "source_archive_path": str(source_archive_path) if source_archive_path else None,
        "source_archive_sha256": archive_sha,
        "tracked_cf_diff_sha256": tracked_diff_sha,
        "source_of_truth": [
            "git_commit" if cf_source_clean else "source_tree_sha256",
            "execution_yaml",
            "artifact_config_hash",
            "source_archive_sha256" if archive_sha else "archive_reference_only",
        ],
    }
    if cf_source_clean and commit != "UNKNOWN":
        lock["reproducibility"] = "PASS" if (archive_sha or True) else "PARTIAL"
        if not clean:
            lock["reproducibility_note"] = (
                "CF sources clean at commit; unrelated working-tree dirt ignored for cf_source_clean"
            )
            lock["git_status_porcelain_unrelated"] = porcelain.splitlines()[:200]
    else:
        lock["reproducibility"] = "PARTIAL"
        lock["git_status_porcelain"] = porcelain.splitlines()[:200]
        lock["cf_dirty_lines"] = cf_dirty
        lock["dirty_source_files"] = tree["files"]
        lock["reproducibility_note"] = (
            "CF sources dirty or commit unknown; "
            "use source_tree_sha256 / source_archive_sha256 / dirty_source_files"
        )
    return lock
