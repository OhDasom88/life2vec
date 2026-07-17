"""Execution provenance for A7 — single-attempt atomicity + artifact manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


REQUIRED_RUN_IDS: Tuple[str, ...] = (
    "decoder_preflight",
    "bundle_bank_build",
    "functional_smoke",
    "curated_fixture",
    "natural_path_a",
    "reconstruction_selection",
    "reconstruction_eval",
    "pytest",
)

REQUIRED_RUN_SEQUENCE: Tuple[str, ...] = REQUIRED_RUN_IDS

REQUIRED_RUN_KEYS = (
    "run_id",
    "entrypoint",
    "git_commit",
    "dependency_tree_sha256",
    "exit_code",
    "pre_dependency_tree_sha256",
    "post_dependency_tree_sha256",
    "execution_attempt_id",
    "orchestrator_invocation_id",
    "final_lock_id",
    "run_sequence",
    "log_sha256",
    "artifact_manifest_sha256",
    "artifacts",
    "config_sha256",
    "stage_a_checkpoint_sha256",
    "vocab_sha256",
)

A8_1_REQUIRED_CONSUMER_RUN_IDS: Tuple[str, ...] = (
    "functional_smoke",
    "curated_fixture",
    "natural_path_a",
    "reconstruction_selection",
    "reconstruction_eval",
)

BANK_REQUIRED_RUN_IDS = frozenset(
    {
        "bundle_bank_build",
        "functional_smoke",
        "curated_fixture",
        "natural_path_a",
        "reconstruction_selection",
        "reconstruction_eval",
    }
)


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_artifact_manifest_sha256(
    *,
    run_id: str,
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    rows = sorted(
        [
            {
                "relative_path": str(a.get("relative_path") or ""),
                "sha256": str(a.get("sha256") or ""),
            }
            for a in artifacts
        ],
        key=lambda r: r["relative_path"],
    )
    payload = {"run_id": str(run_id), "artifacts": rows}
    return sha256_bytes(canonical_json_bytes(payload))


def build_artifact_list(
    root: Path,
    relative_paths: Sequence[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    root = Path(root)
    for rel in relative_paths:
        p = root / rel
        if not p.is_file():
            out.append({"relative_path": rel, "sha256": "", "missing": "true"})
            continue
        out.append({"relative_path": rel, "sha256": file_sha256(p)})
    return out


def dump_runtime_imports(
    *,
    root: Path,
    out_path: Path,
    run_id: str,
    entrypoint: str = "",
) -> List[str]:
    """Write child-process project imports for orchestrator collection."""
    from .source_lock import project_paths_from_sys_modules

    paths = project_paths_from_sys_modules(Path(root))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "run_id": str(run_id),
        "entrypoint": str(entrypoint),
        "runtime_imported_project_paths": paths,
    }
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def load_runtime_imports(path: Path) -> List[str]:
    path = Path(path)
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [str(p) for p in (doc.get("runtime_imported_project_paths") or [])]


def dump_runtime_imports_from_env(root: Path, *, entrypoint: str = "") -> Optional[List[str]]:
    """If CF_M2_RUNTIME_IMPORTS_PATH is set, dump imports there (child entrypoints)."""
    out = os.environ.get("CF_M2_RUNTIME_IMPORTS_PATH")
    if not out:
        return None
    run_id = os.environ.get("CF_M2_RUN_ID", "")
    return dump_runtime_imports(
        root=Path(root),
        out_path=Path(out),
        run_id=run_id,
        entrypoint=entrypoint or os.environ.get("CF_M2_ENTRYPOINT", ""),
    )


def build_run_record(
    *,
    run_id: str,
    entrypoint: str,
    git_commit: str,
    dependency_tree_sha256: str,
    exit_code: int,
    pre_dependency_tree_sha256: Optional[str] = None,
    post_dependency_tree_sha256: Optional[str] = None,
    pre_source_tree_sha256: Optional[str] = None,
    post_source_tree_sha256: Optional[str] = None,
    execution_attempt_id: str,
    orchestrator_invocation_id: str,
    final_lock_id: str,
    run_sequence: int,
    log_sha256: str,
    artifacts: Sequence[Mapping[str, Any]],
    config_sha256: str,
    stage_a_checkpoint_sha256: str,
    vocab_sha256: str,
    bundle_bank_content_sha256: Optional[str] = None,
    runtime_imported_project_paths: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build run record. Canonical domain is dependency_tree_sha256 only.

    Legacy aliases pre/post_source_tree_sha256 are filled with the same
    canonical dependency hashes for backward-compatible readers.
    """
    pre_dep = str(
        pre_dependency_tree_sha256
        if pre_dependency_tree_sha256 is not None
        else pre_source_tree_sha256
        or ""
    )
    post_dep = str(
        post_dependency_tree_sha256
        if post_dependency_tree_sha256 is not None
        else post_source_tree_sha256
        or pre_dep
    )
    arts = [dict(a) for a in artifacts]
    rec: Dict[str, Any] = {
        "run_id": str(run_id),
        "entrypoint": str(entrypoint),
        "git_commit": str(git_commit),
        "dependency_tree_sha256": str(dependency_tree_sha256),
        "exit_code": int(exit_code),
        "pre_dependency_tree_sha256": pre_dep,
        "post_dependency_tree_sha256": post_dep,
        # Alias: same canonical dependency hash (not legacy source_tree)
        "pre_source_tree_sha256": pre_dep,
        "post_source_tree_sha256": post_dep,
        "execution_attempt_id": str(execution_attempt_id),
        "orchestrator_invocation_id": str(orchestrator_invocation_id),
        "final_lock_id": str(final_lock_id),
        "run_sequence": int(run_sequence),
        "log_sha256": str(log_sha256),
        "artifacts": arts,
        "artifact_manifest_sha256": compute_artifact_manifest_sha256(
            run_id=run_id, artifacts=arts
        ),
        "config_sha256": str(config_sha256),
        "stage_a_checkpoint_sha256": str(stage_a_checkpoint_sha256),
        "vocab_sha256": str(vocab_sha256),
        "runtime_imported_project_paths": list(runtime_imported_project_paths or []),
    }
    if bundle_bank_content_sha256 is not None:
        rec["bundle_bank_content_sha256"] = str(bundle_bank_content_sha256)
    if extra:
        rec["extra"] = dict(extra)
    return rec


def _runs_for_attempt(
    runs: Sequence[Mapping[str, Any]], attempt_id: str
) -> List[Mapping[str, Any]]:
    return [r for r in runs if str(r.get("execution_attempt_id")) == str(attempt_id)]


def evaluate_a7_provenance(
    provenance: Mapping[str, Any],
    *,
    final_lock: Optional[Mapping[str, Any]] = None,
    required_run_ids: Sequence[str] = REQUIRED_RUN_IDS,
    expected_sequence: Sequence[str] = REQUIRED_RUN_SEQUENCE,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """A7 PASS only from a single complete execution_attempt_id."""
    errs: List[str] = []
    all_runs = list(provenance.get("runs") or [])
    if provenance.get("rerun_flag_only"):
        errs.append("rerun_flag_only_forbidden")
    if not all_runs:
        return {"a7_pass": False, "errors": ["missing_runs"], "n_runs": 0}

    attempt_id = provenance.get("active_execution_attempt_id")
    if not attempt_id:
        # fallback: most recent attempt id among runs
        ids = [str(r.get("execution_attempt_id") or "") for r in all_runs]
        ids = [i for i in ids if i]
        attempt_id = ids[-1] if ids else None
    if not attempt_id:
        errs.append("missing_execution_attempt_id")
        return {"a7_pass": False, "errors": errs, "n_runs": len(all_runs)}

    runs = _runs_for_attempt(all_runs, str(attempt_id))
    if not runs:
        errs.append("no_runs_for_active_attempt")
        return {
            "a7_pass": False,
            "errors": errs,
            "n_runs": 0,
            "execution_attempt_id": attempt_id,
        }

    req = list(required_run_ids)
    present = [str(r.get("run_id")) for r in runs]
    # exact once each
    from collections import Counter

    counts = Counter(present)
    for rid in req:
        if counts.get(rid, 0) == 0:
            errs.append(f"missing_required_run:{rid}")
        elif counts.get(rid, 0) > 1:
            errs.append(f"duplicate_required_run:{rid}")
    for rid, n in counts.items():
        if rid not in req and n:
            # extras allowed only as superset of known required; non-required ok
            pass

    # sequence order among required runs
    ordered = sorted(runs, key=lambda r: int(r.get("run_sequence") or 0))
    required_ordered = [str(r.get("run_id")) for r in ordered if str(r.get("run_id")) in req]
    if required_ordered != list(expected_sequence):
        # allow if sequences match expected mapping by run_sequence numbers
        seq_map = {str(r.get("run_id")): int(r.get("run_sequence") or -1) for r in runs}
        expected_seqs = list(range(1, len(expected_sequence) + 1))
        actual_seqs = [seq_map.get(rid, -1) for rid in expected_sequence]
        if actual_seqs != expected_seqs:
            errs.append("run_sequence_mismatch")

    lock_id = None
    commit = None
    dep = None
    cfg = None
    ckpt = None
    vocab = None
    for i, run in enumerate(runs):
        for k in REQUIRED_RUN_KEYS:
            if run.get(k) in (None, ""):
                errs.append(f"run[{i}].missing_{k}")
        if int(run.get("exit_code", 1)) != 0:
            errs.append(f"run[{i}].nonzero_exit")
        pre_dep = str(
            run.get("pre_dependency_tree_sha256") or run.get("pre_source_tree_sha256") or ""
        )
        post_dep = str(
            run.get("post_dependency_tree_sha256") or run.get("post_source_tree_sha256") or ""
        )
        if pre_dep != post_dep:
            errs.append(f"run[{i}].pre_post_mismatch")
        run_dep = str(run.get("dependency_tree_sha256") or "")
        if run_dep and pre_dep and run_dep != pre_dep:
            errs.append(f"run[{i}].dependency_vs_pre_mismatch")

        arts = list(run.get("artifacts") or [])
        expected_man = compute_artifact_manifest_sha256(
            run_id=str(run.get("run_id")), artifacts=arts
        )
        if str(run.get("artifact_manifest_sha256")) != expected_man:
            errs.append(f"run[{i}].artifact_manifest_mismatch")
        for a in arts:
            rel = str(a.get("relative_path") or "")
            sha = str(a.get("sha256") or "")
            if not rel or not sha:
                errs.append(f"run[{i}].artifact_incomplete:{rel}")
                continue
            if root is not None:
                p = Path(root) / rel
                if not p.is_file():
                    errs.append(f"run[{i}].artifact_missing_file:{rel}")
                elif file_sha256(p) != sha:
                    errs.append(f"run[{i}].artifact_hash_mismatch:{rel}")

        rid = str(run.get("run_id"))
        if rid in BANK_REQUIRED_RUN_IDS and not run.get("bundle_bank_content_sha256"):
            errs.append(f"run[{i}].missing_bundle_bank_content_sha256")

        # cross-run identity within attempt
        if lock_id is None:
            lock_id = run.get("final_lock_id")
            commit = run.get("git_commit")
            dep = run.get("dependency_tree_sha256")
            cfg = run.get("config_sha256")
            ckpt = run.get("stage_a_checkpoint_sha256")
            vocab = run.get("vocab_sha256")
        else:
            if run.get("final_lock_id") != lock_id:
                errs.append(f"run[{i}].final_lock_id_mismatch")
            if run.get("git_commit") != commit:
                errs.append(f"run[{i}].git_commit_mismatch")
            if run.get("dependency_tree_sha256") != dep:
                errs.append(f"run[{i}].dependency_tree_mismatch")
            if run.get("config_sha256") != cfg:
                errs.append(f"run[{i}].config_sha256_mismatch")
            if run.get("stage_a_checkpoint_sha256") != ckpt:
                errs.append(f"run[{i}].checkpoint_sha256_mismatch")
            if run.get("vocab_sha256") != vocab:
                errs.append(f"run[{i}].vocab_sha256_mismatch")
            if run.get("execution_attempt_id") != attempt_id:
                errs.append(f"run[{i}].attempt_id_mismatch")

    if final_lock:
        lock_dep = str(final_lock.get("dependency_tree_sha256") or "")
        if dep and lock_dep and str(dep) != lock_dep:
            errs.append("provenance_vs_final_lock_dependency_mismatch")
        # Bind each run's pre/post dependency hash to final_lock.dependency_tree_sha256
        if lock_dep:
            for i, run in enumerate(runs):
                pre_dep = str(
                    run.get("pre_dependency_tree_sha256")
                    or run.get("pre_source_tree_sha256")
                    or ""
                )
                post_dep = str(
                    run.get("post_dependency_tree_sha256")
                    or run.get("post_source_tree_sha256")
                    or ""
                )
                if pre_dep and pre_dep != lock_dep:
                    errs.append(f"run[{i}].pre_vs_final_lock_dependency_mismatch")
                if post_dep and post_dep != lock_dep:
                    errs.append(f"run[{i}].post_vs_final_lock_dependency_mismatch")
        if commit and final_lock.get("git_commit"):
            if str(commit) != str(final_lock["git_commit"]):
                errs.append("provenance_vs_final_lock_commit_mismatch")
        if final_lock.get("cf_source_clean") is False:
            errs.append("final_lock_cf_source_dirty")
        if cfg and final_lock.get("config_sha256"):
            if str(cfg) != str(final_lock["config_sha256"]):
                errs.append("provenance_vs_final_lock_config_mismatch")
        if ckpt and final_lock.get("stage_a_checkpoint_sha256"):
            if str(ckpt) != str(final_lock["stage_a_checkpoint_sha256"]):
                errs.append("provenance_vs_final_lock_checkpoint_mismatch")
        if vocab and final_lock.get("vocab_sha256"):
            if str(vocab) != str(final_lock["vocab_sha256"]):
                errs.append("provenance_vs_final_lock_vocab_mismatch")
        if lock_id and final_lock.get("final_lock_id"):
            if str(lock_id) != str(final_lock["final_lock_id"]):
                errs.append("provenance_vs_final_lock_id_mismatch")
        # Do NOT compare legacy source_tree_sha256 against dependency hashes

    # nonempty runtime import union required for A7 when provenance tracks imports
    union = list(provenance.get("runtime_imported_project_paths_union") or [])
    if not union:
        derived: Set[str] = set()
        for r in runs:
            derived.update(str(p) for p in (r.get("runtime_imported_project_paths") or []))
        union = sorted(derived)
    if not union:
        errs.append("empty_runtime_imported_project_paths_union")

    # Per-required-run nonempty runtime import evidence + artifact SHA binding
    by_run_imports = dict(provenance.get("runtime_imports_by_run") or {})
    for i, run in enumerate(runs):
        rid = str(run.get("run_id") or "")
        if rid not in req:
            continue
        paths_list = list(run.get("runtime_imported_project_paths") or [])
        if not paths_list and rid in by_run_imports:
            paths_list = list(by_run_imports.get(rid) or [])
        if not paths_list:
            errs.append(f"run[{i}].missing_runtime_imports")
            continue
        art_name = f"runtime_imports_{rid}.json"
        art_entry = None
        for a in list(run.get("artifacts") or []):
            if Path(str(a.get("relative_path") or "")).name == art_name:
                art_entry = a
                break
        if root is None:
            # Unit tests without filesystem: nonempty paths sufficient;
            # orchestrator always passes root and must bind artifact SHA.
            continue
        if art_entry is None:
            errs.append(f"run[{i}].runtime_imports_not_in_artifact_manifest")
            continue
        p = Path(root) / str(art_entry.get("relative_path") or "")
        if not p.is_file():
            errs.append(f"run[{i}].runtime_imports_artifact_missing")
            continue
        actual_sha = file_sha256(p)
        if str(art_entry.get("sha256") or "") != actual_sha:
            errs.append(f"run[{i}].runtime_imports_artifact_hash_mismatch")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            errs.append(f"run[{i}].runtime_imports_unreadable")
            continue
        if str(doc.get("run_id") or "") != rid:
            errs.append(f"run[{i}].runtime_imports_run_id_mismatch")
        file_paths = [str(x) for x in (doc.get("runtime_imported_project_paths") or [])]
        if file_paths != [str(x) for x in paths_list]:
            errs.append(f"run[{i}].runtime_imports_paths_mismatch")
        if len(file_paths) <= 0:
            errs.append(f"run[{i}].runtime_imports_empty_file")

    # reject mixed-attempt completion via union of all runs
    other_attempts = {
        str(r.get("execution_attempt_id"))
        for r in all_runs
        if str(r.get("execution_attempt_id")) != str(attempt_id)
    }
    if other_attempts:
        # historical attempts ok; must not use them for completeness
        pass

    passed = len(errs) == 0 and all(counts.get(rid, 0) == 1 for rid in req)
    return {
        "a7_pass": bool(passed),
        "errors": errs,
        "n_runs": len(runs),
        "execution_attempt_id": attempt_id,
        "required_run_ids": list(req),
        "present_run_ids": present,
    }


def write_execution_provenance(path: Path, doc: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(doc), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_execution_provenance(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_execution_run(
    path: Path,
    run: Mapping[str, Any],
    *,
    active_execution_attempt_id: Optional[str] = None,
) -> Dict[str, Any]:
    doc = load_execution_provenance(path)
    runs = list(doc.get("runs") or [])
    runs.append(dict(run))
    doc["runs"] = runs
    attempt = active_execution_attempt_id or run.get("execution_attempt_id")
    if attempt:
        doc["active_execution_attempt_id"] = str(attempt)
    doc["git_commit"] = run.get("git_commit") or doc.get("git_commit")
    doc["dependency_tree_sha256"] = run.get("dependency_tree_sha256") or doc.get(
        "dependency_tree_sha256"
    )
    doc["pre_source_tree_sha256"] = run.get("pre_source_tree_sha256") or doc.get(
        "pre_source_tree_sha256"
    )
    doc["post_source_tree_sha256"] = run.get("post_source_tree_sha256") or doc.get(
        "post_source_tree_sha256"
    )
    doc["rerun_flag_only"] = False
    # accumulate runtime imports by run
    by_run = dict(doc.get("runtime_imports_by_run") or {})
    by_run[str(run.get("run_id"))] = list(run.get("runtime_imported_project_paths") or [])
    doc["runtime_imports_by_run"] = by_run
    union: Set[str] = set()
    for paths in by_run.values():
        union.update(str(p) for p in paths)
    doc["runtime_imported_project_paths_union"] = sorted(union)
    write_execution_provenance(path, doc)
    return doc


def evaluate_a2_functional_smoke(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """A2: dedicated functional smoke artifact only."""
    errs: List[str] = []
    if not artifact:
        return {"a2_pass": False, "errors": ["missing_functional_smoke_artifact"]}
    if str(artifact.get("run_type")) != "FUNCTIONAL_DECODER_SMOKE":
        errs.append("invalid_run_type")
    if int(artifact.get("decoder_forward_call_count") or 0) < 1:
        errs.append("decoder_forward_call_count")
    if int(artifact.get("unique_scored_bundle_count") or 0) < 2:
        errs.append("unique_scored_bundle_count")
    if artifact.get("cf_evaluation_performed") is True:
        errs.append("cf_evaluation_must_be_false")
    if artifact.get("functional_smoke_passed") is not True:
        errs.append("functional_smoke_passed_not_true")
    return {
        "a2_pass": len(errs) == 0,
        "errors": errs,
    }
