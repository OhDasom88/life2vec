"""Atomic pytest test-run manifest (source hash bound to log + JUnit)."""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .source_lock import (
    SCOPE_VERSION_V2,
    collect_declared_dependency_paths,
    compute_dependency_tree_sha256,
    file_sha256,
)

TEST_RUN_MANIFEST_VERSION = "p0_gate_test_run_v1"
COUNTERFACTUAL_M1_SUITE = "tests/v2/counterfactual_m1/"

# Fixed relative junit path under root (absolute path is composed at run time).
FIXED_JUNIT_REL = "reports/pytest_counterfactual_m1.xml"
FIXED_LOG_REL = "reports/pytest_counterfactual_m1.log"

FORBIDDEN_PYTEST_FLAGS = frozenset(
    {
        "-k",
        "-m",
        "--ignore",
        "--ignore-glob",
        "--deselect",
        "--lf",
        "--ff",
        "--collect-only",
    }
)


def fixed_pytest_command(*, python_executable: str, junit_xml_absolute: Path) -> List[str]:
    """Exact argv template for full counterfactual_m1 suite (no filters)."""
    return [
        str(python_executable),
        "-m",
        "pytest",
        COUNTERFACTUAL_M1_SUITE,
        "-q",
        "--tb=line",
        f"--junitxml={Path(junit_xml_absolute).resolve()}",
    ]


def parse_junit_counts(junit_path: Path) -> Dict[str, int]:
    """Parse pytest --junitxml counts from testsuite(s)."""
    tree = ET.parse(Path(junit_path))
    root = tree.getroot()
    suites = []
    if root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    elif root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.iter("testsuite"))
    if not suites:
        raise ValueError(f"junit_no_testsuite:{junit_path}")

    collected = 0
    failures = 0
    errors = 0
    skipped = 0
    for suite in suites:
        collected += int(suite.attrib.get("tests") or 0)
        failures += int(suite.attrib.get("failures") or 0)
        errors += int(suite.attrib.get("errors") or 0)
        skipped += int(suite.attrib.get("skipped") or 0)
    passed = max(collected - failures - errors - skipped, 0)
    return {
        "collected_count": collected,
        "passed_count": passed,
        "failed_count": failures,
        "error_count": errors,
        "skipped_count": skipped,
    }


def sha256_path(path: Path) -> str:
    return file_sha256(Path(path))


def sha256_json(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_dependency_tree_sha256(root: Path) -> str:
    root = Path(root)
    declared = collect_declared_dependency_paths(root)
    sha, _ = compute_dependency_tree_sha256(
        root,
        relative_paths=declared,
        scope_version=SCOPE_VERSION_V2,
    )
    return sha


def _flag_name(token: str) -> str:
    """Normalize `--foo=bar` / `-k` to the flag key."""
    t = str(token)
    if t.startswith("--") and "=" in t:
        return t.split("=", 1)[0]
    return t


def command_has_forbidden_suite_shrink_flags(test_command: Sequence[str]) -> List[str]:
    """Scan pytest argv only (tokens after the 'pytest' entry), not ``python -m``."""
    found: List[str] = []
    cmd = [str(x) for x in test_command]
    try:
        pytest_idx = cmd.index("pytest")
    except ValueError:
        # No pytest token — treat whole command as suspect scan range
        pytest_idx = -1
    for tok in cmd[pytest_idx + 1 :]:
        name = _flag_name(tok)
        if name in FORBIDDEN_PYTEST_FLAGS:
            found.append(name)
    return found


def command_matches_fixed_full_suite(
    test_command: Sequence[str],
    *,
    root: Path,
    junit_xml_path: Optional[Path] = None,
    python_executable: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Require exact fixed argv template (no suite-path-only substring match)."""
    errs: List[str] = []
    root = Path(root).resolve()
    junit = Path(junit_xml_path) if junit_xml_path is not None else root / FIXED_JUNIT_REL
    actual = [str(x) for x in test_command]

    forbidden = command_has_forbidden_suite_shrink_flags(actual)
    if forbidden:
        errs.append(f"test_command_forbidden_flags:{','.join(sorted(set(forbidden)))}")

    if not actual:
        errs.append("test_command_empty")
        return False, errs

    py = str(actual[0]) if python_executable is None else str(python_executable)
    # Prefer recorded argv[0] as the executable that was run
    if python_executable is None:
        py = actual[0]
    expected = fixed_pytest_command(
        python_executable=py,
        junit_xml_absolute=junit.resolve(),
    )
    if actual != expected:
        errs.append("test_command_not_fixed_full_suite_argv")

    return len(errs) == 0, errs


# Backward-compatible name used by older call sites / tests
def command_covers_full_counterfactual_m1_suite(test_command: Sequence[str]) -> bool:
    forbidden = command_has_forbidden_suite_shrink_flags(test_command)
    if forbidden:
        return False
    # Weak check only for legacy callers; gate uses command_matches_fixed_full_suite
    cmd = [str(x) for x in test_command]
    if len(cmd) < 6:
        return False
    return (
        "-m" in cmd
        and "pytest" in cmd
        and COUNTERFACTUAL_M1_SUITE.rstrip("/") in " ".join(cmd).replace("\\", "/")
        and "-q" in cmd
        and "--tb=line" in cmd
        and any(str(x).startswith("--junitxml=") for x in cmd)
    )


def build_test_run_manifest(
    *,
    root: Path,
    test_command: Sequence[str],
    pre_dependency_tree_sha256: str,
    post_dependency_tree_sha256: str,
    exit_code: int,
    pytest_log_path: Path,
    junit_xml_path: Path,
    source_scope_version: str = SCOPE_VERSION_V2,
) -> Dict[str, Any]:
    root = Path(root)
    log_rel = str(Path(pytest_log_path).resolve().relative_to(root.resolve()))
    junit_rel = str(Path(junit_xml_path).resolve().relative_to(root.resolve()))
    counts = parse_junit_counts(junit_xml_path)
    return {
        "test_run_manifest_version": TEST_RUN_MANIFEST_VERSION,
        "test_command": [str(x) for x in test_command],
        "working_directory": str(root.resolve()),
        "source_scope_version": source_scope_version,
        "pre_dependency_tree_sha256": str(pre_dependency_tree_sha256),
        "post_dependency_tree_sha256": str(post_dependency_tree_sha256),
        "exit_code": int(exit_code),
        "pytest_log_path": log_rel,
        "pytest_log_sha256": sha256_path(pytest_log_path),
        "junit_xml_path": junit_rel,
        "junit_xml_sha256": sha256_path(junit_xml_path),
        **counts,
    }


def write_test_run_manifest(path: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(manifest)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def load_test_run_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_test_run_manifest(
    *,
    root: Path,
    manifest: Mapping[str, Any],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Verify atomic binding of tested source ↔ log ↔ JUnit. Gate uses this only."""
    root = Path(root).resolve()
    errs: List[str] = []
    details: Dict[str, Any] = {}

    if str(manifest.get("test_run_manifest_version") or "") != TEST_RUN_MANIFEST_VERSION:
        errs.append("test_run_manifest_version_mismatch")

    pre = str(manifest.get("pre_dependency_tree_sha256") or "")
    post = str(manifest.get("post_dependency_tree_sha256") or "")
    if not pre or not post:
        errs.append("test_run_missing_dependency_hashes")
    elif pre != post:
        errs.append("test_run_pre_post_dependency_mismatch")

    current = current_dependency_tree_sha256(root)
    details["current_dependency_tree_sha256"] = current
    details["tested_dependency_tree_sha256"] = pre
    if pre and current != pre:
        errs.append("dependency_tree_does_not_match_tested_source")

    if int(manifest.get("exit_code", 1)) != 0:
        errs.append("test_run_nonzero_exit")

    cmd = list(manifest.get("test_command") or [])
    junit_rel = str(manifest.get("junit_xml_path") or FIXED_JUNIT_REL)
    junit_for_cmd = root / junit_rel
    ok_cmd, cmd_errs = command_matches_fixed_full_suite(
        cmd,
        root=root,
        junit_xml_path=junit_for_cmd,
        python_executable=cmd[0] if cmd else sys.executable,
    )
    if not ok_cmd:
        errs.extend(cmd_errs)

    log_rel = str(manifest.get("pytest_log_path") or "")
    log_path = root / log_rel if log_rel else None
    junit_path = root / junit_rel if junit_rel else None
    if not log_path or not log_path.is_file():
        errs.append("pytest_log_missing")
    else:
        actual_log = sha256_path(log_path)
        details["pytest_log_sha256_recomputed"] = actual_log
        if actual_log != str(manifest.get("pytest_log_sha256") or ""):
            errs.append("pytest_log_sha_mismatch")

    if not junit_path or not junit_path.is_file():
        errs.append("junit_xml_missing")
    else:
        actual_junit = sha256_path(junit_path)
        details["junit_xml_sha256_recomputed"] = actual_junit
        if actual_junit != str(manifest.get("junit_xml_sha256") or ""):
            errs.append("junit_xml_sha_mismatch")
        try:
            recomputed = parse_junit_counts(junit_path)
            details["junit_counts_recomputed"] = recomputed
            for key in (
                "collected_count",
                "passed_count",
                "failed_count",
                "error_count",
                "skipped_count",
            ):
                if int(manifest.get(key, -1)) != int(recomputed.get(key, -2)):
                    errs.append(f"junit_count_mismatch:{key}")
            if int(recomputed.get("collected_count") or 0) <= 0:
                errs.append("collected_count_not_positive")
        except Exception as exc:
            errs.append(f"junit_parse_error:{exc}")

    if int(manifest.get("collected_count") or 0) <= 0:
        errs.append("collected_count_not_positive")
    if int(manifest.get("failed_count") or 0) != 0:
        errs.append("test_run_has_failures")
    if int(manifest.get("error_count") or 0) != 0:
        errs.append("test_run_has_errors")

    ok = len(errs) == 0
    details["full_test_run_verified"] = ok
    details["dependency_tree_matches_tested_source"] = (
        bool(pre) and pre == post and pre == current
    )
    return ok, errs, details
