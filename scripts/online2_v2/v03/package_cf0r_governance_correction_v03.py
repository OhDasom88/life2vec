#!/usr/bin/env python3
"""Build a correction-only CF-0R governance package.

This package does not rewrite or re-run any case result.  It binds corrected
governance prose and epsilon row evidence to immutable hashes of the 55 source
case payloads taken from the original CHECKSUMS.sha256 map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "reports/cf0r_schema_recovery/CF0R_COMPLETE_20260718T140009Z"
DEFAULT_SOURCE_ARCHIVE = (
    ROOT / "reports/cf0r_schema_recovery/CF0R_COMPLETE_20260718T140009Z.tar.gz"
)
DEFAULT_RUN_ROOTS = (
    ROOT
    / "outputs/m1_m2_prereq/funnel_audit/case_runs/cf0_funnel_20260718T130655Z",
    ROOT
    / "outputs/m1_m2_prereq/funnel_audit_cf0r/case_runs/cf0_funnel_20260718T135829Z",
)
DEFAULT_CALIBRATION = ROOT / "outputs/cf_calibration/noop_noise.json"
SNAPSHOT_FILES = (
    "scripts/online2_v2/v03/package_cf0r_governance_correction_v03.py",
    "scripts/online2_v2/v03/package_cf0r_complete_v03.py",
    "scripts/online2_v2/v03/run_cf0r_epsilon_sensitivity_v03.py",
    "scripts/online2_v2/v03/merge_cf0r_composite_v03.py",
    "tests/v2/counterfactual_m1/test_cf0r_governance_correction.py",
    "tests/v2/counterfactual_m1/test_cf0_funnel_accounting.py",
    "tests/v2/counterfactual_m1/test_cf0r_schema_dispatch.py",
    "tests/v2/counterfactual_m1/test_cf0r_decoder_warning.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/funnel_accounting.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/cf_validity.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/crossfit_critic.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/noop_calibration.py",
    "src/online2/v2/finetune_v03/counterfactual/candidates/path_a_schema_dispatch.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_checksums(checksums_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(None, 1)
        mapping[relative.strip()] = digest.strip()
    return mapping


def expected_case_payload_hashes(checksums_path: Path) -> Dict[str, str]:
    mapping = parse_checksums(checksums_path)
    cases = {
        Path(relative).stem: digest
        for relative, digest in mapping.items()
        if relative.startswith("per_case/") and relative.endswith(".json")
    }
    if len(cases) != 55:
        raise ValueError(
            f"expected 55 per_case hashes in CHECKSUMS.sha256, found {len(cases)}"
        )
    return cases


def case_payload_hashes(per_case_dir: Path) -> Dict[str, str]:
    return {
        path.stem: file_sha256(path)
        for path in sorted(per_case_dir.glob("*.json"))
    }


def verify_case_payload_hashes(
    *,
    per_case_dir: Path,
    expected: Mapping[str, str],
) -> Tuple[bool, Dict[str, str]]:
    actual = case_payload_hashes(per_case_dir)
    mismatches = {
        case_id: f"expected={expected[case_id]} actual={actual.get(case_id)}"
        for case_id in sorted(expected)
        if actual.get(case_id) != expected[case_id]
    }
    for case_id in sorted(set(actual) - set(expected)):
        mismatches[case_id] = "unexpected_case_payload"
    return len(mismatches) == 0 and len(actual) == 55, mismatches


def _iter_candidate_rows(run_roots: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    seen_paths = set()
    for root in run_roots:
        for path in sorted(Path(root).glob("**/path_a_m2_cf_results.jsonl")):
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            case_id = path.parents[1].name
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                if bool(row.get("is_noop")):
                    continue
                yield {
                    **row,
                    "_case_id": case_id,
                    "_source_path": display_path(path),
                    "_source_line": line_number,
                }


def _fold_sign_agrees(row: Mapping[str, Any], direction: str) -> bool:
    folds = row.get("delta_r_folds_search") or row.get("delta_r_folds") or []
    try:
        values = [float(value) for value in folds]
    except (TypeError, ValueError):
        return False
    if not values:
        return False
    if direction == "decrease":
        return all(value < 0.0 for value in values)
    return all(value > 0.0 for value in values)


def calibration_provenance(calibration_path: Path) -> Dict[str, Any]:
    if not calibration_path.exists():
        return {
            "calibration_path": display_path(calibration_path),
            "calibration_present": False,
            "noise_floor": 8.374e-7,
            "noise_floor_source": "HARDCODED_FALLBACK",
        }
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    noise = None
    for key in ("noise_floor", "noop_noise_floor", "delta_r_noise_floor"):
        if payload.get(key) is not None:
            noise = float(payload[key])
            break
    if noise is None:
        # Prefer measured max abs noop delta when present; otherwise p99×10.
        if payload.get("noop_delta_abs_max") is not None:
            noise = float(payload["noop_delta_abs_max"])
            source = "noop_delta_abs_max"
        elif payload.get("noop_delta_p99") is not None:
            noise = float(payload["noop_delta_p99"]) * float(
                payload.get("noop_noise_multiplier") or 10.0
            )
            source = "noop_delta_p99_x_multiplier"
        else:
            noise = 8.374e-7
            source = "HARDCODED_FALLBACK"
    else:
        source = "noise_floor_field"
    return {
        "calibration_path": display_path(calibration_path),
        "calibration_present": True,
        "calibration_sha256": file_sha256(calibration_path),
        "calibration_hash": payload.get("calibration_hash"),
        "noise_floor": float(noise),
        "noise_floor_source": source,
        "effective_epsilon": payload.get("effective_epsilon"),
        "effective_epsilon_frozen": payload.get("effective_epsilon_frozen"),
    }


def epsilon_row_evidence(
    *,
    run_roots: Sequence[Path],
    epsilons: Sequence[float],
    calibration_path: Path = DEFAULT_CALIBRATION,
) -> Dict[str, Any]:
    rows = list(_iter_candidate_rows(run_roots))
    calibration = calibration_provenance(Path(calibration_path))
    evidence: Dict[str, Any] = {}
    for epsilon in epsilons:
        accepted: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for row in rows:
            if str(row.get("operational_eligibility")) == "OBSERVATIONAL_SENSITIVITY_ONLY":
                continue
            if str(row.get("source")) == "schema_categorical_adjacent":
                continue
            delta = row.get("delta_r_search")
            if delta is None:
                continue
            delta_f = float(delta)
            direction = None
            if delta_f <= -float(epsilon) and _fold_sign_agrees(row, "decrease"):
                direction = "decrease"
            elif delta_f >= float(epsilon) and _fold_sign_agrees(row, "increase"):
                direction = "increase"
            if direction is None:
                continue
            candidate_id = str(
                row.get("dedup_candidate_id") or row.get("candidate_id") or ""
            )
            key = (row["_case_id"], candidate_id, direction)
            if key in accepted:
                continue
            accepted[key] = {
                "case_id": row["_case_id"],
                "candidate_id": candidate_id,
                "direction": direction,
                "delta_r_search": delta_f,
                "delta_r_folds_search": list(
                    row.get("delta_r_folds_search")
                    or row.get("delta_r_folds")
                    or []
                ),
                "source": str(row.get("source") or ""),
                "source_path": row["_source_path"],
                "source_line": row["_source_line"],
            }
        evidence[str(float(epsilon))] = {
            "epsilon": float(epsilon),
            "row_count": len(accepted),
            "rows": sorted(
                accepted.values(),
                key=lambda item: (
                    item["case_id"],
                    item["candidate_id"],
                    item["direction"],
                ),
            ),
        }
    return {
        "role": "ROW_LEVEL_EVIDENCE_FOR_POST_HOC_DIAGNOSTIC",
        "threshold_selected_from_rows": False,
        "current_operational_epsilon": 0.01,
        "epsilon_policy_status": "UNCHANGED_NOT_RETUNED",
        "case_run_roots": [display_path(Path(root)) for root in run_roots],
        "calibration": calibration,
        "grid": evidence,
    }


def corrected_report(source_report: str) -> str:
    stale = (
        "- `CIRCULAR_FEATURE_LINEAR_BIN_UNAVAILABLE` (`wind_direction_deg`): "
        "not fixed in CF-0; when locus selection lands on this feature, Path A "
        "raises and the case is `FAIL_EXECUTION_ERROR` (blocking for that case)."
    )
    corrected = (
        "- `CIRCULAR_FEATURE_LINEAR_BIN_UNAVAILABLE` is historical parent-run "
        "context only. CF-0R added schema categorical dispatch; the final "
        "55-case composite has `case_execution_error_count=0`. No case payload "
        "is changed by this governance correction."
    )
    text = source_report.replace(stale, corrected)
    text = text.replace(
        "- CF-0 stops here. Threshold relaxation / action-bundle work require separate approval.",
        "- CF-0R composite is COMPLETE_55_OF_55. Threshold relaxation / action-bundle "
        "work still require separate approval and are not authorized by this "
        "governance correction.",
    )
    if "historical parent-run context only" not in text:
        text += (
            "\n- Governance correction: the final composite has no circular-schema "
            "execution errors.\n"
        )
    return text.replace(
        "# CF-0 Candidate Funnel Audit Report",
        "# CF-0R Candidate Funnel Audit Report — Governance-Corrected View",
        1,
    )


def _write_checksums(directory: Path) -> None:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            lines.append(
                f"{file_sha256(path)}  {path.relative_to(directory).as_posix()}"
            )
    (directory / "CHECKSUMS.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _copy_snapshots(output_dir: Path) -> List[str]:
    snap = output_dir / "code_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    copied = []
    for relative in SNAPSHOT_FILES:
        src = ROOT / relative
        if not src.exists():
            continue
        dst = snap / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(relative)
    return copied


def build_package(
    *,
    source_package: Path,
    output_dir: Path,
    run_roots: Sequence[Path],
    calibration_path: Path = DEFAULT_CALIBRATION,
    source_archive: Optional[Path] = None,
) -> Dict[str, Any]:
    per_case = source_package / "per_case"
    expected = expected_case_payload_hashes(source_package / "CHECKSUMS.sha256")
    verified, mismatches = verify_case_payload_hashes(
        per_case_dir=per_case, expected=expected
    )
    if not verified:
        raise ValueError(
            "source case payload hashes do not match CHECKSUMS.sha256: "
            + json.dumps(mismatches, ensure_ascii=False)
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "funnel_audit_summary.json",
        "epsilon_sensitivity_diagnostic.json",
        "audit_manifest.json",
        "SOURCE_REVISION.json",
    ):
        src = source_package / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    (output_dir / "FUNNEL_AUDIT_REPORT.md").write_text(
        corrected_report(
            (source_package / "FUNNEL_AUDIT_REPORT.md").read_text(encoding="utf-8")
        ),
        encoding="utf-8",
    )
    (output_dir / "CASE_PAYLOAD_HASHES.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    epsilon_source = json.loads(
        (source_package / "epsilon_sensitivity_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    epsilons = [
        float(block["epsilon"])
        for block in (epsilon_source.get("grid") or {}).values()
    ]
    evidence = epsilon_row_evidence(
        run_roots=run_roots,
        epsilons=epsilons,
        calibration_path=calibration_path,
    )
    (output_dir / "EPSILON_ROW_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    copied = _copy_snapshots(output_dir)
    test_log = output_dir / "test_log.txt"
    proc = subprocess.run(
        [
            str(Path.home() / "miniconda3/envs/life2vec/bin/python"),
            "-m",
            "pytest",
            "tests/v2/counterfactual_m1/test_cf0_funnel_accounting.py",
            "tests/v2/counterfactual_m1/test_cf0r_schema_dispatch.py",
            "tests/v2/counterfactual_m1/test_cf0r_decoder_warning.py",
            "tests/v2/counterfactual_m1/test_cf0r_governance_correction.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    test_log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    after_verified, after_mismatches = verify_case_payload_hashes(
        per_case_dir=per_case, expected=expected
    )
    archive_path = Path(source_archive) if source_archive else DEFAULT_SOURCE_ARCHIVE
    manifest = {
        "package_role": "CF0R_GOVERNANCE_CORRECTION_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_package": display_path(source_package),
        "source_package_checksums_sha256": file_sha256(
            source_package / "CHECKSUMS.sha256"
        ),
        "source_archive": display_path(archive_path) if archive_path.exists() else None,
        "source_archive_sha256": (
            file_sha256(archive_path) if archive_path.exists() else None
        ),
        "case_result_payload_changed": False,
        "case_payload_count": len(expected),
        "case_payload_hashes_verified": after_verified,
        "case_payload_hash_mismatches": after_mismatches,
        "internal_per_case_sha256_self_reference_corrected": False,
        "threshold_changed": False,
        "current_operational_epsilon": 0.01,
        "epsilon_grid_role": "POST_HOC_DIAGNOSTIC_ONLY",
        "single_execution_lineage_claimed": False,
        "lineage_role": "READ_ONLY_GOVERNANCE_CORRECTION_OF_COMPOSITE_47_PLUS_8",
        "parent_lineage_is_archive_id_not_git_commit": True,
        "cf0r_cases_rerun": False,
        "per_case_directory_copied": False,
        "snapshot_files": copied,
        "test_exit_code": proc.returncode,
        "push_allowed": False,
    }
    (output_dir / "CORRECTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# CF-0R Governance Correction-Only Package",
                "",
                "- No case was re-run or rewritten.",
                "- The 55 source payload hashes are bound from the original CHECKSUMS map.",
                "- Internal JSON `per_case_sha256` self-reference mismatches are left untouched.",
                "- Epsilon remains 0.01; row evidence is diagnostic only.",
                "- Composite 47+8 lineage is retained and is not represented as one execution.",
                "- Push is not authorized.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_checksums(output_dir)
    if not after_verified or proc.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "case_payload_hashes_verified": after_verified,
                    "mismatches": after_mismatches,
                    "test_exit_code": proc.returncode,
                },
                ensure_ascii=False,
            )
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=ROOT / "reports/cf0r_schema_recovery",
    )
    parser.add_argument("--case-run-root", type=Path, action="append", default=[])
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.report_root / f"CF0R_GOVERNANCE_CORRECTION_{stamp}"
    manifest = build_package(
        source_package=args.source_package,
        output_dir=output_dir,
        run_roots=args.case_run_root or list(DEFAULT_RUN_ROOTS),
        calibration_path=args.calibration,
        source_archive=args.source_archive,
    )
    archive = output_dir.with_suffix(".tar.gz")
    subprocess.check_call(
        ["tar", "-czf", str(archive), "-C", str(output_dir.parent), output_dir.name]
    )
    archive_hash = file_sha256(archive)
    Path(str(archive) + ".sha256").write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "CF0R_GOVERNANCE_CORRECTION_COMPLETE",
                "package_dir": str(output_dir),
                "archive": str(archive),
                "archive_sha256": archive_hash,
                "manifest": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
