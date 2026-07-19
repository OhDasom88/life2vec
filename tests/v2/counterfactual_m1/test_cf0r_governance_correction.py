"""CF-0R governance correction must leave all case payloads immutable."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.online2_v2.v03.package_cf0r_governance_correction_v03 import (
    build_package,
    case_payload_hashes,
    corrected_report,
    epsilon_row_evidence,
    expected_case_payload_hashes,
    file_sha256,
)


def test_corrected_report_removes_stale_schema_warning():
    stale = (
        "# CF-0 Candidate Funnel Audit Report\n"
        "- `CIRCULAR_FEATURE_LINEAR_BIN_UNAVAILABLE` (`wind_direction_deg`): "
        "not fixed in CF-0; when locus selection lands on this feature, Path A "
        "raises and the case is `FAIL_EXECUTION_ERROR` (blocking for that case).\n"
        "- CF-0 stops here. Threshold relaxation / action-bundle work require separate approval.\n"
    )
    corrected = corrected_report(stale)
    assert "Governance-Corrected View" in corrected
    assert "historical parent-run context only" in corrected
    assert "not fixed in CF-0" not in corrected
    assert "COMPLETE_55_OF_55" in corrected


def test_epsilon_row_evidence_is_non_authorizing(tmp_path: Path):
    run = tmp_path / "CASE_A" / "results"
    run.mkdir(parents=True)
    (run / "path_a_m2_cf_results.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "c1",
                "is_noop": False,
                "source": "adjacent_bin",
                "delta_r_search": 0.02,
                "delta_r_folds_search": [0.01, 0.03],
            }
        )
        + "\n"
        + json.dumps(
            {
                "candidate_id": "c1",
                "is_noop": False,
                "source": "adjacent_bin",
                "delta_r_search": 0.02,
                "delta_r_folds_search": [0.01, 0.03],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = epsilon_row_evidence(run_roots=[tmp_path], epsilons=[0.01])
    assert evidence["threshold_selected_from_rows"] is False
    assert evidence["current_operational_epsilon"] == 0.01
    assert evidence["grid"]["0.01"]["row_count"] == 1
    assert "calibration" in evidence


def test_build_correction_package_preserves_payload_hashes(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    per_case = source / "per_case"
    per_case.mkdir(parents=True)
    checksum_lines = []
    for index in range(55):
        path = per_case / f"case_{index:02d}.json"
        path.write_text(json.dumps({"case": index}) + "\n", encoding="utf-8")
        checksum_lines.append(f"{file_sha256(path)}  per_case/{path.name}")
    (source / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    (source / "funnel_audit_summary.json").write_text("{}\n", encoding="utf-8")
    (source / "epsilon_sensitivity_diagnostic.json").write_text(
        json.dumps({"grid": {"0.01": {"epsilon": 0.01}}}) + "\n",
        encoding="utf-8",
    )
    (source / "FUNNEL_AUDIT_REPORT.md").write_text(
        "# CF-0 Candidate Funnel Audit Report\n", encoding="utf-8"
    )
    (source / "SOURCE_REVISION.json").write_text("{}\n", encoding="utf-8")
    (source / "audit_manifest.json").write_text("{}\n", encoding="utf-8")

    # Avoid running the full pytest suite inside the unit test.
    monkeypatch.setattr(
        "scripts.online2_v2.v03.package_cf0r_governance_correction_v03.subprocess.run",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": "ok\n", "stderr": ""}
        )(),
    )

    before = case_payload_hashes(per_case)
    expected = expected_case_payload_hashes(source / "CHECKSUMS.sha256")
    assert before == expected
    out = tmp_path / "correction"
    manifest = build_package(
        source_package=source,
        output_dir=out,
        run_roots=[],
        source_archive=tmp_path / "missing.tar.gz",
    )
    after = case_payload_hashes(per_case)
    assert before == after
    assert manifest["case_result_payload_changed"] is False
    assert manifest["case_payload_hashes_verified"] is True
    assert manifest["threshold_changed"] is False
    assert manifest["per_case_directory_copied"] is False
    assert not (out / "per_case").exists()
    assert (out / "CASE_PAYLOAD_HASHES.json").exists()
    assert (out / "EPSILON_ROW_EVIDENCE.json").exists()
