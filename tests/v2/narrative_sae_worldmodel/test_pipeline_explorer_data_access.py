"""ui/pipeline_explorer/data_access.py 회귀 테스트 (Streamlit 의존성 없음)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/pipeline_explorer"
)
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import data_access as da  # noqa: E402


def test_parse_token_trace_aligns_tokens_roles_and_groups():
    sentence = "[EVENT_SEP] VIEW|ENVIRONMENT FEATURE|CO2_PPM VALUE_ABS|ABS_B06"
    groups = json.dumps(["NONE", "NONE", "meas_abc", "meas_abc"])
    roles = json.dumps(["sep", "meta", "feature_identity", "value_abs"])

    entries = da.parse_token_trace(sentence, groups, roles)

    assert [e.token for e in entries] == [
        "[EVENT_SEP]", "VIEW|ENVIRONMENT", "FEATURE|CO2_PPM", "VALUE_ABS|ABS_B06",
    ]
    assert [e.role for e in entries] == ["sep", "meta", "feature_identity", "value_abs"]
    assert [e.group_id for e in entries] == ["NONE", "NONE", "meas_abc", "meas_abc"]
    assert [e.position for e in entries] == [0, 1, 2, 3]


def test_parse_token_trace_defaults_when_side_channels_shorter_than_tokens():
    # 실제 데이터에서 side-channel과 토큰 길이가 어긋나는 경우가 있어(예: 잘림) 방어적으로 처리해야 한다.
    sentence = "A B C"
    entries = da.parse_token_trace(sentence, json.dumps(["NONE"]), json.dumps(["sep"]))

    assert len(entries) == 3
    assert entries[0].role == "sep"
    assert entries[1].role == "meta"  # 기본값
    assert entries[2].group_id == "NONE"  # 기본값


def test_parse_token_trace_handles_empty_side_channels():
    entries = da.parse_token_trace("A B", "", "")
    assert [e.role for e in entries] == ["meta", "meta"]
    assert [e.group_id for e in entries] == ["NONE", "NONE"]


def test_checkpoint_path_for_run_prefers_step_snapshot_over_best(tmp_path, monkeypatch):
    ablation_root = tmp_path / "narrative_ablation"
    run_dir = ablation_root / "control"
    run_dir.mkdir(parents=True)
    (run_dir / "best.ckpt").write_bytes(b"best")
    (run_dir / "checkpoint_step_5000.pt").write_bytes(b"step5000")

    monkeypatch.setattr(da, "ABLATION_RUN_ROOT", ablation_root)

    path = da.checkpoint_path_for_run("control", step=5000)

    assert path.name == "checkpoint_step_5000.pt"


def test_checkpoint_path_for_run_falls_back_to_best_when_step_missing(tmp_path, monkeypatch):
    ablation_root = tmp_path / "narrative_ablation"
    run_dir = ablation_root / "dedup_reduced"
    run_dir.mkdir(parents=True)
    (run_dir / "best.ckpt").write_bytes(b"best")

    monkeypatch.setattr(da, "ABLATION_RUN_ROOT", ablation_root)

    path = da.checkpoint_path_for_run("dedup_reduced", step=5000)

    assert path.name == "best.ckpt"


def test_checkpoint_path_for_run_original_ignores_ablation_root(tmp_path, monkeypatch):
    original_dir = tmp_path / "full_event_grain_earlystop"
    original_dir.mkdir(parents=True)
    monkeypatch.setattr(da, "ORIGINAL_RUN_DIR", original_dir)

    path = da.checkpoint_path_for_run("original(full_event_grain_earlystop)")

    assert path == original_dir / "best.ckpt"


def test_list_local_runs_reads_manifests_for_original_and_ablation_arms(tmp_path, monkeypatch):
    original_dir = tmp_path / "original"
    original_dir.mkdir()
    (original_dir / "run_manifest_v2.json").write_text(json.dumps({"final_step": 18600}), encoding="utf-8")

    ablation_root = tmp_path / "narrative_ablation"
    control_dir = ablation_root / "control"
    control_dir.mkdir(parents=True)
    (control_dir / "run_manifest_v2.json").write_text(json.dumps({"final_step": 5000}), encoding="utf-8")

    monkeypatch.setattr(da, "ORIGINAL_RUN_DIR", original_dir)
    monkeypatch.setattr(da, "ABLATION_RUN_ROOT", ablation_root)
    monkeypatch.setattr(da, "ABLATION_ARMS", ("control", "sop_balanced"))  # sop_balanced has no manifest here

    runs = da.list_local_runs()

    names = [r["name"] for r in runs]
    assert "original(full_event_grain_earlystop)" in names
    assert "control" in names
    assert "sop_balanced" not in names  # 매니페스트 없으면 목록에서 빠져야 한다(조용히 잘못된 값 채우지 않음)
    control_run = next(r for r in runs if r["name"] == "control")
    assert control_run["final_step"] == 5000


def test_lookup_raw_rows_returns_empty_when_table_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "RAW_DATA_ROOT", tmp_path / "nonexistent")
    da._load_raw_zone_csv.cache_clear()
    da._load_raw_combined_csv.cache_clear()

    result = da.lookup_raw_rows("F000000", "1", "2025-03-22 00:00:00")

    assert result == {}
