from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = ROOT / "conf/m1/cf1s_policies"


@pytest.fixture(scope="session")
def policy_dir() -> Path:
    return POLICY_DIR


@pytest.fixture
def sample_events():
    events = []
    for feature in ("circulation_fan", "fcu_fan", "co2_supply"):
        for t in range(6):
            events.append(
                {
                    "event_id": f"{feature}_{t}",
                    "feature_id": feature,
                    "mg_id": f"MG_{feature}_{t}",
                    "event_time": f"2025-01-01T{t:02d}:00:00",
                    "event_time_epoch": float(t * 3600),
                    "same_time_group_id": f"2025-01-01T{t:02d}:00:00",
                    "observed_raw": float(t % 2),
                    "raw_reconstructable": True,
                    "atomic_schema_edit_possible": True,
                    "token_indices": [hash((feature, t)) % 10000],
                    "member_token_count": 1,
                    "absolute_saliency": float(20 - t),
                    "signed_saliency": float(20 - t),
                    "fold_stable": t < 2,
                    "method_stable": True,
                    "perturbation_stable": True,
                    "saliency_evaluable": True,
                }
            )
        # Duplicate raw MG via second token view
        events.append(
            {
                **events[-6],
                "token_indices": [1, 2, 3],
                "member_token_count": 3,
                "absolute_saliency": 0.001,
            }
        )
    # Forbidden feature should never enter universe
    events.append(
        {
            "event_id": "label_0",
            "feature_id": "diagnosis_label",
            "mg_id": "MG_label",
            "event_time": "2025-01-01T00:00:00",
            "event_time_epoch": 0.0,
            "observed_raw": 1.0,
            "raw_reconstructable": True,
            "atomic_schema_edit_possible": True,
            "absolute_saliency": 999.0,
            "fold_stable": True,
            "method_stable": True,
            "perturbation_stable": True,
            "saliency_evaluable": True,
        }
    )
    return events
