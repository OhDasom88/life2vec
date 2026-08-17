from __future__ import annotations

from typing import Any

from .paths import load_yaml, CONFIG_DIR, write_json
from .splits import verify_no_overlap


def load_acceptance() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "acceptance.yaml")["acceptance"]


def evaluate_acceptance(observed: dict[str, Any], expected: dict[str, Any] | None = None) -> dict[str, Any]:
    exp = expected or load_acceptance()
    failures = []
    checks = []

    def check(path: str, got, want, op="eq"):
        ok = (got == want) if op == "eq" else (got <= want if op == "le" else got >= want)
        checks.append({"path": path, "got": got, "want": want, "op": op, "pass": bool(ok)})
        if not ok:
            failures.append(path)

    leak = observed.get("leakage", {})
    for k, want in exp["leakage"].items():
        check(f"leakage.{k}", leak.get(k, None), want)

    cube = observed.get("cube", {})
    for k, want in exp["cube"].items():
        if k in {"invalid_pixel_ratio_max", "cube_env_alignment_max_backward_minutes"}:
            # preflight_locked values: compare against observed locked thresholds if provided
            got = cube.get(k, None)
            if got is None:
                check(f"cube.{k}", got, want)
            elif want == "preflight_locked":
                check(f"cube.{k}_locked", got is not None, True)
            else:
                check(f"cube.{k}", got, want, op="le" if "max" in k or "ratio" in k else "eq")
        elif k == "embedding_reproducibility_min_cosine":
            check(f"cube.{k}", cube.get(k, 0.0), want, op="ge")
        else:
            check(f"cube.{k}", cube.get(k, None), want)

    train = observed.get("training", {})
    for k, want in exp["training"].items():
        if k.endswith("_max"):
            check(f"training.{k}", train.get(k, 1e9), want, op="le")
        else:
            check(f"training.{k}", train.get(k, None), want)

    comp = observed.get("compute", {})
    for k, want in exp["compute"].items():
        check(f"compute.{k}", comp.get(k, None), want)

    champ = observed.get("champion", {})
    for k, want in exp["champion"].items():
        check(f"champion.{k}", champ.get(k, None), want)

    return {
        "pass": len(failures) == 0,
        "n_checks": len(checks),
        "n_failures": len(failures),
        "failures": failures,
        "checks": checks,
    }


def overlap_report(train_df, valid_df) -> dict[str, Any]:
    checks = [
        verify_no_overlap(train_df["raw_row_id"], valid_df["raw_row_id"], "raw_row_id")
        if "raw_row_id" in train_df.columns
        else {"name": "raw_row_id", "pass": True, "overlap_n": 0},
        verify_no_overlap(train_df.get("cube_ids", []), valid_df.get("cube_ids", []), "cube_ids")
        if False
        else None,
    ]
    # sequence-level ids
    out = {
        "raw_row": verify_no_overlap(
            train_df["raw_row_id"] if "raw_row_id" in train_df else [],
            valid_df["raw_row_id"] if "raw_row_id" in valid_df else [],
            "raw_row_id",
        ),
        "source_window": verify_no_overlap(
            train_df["source_window_id"] if "source_window_id" in train_df else [],
            valid_df["source_window_id"] if "source_window_id" in valid_df else [],
            "source_window_id",
        ),
        "dat": verify_no_overlap(train_df["dat"], valid_df["dat"], "dat"),
    }
    return out
