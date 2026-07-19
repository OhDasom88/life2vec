#!/usr/bin/env python3
"""Authorize Development3 production smoke — trust-root PRE_EXECUTION only."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-policy-sha")
    parser.add_argument("--expected-code-sha")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stable-lock-manifest",
        type=Path,
        required=True,
        help="CF1S_STABLE_LOCK_MANIFEST_V2 with artifact_sources for fresh hashing.",
    )
    parser.add_argument(
        "--trust-root",
        type=Path,
        default=ROOT / "conf/m1/cf1s_policies/CF1S_DEVELOPMENT_TRUST_ROOT_V1.json",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=ROOT / "outputs/cf1s_core/CF1S_CORE_PREFLIGHT.json",
    )
    parser.add_argument(
        "--authorize-execution",
        action="store_true",
        help="Issue signed PRE_EXECUTION_AUTHORIZATION when SHA+preflight+trust-root gates pass.",
    )
    parser.add_argument(
        "--expires-hours",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_authorization import (
        TRUST_ROOT_ENV,
        issue_pre_execution_authorization,
        load_trust_root,
        tree_manifest_sha,
        verify_trust_root_env,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import sha256_file
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
        recompute_stable_lock_manifest,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_preflight import (
        validate_measured_preflight_contract,
    )

    stable_path = (
        args.stable_lock_manifest
        if args.stable_lock_manifest.is_absolute()
        else ROOT / args.stable_lock_manifest
    )
    expected_stable = json.loads(stable_path.read_text(encoding="utf-8"))
    observed_stable = recompute_stable_lock_manifest(expected_stable, root=ROOT)
    observed_code = observed_stable["code_tree_sha256"]
    observed_policy = observed_stable["policy_tree_sha256"]
    code_ok = args.expected_code_sha in (None, observed_code)
    policy_ok = args.expected_policy_sha in (None, observed_policy)

    preflight_ok = False
    preflight_path = args.preflight if args.preflight.is_absolute() else ROOT / args.preflight
    if preflight_path.exists():
        pre = json.loads(preflight_path.read_text(encoding="utf-8"))
        try:
            validate_measured_preflight_contract(list(pre.get("checks") or []))
            preflight_ok = True
        except Exception:
            preflight_ok = False

    trust_path = args.trust_root if args.trust_root.is_absolute() else ROOT / args.trust_root
    trust_env_ok = False
    trust_sha = ""
    try:
        trust_sha = verify_trust_root_env(trust_path)
        trust_env_ok = True
    except Exception as exc:
        trust_err = str(exc)
    else:
        trust_err = ""

    verified = bool(code_ok and policy_ok and preflight_ok and trust_env_ok)
    authorize = bool(args.authorize_execution and verified)

    out = (
        args.out
        if args.out and args.out.is_absolute()
        else (
            ROOT / args.out
            if args.out
            else ROOT
            / "outputs/cf1s_core/authorizations"
            / f"{args.run_id}.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    if authorize:
        priv = os.environ.get("CF1S_DEVELOPMENT_SIGNING_KEY_HEX")
        if not priv:
            raise SystemExit(
                "CF1S_DEVELOPMENT_SIGNING_KEY_HEX required to issue signed authorization"
            )
        root = load_trust_root(trust_path)
        artifact = issue_pre_execution_authorization(
            trust_root=root,
            trust_root_sha256=trust_sha,
            issuer_id="cf1s-development-issuer-v1",
            key_id="dev-key-1",
            private_key_hex=priv,
            stable_lock_manifest=observed_stable,
            run_id=args.run_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=float(args.expires_hours)),
        )
    else:
        artifact = {
            "artifact_kind": "PRE_EXECUTION_AUTHORIZATION_UNSIGNED",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "authorization_verified": verified,
            "development_production_execution_authorized": False,
            "evidence_output_authorized": False,
            "recommendation_authorized": False,
            "primary32_execution_authorized": False,
            "problem20_execution_authorized": False,
            "trajectory_authorized": False,
            "expected_code_sha256": args.expected_code_sha,
            "observed_code_sha256": observed_code,
            "code_sha_match": code_ok,
            "expected_policy_sha256": args.expected_policy_sha,
            "observed_policy_sha256": observed_policy,
            "policy_sha_match": policy_ok,
            "preflight_path": str(preflight_path),
            "preflight_ok": preflight_ok,
            "trust_root_env_ok": trust_env_ok,
            "trust_root_error": trust_err,
            "run_id": args.run_id,
            "stable_lock_sha256": observed_stable["stable_lock_sha256"],
            "stable_lock_manifest": observed_stable,
            "cohort": "development3",
            "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
            "interpretation_scope": "Development3 selection-blind contract verification evidence",
            "notes": (
                "Signed authorization requires --authorize-execution, SHA match, "
                "preflight PASS, CF1S_TRUST_ROOT_SHA256, and CF1S_DEVELOPMENT_SIGNING_KEY_HEX."
            ),
        }

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )

    if out.exists():
        raise SystemExit(f"authorization output already exists; refuse overwrite: {out}")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(artifact) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "out": str(out),
                "authorized": authorize,
                "observed_code_sha256": observed_code,
                "observed_policy_sha256": observed_policy,
            },
            indent=2,
        )
    )
    if args.authorize_execution and not verified:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
