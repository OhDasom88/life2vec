#!/usr/bin/env python3
"""Diagnostic (read-only, no evidence claims): reproduce
build_multievent_candidates_for_case for the 3 Development3 cases and report
exactly which PAIR bundle's atoms diverge from its declared parents' atoms."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_candidates import (
        build_multievent_candidates_for_case,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_sha256,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_runtime import (
        apply_exact_byte_deterministic_runtime,
        load_case_events,
    )

    apply_exact_byte_deterministic_runtime(before_cuda_init=True)
    cfg = yaml.safe_load((ROOT / "conf/m1/cf1s_core_smoke.yaml").read_text(encoding="utf-8"))
    man = json.loads((ROOT / str(cfg["development_manifest_path"])).read_text(encoding="utf-8"))
    case_ids = list(man["ordered_case_ids"])

    for case_id in case_ids:
        print(f"=== case {case_id} ===")
        events = load_case_events(cfg, case_id)
        for ev in events:
            if not getattr(ev, "sentence_tokens", None):
                toks = list(getattr(ev, "tokens", None) or [])
                ev.sentence_tokens = toks
                ev.token_count = len(toks)
        candidates = build_multievent_candidates_for_case(
            cfg,
            case_id=case_id,
            events=events,
            max_two_event=min(2, int((cfg.get("cf1s_core") or {}).get("max_two_event_families_per_case") or 2)),
            top_k_events=int((cfg.get("attribution") or {}).get("event_preselect_top_k") or 8),
        )
        atomic_by_cid = {
            c["candidate_id"]: c for c in candidates if c.get("kind") == "ATOMIC"
        }
        print(f"  total candidates: {len(candidates)} "
              f"(ATOMIC={sum(1 for c in candidates if c['kind']=='ATOMIC')}, "
              f"PAIR={sum(1 for c in candidates if c['kind']=='PAIR')})")
        for c in candidates:
            if c.get("kind") != "PAIR":
                continue
            bundle_atoms = {
                canonical_json_sha256(a) if isinstance(a, dict) else str(a)
                for a in (c.get("atomics") or [])
            }
            parent_ids = c.get("parent_candidate_ids") or []
            parent_atoms = set()
            parent_feature_ids = []
            for pid in parent_ids:
                parent = atomic_by_cid.get(pid)
                if parent is None:
                    print(f"  PAIR {c['candidate_id']}: parent {pid} NOT FOUND in atomic_by_cid")
                    continue
                parent_feature_ids.append(parent.get("atomics", [{}])[0].get("feature_id"))
                for a in parent.get("atomics") or []:
                    parent_atoms.add(canonical_json_sha256(a) if isinstance(a, dict) else str(a))
            bundle_feature_ids = [a.get("feature_id") for a in (c.get("atomics") or [])]
            match = bundle_atoms == parent_atoms
            print(
                f"  PAIR {c['candidate_id']}: event_ids={c.get('event_ids')} "
                f"bundle_features={bundle_feature_ids} parent_ids={parent_ids} "
                f"parent_features={parent_feature_ids} ATOMS_MATCH={match}"
            )
            if not match:
                print(f"    bundle_atoms_sha={sorted(bundle_atoms)}")
                print(f"    parent_atoms_sha={sorted(parent_atoms)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
