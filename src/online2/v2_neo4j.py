"""Versioned Neo4j persistence for V2 registries (does not mutate raw atoms)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .neo4j import GraphOperator, Neo4jConfig


V2_CONSTRAINTS = {
    "online2_v2_binning_registry_id": ("BinningRegistryV2", "registry_id"),
    "online2_v2_vocabulary_id": ("VocabularyV2", "vocabulary_id"),
    "online2_v2_materialization_run_id": ("MaterializationRunV2", "run_id"),
    "online2_v2_tokenization_result_id": ("TokenizationResultV2", "result_id"),
    "online2_v2_external_embedding_id": ("ExternalEmbeddingV2", "embedding_id"),
}


def ensure_v2_schema(operator: GraphOperator) -> None:
    with operator.driver.session(database=operator.config.database) as session:
        for name, (label, prop) in V2_CONSTRAINTS.items():
            session.run(
                f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )


def load_v2_registry_refs(build_dir: Path, operator: GraphOperator) -> dict[str, Any]:
    build_dir = build_dir.resolve()
    binning = json.loads((build_dir / "binning_registry_v2_transductive.json").read_text())
    vocab = json.loads((build_dir / "vocab_v2.json").read_text() if (build_dir / "vocab_v2.json").exists() else (build_dir / "life2vec_token_registry_v2.json").read_text())
    manifest = json.loads((build_dir / "build_manifest_v2.json").read_text())
    ensure_v2_schema(operator)
    registry_id = f"binning:v2_transductive:{binning['meta']['registry_hash'][:16]}"
    vocab_id = f"vocab:v2:{manifest.get('git_commit_hash', 'unknown')[:12]}"
    run_id = f"materialization:v2:{manifest.get('git_commit_hash', 'unknown')[:12]}"
    with operator.driver.session(database=operator.config.database) as session:
        session.run(
            """
            MERGE (b:BinningRegistryV2 {registry_id:$id})
            SET b.version='v2_transductive',
                b.registry_hash=$hash,
                b.fit_scope=$fit_scope,
                b.training_mode=$mode,
                b.contains_problem_observations=true,
                b.contains_problem_images=true,
                b.contains_problem_hidden_targets=false,
                b.artifact_uri=$uri
            """,
            id=registry_id,
            hash=binning["meta"]["registry_hash"],
            fit_scope=binning["meta"]["fit_scope"],
            mode="transductive_public_pretraining",
            uri=str(build_dir / "binning_registry_v2_transductive.json"),
        )
        session.run(
            """
            MERGE (v:VocabularyV2 {vocabulary_id:$id})
            SET v.version='v2',
                v.size=$size,
                v.training_mode=$mode,
                v.contains_problem_hidden_targets=false,
                v.artifact_uri=$uri
            """,
            id=vocab_id,
            size=len(vocab.get("tokens", [])),
            mode="transductive_public_pretraining",
            uri=str(build_dir / "vocab_v2.json"),
        )
        session.run(
            """
            MERGE (m:MaterializationRunV2 {run_id:$id})
            SET m.version='v2',
                m.git_commit_hash=$git,
                m.training_mode=$mode,
                m.created_at=$created,
                m.artifact_dir=$dir
            WITH m
            MATCH (b:BinningRegistryV2 {registry_id:$bin_id})
            MATCH (v:VocabularyV2 {vocabulary_id:$vocab_id})
            MERGE (m)-[:USED_BINNING]->(b)
            MERGE (m)-[:USED_VOCAB]->(v)
            """,
            id=run_id,
            git=manifest.get("git_commit_hash", ""),
            mode="transductive_public_pretraining",
            created=manifest.get("created_at", ""),
            dir=str(build_dir),
            bin_id=registry_id,
            vocab_id=vocab_id,
        )
    return {
        "binning_registry_id": registry_id,
        "vocabulary_id": vocab_id,
        "materialization_run_id": run_id,
        "status": "PASS",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    config = Neo4jConfig.from_env()
    operator = GraphOperator.connect(config)
    try:
        print(json.dumps(load_v2_registry_refs(args.build_dir, operator), indent=2))
    finally:
        operator.driver.close()


if __name__ == "__main__":
    main()
