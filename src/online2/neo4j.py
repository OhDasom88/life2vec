"""Safe Neo4j operations. Importing this module never opens a connection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import pyarrow.parquet as pq

from .catalog import write_json

CONSTRAINTS = {
    "online2_dataset_id": ("Dataset", "dataset_id"),
    "online2_source_file_id": ("SourceFile", "source_file_id"),
    "online2_source_row_id": ("SourceRow", "source_row_id"),
    "online2_cell_id": ("CellOccurrence", "cell_id"),
    "online2_atomic_value_id": ("AtomicValue", "atomic_value_id"),
    "online2_token_id": ("CategoricalToken", "token_id"),
    "online2_binning_registry_id": ("BinningRegistry", "registry_id"),
    "online2_event_id": ("Event", "event_id"),
    "online2_segment_id": ("Event", "segment_id"),
    "online2_time_group_id": ("SameTimeGroup", "same_time_group_id"),
    "online2_narrative_id": ("Narrative", "narrative_id"),
    "online2_sequence_id": ("Sequence", "sequence_id"),
    "online2_corpus_id": ("Corpus", "corpus_id"),
    "online2_build_run_id": ("BuildRun", "build_id"),
}
INDEXES = {
    "online2_event_timestamp": ("Event", "observation_timestamp"),
    "online2_event_farm": ("Event", "spatial_scope_id"),
    "online2_sequence_narrative": ("Sequence", "narrative_id"),
    "online2_cell_farm": ("CellOccurrence", "farm_id"),
    "online2_token_string": ("CategoricalToken", "token_string"),
}

ONLINE1_LEGACY_LABELS = frozenset({"SoilSample", "Cell", "EnvSample"})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        uri = os.environ.get("NEO4J_URI", "").strip()
        if not uri:
            host = os.environ.get("NEO4J_HOST", "").strip()
            bolt = os.environ.get("NEO4J_BOLT", "7687").strip()
            if host:
                uri = f"bolt://{host}:{bolt}"
        database = (
            os.environ.get("NEO4J_DATABASE", "").strip()
            or os.environ.get("NEO4j_DB", "").strip()
            or os.environ.get("NEO4J_DB", "").strip()
        )
        values = {
            "uri": uri,
            "user": os.environ.get("NEO4J_USER", "").strip(),
            "password": os.environ.get("NEO4J_PASSWORD", ""),
            "database": database,
        }
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise ValueError(f"missing Neo4j environment values: {', '.join(missing)}")
        return cls(**values)

    def public_dict(self) -> dict[str, str]:
        return {
            "uri": self.uri,
            "user": self.user,
            "password": "<redacted>",
            "database": self.database,
        }


class GraphOperator:
    def __init__(self, config: Neo4jConfig, driver: Any) -> None:
        self.config = config
        self.driver = driver

    @classmethod
    def connect(cls, config: Neo4jConfig) -> "GraphOperator":
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
        return cls(config, driver)

    def preflight(self) -> dict[str, Any]:
        self.driver.verify_connectivity()
        with self.driver.session(database=self.config.database) as session:
            version = session.run(
                "CALL dbms.components() YIELD name, versions "
                "RETURN name, versions[0] AS version"
            ).single()
            session.run("RETURN 1").consume()
        return {
            "database": self.config.database,
            "server": dict(version) if version else {},
            "credentials_redacted": True,
        }

    def report(self) -> dict[str, Any]:
        with self.driver.session(database=self.config.database) as session:
            labels = [
                dict(record)
                for record in session.run(
                    "MATCH (n) UNWIND labels(n) AS label "
                    "RETURN label, count(*) AS count ORDER BY label"
                )
            ]
            relationships = [
                dict(record)
                for record in session.run(
                    "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
                )
            ]
            constraints = [dict(record) for record in session.run("SHOW CONSTRAINTS")]
            indexes = [dict(record) for record in session.run("SHOW INDEXES")]
            signatures = [
                dict(record)
                for record in session.run(
                    "MATCH (d:Dataset) RETURN d.dataset_tag AS dataset_tag, "
                    "d.dataset_id AS dataset_id LIMIT 20"
                )
            ]
            legacy_catalogs = [
                dict(record)
                for record in session.run(
                    "MATCH (c:NarrativeCatalog) RETURN c.name AS name, "
                    "c.source AS source, c.count AS count LIMIT 20"
                )
            ]
            natural_key_samples = [
                dict(record)
                for record in session.run(
                    "MATCH (c:Cell) RETURN c.id AS id, c.dat_id AS dat_id, "
                    "c.split AS split LIMIT 20"
                )
            ]
            online1_labels = [
                record["label"]
                for record in session.run(
                    "MATCH (n) WHERE n.dataset_tag = 'online1' "
                    "UNWIND labels(n) AS label RETURN DISTINCT label ORDER BY label"
                )
            ]
        return _json_safe(
            {
                "database": self.config.database,
                "labels": labels,
                "relationship_types": relationships,
                "constraints": constraints,
                "indexes": indexes,
                "dataset_signatures": signatures,
                "legacy_catalogs": legacy_catalogs,
                "natural_key_samples": natural_key_samples,
                "online1_labels": online1_labels,
                "code_version": os.environ.get("GIT_COMMIT", "working-tree"),
            }
        )

    def _has_online1_signature(self, before: Mapping[str, Any]) -> bool:
        signatures = {
            item.get("dataset_tag") for item in before.get("dataset_signatures") or []
        }
        if "online1" in signatures:
            return True
        legacy_names = {
            item.get("name") for item in before.get("legacy_catalogs") or []
        }
        if "online1_v1" in legacy_names:
            return True
        present_labels = {item.get("label") for item in before.get("labels") or []}
        return bool(ONLINE1_LEGACY_LABELS & present_labels)

    def purge(
        self,
        report_path: Path,
        execute: bool = False,
        database: Optional[str] = None,
        expected_dataset: Optional[str] = None,
        batch_size: int = 10_000,
    ) -> dict[str, Any]:
        before = self.report()
        write_json(report_path.with_name(report_path.stem + "_pre.json"), before)
        if not execute:
            return {"dry_run": True, "executed": False, "pre_report": before}
        if not database or database != self.config.database:
            raise ValueError("--database must exactly match NEO4J_DATABASE")
        if expected_dataset != "online1":
            raise ValueError("--expected-dataset online1 is required")
        if not self._has_online1_signature(before):
            raise RuntimeError("online1 legacy signature was not found; refusing purge")
        signatures = {
            item.get("dataset_tag") for item in before.get("dataset_signatures") or []
        }
        if signatures - {"online1", None}:
            raise RuntimeError(
                "mixed dataset signatures found; refusing full-database purge"
            )
        with self.driver.session(database=database) as session:
            while True:
                deleted = session.run(
                    "MATCH (n) WITH n LIMIT $batch "
                    "DETACH DELETE n RETURN count(n) AS deleted",
                    batch=batch_size,
                ).single()["deleted"]
                if not deleted:
                    break
            # Full-database replacement: drop every non-LOOKUP/FULLTEXT schema
            # object that belonged to the pre-delete graph. Do not invent names;
            # only act on SHOW CONSTRAINTS / SHOW INDEXES rows.
            pre_labels = {item["label"] for item in before["labels"]}
            schema_objects = []
            for object_kind, items in (
                ("CONSTRAINT", before["constraints"]),
                ("INDEX", before["indexes"]),
            ):
                for item in items:
                    if item.get("type") in {"LOOKUP", "FULLTEXT"}:
                        continue
                    name = item.get("name") or ""
                    labels = set(item.get("labelsOrTypes") or [])
                    if not name:
                        continue
                    if "online1" in name.lower() or labels & pre_labels or not labels:
                        schema_objects.append((object_kind, name))
            for object_kind, name in sorted(set(schema_objects)):
                escaped = name.replace("`", "``")
                session.run(f"DROP {object_kind} `{escaped}` IF EXISTS").consume()
        after = self.report()
        write_json(report_path.with_name(report_path.stem + "_post.json"), after)
        remaining_nodes = sum(item["count"] for item in after["labels"])
        remaining_relationships = sum(
            item["count"] for item in after["relationship_types"]
        )
        remaining_user_schema = [
            item.get("name")
            for item in after["constraints"] + after["indexes"]
            if item.get("type") not in {"LOOKUP", "FULLTEXT"}
        ]
        if remaining_nodes or remaining_relationships or remaining_user_schema:
            raise RuntimeError("post-purge zero-count validation failed")
        return {
            "dry_run": False,
            "executed": True,
            "pre_report": before,
            "post_report": after,
        }

    def ensure_schema(self) -> None:
        with self.driver.session(database=self.config.database) as session:
            for name, (label, property_name) in CONSTRAINTS.items():
                # Event has two uniqueness constraints; Neo4j allows both.
                session.run(
                    f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) "
                    f"REQUIRE n.{property_name} IS UNIQUE"
                ).consume()
            for name, (label, property_name) in INDEXES.items():
                session.run(
                    f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) "
                    f"ON (n.{property_name})"
                ).consume()
            session.run("CALL db.awaitIndexes(300)").consume()

    def merge_nodes(
        self,
        label: str,
        key: str,
        rows: Iterable[Mapping[str, Any]],
        batch_size: int = 1_000,
    ) -> int:
        approved_labels = {value[0] for value in CONSTRAINTS.values()}
        approved_keys = {value[1] for value in CONSTRAINTS.values()}
        if label not in approved_labels:
            raise ValueError(f"unapproved label: {label}")
        if key not in approved_keys:
            raise ValueError(f"unapproved merge key: {key}")
        query = (
            f"UNWIND $rows AS row MERGE (n:{label} {{{key}: row.{key}}}) "
            "SET n += row RETURN count(n) AS touched"
        )
        return self._run_batches(query, rows, batch_size)

    def _run_batches(
        self,
        query: str,
        rows: Iterable[Mapping[str, Any]],
        batch_size: int,
        fixed_params: Optional[Mapping[str, Any]] = None,
    ) -> int:
        touched = 0
        batch: list[dict[str, Any]] = []
        fixed = dict(fixed_params or {})
        with self.driver.session(database=self.config.database) as session:
            for row in rows:
                batch.append(dict(row))
                if len(batch) >= batch_size:
                    touched += session.run(query, rows=batch, **fixed).single()["touched"]
                    batch.clear()
            if batch:
                touched += session.run(query, rows=batch, **fixed).single()["touched"]
        return touched

    def load_build(
        self,
        build_dir: Path,
        batch_size: int = 1_000,
        relationships_only: bool = False,
    ) -> dict[str, int]:
        """Idempotently load deterministic node and lineage tables with UNWIND."""
        build_dir = Path(build_dir).resolve()
        manifest = json.loads((build_dir / "build_manifest.json").read_text(encoding="utf-8"))
        build_id = str(manifest["build_id"])
        schema_version = str(manifest["schema_version"])
        dataset_id = "online2"
        corpus_id = f"online2:{build_id}"
        binning = json.loads((build_dir / "binning_registry.json").read_text(encoding="utf-8"))
        registry_id = f"binning:{binning.get('registry_version', '1')}"

        counts: dict[str, int] = {}
        if not relationships_only:
            counts["Dataset"] = self.merge_nodes(
                "Dataset",
                "dataset_id",
                [
                    {
                        "dataset_id": dataset_id,
                        "dataset_tag": "online2",
                        "schema_version": schema_version,
                        "build_id": build_id,
                    }
                ],
                batch_size,
            )
            counts["BuildRun"] = self.merge_nodes(
                "BuildRun",
                "build_id",
                [
                    {
                        "build_id": build_id,
                        "dataset_tag": "online2",
                        "schema_version": schema_version,
                        "source_checksum": manifest.get("registry_hashes", {}).get(
                            "normalized_catalog", ""
                        ),
                    }
                ],
                batch_size,
            )
            counts["Corpus"] = self.merge_nodes(
                "Corpus",
                "corpus_id",
                [
                    {
                        "corpus_id": corpus_id,
                        "dataset_tag": "online2",
                        "schema_version": schema_version,
                        "build_id": build_id,
                    }
                ],
                batch_size,
            )
            counts["BinningRegistry"] = self.merge_nodes(
                "BinningRegistry",
                "registry_id",
                [
                    {
                        "registry_id": registry_id,
                        "dataset_tag": "online2",
                        "schema_version": schema_version,
                        "build_id": build_id,
                        "registry_hash": binning.get("registry_hash", ""),
                        "registry_version": binning.get("registry_version", "1"),
                        "fit_population_hash": binning.get("fit_population_hash", ""),
                    }
                ],
                batch_size,
            )

            catalog_path = build_dir / "normalized_catalog.csv"
            if catalog_path.exists():
                import csv

                narratives = []
                with catalog_path.open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if str(row.get("status", "")).upper() != "ACTIVE":
                            continue
                        narratives.append(
                            {
                                "narrative_id": row.get("narrative_id")
                                or row.get("template_id")
                                or row.get("id"),
                                "dataset_tag": "online2",
                                "schema_version": schema_version,
                                "build_id": build_id,
                                "status": "ACTIVE",
                            }
                        )
                if narratives:
                    counts["Narrative"] = self.merge_nodes(
                        "Narrative", "narrative_id", narratives, batch_size
                    )

            node_specs = {
                "source_files.parquet": ("SourceFile", "source_file_id"),
                "source_rows.parquet": ("SourceRow", "source_row_id"),
                "cell_occurrences.parquet": ("CellOccurrence", "cell_id"),
                "atomic_values.parquet": ("AtomicValue", "atomic_value_id"),
                "events.parquet": ("Event", "event_id"),
                "same_time_groups.parquet": ("SameTimeGroup", "same_time_group_id"),
                "sequences.parquet": ("Sequence", "sequence_id"),
            }
            for filename, (label, key) in node_specs.items():
                path = build_dir / filename
                if not path.exists():
                    continue
                counts[label] = self.merge_nodes(
                    label, key, parquet_rows(path), batch_size
                )

            token_rows: dict[str, dict[str, Any]] = {}
            mappings_path = build_dir / "cell_token_mappings.parquet"
            if mappings_path.exists():
                for row in parquet_rows(mappings_path):
                    token_rows.setdefault(
                        row["token_id"],
                        {
                            "token_id": row["token_id"],
                            "token_string": row["token_string"],
                            "registry_version": row.get("registry_version", "1"),
                            "tokenization_kind": row.get("tokenization_kind", ""),
                            "rule_hash": row.get("rule_hash", ""),
                            "dataset_tag": "online2",
                            "schema_version": schema_version,
                            "build_id": build_id,
                        },
                    )
            event_tokens_path = build_dir / "event_tokens.parquet"
            if event_tokens_path.exists():
                for row in parquet_rows(
                    event_tokens_path, columns=["token_id", "token_string", "role"]
                ):
                    token_rows.setdefault(
                        row["token_id"],
                        {
                            "token_id": row["token_id"],
                            "token_string": row["token_string"],
                            "registry_version": "1",
                            "tokenization_kind": "EVENT",
                            "rule_hash": "",
                            "dataset_tag": "online2",
                            "schema_version": schema_version,
                            "build_id": build_id,
                        },
                    )
            if token_rows:
                counts["CategoricalToken"] = self.merge_nodes(
                    "CategoricalToken", "token_id", token_rows.values(), batch_size
                )

        with self.driver.session(database=self.config.database) as session:
            session.run(
                "MATCH (d:Dataset {dataset_id:$dataset_id}), "
                "(b:BuildRun {build_id:$build_id}), "
                "(c:Corpus {corpus_id:$corpus_id}), "
                "(r:BinningRegistry {registry_id:$registry_id}) "
                "MERGE (b)-[:BUILT]->(d) "
                "MERGE (b)-[:PRODUCED]->(c) "
                "MERGE (b)-[:USED_REGISTRY]->(r)",
                dataset_id=dataset_id,
                build_id=build_id,
                corpus_id=corpus_id,
                registry_id=registry_id,
            ).consume()

        counts["relationship:HAS_SOURCE_FILE"] = self._run_batches(
            "UNWIND $rows AS row "
            "MATCH (d:Dataset {dataset_id:$dataset_id}) "
            "MATCH (f:SourceFile {source_file_id:row.source_file_id}) "
            "MERGE (d)-[:HAS_SOURCE_FILE]->(f) RETURN count(f) AS touched",
            (
                {"source_file_id": row["source_file_id"]}
                for row in parquet_rows(
                    build_dir / "source_files.parquet", columns=["source_file_id"]
                )
            ),
            batch_size,
            fixed_params={"dataset_id": dataset_id},
        )
        counts["relationship:CONTAINS_SEQUENCE"] = self._run_batches(
            "UNWIND $rows AS row "
            "MATCH (c:Corpus {corpus_id:$corpus_id}) "
            "MATCH (s:Sequence {sequence_id:row.sequence_id}) "
            "MERGE (c)-[:CONTAINS_SEQUENCE]->(s) RETURN count(s) AS touched",
            (
                {"sequence_id": row["sequence_id"]}
                for row in parquet_rows(
                    build_dir / "sequences.parquet", columns=["sequence_id"]
                )
            ),
            batch_size,
            fixed_params={"corpus_id": corpus_id},
        )
        counts["relationship:MATERIALIZES"] = self._run_batches(
            "UNWIND $rows AS row "
            "MATCH (n:Narrative {narrative_id:row.narrative_id}) "
            "MATCH (s:Sequence {sequence_id:row.sequence_id}) "
            "MERGE (n)-[:MATERIALIZES]->(s) RETURN count(s) AS touched",
            (
                {
                    "sequence_id": row["sequence_id"],
                    "narrative_id": row["narrative_id"],
                }
                for row in parquet_rows(
                    build_dir / "sequences.parquet",
                    columns=["sequence_id", "narrative_id"],
                )
            ),
            batch_size,
        )

        relationship_specs = [
            (
                "source_rows.parquet",
                "UNWIND $rows AS row MATCH (a:SourceFile {source_file_id:row.source_file_id}) "
                "MATCH (b:SourceRow {source_row_id:row.source_row_id}) "
                "MERGE (a)-[:HAS_ROW]->(b) RETURN count(b) AS touched",
            ),
            (
                "cell_occurrences.parquet",
                "UNWIND $rows AS row MATCH (a:SourceRow {source_row_id:row.source_row_id}) "
                "MATCH (b:CellOccurrence {cell_id:row.cell_id}) "
                "MERGE (a)-[:HAS_CELL]->(b) RETURN count(b) AS touched",
            ),
            (
                "cell_token_mappings.parquet",
                "UNWIND $rows AS row MATCH (c:CellOccurrence {cell_id:row.cell_id}) "
                "MATCH (a:AtomicValue {atomic_value_id:row.atomic_value_id}) "
                "MATCH (t:CategoricalToken {token_id:row.token_id}) "
                "MERGE (c)-[:HAS_ATOMIC_VALUE]->(a) "
                "MERGE (c)-[e:ENCODED_AS]->(t) SET e.policy_id=row.policy_id, "
                "e.registry_version=row.registry_version, e.rule_hash=row.rule_hash "
                "MERGE (a)-[m:MAPS_TO]->(t) SET m.policy_id=row.policy_id, "
                "m.registry_version=row.registry_version, m.rule_hash=row.rule_hash "
                "RETURN count(c) AS touched",
            ),
            (
                "event_tokens.parquet",
                "UNWIND $rows AS row MATCH (e:Event {event_id:row.event_id}) "
                "MATCH (t:CategoricalToken {token_id:row.token_id}) "
                "MERGE (e)-[h:HAS_TOKEN]->(t) SET h.position=row.position, h.role=row.role "
                "WITH row,e OPTIONAL MATCH (c:CellOccurrence {cell_id:row.cell_id}) "
                "FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | "
                "MERGE (e)-[x:CONTAINS_CELL]->(c) SET x.position=row.position, x.role=row.role) "
                "RETURN count(e) AS touched",
            ),
            (
                "events.parquet",
                "UNWIND $rows AS row MATCH (g:SameTimeGroup "
                "{same_time_group_id:row.same_time_group_id}) "
                "MATCH (e:Event {event_id:row.event_id}) "
                "MERGE (g)-[:HAS_EVENT]->(e) RETURN count(e) AS touched",
            ),
            (
                "sequence_segments.parquet",
                "UNWIND $rows AS row MATCH (s:Sequence {sequence_id:row.sequence_id}) "
                "MATCH (e:Event {event_id:row.event_id}) "
                "MERGE (s)-[h:HAS_EVENT]->(e) SET h.position=row.position, "
                "h.time_group_rank=row.time_group_rank, "
                "h.position_in_group=row.position_in_group, h.evidence_role=row.evidence_role "
                "RETURN count(e) AS touched",
            ),
        ]
        for filename, query in relationship_specs:
            path = build_dir / filename
            if not path.exists():
                continue
            rel_batch = (
                min(batch_size, 400)
                if any(key in filename for key in ("mappings", "segments", "event_tokens"))
                else batch_size
            )
            counts[f"relationship:{filename}"] = self._run_batches(
                query, parquet_rows(path), rel_batch
            )
        return counts

    def validate(self, expected_counts: Mapping[str, int]) -> dict[str, Any]:
        actual = {}
        approved = {value[0] for value in CONSTRAINTS.values()}
        with self.driver.session(database=self.config.database) as session:
            for label in expected_counts:
                if label not in approved:
                    raise ValueError(f"unapproved label: {label}")
                actual[label] = session.run(
                    f"MATCH (n:{label}) RETURN count(n) AS count"
                ).single()["count"]
        mismatches = {
            label: {"expected": expected_counts[label], "actual": actual[label]}
            for label in expected_counts
            if expected_counts[label] != actual[label]
        }
        return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}


def parquet_rows(path: Path, columns: Optional[list[str]] = None) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns):
        yield from batch.to_pylist()
