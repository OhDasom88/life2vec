"""Command-line interface for offline builds and explicit Neo4j operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .builder import build_corpus
from .catalog import normalize_catalog
from .inventory import build_inventory
from .neo4j import GraphOperator, Neo4jConfig
from .training_export import export_training_manifest


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="scripts/online2")
    root.add_argument(
        "--data-root",
        type=_path,
        default=_path("datasets/agrichallenge/online2"),
    )
    commands = root.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory")
    inventory.add_argument("--output", type=_path, required=True)

    catalog = commands.add_parser("normalize-catalog")
    catalog.add_argument("--output", type=_path, required=True)

    build = commands.add_parser("build")
    build.add_argument("--output", type=_path, required=True)
    build.add_argument("--batch-size", type=int, default=10_000)

    export = commands.add_parser("export-training")
    export.add_argument("--build-dir", type=_path, required=True)

    neo4j = commands.add_parser("neo4j")
    graph_commands = neo4j.add_subparsers(dest="graph_command", required=True)
    graph_commands.add_parser("preflight")
    report = graph_commands.add_parser("report")
    report.add_argument("--output", type=_path, required=True)
    purge = graph_commands.add_parser("purge")
    purge.add_argument("--report", type=_path, required=True)
    purge.add_argument("--execute", action="store_true")
    purge.add_argument("--database")
    purge.add_argument("--expected-dataset")
    purge.add_argument("--batch-size", type=int, default=10_000)
    graph_commands.add_parser("ensure-schema")
    load = graph_commands.add_parser("load")
    load.add_argument("--build-dir", type=_path, required=True)
    load.add_argument("--batch-size", type=int, default=1_000)
    load.add_argument(
        "--relationships-only",
        action="store_true",
        help="Skip node MERGE and only (re)load lineage relationships",
    )
    validate = graph_commands.add_parser("validate")
    validate.add_argument("--expected-counts", type=_path, required=True)
    return root


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "inventory":
        _print(build_inventory(args.data_root, args.output))
        return 0
    if args.command == "normalize-catalog":
        _print(
            normalize_catalog(
                args.data_root / "online2_narrative_catalog.csv", args.output
            )
        )
        return 0
    if args.command == "build":
        _print(build_corpus(args.data_root, args.output, args.batch_size))
        return 0
    if args.command == "export-training":
        _print(export_training_manifest(args.build_dir))
        return 0

    config = Neo4jConfig.from_env()
    operator = GraphOperator.connect(config)
    try:
        if args.graph_command == "preflight":
            _print(operator.preflight())
        elif args.graph_command == "report":
            payload = operator.report()
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            _print({"database": payload["database"], "label_count": len(payload["labels"])})
        elif args.graph_command == "purge":
            _print(
                operator.purge(
                    args.report,
                    args.execute,
                    args.database,
                    args.expected_dataset,
                    args.batch_size,
                )
            )
        elif args.graph_command == "ensure-schema":
            operator.ensure_schema()
            _print({"status": "PASS", "database": config.database})
        elif args.graph_command == "load":
            _print(
                operator.load_build(
                    args.build_dir,
                    args.batch_size,
                    relationships_only=args.relationships_only,
                )
            )
        elif args.graph_command == "validate":
            expected = json.loads(args.expected_counts.read_text(encoding="utf-8"))
            _print(operator.validate(expected))
        else:
            raise AssertionError(args.graph_command)
    finally:
        operator.driver.close()
    return 0
