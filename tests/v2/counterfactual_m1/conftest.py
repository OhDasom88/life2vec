import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

@pytest.fixture(scope="session", autouse=True)
def _dump_runtime_imports_session():
    yield
    from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
        dump_runtime_imports_from_env,
    )
    dump_runtime_imports_from_env(ROOT, entrypoint="pytest")
