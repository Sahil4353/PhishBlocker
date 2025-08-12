import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# IMPORTANT: set env BEFORE importing app modules that create the engine
@pytest.fixture(scope="session", autouse=True)
def _test_env(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("db")
    db_path = tmpdir / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    # quiet logs during tests (flip to "true" if you want to see logs)
    os.environ.setdefault("ENABLE_LOGGING", "false")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    yield
    # no explicit cleanup needed; tmp dir goes away after session


@pytest.fixture(scope="session")
def app():
    # Import after DATABASE_URL is set
    from app.main import create_app

    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)
