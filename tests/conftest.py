import os

import pytest
from fastapi.testclient import TestClient


# 1) Ensure test env is set BEFORE importing app modules that create the engine
@pytest.fixture(scope="session", autouse=True)
def _test_env(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("db")
    db_path = tmpdir / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ.setdefault(
        "ENABLE_LOGGING", "false"
    )  # flip to "true" if you want logs during tests
    os.environ.setdefault("LOG_LEVEL", "INFO")
    yield  # tmpdir cleanup handled by pytest


# 2) Import app only after env is set
@pytest.fixture(scope="session")
def app():
    from app.main import create_app

    return create_app()


# 3) Provide a TestClient to tests
@pytest.fixture()
def client(app):
    return TestClient(app)
