import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def _test_env(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("db")
    db_path = tmpdir / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ.setdefault("ENABLE_LOGGING", "false")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    yield


@pytest.fixture(scope="session")
def app():
    from app.main import create_app

    return create_app()


@pytest.fixture()
def client(app):
    # Ensure tables exist even if lifespan timing changes
    from app.db import engine
    from app.models.base import Base

    Base.metadata.create_all(bind=engine)

    # Use TestClient as a context manager so startup/shutdown run
    with TestClient(app) as c:
        yield c
