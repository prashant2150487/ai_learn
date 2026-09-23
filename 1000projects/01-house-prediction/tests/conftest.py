import os
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///./data/test_app.db")

TEST_DB = Path("data/test_app.db")


def pytest_sessionstart(session):
    TEST_DB.parent.mkdir(parents=True, exist_ok=True)
    if TEST_DB.exists():
        TEST_DB.unlink()
