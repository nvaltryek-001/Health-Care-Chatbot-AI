import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app  # noqa: E402
from app.services import get_bot, get_heart_model  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return create_app({"TESTING": True})


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def bot():
    return get_bot()


@pytest.fixture(scope="session")
def heart():
    return get_heart_model()
