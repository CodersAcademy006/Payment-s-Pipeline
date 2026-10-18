import os

import pytest

from backend import db
from backend.app import create_app

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite://")


@pytest.fixture
def app():
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test", "SQLALCHEMY_DATABASE_URI": DATABASE_URL}
    )
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()
