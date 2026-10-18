import pytest
from sqlalchemy import inspect

from backend import db
from backend.app import create_app
from backend.models.user import User


def test_app_refuses_to_start_without_secret_key():
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app({"SECRET_KEY": None})


def test_blueprints_registered(app):
    assert {"auth", "payment", "notifications"} <= set(app.blueprints)


def test_all_tables_created(app):
    assert {"users", "transactions", "audit_logs", "idempotency_keys"} <= set(
        inspect(db.engine).get_table_names()
    )


def test_password_is_hashed_not_stored(app):
    user = User(username="alice", email="alice@example.com", password="correct horse battery")
    db.session.add(user)
    db.session.commit()

    assert user.password_hash != "correct horse battery"
    assert user.verify_password("correct horse battery")
    assert not user.verify_password("wrong password!!")
