from flask import Flask

from backend import db, models  # noqa: F401  (registers every model with db)
from backend.api import auth, notifications, payment


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.from_object("backend.config")
    app.config.update(overrides or {})
    if not app.config["SECRET_KEY"]:
        raise RuntimeError("SECRET_KEY environment variable is required")

    db.init_app(app)
    for module in (auth, payment, notifications):
        app.register_blueprint(module.bp)
    return app
