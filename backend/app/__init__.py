from flask import Flask
from .api import auth, payment, notifications, analytics

def create_app():
    app = Flask(__name__)
    app.config.from_object('backend.config')

    app.register_blueprint(auth.bp)
    app.register_blueprint(payment.bp)
    app.register_blueprint(notifications.bp)
    app.register_blueprint(analytics.bp)

    return app
