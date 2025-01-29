from flask import Flask
from .api import auth, payment, fraud_detection, notifications, analytics

def create_app():
    app = Flask(__name__)
    app.config.from_object('backend.config.settings')

    app.register_blueprint(auth.bp)
    app.register_blueprint(payment.bp)
    app.register_blueprint(fraud_detection.bp)
    app.register_blueprint(notifications.bp)
    app.register_blueprint(analytics.bp)

    return app
