import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'postgresql://user:password@localhost/paymentdb'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret')
