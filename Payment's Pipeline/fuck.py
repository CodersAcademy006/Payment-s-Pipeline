import os

# Directory structure for the project
directories = [
    'payment_pipeline',  # Root project folder
    'payment_pipeline/backend',
    'payment_pipeline/backend/app',
    'payment_pipeline/backend/api',
    'payment_pipeline/backend/models',
    'payment_pipeline/backend/tests',
    'payment_pipeline/frontend',
    'payment_pipeline/frontend/src',
    'payment_pipeline/frontend/src/components',
    'payment_pipeline/frontend/src/components/Auth',
    'payment_pipeline/frontend/src/components/Dashboard',
    'payment_pipeline/frontend/src/components/Analytics',
    'payment_pipeline/deployment',
    'payment_pipeline/backend/config',  # Add this line to create the config directory
]

# Files for backend
backend_files = {
    'payment_pipeline/backend/app/__init__.py': """from flask import Flask
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
""",
    'payment_pipeline/backend/api/auth.py': """from flask import Blueprint, request, jsonify
from backend.models.user import User
from backend import db

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    new_user = User(username=data['username'], email=data['email'], password=data['password'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email'], password=data['password']).first()
    if user:
        return jsonify({'message': 'Login successful'}), 200
    return jsonify({'message': 'Invalid credentials'}), 401
""",
    'payment_pipeline/backend/api/payment.py': """from flask import Blueprint, request, jsonify
from backend.models.transaction import Transaction
from backend import db

bp = Blueprint('payment', __name__, url_prefix='/payment')

@bp.route('/process', methods=['POST'])
def process_payment():
    data = request.get_json()
    transaction = Transaction(amount=data['amount'], user_id=data['user_id'], status='Success')
    db.session.add(transaction)
    db.session.commit()
    return jsonify({'message': 'Payment processed successfully', 'transaction_id': transaction.id}), 200
""",
    'payment_pipeline/backend/api/fraud_detection.py': """from flask import Blueprint, request, jsonify
import pickle
from backend.models.transaction import Transaction

bp = Blueprint('fraud_detection', __name__, url_prefix='/fraud')

with open('models/fraud_detection_model.pkl', 'rb') as model_file:
    fraud_model = pickle.load(model_file)

@bp.route('/check', methods=['POST'])
def check_fraud():
    data = request.get_json()
    features = [data['feature1'], data['feature2'], data['feature3']]
    fraud_score = fraud_model.predict_proba([features])[0][1]
    result = 'fraud' if fraud_score > 0.8 else 'safe'
    return jsonify({'status': result, 'fraud_score': fraud_score}), 200
""",
    'payment_pipeline/backend/api/notifications.py': """from flask import Blueprint, request, jsonify
import smtplib
from backend.models.transaction import Transaction

bp = Blueprint('notifications', __name__, url_prefix='/notifications')

@bp.route('/send_email', methods=['POST'])
def send_email():
    data = request.get_json()
    send_email_to_user(data['email'], data['subject'], data['body'])
    return jsonify({'message': 'Email sent successfully'}), 200

def send_email_to_user(email, subject, body):
    msg = f"Subject: {subject}\n\n{body}"
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your_email@example.com', 'your_password')
        server.sendmail('your_email@example.com', email, msg)
""",
    'payment_pipeline/backend/models/user.py': """from backend import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'
""",
    'payment_pipeline/backend/models/transaction.py': """from backend import db

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<Transaction {self.id}>'
""",
    'payment_pipeline/backend/models/fraud_logs.py': """from backend import db

class FraudLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey('transaction.id'), nullable=False)
    fraud_score = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<FraudLog {self.id}>'
""",
    'payment_pipeline/backend/config/settings.py': """import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'postgresql://user:password@localhost/paymentdb'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret')
""",
    'payment_pipeline/backend/wsgi.py': """from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
""",
    'payment_pipeline/backend/tests/test_payment.py': """import unittest
from backend.app import create_app
from backend import db

class PaymentTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_process_payment(self):
        response = self.client.post('/payment/process', json={'amount': 100, 'user_id': 1})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Payment processed successfully', response.get_json()['message'])
"""
}

# Files for frontend
frontend_files = {
    'payment_pipeline/frontend/public/index.html': """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Payment Pipeline</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
""",
    'payment_pipeline/frontend/src/components/Auth/Login.js': """import React, { useState } from 'react';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    const response = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await response.json();
    setMessage(data.message);
  };

  return (
    <div>
      <h2>Login</h2>
      <form onSubmit={handleLogin}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit">Login</button>
      </form>
      {message && <p>{message}</p>}
    </div>
  );
};

export default Login;
""",
    'payment_pipeline/frontend/src/components/Auth/Register.js': """import React, { useState } from 'react';

const Register = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');

  const handleRegister = async (e) => {
    e.preventDefault();
    const response = await fetch('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });
    const data = await response.json();
    setMessage(data.message);
  };

  return (
    <div>
      <h2>Register</h2>
      <form onSubmit={handleRegister}>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit">Register</button>
      </form>
      {message && <p>{message}</p>}
    </div>
  );
};

export default Register;
""",
    'payment_pipeline/frontend/src/components/Dashboard/Dashboard.js': """import React from 'react';

const Dashboard = () => {
  return (
    <div>
      <h1>Welcome to Payment Dashboard</h1>
      <p>Manage payments and view analytics.</p>
    </div>
  );
};

export default Dashboard;
""",
    'payment_pipeline/frontend/src/components/Analytics/Analytics.js': """import React from 'react';

const Analytics = () => {
  return (
    <div>
      <h2>Payment Analytics</h2>
      <p>View trends and statistics about the payments.</p>
    </div>
  );
};

export default Analytics;
"""
}

# Function to create directories and files
def create_structure():
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    for file_path, content in backend_files.items():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure parent directories are created
        with open(file_path, 'w') as file:
            file.write(content)
    
    for file_path, content in frontend_files.items():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure parent directories are created
        with open(file_path, 'w') as file:
            file.write(content)

# Create the structure
create_structure()