from flask import Blueprint, request, jsonify
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
