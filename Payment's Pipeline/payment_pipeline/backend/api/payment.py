from flask import Blueprint, request, jsonify
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
