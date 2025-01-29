from flask import Blueprint, request, jsonify
import smtplib
from backend.models.transaction import Transaction

bp = Blueprint('notifications', __name__, url_prefix='/notifications')

@bp.route('/send_email', methods=['POST'])
def send_email():
    data = request.get_json()
    send_email_to_user(data['email'], data['subject'], data['body'])
    return jsonify({'message': 'Email sent successfully'}), 200

def send_email_to_user(email, subject, body):
    msg = f"Subject: {subject}

{body}"
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your_email@example.com', 'your_password')
        server.sendmail('your_email@example.com', email, msg)
