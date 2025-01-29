from backend import db

class FraudLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey('transaction.id'), nullable=False)
    fraud_score = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<FraudLog {self.id}>'
