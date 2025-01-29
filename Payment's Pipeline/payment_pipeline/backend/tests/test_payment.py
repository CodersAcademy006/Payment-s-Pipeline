import unittest
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
