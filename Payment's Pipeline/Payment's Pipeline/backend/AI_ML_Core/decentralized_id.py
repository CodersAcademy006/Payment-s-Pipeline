"""
Secure Payment Pipeline with Decentralized Identity Management
Integrates blockchain-anchored credentials and transaction logging
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from uuid import uuid4
from threading import Lock
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PaymentPipeline")

class DecentralizedID:
    """Enhanced DID system with thread-safe operations"""
    
    def __init__(self, blockchain_conn=None):
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        self.did = self.generate_did()
        self.registry = {}
        self.revocation_list = set()
        self.lock = Lock()

    def generate_did(self) -> str:
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return f"did:payment:{public_bytes.hex()[:16]}"

    def create_vc(self, subject: Dict, expiration_days: int = 30) -> Dict:
        with self.lock:
            credential = {
                "@context": [
                    "https://www.w3.org/2018/credentials/v1",
                    "https://w3id.org/security/suites/ed25519-2020/v1"
                ],
                "id": f"urn:uuid:{uuid4()}",
                "type": ["VerifiableCredential", "PaymentCredential"],
                "issuer": self.did,
                "issuanceDate": datetime.now().isoformat(),
                "expirationDate": (datetime.now() + timedelta(days=expiration_days)).isoformat(),
                "credentialSubject": subject,
                "proof": self._sign_credential(subject)
            }
            return credential

    def _sign_credential(self, payload: Dict) -> Dict:
        message = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = self.private_key.sign(message)
        return {
            "type": "Ed25519Signature2020",
            "created": datetime.now().isoformat(),
            "verificationMethod": self.did,
            "proofValue": signature.hex()
        }

    def verify_vc(self, credential: Dict) -> bool:
        with self.lock:
            try:
                if credential['id'] in self.revocation_list:
                    return False

                proof = credential.pop('proof')
                public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                    bytes.fromhex(credential['issuer'].split(':')[-1])
                )
                message = json.dumps(credential, sort_keys=True).encode('utf-8')
                public_key.verify(bytes.fromhex(proof['proofValue']), message)
                return True
            except (InvalidSignature, KeyError) as e:
                logger.error(f"VC verification failed: {str(e)}")
                return False
            finally:
                credential['proof'] = proof

    def revoke_credential(self, credential_id: str):
        with self.lock:
            self.revocation_list.add(credential_id)
            logger.info(f"Revoked credential {credential_id}")

class User:
    """Financial user entity with DID and payment capabilities"""
    
    def __init__(self, name: str, initial_balance: float = 0.0):
        self.did_system = DecentralizedID()
        self.name = name
        self._balance = initial_balance
        self.lock = Lock()
        self.credential = self.did_system.create_vc(
            subject={
                "id": self.did_system.did,
                "name": self.name,
                "authorization": "payment:full"
            }
        )

    @property
    def balance(self) -> float:
        with self.lock:
            return self._balance

    def deposit(self, amount: float):
        with self.lock:
            if amount > 0:
                self._balance += amount
                logger.info(f"Deposited {amount} to {self.did_system.did}")

    def withdraw(self, amount: float) -> Tuple[bool, str]:
        with self.lock:
            if amount <= 0:
                return False, "Invalid amount"
            if self._balance >= amount:
                self._balance -= amount
                logger.info(f"Withdrew {amount} from {self.did_system.did}")
                return True, "Success"
            return False, "Insufficient funds"

class PaymentPipeline:
    """Core payment processing system with blockchain integration"""
    
    def __init__(self, blockchain):
        self.users: Dict[str, User] = {}
        self.blockchain = blockchain
        self.lock = Lock()

    def register_user(self, user: User):
        with self.lock:
            self.users[user.did_system.did] = user
            self.blockchain.register_did(user.did_system.did)
            logger.info(f"Registered user {user.name} with DID {user.did_system.did}")

    def process_payment(self, sender_did: str, recipient_did: str, amount: float, vc: Dict) -> Dict:
        logger.info(f"Processing payment: {sender_did} -> {recipient_did} ({amount})")
        
        # Input validation
        if amount <= 0:
            return self._error_response("Invalid payment amount")

        # Credential verification
        if not self._verify_payment_credential(sender_did, vc):
            return self._error_response("Invalid payment credential")

        # User verification
        sender = self.users.get(sender_did)
        recipient = self.users.get(recipient_did)
        if not sender or not recipient:
            return self._error_response("User not found")

        # Balance check
        success, message = sender.withdraw(amount)
        if not success:
            return self._error_response(message)

        # Update recipient balance
        recipient.deposit(amount)

        # Blockchain recording
        transaction = {
            'sender': sender_did,
            'recipient': recipient_did,
            'amount': amount,
            'currency': 'USD',
            'timestamp': datetime.now().isoformat()
        }
        self.blockchain.add_transaction(transaction)
        self.blockchain.mine_block()

        logger.info(f"Payment successful: {amount} from {sender_did} to {recipient_did}")
        return {
            "status": "success",
            "transaction_id": transaction['timestamp'],
            "sender_balance": sender.balance,
            "recipient_balance": recipient.balance
        }

    def _verify_payment_credential(self, sender_did: str, vc: Dict) -> bool:
        user = self.users.get(sender_did)
        if not user:
            return False
        return user.did_system.verify_vc(vc)

    def _error_response(self, message: str) -> Dict:
        logger.error(f"Payment failed: {message}")
        return {"status": "error", "message": message}

class PaymentBlockchain:
    """Production-grade blockchain interface for financial transactions"""
    
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.lock = Lock()

    def register_did(self, did: str) -> bool:
        with self.lock:
            self.current_transactions.append({
                'type': 'DID_REGISTRATION',
                'did': did,
                'timestamp': datetime.now().isoformat()
            })
            return True

    def add_transaction(self, transaction: Dict) -> bool:
        with self.lock:
            self.current_transactions.append({
                'type': 'PAYMENT',
                'data': transaction,
                'timestamp': datetime.now().isoformat()
            })
            return True

    def mine_block(self) -> Dict:
        with self.lock:
            block = {
                'index': len(self.chain) + 1,
                'transactions': self.current_transactions.copy(),
                'timestamp': datetime.now().isoformat(),
                'previous_hash': self._hash_block(self.chain[-1] if self.chain else None),
                'nonce': 0  # Simplified for demonstration
            }
            self.chain.append(block)
            self.current_transactions = []
            return block

    def _hash_block(self, block: Optional[Dict]) -> str:
        return hash(json.dumps(block, sort_keys=True)).hexdigest() if block else '0'

if __name__ == "__main__":
    # Initialize system components
    blockchain = PaymentBlockchain()
    payment_pipeline = PaymentPipeline(blockchain)

    # Create and register users
    alice = User("Alice Corporation", initial_balance=1000.0)
    bob = User("Bob Enterprises", initial_balance=500.0)
    
    payment_pipeline.register_user(alice)
    payment_pipeline.register_user(bob)

    # Process payment
    payment_result = payment_pipeline.process_payment(
        sender_did=alice.did_system.did,
        recipient_did=bob.did_system.did,
        amount=300.0,
        vc=alice.credential
    )

    print("\nPayment Result:")
    print(json.dumps(payment_result, indent=2))

    print("\nBalances:")
    print(f"Alice: {alice.balance}")
    print(f"Bob: {bob.balance}")

    # Demonstrate revocation
    alice.did_system.revoke_credential(alice.credential['id'])
    revoked_payment = payment_pipeline.process_payment(
        sender_did=alice.did_system.did,
        recipient_did=bob.did_system.did,
        amount=200.0,
        vc=alice.credential
    )

    print("\nRevoked Credential Payment Result:")
    print(json.dumps(revoked_payment, indent=2))

    print("\nBlockchain Status:")
    print(f"Chain length: {len(blockchain.chain)} blocks")
    print(f"Last block transactions: {len(blockchain.chain[-1]['transactions'])}")