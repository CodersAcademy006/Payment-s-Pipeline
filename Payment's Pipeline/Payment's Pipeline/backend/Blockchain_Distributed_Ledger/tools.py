"""
Blockchain Tools Module for Payment Pipeline System
Contains cryptographic utilities, transaction classes, and blockchain operations
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BlockchainTools")


class BlockchainToolsError(Exception):
    """Base exception for blockchain tools errors"""
    pass


class InvalidTransactionError(BlockchainToolsError):
    """Raised when transaction validation fails"""
    pass


class InsufficientFundsError(BlockchainToolsError):
    """Raised when account has insufficient funds"""
    pass


class Wallet:
    """Digital Wallet implementation with RSA-based cryptography"""
    
    def __init__(self, private_key: Optional[rsa.RSAPrivateKey] = None):
        """
        Initialize wallet with existing or new private key
        Args:
            private_key: Optional existing private key
        """
        self.private_key = private_key or self._generate_private_key()
        self.public_key = self.private_key.public_key()
        
    @staticmethod
    def _generate_private_key() -> rsa.RSAPrivateKey:
        """Generate RSA private key (2048 bits)"""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

    def sign_transaction(self, transaction_hash: str) -> bytes:
        """
        Sign transaction hash with private key
        Args:
            transaction_hash: SHA-256 hash of transaction data
        Returns:
            Digital signature
        """
        try:
            return self.private_key.sign(
                transaction_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except Exception as e:
            logger.error(f"Signing failed: {str(e)}")
            raise BlockchainToolsError("Transaction signing failed") from e

    def get_public_pem(self) -> bytes:
        """Get public key in PEM format"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    @classmethod
    def public_key_from_pem(cls, pem_data: bytes) -> rsa.RSAPublicKey:
        """Load public key from PEM data"""
        return serialization.load_pem_public_key(
            pem_data,
            backend=default_backend()
        )

    def save_private_key(self, filename: str, password: Optional[bytes] = None):
        """Securely store private key to file"""
        encryption = (
            serialization.BestAvailableEncryption(password) if password
            else serialization.NoEncryption()
        )
        
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption
        )
        
        with open(filename, 'wb') as f:
            f.write(pem)
        logger.info(f"Private key saved to {filename}")

    @classmethod
    def load_private_key(cls, filename: str, password: Optional[bytes] = None) -> 'Wallet':
        """Load private key from encrypted file"""
        with open(filename, 'rb') as f:
            pem = f.read()
        
        private_key = load_pem_private_key(
            pem,
            password=password,
            backend=default_backend()
        )
        return cls(private_key)


class Transaction:
    """Blockchain Transaction Class"""
    
    def __init__(self, sender: str, receiver: str, amount: float, 
                 signature: Optional[bytes] = None, timestamp: Optional[float] = None):
        """
        Initialize a new transaction
        Args:
            sender: Sender's public key (PEM)
            receiver: Receiver's public key (PEM)
            amount: Transaction amount
            signature: Digital signature (optional)
            timestamp: Transaction timestamp (optional)
        """
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = timestamp or datetime.now().timestamp()
        self.signature = signature
        self.transaction_id = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of transaction data"""
        transaction_data = {
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'timestamp': self.timestamp
        }
        return hashlib.sha256(
            json.dumps(transaction_data, sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self) -> Dict:
        """Serialize transaction to dictionary"""
        return {
            'transaction_id': self.transaction_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'timestamp': self.timestamp,
            'signature': self.signature.hex() if self.signature else None
        }

    def validate(self, sender_public_key: rsa.RSAPublicKey) -> bool:
        """
        Validate transaction signature
        Args:
            sender_public_key: Sender's public key
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            sender_public_key.verify(
                self.signature,
                self.transaction_id.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            logger.warning("Invalid transaction signature")
            return False
        except Exception as e:
            logger.error(f"Signature validation error: {str(e)}")
            return False


class BlockchainTools:
    """Main interface for blockchain operations"""
    
    @staticmethod
    def create_transaction(wallet: Wallet, receiver_public_pem: bytes, 
                          amount: float) -> Transaction:
        """
        Create and sign a new transaction
        Args:
            wallet: Sender's wallet
            receiver_public_pem: Receiver's public key (PEM)
            amount: Transaction amount
        Returns:
            Signed Transaction object
        """
        try:
            transaction = Transaction(
                sender=wallet.get_public_pem().decode(),
                receiver=receiver_public_pem.decode(),
                amount=amount
            )
            
            transaction.signature = wallet.sign_transaction(transaction.transaction_id)
            logger.info(f"Created transaction {transaction.transaction_id}")
            return transaction
        except Exception as e:
            logger.error(f"Transaction creation failed: {str(e)}")
            raise BlockchainToolsError("Transaction creation failed") from e

    @staticmethod
    def validate_transaction(transaction: Transaction) -> Tuple[bool, str]:
        """
        Validate transaction integrity and signature
        Args:
            transaction: Transaction to validate
        Returns:
            Tuple: (validation status, status message)
        """
        try:
            # Validate transaction hash
            if transaction.transaction_id != transaction.calculate_hash():
                return False, "Invalid transaction hash"

            # Load sender's public key
            sender_public_key = Wallet.public_key_from_pem(
                transaction.sender.encode()
            )

            # Validate signature
            if not transaction.validate(sender_public_key):
                return False, "Invalid digital signature"

            # Add additional business logic validations here
            if transaction.amount <= 0:
                return False, "Invalid transaction amount"

            return True, "Transaction valid"
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def generate_key_pair() -> Tuple[Wallet, bytes]:
        """Generate new wallet and return public key"""
        wallet = Wallet()
        return wallet, wallet.get_public_pem()

    @staticmethod
    def serialize_transaction(transaction: Transaction) -> str:
        """Serialize transaction for network transmission"""
        return json.dumps(transaction.to_dict())

    @staticmethod
    def deserialize_transaction(transaction_data: str) -> Transaction:
        """Deserialize transaction from JSON string"""
        data = json.loads(transaction_data)
        return Transaction(
            sender=data['sender'],
            receiver=data['receiver'],
            amount=data['amount'],
            timestamp=data['timestamp'],
            signature=bytes.fromhex(data['signature']) if data['signature'] else None
        )


# Example usage
if __name__ == "__main__":
    try:
        # Generate sender and receiver wallets
        sender_wallet, sender_public = BlockchainTools.generate_key_pair()
        receiver_wallet, receiver_public = BlockchainTools.generate_key_pair()

        # Create and validate transaction
        transaction = BlockchainTools.create_transaction(
            sender_wallet,
            receiver_public,
            100.0
        )

        # Validate transaction
        is_valid, message = BlockchainTools.validate_transaction(transaction)
        print(f"Transaction valid: {is_valid} - {message}")

        # Serialization demo
        serialized = BlockchainTools.serialize_transaction(transaction)
        deserialized = BlockchainTools.deserialize_transaction(serialized)
        print(f"Deserialized transaction ID: {deserialized.transaction_id}")

    except BlockchainToolsError as e:
        print(f"Blockchain error occurred: {str(e)}")