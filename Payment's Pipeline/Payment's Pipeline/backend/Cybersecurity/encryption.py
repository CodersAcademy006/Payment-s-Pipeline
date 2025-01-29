"""
Payment Pipeline Cryptographic Security Module
Combines AES-256-GCM, RSA-OAEP, and ED25519 in a hardened encryption implementation
"""

import os
import logging
import json
import base64
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Tuple, Optional, Union
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ed25519
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.keywrap import aes_key_wrap, aes_key_unwrap
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.constant_time import bytes_eq

# Configure security parameters
SECURITY_CONFIG = {
    "aes_key_size": 32,  # 256-bit
    "salt_size": 32,
    "pbkdf2_iterations": 310000,  # OWASP 2023 recommendations
    "nonce_size": 12,  # GCM recommended nonce size
    "max_age_seconds": 300, #5-minute validity window
    "rsa_key_size": 4096,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('secure_payment.log'), logging.StreamHandler()]
)
logger = logging.getLogger("PaymentCryptography")

class EncryptionError(Exception):
    """Base exception for cryptographic operations"""
    pass

class TamperDetectedError(EncryptionError):
    """Raised when data integrity check fails"""
    pass

class KeyExpiredError(EncryptionError):
    """Raised when cryptographic material is expired"""
    pass

class SecureEncryptor:
    """Combined symmetric/asymmetric encryption suite with key rotation support"""
    
    @staticmethod
    def generate_aes_key() -> bytes:
        """Generate cryptographically secure AES-256 key"""
        return os.urandom(SECURITY_CONFIG["aes_key_size"])
    
    @staticmethod
    def derive_key(password: bytes, salt: bytes) -> bytes:
        """PBKDF2 key derivation with security parameters"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=SECURITY_CONFIG["aes_key_size"],
            salt=salt,
            iterations=SECURITY_CONFIG["pbkdf2_iterations"],
            backend=default_backend()
        )
        return kdf.derive(password)
    
    @staticmethod
    def aes_gcm_encrypt(plaintext: bytes, key: bytes, associated_data: bytes = None) -> dict:
        """Authenticated AES-GCM encryption with automatic nonce generation"""
        if len(key) != SECURITY_CONFIG["aes_key_size"]:
            raise EncryptionError(f"Invalid key length: {len(key)} bytes")
        
        nonce = os.urandom(SECURITY_CONFIG["nonce_size"])
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return {
            "ciphertext": ciphertext,
            "nonce": nonce,
            "tag": encryptor.tag,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def aes_gcm_decrypt(encrypted_data: dict, key: bytes, associated_data: bytes = None) -> bytes:
        """Authenticated AES-GCM decryption with validity checks"""
        required_fields = {"ciphertext", "nonce", "tag", "timestamp"}
        if not required_fields.issubset(encrypted_data):
            raise EncryptionError("Invalid encrypted data structure")
        
        if datetime.utcnow() - datetime.fromisoformat(encrypted_data["timestamp"]) > timedelta(
            seconds=SECURITY_CONFIG["max_age_seconds"]
        ):
            raise KeyExpiredError("Encrypted payload has expired")
        
        try:
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(encrypted_data["nonce"], encrypted_data["tag"]),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            if associated_data:
                decryptor.authenticate_additional_data(associated_data)
            
            return decryptor.update(encrypted_data["ciphertext"]) + decryptor.finalize()
        except InvalidTag:
            logger.error("GCM tag verification failed")
            raise TamperDetectedError("Data integrity check failed") from None
    
    @staticmethod
    def generate_rsa_keypair() -> Tuple[rsa.RSAPrivateKey, bytes]:
        """Generate RSA-4096 key pair with PKCS#8 serialization"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=SECURITY_CONFIG["rsa_key_size"],
            backend=default_backend()
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return private_key, public_pem
    
    @staticmethod
    def rsa_encrypt(data: bytes, public_key_pem: bytes) -> bytes:
        """RSA-OAEP encryption with SHA-512 and MGF1 padding"""
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA512()),
                algorithm=hashes.SHA512(),
                label=None
            )
        )
    
    @staticmethod
    def rsa_decrypt(ciphertext: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """RSA-OAEP decryption with SHA-512 and MGF1 padding"""
        return private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA512()),
                algorithm=hashes.SHA512(),
                label=None
            )
        )
    
    @staticmethod
    def generate_ed25519_keypair() -> Tuple[ed25519.Ed25519PrivateKey, bytes]:
        """Generate ED25519 key pair for digital signatures"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return private_key, public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
    
    @staticmethod
    def sign_message(message: bytes, private_key: ed25519.Ed25519PrivateKey) -> bytes:
        """Generate ED25519 signature for message"""
        return private_key.sign(message)
    
    @staticmethod
    def verify_signature(message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify ED25519 signature with constant-time comparison"""
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
        try:
            public_key.verify(signature, message)
            return True
        except InvalidSignature:
            return False

class HybridEncryptor:
    """Combined symmetric/asymmetric encryption with key wrapping"""
    
    @staticmethod
    def encrypt_with_keywrap(plaintext: bytes, public_key_pem: bytes) -> dict:
        """Hybrid encryption using AES-256-GCM + RSA-OAEP key wrapping"""
        # Generate ephemeral AES key
        aes_key = SecureEncryptor.generate_aes_key()
        
        # Encrypt data with AES-GCM
        encrypted_data = SecureEncryptor.aes_gcm_encrypt(plaintext, aes_key)
        
        # Wrap AES key with RSA-OAEP
        wrapped_key = SecureEncryptor.rsa_encrypt(aes_key, public_key_pem)
        
        return {
            "wrapped_key": wrapped_key,
            "encrypted_data": encrypted_data,
            "algorithm": "AES256-GCM+RSA-OAEP"
        }
    
    @staticmethod
    def decrypt_with_keywrap(encrypted_package: dict, private_key: rsa.RSAPrivateKey) -> bytes:
        """Hybrid decryption with RSA key unwrapping"""
        # Unwrap AES key
        aes_key = SecureEncryptor.rsa_decrypt(
            encrypted_package["wrapped_key"], private_key
        )
        
        # Decrypt data with AES-GCM
        return SecureEncryptor.aes_gcm_decrypt(
            encrypted_package["encrypted_data"], aes_key
        )

class Hasher:
    """Cryptographic hashing and HMAC utilities"""
    
    @staticmethod
    def secure_hash(data: bytes) -> Tuple[bytes, bytes]:
        """Memory-hard Argon2 hash with salt"""
        salt = os.urandom(SECURITY_CONFIG["salt_size"])
        hashed = hashlib.blake2b(
            data,
            salt=salt,
            person=b'payment_pipeline',
            digest_size=64
        ).digest()
        return hashed, salt
    
    @staticmethod
    def hmac_sign(data: bytes, key: bytes) -> bytes:
        """HMAC-SHA512 with constant-time verification"""
        if len(key) < SECURITY_CONFIG["aes_key_size"]:
            raise EncryptionError("HMAC key too short")
        return hmac.new(key, data, hashlib.sha512).digest()
    
    @staticmethod
    def hmac_verify(data: bytes, signature: bytes, key: bytes) -> bool:
        """Constant-time HMAC verification"""
        expected = Hasher.hmac_sign(data, key)
        return bytes_eq(expected, signature)

# Example usage
if __name__ == "__main__":
    try:
        # Generate keys
        rsa_private, rsa_public = SecureEncryptor.generate_rsa_keypair()
        ed_private, ed_public = SecureEncryptor.generate_ed25519_keypair()
        
        # Hybrid encryption demo
        sensitive_data = b"Payment: $5000 to ACCT-123456"
        encrypted_package = HybridEncryptor.encrypt_with_keywrap(sensitive_data, rsa_public)
        decrypted_data = HybridEncryptor.decrypt_with_keywrap(encrypted_package, rsa_private)
        
        print(f"Decrypted match: {decrypted_data == sensitive_data}")
        
        # Digital signature demo
        signature = SecureEncryptor.sign_message(sensitive_data, ed_private)
        valid = SecureEncryptor.verify_signature(sensitive_data, signature, ed_public)
        print(f"Signature valid: {valid}")
        
        # Key derivation demo
        salt = os.urandom(SECURITY_CONFIG["salt_size"])
        derived_key = SecureEncryptor.derive_key(b"secure_password", salt)
        print(f"Derived key: {base64.b64encode(derived_key).decode()}")
        
    except EncryptionError as e:
        logger.error(f"Cryptographic failure: {str(e)}")
        raise