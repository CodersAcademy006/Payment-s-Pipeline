"""
Post-Quantum Cryptography Payment Pipeline with Hybrid Kyber/Dilithium Implementation
"""

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import kyber, dilithium, ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.types import PublicKeyTypes
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.exceptions import InvalidSignature, InvalidTag
import os
import hashlib
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumSecurePayment")

class QuantumSecurePayment:
    def __init__(self):
        """Initialize with Kyber, Dilithium, and ECC key pairs"""
        try:
            self.kyber_private_key = kyber.generate_private_key()
            self.dilithium_private_key = dilithium.generate_private_key()
            self.ecc_private_key = ec.generate_private_key(ec.SECP521R1())
            logger.info("Generated all cryptographic key pairs")
        except Exception as e:
            logger.error(f"Key generation failed: {str(e)}")
            raise

    # Key Serialization Methods
    def get_kyber_public_key(self) -> kyber.KyberPublicKey:
        return self.kyber_private_key.public_key()

    def get_dilithium_public_key(self) -> dilithium.DilithiumPublicKey:
        return self.dilithium_private_key.public_key()

    def get_ecc_public_key(self) -> ec.EllipticCurvePublicKey:
        return self.ecc_private_key.public_key()

    def serialize_kyber_public_key(self) -> bytes:
        return self.get_kyber_public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

    # Encryption/Decryption Methods
    def quantum_encrypt(self, recipient_kyber_pubkey: kyber.KyberPublicKey, 
                       plaintext: bytes) -> Tuple[bytes, bytes, bytes]:
        """Kyber KEM + AES-GCM encryption with forward secrecy"""
        try:
            # Kyber KEM
            kem_ciphertext, shared_secret = recipient_kyber_pubkey.encrypt()
            
            # Derive symmetric key
            symmetric_key = HKDF(
                algorithm=hashes.SHA512(),
                length=32,
                salt=None,
                info=b'kyber_aes_gcm'
            ).derive(shared_secret)

            # AES-GCM encryption
            nonce = os.urandom(12)
            ciphertext = AESGCM(symmetric_key).encrypt(nonce, plaintext, None)
            
            return kem_ciphertext, nonce, ciphertext
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise

    def quantum_decrypt(self, kem_ciphertext: bytes, nonce: bytes, 
                       ciphertext: bytes) -> bytes:
        """Kyber KEM + AES-GCM decryption"""
        try:
            # Kyber decapsulation
            shared_secret = self.kyber_private_key.decapsulate(kem_ciphertext)
            
            # Derive symmetric key
            symmetric_key = HKDF(
                algorithm=hashes.SHA512(),
                length=32,
                salt=None,
                info=b'kyber_aes_gcm'
            ).derive(shared_secret)

            # AES-GCM decryption
            return AESGCM(symmetric_key).decrypt(nonce, ciphertext, None)
        except (InvalidTag, ValueError) as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise

    # Hybrid Key Exchange
    def hybrid_key_exchange(self, 
                           recipient_ecc_pubkey: ec.EllipticCurvePublicKey,
                           recipient_kyber_pubkey: kyber.KyberPublicKey
                           ) -> Tuple[bytes, bytes, bytes]:
        """Combined ECDH (P-521) + Kyber-768 Key Exchange"""
        try:
            # ECDH Key Exchange
            ecc_shared_secret = self.ecc_private_key.exchange(
                ec.ECDH(), recipient_ecc_pubkey
            )
            
            # Kyber KEM
            kem_ciphertext, kyber_shared_secret = recipient_kyber_pubkey.encrypt()
            
            # Combine secrets using HKDF
            combined_secret = HKDF(
                algorithm=hashes.SHA3_512(),
                length=64,
                salt=None,
                info=b'hybrid_key_derivation'
            ).derive(ecc_shared_secret + kyber_shared_secret)
            
            return kem_ciphertext, combined_secret
        except Exception as e:
            logger.error(f"Hybrid key exchange failed: {str(e)}")
            raise

    # Digital Signatures
    def sign_transaction(self, transaction_data: bytes) -> bytes:
        """Dilithium signature with SHA3-512 hashing"""
        try:
            digest = hashlib.sha3_512(transaction_data).digest()
            return self.dilithium_private_key.sign(digest)
        except Exception as e:
            logger.error(f"Signing failed: {str(e)}")
            raise

    @staticmethod
    def verify_signature(public_key: dilithium.DilithiumPublicKey,
                        transaction_data: bytes,
                        signature: bytes) -> bool:
        """Verify Dilithium signature"""
        try:
            digest = hashlib.sha3_512(transaction_data).digest()
            public_key.verify(signature, digest)
            return True
        except InvalidSignature:
            logger.warning("Invalid signature detected")
            return False

    # Key Serialization
    def serialize_hybrid_public_keys(self) -> Tuple[bytes, bytes, bytes]:
        """Serialize all public keys for transmission"""
        return (
            self.serialize_kyber_public_key(),
            self.get_ecc_public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint
            ),
            self.get_dilithium_public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

    @staticmethod
    def deserialize_hybrid_public_keys(kyber_bytes: bytes,
                                      ecc_bytes: bytes,
                                      dilithium_bytes: bytes
                                      ) -> Tuple[kyber.KyberPublicKey,
                                                ec.EllipticCurvePublicKey,
                                                dilithium.DilithiumPublicKey]:
        """Deserialize public keys from bytes"""
        kyber_pubkey = kyber.KyberPublicKey.from_public_bytes(kyber_bytes)
        ecc_pubkey = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP521R1(), ecc_bytes
        )
        dilithium_pubkey = dilithium.DilithiumPublicKey.from_public_bytes(dilithium_bytes)
        return kyber_pubkey, ecc_pubkey, dilithium_pubkey

# Example Usage
if __name__ == "__main__":
    try:
        # Initialize payment endpoints
        merchant = QuantumSecurePayment()
        customer = QuantumSecurePayment()

        # Key exchange
        kyber_pub, ecc_pub, dilithium_pub = merchant.serialize_hybrid_public_keys()
        merchant_pubkeys = merchant.deserialize_hybrid_public_keys(kyber_pub, ecc_pub, dilithium_pub)

        # Customer sends encrypted payment data
        payment_data = b"Payment: $42.00|Nonce:12345"
        kem_ct, nonce, aes_ct = customer.quantum_encrypt(merchant_pubkeys[0], payment_data)
        signature = customer.sign_transaction(payment_data)

        # Merchant processes payment
        decrypted_data = merchant.quantum_decrypt(kem_ct, nonce, aes_ct)
        is_valid = QuantumSecurePayment.verify_signature(
            merchant_pubkeys[2], decrypted_data, signature
        )

        print(f"Payment valid: {is_valid}, Data: {decrypted_data.decode()}")

    except Exception as e:
        logger.error(f"Payment processing failed: {str(e)}")
        raise