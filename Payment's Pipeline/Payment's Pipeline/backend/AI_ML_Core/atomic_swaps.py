"""
Atomic cross-chain swaps implementation with hash time-locked contracts (HTLC)
Supports multi-chain interoperability through blockchain adapters with rigorous error handling
"""

import hashlib
import json
import logging
import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Configure logging
logger = logging.getLogger("AtomicSwap")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# Custom Exceptions
class AtomicSwapError(Exception):
    """Base exception for atomic swap operations"""


class InvalidSecretError(AtomicSwapError):
    """Raised when an invalid secret is provided"""


class ContractError(AtomicSwapError):
    """Base exception for contract-related errors"""


class ContractExpiredError(ContractError):
    """Raised when operating on an expired contract"""


class ContractValidationError(ContractError):
    """Raised when contract validation fails"""


class BlockchainAdapter(ABC):
    """Abstract base class for blockchain adapters"""

    @abstractmethod
    def deploy_contract(self, contract: Dict) -> str:
        """Deploy HTLC contract to blockchain"""
        pass

    @abstractmethod
    def validate_address(self, address: str) -> bool:
        """Validate blockchain address format"""
        pass

    @abstractmethod
    def get_contract_status(self, tx_hash: str) -> str:
        """Get current status of deployed contract"""
        pass


class MockBlockchain(BlockchainAdapter):
    """Mock blockchain implementation for testing"""

    def __init__(self, name: str):
        self.name = name
        self.contracts: Dict[str, Dict] = {}

    def deploy_contract(self, contract: Dict) -> str:
        """Store contract in memory and return mock tx hash"""
        tx_hash = f"0x{hashlib.sha256(json.dumps(contract).encode()).hexdigest()[:64]}"
        self.contracts[tx_hash] = contract
        logger.info(f"Deployed contract on {self.name}: {tx_hash}")
        return tx_hash

    def validate_address(self, address: str) -> bool:
        """Mock address validation"""
        return address.startswith("0x") and len(address) == 42

    def get_contract_status(self, tx_hash: str) -> str:
        """Get contract status from mock storage"""
        return self.contracts.get(tx_hash, {}).get("status", "not_found")


class AtomicSwap:
    """HTLC-based atomic swap coordinator with multi-chain support"""

    def __init__(
        self,
        initiator_chain: BlockchainAdapter,
        participant_chain: BlockchainAdapter,
        initiator_address: str,
        participant_address: str,
        timeout: int = 3600,
        timeout_buffer: int = 300,
    ):
        if not initiator_chain.validate_address(initiator_address):
            raise ContractValidationError("Invalid initiator address")
        if not participant_chain.validate_address(participant_address):
            raise ContractValidationError("Invalid participant address")

        self.initiator_chain = initiator_chain
        self.participant_chain = participant_chain
        self.initiator_address = initiator_address
        self.participant_address = participant_address
        self.timeout = timeout
        self.timeout_buffer = timeout_buffer

        # Generate cryptographic secrets
        self.secret = secrets.token_hex(32)
        self.secret_hash = self._generate_hash(self.secret)
        logger.debug("Generated new swap secret")

    def _generate_hash(self, secret: str) -> str:
        """Generate SHA-256 hash of secret"""
        return hashlib.sha256(secret.encode()).hexdigest()

    def initiate_swap(self, amount: float) -> Tuple[str, Dict]:
        """Create initial HTLC on initiator chain"""
        if amount <= 0:
            raise ContractValidationError("Swap amount must be positive")

        contract = {
            "amount": amount,
            "recipient": self.participant_address,
            "secret_hash": self.secret_hash,
            "timeout": (datetime.now() + timedelta(seconds=self.timeout)).isoformat(),
            "status": "pending",
            "chain": self.initiator_chain.__class__.__name__,
        }

        try:
            tx_hash = self.initiator_chain.deploy_contract(contract)
            logger.info(f"Initiator contract deployed: {tx_hash}")
            return tx_hash, contract
        except Exception as e:
            logger.error(f"Failed to deploy initiator contract: {str(e)}")
            raise ContractError("Initiator contract deployment failed") from e

    def participate_swap(self, amount: float, initiator_tx_hash: str) -> Tuple[str, Dict]:
        """Create reciprocal HTLC on participant chain"""
        initiator_contract = self.initiator_chain.contracts.get(initiator_tx_hash)
        if not initiator_contract:
            raise ContractValidationError("Initiator contract not found")

        if initiator_contract["secret_hash"] != self.secret_hash:
            raise InvalidSecretError("Secret hash mismatch")

        participant_timeout = (
            datetime.fromisoformat(initiator_contract["timeout"])
            - timedelta(seconds=self.timeout_buffer)
        ).isoformat()

        contract = {
            "amount": amount,
            "recipient": self.initiator_address,
            "secret_hash": self.secret_hash,
            "timeout": participant_timeout,
            "status": "pending",
            "chain": self.participant_chain.__class__.__name__,
        }

        try:
            tx_hash = self.participant_chain.deploy_contract(contract)
            logger.info(f"Participant contract deployed: {tx_hash}")
            return tx_hash, contract
        except Exception as e:
            logger.error(f"Failed to deploy participant contract: {str(e)}")
            raise ContractError("Participant contract deployment failed") from e

    def redeem_swap(self, tx_hash: str, secret: str) -> bool:
        """Redeem funds from target contract"""
        contract = self.participant_chain.contracts.get(tx_hash)
        if not contract:
            raise ContractValidationError("Contract not found")

        if self._generate_hash(secret) != contract["secret_hash"]:
            raise InvalidSecretError("Invalid redemption secret")

        if datetime.fromisoformat(contract["timeout"]) < datetime.now():
            raise ContractExpiredError("Contract has expired")

        contract["status"] = "completed"
        logger.info(f"Successfully redeemed contract: {tx_hash}")
        return True

    def refund_swap(self, tx_hash: str) -> bool:
        """Refund expired contract"""
        contract = self.initiator_chain.contracts.get(tx_hash)
        if not contract:
            raise ContractValidationError("Contract not found")

        if datetime.fromisoformat(contract["timeout"]) > datetime.now():
            raise ContractError("Contract not yet expired")

        contract["status"] = "refunded"
        logger.info(f"Successfully refunded contract: {tx_hash}")
        return True


if __name__ == "__main__":
    # Example usage with mock blockchains
    try:
        # Initialize mock blockchains
        ethereum = MockBlockchain("Ethereum")
        algorand = MockBlockchain("Algorand")

        # Create swap instance
        swap = AtomicSwap(
            initiator_chain=ethereum,
            participant_chain=algorand,
            initiator_address="0xInitiatorAddress",
            participant_address="0xParticipantAddress",
            timeout=3600,
        )

        # Initiate swap on Ethereum
        tx1, contract1 = swap.initiate_swap(1.5)
        logger.info(f"Ethereum contract created: {json.dumps(contract1, indent=2)}")

        # Participate swap on Algorand
        tx2, contract2 = swap.participate_swap(150, tx1)
        logger.info(f"Algorand contract created: {json.dumps(contract2, indent=2)}")

        # Redeem swap
        success = swap.redeem_swap(tx2, swap.secret)
        logger.info(f"Redeem successful: {success}")

        # Check contract statuses
        logger.info(f"Ethereum contract status: {ethereum.get_contract_status(tx1)}")
        logger.info(f"Algorand contract status: {algorand.get_contract_status(tx2)}")

    except AtomicSwapError as e:
        logger.error(f"Atomic swap failed: {str(e)}")
        exit(1)