"""
Advanced consensus algorithms implementation for distributed payment systems
Supports PoW, PoS, DPoS, PBFT with configurable parameters and performance metrics
"""

import hashlib
import logging
import random
import time
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Type

logger = logging.getLogger("Consensus")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Custom Exceptions
class ConsensusError(Exception):
    """Base exception for consensus-related errors"""

class InvalidBlockError(ConsensusError):
    """Raised when block validation fails"""

class InsufficientStakeError(ConsensusError):
    """Raised when validator has insufficient stake"""

class NotEnoughParticipantsError(ConsensusError):
    """Raised when minimum node count not met"""

@dataclass
class Block:
    index: int
    timestamp: float
    transactions: List[Dict]
    previous_hash: str
    nonce: int = 0
    validator: Optional[str] = None
    signature: Optional[str] = None
    hash: Optional[str] = None

    def compute_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "validator": self.validator
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class ConsensusAlgorithm(ABC):
    """Abstract base class for consensus implementations"""
    
    def __init__(self, node_id: str, difficulty: int = 4):
        self.node_id = node_id
        self.difficulty = difficulty
        self.metrics = {
            "blocks_validated": 0,
            "validation_time": [],
            "consensus_rounds": 0
        }

    @abstractmethod
    def validate_block(self, block: Block, previous_block: Block) -> bool:
        """Validate block according to consensus rules"""
        pass

    @abstractmethod
    def create_block(self, transactions: List[Dict], previous_block: Block) -> Block:
        """Create new block according to consensus rules"""
        pass

    def get_metrics(self) -> Dict:
        """Return performance metrics"""
        return {
            **self.metrics,
            "avg_validation_time": (
                sum(self.metrics["validation_time"]) / 
                len(self.metrics["validation_time"]) 
                if self.metrics["validation_time"] else 0
            )
        }

class ProofOfWork(ConsensusAlgorithm):
    """Proof of Work implementation with dynamic difficulty"""
    
    def create_block(self, transactions: List[Dict], previous_block: Block) -> Block:
        start_time = time.time()
        new_block = Block(
            index=previous_block.index + 1,
            timestamp=time.time(),
            transactions=transactions,
            previous_hash=previous_block.hash,
            nonce=0
        )

        prefix = '0' * self.difficulty
        while True:
            new_block.hash = new_block.compute_hash()
            if new_block.hash.startswith(prefix):
                break
            new_block.nonce += 1

        self.metrics["validation_time"].append(time.time() - start_time)
        self.metrics["blocks_validated"] += 1
        return new_block

    def validate_block(self, block: Block, previous_block: Block) -> bool:
        if block.previous_hash != previous_block.hash:
            logger.error("Block chain link broken")
            return False
            
        if not block.hash.startswith('0' * self.difficulty):
            logger.error("Proof of Work validation failed")
            return False
            
        if block.hash != block.compute_hash():
            logger.error("Block hash mismatch")
            return False
            
        return True

class ProofOfStake(ConsensusAlgorithm):
    """Proof of Stake implementation with weighted validation"""
    
    def __init__(self, node_id: str, stake: int = 100):
        super().__init__(node_id)
        self.stake = stake
        self.validators = defaultdict(int)

    def register_validator(self, node_id: str, stake: int):
        if stake <= 0:
            raise ValueError("Stake must be positive")
        self.validators[node_id] = stake

    def create_block(self, transactions: List[Dict], previous_block: Block) -> Block:
        if self.validators[self.node_id] < self.stake:
            raise InsufficientStakeError(f"Node {self.node_id} has insufficient stake")

        start_time = time.time()
        validator = self._select_validator()
        
        new_block = Block(
            index=previous_block.index + 1,
            timestamp=time.time(),
            transactions=transactions,
            previous_hash=previous_block.hash,
            validator=validator
        )
        
        new_block.hash = new_block.compute_hash()
        self.metrics["validation_time"].append(time.time() - start_time)
        self.metrics["blocks_validated"] += 1
        return new_block

    def _select_validator(self) -> str:
        total_stake = sum(self.validators.values())
        selection = random.uniform(0, total_stake)
        current = 0
        
        for validator, stake in self.validators.items():
            current += stake
            if selection <= current:
                return validator
        return list(self.validators.keys())[-1]

    def validate_block(self, block: Block, previous_block: Block) -> bool:
        if block.validator not in self.validators:
            logger.error("Invalid validator")
            return False
            
        if self.validators[block.validator] < self.stake:
            logger.error("Validator has insufficient stake")
            return False
            
        if block.hash != block.compute_hash():
            logger.error("Block hash mismatch")
            return False
            
        return True

class DelegatedProofOfStake(ProofOfStake):
    """Delegated Proof of Stake with witness elections"""
    
    def __init__(self, node_id: str, num_delegates: int = 21):
        super().__init__(node_id)
        self.num_delegates = num_delegates
        self.delegates = set()

    def elect_delegates(self):
        sorted_validators = sorted(
            self.validators.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.num_delegates]
        self.delegates = {delegate[0] for delegate in sorted_validators}

    def create_block(self, transactions: List[Dict], previous_block: Block) -> Block:
        if self.node_id not in self.delegates:
            raise ConsensusError("Node not in current delegate set")

        return super().create_block(transactions, previous_block)

    def validate_block(self, block: Block, previous_block: Block) -> bool:
        if block.validator not in self.delegates:
            logger.error("Block not from elected delegate")
            return False
        return super().validate_block(block, previous_block)

class PBFT(ConsensusAlgorithm):
    """Practical Byzantine Fault Tolerance implementation"""
    
    def __init__(self, node_id: str, nodes: List[str], fault_tolerance: float = 0.33):
        super().__init__(node_id)
        self.nodes = nodes
        self.fault_tolerance = fault_tolerance
        self.messages = defaultdict(dict)
        self.min_nodes = max(1, int(len(nodes) * (1 - fault_tolerance)))

    def validate_block(self, block: Block, previous_block: Block) -> bool:
        # PBFT validation requires 3-phase commit
        return self._pbft_consensus(block)

    def _pbft_consensus(self, block: Block) -> bool:
        # Phase 1: Pre-prepare
        if not self._verify_pre_prepare(block):
            return False

        # Phase 2: Prepare
        prepare_messages = self._collect_messages("prepare", block.hash)
        if len(prepare_messages) < 2 * self.min_nodes // 3:
            return False

        # Phase 3: Commit
        commit_messages = self._collect_messages("commit", block.hash)
        if len(commit_messages) < self.min_nodes:
            return False

        return True

    def _verify_pre_prepare(self, block: Block) -> bool:
        # In real implementation this would verify primary node signature
        return block.hash == block.compute_hash()

    def _collect_messages(self, msg_type: str, block_hash: str) -> List:
        return [
            msg for msg in self.messages.get(block_hash, [])
            if msg["type"] == msg_type
        ]

class ConsensusFactory:
    """Factory for creating consensus algorithm instances"""
    
    algorithms = {
        "pow": ProofOfWork,
        "pos": ProofOfStake,
        "dpos": DelegatedProofOfStake,
        "pbft": PBFT
    }

    @classmethod
    def create_consensus(
        cls,
        algorithm: str,
        node_id: str,
        **kwargs
    ) -> ConsensusAlgorithm:
        algo_class = cls.algorithms.get(algorithm.lower())
        if not algo_class:
            raise ValueError(f"Unsupported consensus algorithm: {algorithm}")
        return algo_class(node_id, **kwargs)

# Example Usage
if __name__ == "__main__":
    try:
        # Initialize network
        nodes = [f"node_{i}" for i in range(10)]
        
        # Create PoW instance
        pow_node = ConsensusFactory.create_consensus("pow", "miner_node", difficulty=5)
        
        # Create PoS network
        pos_node = ConsensusFactory.create_consensus("pos", "validator_node", stake=100)
        pos_node.register_validator("validator_node", 1000)
        pos_node.register_validator("other_validator", 500)
        
        # Simulate block creation
        genesis_block = Block(0, time.time(), [], "0")
        genesis_block.hash = genesis_block.compute_hash()
        
        # PoW Mining
        pow_block = pow_node.create_block(
            [{"from": "A", "to": "B", "amount": 10}],
            genesis_block
        )
        print(f"PoW Block mined: {pow_block.hash}")
        
        # PoS Validation
        pos_block = pos_node.create_block(
            [{"from": "C", "to": "D", "amount": 20}],
            genesis_block
        )
        print(f"PoS Block created by: {pos_block.validator}")
        
        # DPOS Election
        dpos_node = ConsensusFactory.create_consensus("dpos", "delegate_node", num_delegates=5)
        for n in nodes:
            dpos_node.register_validator(n, random.randint(100, 1000))
        dpos_node.elect_delegates()
        print(f"Elected delegates: {dpos_node.delegates}")
        
        # PBFT Network
        pbft_node = ConsensusFactory.create_consensus(
            "pbft",
            "pbft_node",
            nodes=nodes,
            fault_tolerance=0.25
        )
        pbft_block = Block(1, time.time(), [], genesis_block.hash)
        pbft_block.hash = pbft_block.compute_hash()
        print("PBFT Consensus:", pbft_node.validate_block(pbft_block, genesis_block))
        
    except ConsensusError as e:
        logger.error(f"Consensus process failed: {str(e)}")