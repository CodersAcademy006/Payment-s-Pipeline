"""
Blockchain-based Smart Contracts for Payment Pipeline System
Author: AI Assistant
Date: [Current Date]
Version: 1.2.0
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import uuid
from dataclasses import dataclass, field

# Configure logging (would need proper logging setup in real implementation)
# import logging
# logger = logging.getLogger(__name__)

@dataclass
class SmartContract:
    """
    A blockchain-based smart contract for payment processing with escrow capabilities,
    dispute resolution, and audit trails.
    """
    contract_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parties: List[str] = field(default_factory=list)
    payment_amount: float = 0.0
    conditions: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, active, completed, disputed, canceled
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    ledger: List[Tuple[datetime, str, dict]] = field(default_factory=list)
    escrow_balance: float = 0.0

    def __post_init__(self):
        self._validate_initial_conditions()
        self._record_ledger_event("CONTRACT_CREATED", {
            "parties": self.parties,
            "amount": self.payment_amount
        })

    def _validate_initial_conditions(self):
        """Validate initial contract parameters"""
        if len(self.parties) < 2:
            raise ValueError("Contract requires at least two parties")
        if self.payment_amount <= 0:
            raise ValueError("Payment amount must be positive")
        if not all(isinstance(party, str) for party in self.parties):
            raise TypeError("All parties must be string identifiers")

    def _record_ledger_event(self, event_type: str, metadata: dict):
        """Record immutable ledger event with timestamp"""
        event_time = datetime.utcnow()
        self.ledger.append((event_time, event_type, metadata))
        self.updated_at = event_time
        # In real implementation, this would be hashed and added to blockchain
        self._create_block(event_time, event_type, metadata)

    def _create_block(self, timestamp: datetime, event_type: str, data: dict):
        """Simulate blockchain block creation"""
        block_data = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "data": data,
            "previous_hash": self._get_previous_block_hash(),
        }
        block_hash = self._hash_block(block_data)
        self.ledger[-1] = (timestamp, event_type, {**data, "block_hash": block_hash})

    def _get_previous_block_hash(self) -> str:
        """Get hash of the previous block in the chain"""
        if len(self.ledger) == 0:
            return "0"
        return self.ledger[-1][2].get("block_hash", "0")

    @staticmethod
    def _hash_block(block_data: dict) -> str:
        """Create SHA-256 hash of block data"""
        block_string = json.dumps(block_data, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def deposit_to_escrow(self, amount: float, depositor: str):
        """Deposit funds into escrow account"""
        if self.status != "pending":
            raise ValueError("Escrow deposits only allowed in pending state")
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self.escrow_balance += amount
        self._record_ledger_event("ESCROW_DEPOSIT", {
            "depositor": depositor,
            "amount": amount,
            "new_balance": self.escrow_balance
        })

    def release_payment(self, releasing_party: str) -> bool:
        """
        Release payment to recipient when conditions are met
        Returns True if payment was successfully released
        """
        if self.status != "active":
            raise ValueError("Contract must be active to release payment")
        if not self._conditions_met():
            raise ValueError("Cannot release payment - conditions not met")
        if self.escrow_balance < self.payment_amount:
            raise ValueError("Insufficient escrow balance")

        # Execute payment (simulated)
        self.escrow_balance -= self.payment_amount
        self.status = "completed"
        self._record_ledger_event("PAYMENT_RELEASED", {
            "released_by": releasing_party,
            "amount": self.payment_amount,
            "recipient": self.parties[1],  # Assuming second party is recipient
            "remaining_balance": self.escrow_balance
        })
        return True

    def add_condition(self, condition: str):
        """Add new condition to the smart contract"""
        if self.status != "pending":
            raise ValueError("Cannot add conditions after contract activation")
        self.conditions.append(condition)
        self._record_ledger_event("CONDITION_ADDED", {
            "condition": condition,
            "total_conditions": len(self.conditions)
        })

    def activate_contract(self):
        """Activate the contract after all parties agree"""
        if self.status != "pending":
            raise ValueError("Contract already activated")
        if self.escrow_balance < self.payment_amount:
            raise ValueError("Full escrow amount must be deposited before activation")
        
        self.status = "active"
        self._record_ledger_event("CONTRACT_ACTIVATED", {
            "escrow_balance": self.escrow_balance,
            "conditions_count": len(self.conditions)
        })

    def raise_dispute(self, disputing_party: str, reason: str):
        """Raise a dispute about contract execution"""
        if self.status not in ["active", "pending"]:
            raise ValueError("Disputes cannot be raised in current state")
        
        self.status = "disputed"
        self._record_ledger_event("DISPUTE_RAISED", {
            "disputing_party": disputing_party,
            "reason": reason
        })

    def resolve_dispute(self, resolution: str, payment_amount: Optional[float] = None):
        """Resolve dispute and execute final decision"""
        if self.status != "disputed":
            raise ValueError("No active dispute to resolve")
        
        # Execute resolution
        if payment_amount:
            self.payment_amount = payment_amount
        
        self.status = "completed" if self._conditions_met() else "canceled"
        self._record_ledger_event("DISPUTE_RESOLVED", {
            "resolution": resolution,
            "final_status": self.status,
            "payment_amount": self.payment_amount
        })

    def _conditions_met(self) -> bool:
        """Check if all conditions are met (simulated implementation)"""
        # In real implementation, this would interface with oracle services
        # and execute actual condition verification logic
        return len(self.conditions) > 0 and all(
            condition.startswith("verified:") for condition in self.conditions
        )

    def audit_contract(self) -> Dict:
        """Generate audit report for the contract"""
        return {
            "contract_id": self.contract_id,
            "status": self.status,
            "escrow_balance": self.escrow_balance,
            "total_conditions": len(self.conditions),
            "conditions_met": self._conditions_met(),
            "last_updated": self.updated_at.isoformat(),
            "ledger_entries": len(self.ledger)
        }

    def get_contract_state(self) -> Dict:
        """Get current state of the contract"""
        return {
            "contract_id": self.contract_id,
            "parties": self.parties,
            "payment_amount": self.payment_amount,
            "status": self.status,
            "escrow_balance": self.escrow_balance,
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

# Example Usage
if __name__ == "__main__":
    # Initialize contract
    contract = SmartContract(
        parties=["client@company.com", "freelancer@example.com"],
        payment_amount=5000.00
    )

    # Add conditions
    contract.add_condition("verified:deliverables_completed")
    contract.add_condition("verified:client_approval")

    # Deposit escrow
    contract.deposit_to_escrow(5000.00, "client@company.com")

    # Activate contract
    contract.activate_contract()

    # Simulate conditions being met
    contract.conditions = ["verified:deliverables_completed", "verified:client_approval"]

    # Release payment
    try:
        if contract.release_payment("system"):
            print("Payment successfully released!")
    except Exception as e:
        print(f"Payment failed: {str(e)}")

    # Generate audit report
    print("\nAudit Report:")
    print(json.dumps(contract.audit_contract(), indent=2))

    # Display final state
    print("\nFinal Contract State:")
    print(json.dumps(contract.get_contract_state(), indent=2))