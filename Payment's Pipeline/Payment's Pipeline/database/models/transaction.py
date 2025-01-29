"""
Payment Transaction Model - Core of Payment Processing System
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from decimal import Decimal

from sqlalchemy import (
    Column,
    String,
    Numeric,
    Enum as SQLEnum,
    DateTime,
    ForeignKey,
    Index,
    CheckConstraint,
    Text,
    JSON
)
from sqlalchemy.orm import declarative_base, validates
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class TransactionStatus(str, Enum):
    """Enum representing all possible transaction states"""
    PENDING = 'pending'
    COMPLETED = 'completed'
    FAILED = 'failed'
    REFUNDED = 'refunded'
    CANCELLED = 'cancelled'
    CHARGEBACK = 'chargeback'
    REQUIRES_ACTION = 'requires_action'

class TransactionType(str, Enum):
    """Enum representing transaction types"""
    AUTHORIZE = 'authorize'
    CAPTURE = 'capture'
    REFUND = 'refund'
    CHARGEBACK = 'chargeback'

class CurrencyCode(str, Enum):
    """ISO 4217 currency codes"""
    USD = 'USD'
    EUR = 'EUR'
    GBP = 'GBP'
    JPY = 'JPY'
    # Add other supported currencies

class PaymentMethodType(str, Enum):
    """Supported payment methods"""
    CREDIT_CARD = 'credit_card'
    BANK_TRANSFER = 'bank_transfer'
    DIGITAL_WALLET = 'digital_wallet'
    CRYPTO = 'crypto'

class TransactionError(Exception):
    """Base exception for transaction-related errors"""
    pass

class TransactionNotFound(TransactionError):
    """Raised when transaction is not found"""
    pass

class InvalidTransactionState(TransactionError):
    """Raised for invalid state transitions"""
    pass

class Transaction(Base):
    """
    Core transaction model representing financial operations in the payment pipeline.
    
    Features:
    - Idempotency support
    - Multi-currency support
    - Audit logging
    - Status tracking with state machine
    - Error handling with retry capabilities
    """
    __tablename__ = 'transactions'
    __table_args__ = (
        Index('idx_user_transactions', 'user_id', 'status'),
        Index('idx_transaction_created', 'created_at'),
        Index('idx_idempotency_key', 'idempotency_key', unique=True),
        CheckConstraint('amount >= 0', name='check_amount_positive'),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    amount = Column(Numeric(precision=20, scale=4), nullable=False)
    currency = Column(SQLEnum(CurrencyCode), nullable=False)
    status = Column(SQLEnum(TransactionStatus), default=TransactionStatus.PENDING, nullable=False)
    payment_method_type = Column(SQLEnum(PaymentMethodType), nullable=False)
    payment_method_details = Column(JSON, nullable=False)
    processor_reference = Column(String(255))  # External processor ID
    idempotency_key = Column(String(255), unique=True)
    transaction_type = Column(SQLEnum(TransactionType), nullable=False)
    related_transaction_id = Column(UUID(as_uuid=True), ForeignKey('transactions.id'))
    metadata = Column(JSON)  # Additional transaction context
    processor_response = Column(JSON)  # Raw processor response
    error_details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships would be defined here if using ORM relationships

    @validates('status')
    def validate_status_transition(self, key, new_status):
        """Validate state transitions using state machine rules"""
        valid_transitions = {
            TransactionStatus.PENDING: [
                TransactionStatus.COMPLETED,
                TransactionStatus.FAILED,
                TransactionStatus.CANCELLED,
                TransactionStatus.REQUIRES_ACTION
            ],
            TransactionStatus.REQUIRES_ACTION: [
                TransactionStatus.COMPLETED,
                TransactionStatus.FAILED,
                TransactionStatus.CANCELLED
            ],
            TransactionStatus.COMPLETED: [
                TransactionStatus.REFUNDED,
                TransactionStatus.CHARGEBACK
            ],
            TransactionStatus.FAILED: [TransactionStatus.PENDING],
            TransactionStatus.REFUNDED: [],
            TransactionStatus.CHARGEBACK: [],
            TransactionStatus.CANCELLED: []
        }

        if self.status == new_status:
            return new_status  # No state change

        if new_status not in valid_transitions.get(self.status, []):
            raise InvalidTransactionState(
                f"Invalid status transition from {self.status} to {new_status}"
            )

        return new_status

    @classmethod
    def create_transaction(
        cls,
        *,
        user_id: uuid.UUID,
        amount: Decimal,
        currency: CurrencyCode,
        payment_method_type: PaymentMethodType,
        payment_method_details: Dict[str, Any],
        transaction_type: TransactionType,
        idempotency_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Transaction':
        """Factory method for creating new transactions with validation"""
        if not idempotency_key:
            idempotency_key = str(uuid.uuid4())

        return cls(
            user_id=user_id,
            amount=amount,
            currency=currency,
            payment_method_type=payment_method_type,
            payment_method_details=payment_method_details,
            transaction_type=transaction_type,
            idempotency_key=idempotency_key,
            metadata=metadata or {},
            status=TransactionStatus.PENDING
        )

    def update_status(
        self,
        new_status: TransactionStatus,
        *,
        processor_response: Optional[Dict[str, Any]] = None,
        error_details: Optional[str] = None
    ) -> None:
        """Update transaction status with audit trail"""
        self.status = new_status
        self.processor_response = processor_response
        self.error_details = error_details

        if new_status in (TransactionStatus.COMPLETED, TransactionStatus.FAILED):
            self.processing_cleanup()

    def processing_cleanup(self) -> None:
        """Perform cleanup tasks after final processing"""
        # Example: Clear sensitive data from payment method details
        self.payment_method_details = {
            'last4': self.payment_method_details.get('last4'),
            'brand': self.payment_method_details.get('brand')
        }

    def request_refund(
        self,
        *,
        amount: Optional[Decimal] = None,
        reason: Optional[str] = None
    ) -> 'Transaction':
        """Create refund transaction linked to this transaction"""
        if self.status != TransactionStatus.COMPLETED:
            raise InvalidTransactionState("Only completed transactions can be refunded")

        refund_amount = amount or self.amount

        return Transaction(
            user_id=self.user_id,
            amount=refund_amount,
            currency=self.currency,
            payment_method_type=self.payment_method_type,
            payment_method_details=self.payment_method_details,
            transaction_type=TransactionType.REFUND,
            related_transaction_id=self.id,
            metadata={
                'refund_reason': reason,
                'original_transaction_id': str(self.id)
            }
        )

    def add_chargeback(self, reason: str) -> 'Transaction':
        """Create chargeback transaction"""
        return Transaction(
            user_id=self.user_id,
            amount=self.amount,
            currency=self.currency,
            payment_method_type=self.payment_method_type,
            payment_method_details=self.payment_method_details,
            transaction_type=TransactionType.CHARGEBACK,
            related_transaction_id=self.id,
            metadata={
                'chargeback_reason': reason,
                'original_transaction_id': str(self.id)
            }
        )

    def __repr__(self) -> str:
        return (
            f"<Transaction(id={self.id}, "
            f"amount={self.amount} {self.currency}, "
            f"status={self.status}, "
            f"type={self.transaction_type})>"
        )