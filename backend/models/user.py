"""
User Model for Payment Pipeline System
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import Column, String, DateTime, Boolean, Enum as SQLEnum, JSON, Index, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from passlib.context import CryptContext
import pyotp

Base = declarative_base()

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRole(str, Enum):
    CUSTOMER = "CUSTOMER"
    MERCHANT = "MERCHANT"
    ADMIN = "ADMIN"
    SUPPORT = "SUPPORT"

class AccountStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DELETED = "DELETED"

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("idx_email", "email"),
        Index("idx_account_status", "account_status"),
        CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$'", name="valid_email"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.CUSTOMER)
    is_kyc_verified = Column(Boolean, default=False, nullable=False)
    kyc_verified_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    two_factor_enabled = Column(Boolean, default=False, nullable=False)
    two_factor_secret = Column(String(255), nullable=True)
    account_status = Column(SQLEnum(AccountStatus), default=AccountStatus.ACTIVE, nullable=False)
    metadata = Column(JSON, nullable=True)  # For additional custom fields

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'password' in kwargs:
            self.set_password(kwargs['password'])

    def set_password(self, password: str):
        """Securely hash and store password"""
        if len(password) < 12:
            raise ValueError("Password must be at least 12 characters")
        self.password_hash = pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        return pwd_context.verify(password, self.password_hash)

    def generate_2fa_secret(self) -> str:
        """Generate a new 2FA secret"""
        self.two_factor_secret = pyotp.random_base32()
        return self.two_factor_secret

    def verify_2fa(self, token: str) -> bool:
        """Verify 2FA token"""
        if not self.two_factor_secret:
            return False
        totp = pyotp.TOTP(self.two_factor_secret)
        return totp.verify(token)

    def rotate_2fa_secret(self) -> str:
        """Generate new 2FA secret and disable 2FA until verified"""
        self.two_factor_enabled = False
        return self.generate_2fa_secret()

    def update_email(self, new_email: str):
        """Secure email update procedure"""
        self.email = new_email.lower().strip()
        self.is_kyc_verified = False  # Require re-verification after email change

    def mark_kyc_verified(self):
        """Mark KYC verification with timestamp"""
        self.is_kyc_verified = True
        self.kyc_verified_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Safe serialization for public use"""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
            "account_status": self.account_status.value,
            "is_kyc_verified": self.is_kyc_verified,
            "two_factor_enabled": self.two_factor_enabled
        }

    def __repr__(self):
        return f"<User {self.username} ({self.email})>"

# Example Usage:
# user = User(username='payment_user', email='user@payment.com')
# user.set_password('StrongPassword123!')
# user.generate_2fa_secret()
# print(user.verify_2fa('123456'))  # Verify 2FA code