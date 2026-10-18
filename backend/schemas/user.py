"""
User Schema Module for Payment Pipeline System
Defines Pydantic schemas, security utilities, and validation logic
"""

import re
import secrets
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

# Constants
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-.]{3,50}$")
PASSWORD_MIN_LENGTH = 12
ALLOWED_EMAIL_DOMAINS = {"example.com", "yourcompany.com"}  # Add production domains


class UserBase(BaseModel):
    """Base user schema with common attributes"""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        examples=["secure_user123"],
        description="Unique username containing letters, numbers, and allowed symbols (_-.)",
    )
    email: EmailStr = Field(
        ..., examples=["user@yourcompany.com"], description="Valid corporate email address"
    )

    @field_validator("username")
    @classmethod
    def validate_username_format(cls, value):
        """Validate username format using regex"""
        if not USERNAME_REGEX.match(value):
            raise ValueError(
                "Username must be 3-50 characters containing only: a-z, A-Z, 0-9, '_', '-', '.'"
            )
        return value.lower()

    @field_validator("email")
    @classmethod
    def validate_email_domain(cls, value):
        """Validate corporate email domain (remove in production if not needed)"""
        if not value.endswith(tuple(ALLOWED_EMAIL_DOMAINS)):
            raise ValueError("Invalid email domain. Use corporate email.")
        return value.lower()


class UserCreate(UserBase):
    """User creation schema with password validation"""

    password: str = Field(
        ...,
        min_length=PASSWORD_MIN_LENGTH,
        examples=["Strong!PassPhrase1234"],
        description=(
            f"Minimum {PASSWORD_MIN_LENGTH} chars with uppercase, lowercase, number, "
            "and special character"
        ),
    )
    password_confirmation: str = Field(..., examples=["Strong!PassPhrase1234"])

    @model_validator(mode="after")
    def validate_passwords_match(self):
        """Ensure password and confirmation match"""
        if self.password != self.password_confirmation:
            raise ValueError("Passwords do not match")
        return self

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, value):
        """Enforce strong password requirements"""
        errors = []

        if len(value) < PASSWORD_MIN_LENGTH:
            errors.append(f"Minimum {PASSWORD_MIN_LENGTH} characters")
        if not re.search(r"[A-Z]", value):
            errors.append("At least one uppercase letter")
        if not re.search(r"[a-z]", value):
            errors.append("At least one lowercase letter")
        if not re.search(r"\d", value):
            errors.append("At least one digit")
        if not re.search(r"[!@#$%^&*()\-_=+{};:,<.>?]", value):
            errors.append("At least one special character")
        if re.search(r"(.)\1{3,}", value):
            errors.append("No repeated characters (4+ times)")

        if errors:
            raise ValueError("Password requirements failed: " + ", ".join(errors))
        return value


class UserUpdate(BaseModel):
    """User update schema with optional fields"""

    username: str | None = Field(None, min_length=3, max_length=50, examples=["new_secure_user123"])
    email: EmailStr | None = Field(None, examples=["new.user@yourcompany.com"])
    password: str | None = Field(
        None, min_length=PASSWORD_MIN_LENGTH, examples=["NewStrong!PassPhrase1234"]
    )


class UserResponse(UserBase):
    """User response schema (public-facing)"""

    id: str
    is_active: bool
    is_verified: bool
    roles: list[str]
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "secure_user123",
                "email": "user@yourcompany.com",
                "is_active": True,
                "is_verified": True,
                "roles": ["user"],
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z",
                "last_login": "2023-01-01T00:00:00Z",
            }
        },
    )


class UserInDB(UserResponse):
    """Internal user schema with sensitive data"""

    hashed_password: str
    verification_token: str | None
    password_reset_token: str | None
    failed_login_attempts: int = 0
    totp_secret: str | None
    mfa_enabled: bool = False

    model_config = ConfigDict(from_attributes=True)


class UserLogin(BaseModel):
    """User login schema"""

    username_or_email: str = Field(..., examples=["secure_user123 or user@yourcompany.com"])
    password: str = Field(..., examples=["Strong!PassPhrase1234"])
    mfa_code: str | None = Field(None, examples=["123456"])


class PasswordResetRequest(BaseModel):
    """Password reset request schema"""

    email: EmailStr = Field(..., examples=["user@yourcompany.com"])


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema"""

    reset_token: str = Field(...)
    new_password: str = Field(
        ..., min_length=PASSWORD_MIN_LENGTH, examples=["NewStrong!PassPhrase1234"]
    )


def generate_password_reset_token() -> str:
    """Generate secure password reset token"""
    return secrets.token_urlsafe(64)


def validate_password_strength(password: str) -> None:
    """Reusable password validation function"""
    UserCreate.validate_password_complexity(password)
