from sqlalchemy import JSON, Column, DateTime, String

from backend import db, utcnow


class IdempotencyKey(db.Model):
    """Stored response for a client-supplied Idempotency-Key, so retries never charge twice."""

    __tablename__ = "idempotency_keys"

    key = Column(String(64), primary_key=True)
    user_id = Column(String(36), nullable=False)
    request_hash = Column(String(64), nullable=False)
    response = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
