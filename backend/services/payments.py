"""Idempotency and rate limiting for payment requests."""

import hashlib
import json
from collections import defaultdict, deque
from datetime import timedelta

from backend import db, utcnow
from backend.models.idempotency_key import IdempotencyKey


class PaymentError(Exception):
    """Base payment error"""


class RateLimitExceededError(PaymentError):
    """Rate limit exceeded"""


class IdempotencyConflictError(PaymentError):
    """Idempotency key reused with a different request body"""


def _request_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def find_idempotent_response(key: str, payload: dict) -> dict | None:
    """Return the stored response for a retried request, or None if the key is new."""
    stored = db.session.get(IdempotencyKey, key)
    if stored is None:
        return None
    if stored.request_hash != _request_hash(payload):
        raise IdempotencyConflictError("Idempotency key reused with a different request")
    return stored.response


def store_idempotent_response(
    key: str, user_id: str, payload: dict, response: dict, ttl_days: int = 7
) -> None:
    db.session.add(
        IdempotencyKey(
            key=key,
            user_id=user_id,
            request_hash=_request_hash(payload),
            response=response,
            expires_at=utcnow() + timedelta(days=ttl_days),
        )
    )
    db.session.commit()


class RateLimiter:
    """Sliding-window limit per user.

    ponytail: in-process memory, so limits are per worker; move to Redis when running more than one.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.hits: defaultdict[str, deque] = defaultdict(deque)

    def check(self, user_id: str) -> None:
        now = utcnow()
        hits = self.hits[user_id]
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.max_requests:
            raise RateLimitExceededError("Too many requests")
        hits.append(now)
