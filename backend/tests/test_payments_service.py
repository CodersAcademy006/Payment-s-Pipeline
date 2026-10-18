import pytest

from backend.services.payments import (
    IdempotencyConflictError,
    RateLimiter,
    RateLimitExceededError,
    find_idempotent_response,
    store_idempotent_response,
)

PAYLOAD = {"amount": 50000, "currency": "INR"}


def test_retry_with_same_key_returns_stored_response(app):
    assert find_idempotent_response("key-1", PAYLOAD) is None
    store_idempotent_response("key-1", "user-1", PAYLOAD, {"order_id": "order_1"})

    assert find_idempotent_response("key-1", PAYLOAD) == {"order_id": "order_1"}


def test_same_key_with_different_body_is_rejected(app):
    store_idempotent_response("key-1", "user-1", PAYLOAD, {"order_id": "order_1"})

    with pytest.raises(IdempotencyConflictError):
        find_idempotent_response("key-1", {**PAYLOAD, "amount": 99})


def test_rate_limiter_blocks_after_limit_per_user():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.check("user-1")
    limiter.check("user-1")

    with pytest.raises(RateLimitExceededError):
        limiter.check("user-1")
    limiter.check("user-2")
