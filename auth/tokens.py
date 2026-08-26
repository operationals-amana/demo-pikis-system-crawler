"""
JWT access tokens and hashed, rotating refresh tokens.

Refresh tokens are stored as sha256 hashes, never in plaintext: a database leak must
not also be a session leak. Rotation on every refresh means a stolen refresh token is
usable at most once before the legitimate client's next refresh invalidates it.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import (
    ACCESS_TOKEN_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET,
    REFRESH_TOKEN_DAYS,
)


def _require_secret() -> str:
    if not JWT_SECRET:
        # Refusing to start beats minting tokens signed with an empty string.
        raise RuntimeError("JWT_SECRET is not set; refusing to issue tokens")
    return JWT_SECRET


def create_access_token(user: dict[str, Any]) -> tuple[str, int]:
    """Return (token, expires_in_seconds). Claims stay thin -- see decode_access_token."""
    expires_in = ACCESS_TOKEN_MINUTES * 60
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "role": user["role"],
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
    }
    return jwt.encode(payload, _require_secret(), algorithm=JWT_ALGORITHM), expires_in


def decode_access_token(token: str) -> dict[str, Any] | None:
    """
    Verify and decode. Returns None on any failure -- callers turn that into a 401.

    The claims are deliberately thin: role is re-read from the database on every
    request that cares, so deactivating a user takes effect immediately rather than
    at token expiry.
    """
    try:
        return jwt.decode(token, _require_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_refresh_token(db: Session, user_id: str, user_agent: str | None = None) -> str:
    raw = secrets.token_urlsafe(48)
    db.execute(
        sql(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at, user_agent) "
            "VALUES (CAST(:u AS uuid), :h, now() + make_interval(days => :d), :ua)"
        ),
        {"u": user_id, "h": _hash(raw), "d": REFRESH_TOKEN_DAYS, "ua": (user_agent or "")[:300]},
    )
    db.commit()
    return raw


def rotate_refresh_token(db: Session, raw: str, user_agent: str | None = None) -> tuple[str, str] | None:
    """
    Consume one refresh token and issue its replacement.

    Returns (user_id, new_raw_token) or None when the token is unknown, expired or
    already revoked.
    """
    row = db.execute(
        sql(
            "SELECT id::text, user_id::text FROM refresh_tokens "
            "WHERE token_hash = :h AND revoked_at IS NULL AND expires_at > now()"
        ),
        {"h": _hash(raw)},
    ).first()
    if not row:
        return None
    token_id, user_id = row
    new_raw = secrets.token_urlsafe(48)
    db.execute(
        sql(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at, user_agent) "
            "VALUES (CAST(:u AS uuid), :h, now() + make_interval(days => :d), :ua)"
        ),
        {"u": user_id, "h": _hash(new_raw), "d": REFRESH_TOKEN_DAYS, "ua": (user_agent or "")[:300]},
    )
    db.execute(
        sql("UPDATE refresh_tokens SET revoked_at = now() WHERE id = CAST(:i AS uuid)"),
        {"i": token_id},
    )
    db.commit()
    return user_id, new_raw


def revoke_refresh_token(db: Session, raw: str) -> None:
    db.execute(
        sql("UPDATE refresh_tokens SET revoked_at = now() WHERE token_hash = :h AND revoked_at IS NULL"),
        {"h": _hash(raw)},
    )
    db.commit()
