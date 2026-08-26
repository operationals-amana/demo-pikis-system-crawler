"""Request dependencies: database session, current user, admin gate."""

from typing import Any, Iterator

from fastapi import Depends, Header
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.errors import forbidden, unauthorized
from auth.tokens import decode_access_token
from db.engine import SessionLocal


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise unauthorized()
    claims = decode_access_token(authorization.split(" ", 1)[1].strip())
    if not claims:
        raise unauthorized("Session expired")

    # Re-read the user rather than trusting the token's claims. This is what makes
    # deactivating an account take effect immediately instead of at token expiry.
    row = db.execute(
        sql(
            "SELECT id::text, email, full_name, role, is_active FROM users "
            "WHERE id = CAST(:i AS uuid)"
        ),
        {"i": claims.get("sub")},
    ).mappings().first()
    if not row or not row["is_active"]:
        raise unauthorized("Account is not active")
    return dict(row)


def require_admin(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user.get("role") != "admin":
        raise forbidden("Administrator access required")
    return user


def optional_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any] | None:
    """For endpoints that work signed-out but personalise when signed in."""
    if not authorization:
        return None
    try:
        return current_user(authorization, db)
    except Exception:  # noqa: BLE001 -- optional means optional
        return None
