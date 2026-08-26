"""Authentication: register, login, refresh, logout, me."""

from typing import Any

from fastapi import APIRouter, Depends, Header
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import ALLOW_SELF_REGISTER
from app.deps import current_user, get_db
from app.errors import bad_request, forbidden, unauthorized
from app.schemas import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from auth.passwords import hash_password, verify_password
from auth.tokens import (
    create_access_token,
    issue_refresh_token,
    revoke_refresh_token,
    rotate_refresh_token,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _token_response(db: Session, user: dict[str, Any], user_agent: str | None) -> TokenResponse:
    access, expires_in = create_access_token(user)
    refresh = issue_refresh_token(db, user["id"], user_agent)
    db.execute(
        sql("UPDATE users SET last_login_at = now() WHERE id = CAST(:i AS uuid)"),
        {"i": user["id"]},
    )
    db.commit()
    return TokenResponse(
        access_token=access,
        expires_in=expires_in,
        refresh_token=refresh,
        user=UserOut(**{k: user[k] for k in ("id", "email", "full_name", "role")}),
    )


@router.post("/register", response_model=TokenResponse)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
    user_agent: str | None = Header(default=None),
) -> TokenResponse:
    existing = db.execute(sql("SELECT count(*) FROM users")).scalar() or 0
    # The very first account always becomes admin -- otherwise a fresh deployment has
    # no way to reach the admin dashboard. After that, registration honours the flag.
    if existing and not ALLOW_SELF_REGISTER:
        raise forbidden("Self-registration is disabled")

    taken = db.execute(
        sql("SELECT 1 FROM users WHERE lower(email) = lower(:e)"), {"e": payload.email}
    ).scalar()
    if taken:
        raise bad_request("That email is already registered")

    row = db.execute(
        sql(
            "INSERT INTO users (email, password_hash, full_name, role) "
            "VALUES (:e, :p, :n, :r) RETURNING id::text, email, full_name, role"
        ),
        {
            "e": payload.email,
            "p": hash_password(payload.password),
            "n": payload.full_name,
            "r": "admin" if not existing else "researcher",
        },
    ).mappings().first()
    db.commit()
    return _token_response(db, dict(row), user_agent)


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
    user_agent: str | None = Header(default=None),
) -> TokenResponse:
    row = db.execute(
        sql(
            "SELECT id::text, email, full_name, role, password_hash, is_active "
            "FROM users WHERE lower(email) = lower(:e)"
        ),
        {"e": payload.email},
    ).mappings().first()

    # One message for both "no such user" and "wrong password": distinguishing them
    # tells an attacker which emails are registered.
    if not row or not verify_password(payload.password, row["password_hash"]):
        raise unauthorized("Incorrect email or password")
    if not row["is_active"]:
        raise unauthorized("Account is not active")

    return _token_response(db, dict(row), user_agent)


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    payload: RefreshRequest,
    db: Session = Depends(get_db),
    user_agent: str | None = Header(default=None),
) -> TokenResponse:
    rotated = rotate_refresh_token(db, payload.refresh_token, user_agent)
    if not rotated:
        raise unauthorized("Refresh token is invalid or expired")
    user_id, new_refresh = rotated

    row = db.execute(
        sql("SELECT id::text, email, full_name, role, is_active FROM users WHERE id = CAST(:i AS uuid)"),
        {"i": user_id},
    ).mappings().first()
    if not row or not row["is_active"]:
        raise unauthorized("Account is not active")

    access, expires_in = create_access_token(dict(row))
    return TokenResponse(
        access_token=access,
        expires_in=expires_in,
        refresh_token=new_refresh,
        user=UserOut(**{k: row[k] for k in ("id", "email", "full_name", "role")}),
    )


@router.post("/logout", status_code=204)
def logout(payload: RefreshRequest, db: Session = Depends(get_db)) -> None:
    revoke_refresh_token(db, payload.refresh_token)


@router.get("/me", response_model=UserOut)
def me(user: dict[str, Any] = Depends(current_user)) -> UserOut:
    return UserOut(**{k: user[k] for k in ("id", "email", "full_name", "role")})
