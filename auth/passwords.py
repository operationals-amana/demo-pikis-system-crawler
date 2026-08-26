"""
bcrypt directly, not passlib.

passlib 1.7.4 is unmaintained and crashes on bcrypt >= 4.1 -- it reads
bcrypt.__about__.__version__, which was removed in 4.1. The usual workaround is
pinning bcrypt to 4.0.1 forever. Using bcrypt's own API is fifteen lines and has no
such ceiling.
"""

import bcrypt

# bcrypt silently truncates at 72 BYTES (not characters). Rejecting explicitly beats
# a password that works but ignores everything past the limit.
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"password exceeds {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:MAX_PASSWORD_BYTES], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed stored hash must read as "wrong password", not as a 500 that
        # tells an attacker the account exists.
        return False
