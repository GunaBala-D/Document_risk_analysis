"""API-key authentication with role-based permissions.

V1 deliberately avoids an identity provider: each user gets a random API key; only its
SHA-256 hash is stored (keys are 32+ random bytes, so a slow password hash is unnecessary).
A production deployment would put SSO/OIDC in front and map IdP groups to these roles.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import User
from database.session import get_session

ANALYST = {"documents:read", "documents:write", "feedback:write", "analytics:read", "models:read",
           "monitoring:read"}
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "compliance_analyst": ANALYST,
    "finance_analyst": ANALYST,
    "procurement_analyst": ANALYST,
    "risk_manager": ANALYST | {"feedback:validate", "monitoring:run"},
    "admin": ANALYST | {"feedback:validate", "monitoring:run", "admin"},
}


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def generate_key() -> str:
    return "dre_" + secrets.token_urlsafe(32)


def get_current_user(x_api_key: str | None = Header(default=None), session: Session = Depends(get_session)) -> User:
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header",
                            headers={"WWW-Authenticate": "APIKey"})
    user = session.scalar(select(User).where(User.api_key_hash == hash_key(x_api_key)))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key", headers={"WWW-Authenticate": "APIKey"})
    return user


def require(permission: str):
    def checker(user: User = Depends(get_current_user)) -> User:
        if permission not in ROLE_PERMISSIONS.get(user.role, set()):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Role '{user.role}' lacks permission '{permission}'")
        return user

    return checker
