"""
Shared FastAPI dependencies: who is calling.
"""

from typing import Optional

from fastapi import Header, HTTPException

from database.models import get_user_by_session


def bearer_token(authorization: str) -> str:
    return authorization.replace("Bearer ", "").strip()


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Extract user from Bearer token if provided."""
    if not authorization:
        return None
    return get_user_by_session(bearer_token(authorization))


def require_user(authorization: Optional[str] = Header(None)) -> dict:
    """Dependency: the signed-in user, or 401."""
    user = get_current_user_optional(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user
