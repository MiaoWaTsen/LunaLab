"""Accounts and sessions."""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from backend.deps import bearer_token, get_current_user_optional
from backend.schemas import AuthResponse, LoginRequest, RegisterRequest, UserProfile
from database.models import authenticate_user, create_session, create_user, delete_session

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """Register a new user commander."""
    try:
        user = create_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_session(user["id"])
    return {"token": token, "user": user}


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Authenticate commander and return session token."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_session(user["id"])
    return {"token": token, "user": user}


@router.get("/me", response_model=UserProfile)
def get_me(authorization: Optional[str] = Header(None)):
    """Retrieve logged in user profile."""
    user = get_current_user_optional(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


@router.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    """Invalidate current session token."""
    if authorization:
        delete_session(bearer_token(authorization))
    return {"detail": "Logged out successfully."}
