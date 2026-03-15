"""Authentication routes — register, login, current user, update profile."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token, verify_token
from app.models.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
    UserUpdate,
    VALID_TRADE_MODES,
    VALID_USER_MODES,
    VALID_RISK_TOLERANCES,
)
from app.services.tax_calculator import VALID_PROVINCES

VALID_TAX_TREATMENTS = {"auto", "capital_gains", "business_income"}
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ------------------------------------------------------------------
# Dependency: get current user from JWT token
# ------------------------------------------------------------------

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    username: str | None = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
        )
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Create a new user account."""
    # Check for duplicate username
    result = await db.execute(select(User).where(User.username == body.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )
    user = User(
        username=body.username,
        hashed_password=pwd_context.hash(body.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate and return a JWT token."""
    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()
    if user is None or not pwd_context.verify(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(data={"sub": user.username})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return user


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UserUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user's profile/preferences."""
    updates = body.model_dump(exclude_unset=True)

    # Validate enum-like fields
    if "trade_mode" in updates:
        if updates["trade_mode"] not in VALID_TRADE_MODES:
            raise HTTPException(400, f"Invalid trade_mode. Must be one of {VALID_TRADE_MODES}")
        if updates["trade_mode"] == "day_trade":
            raise HTTPException(400, "Day trade mode is coming soon — requires a paid data provider.")
    if "user_mode" in updates:
        if updates["user_mode"] not in VALID_USER_MODES:
            raise HTTPException(400, f"Invalid user_mode. Must be one of {VALID_USER_MODES}")
    if "risk_tolerance" in updates:
        if updates["risk_tolerance"] not in VALID_RISK_TOLERANCES:
            raise HTTPException(400, f"Invalid risk_tolerance. Must be one of {VALID_RISK_TOLERANCES}")

    # Validate tax fields
    if "tax_province" in updates:
        prov = updates["tax_province"]
        if prov is not None and prov not in VALID_PROVINCES:
            raise HTTPException(
                400,
                f"Invalid tax_province '{prov}'. Must be one of: {sorted(VALID_PROVINCES)}",
            )
    if "tax_annual_income" in updates:
        income = updates["tax_annual_income"]
        if income is not None and income < 0:
            raise HTTPException(400, "tax_annual_income must be >= 0")
    if "tax_treatment" in updates:
        treat = updates["tax_treatment"]
        if treat not in VALID_TAX_TREATMENTS:
            raise HTTPException(
                400,
                f"Invalid tax_treatment '{treat}'. Must be one of: {sorted(VALID_TAX_TREATMENTS)}",
            )

    for field, value in updates.items():
        setattr(user, field, value)

    await db.commit()
    await db.refresh(user)
    return user
