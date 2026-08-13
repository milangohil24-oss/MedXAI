import datetime
import os
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from .database import users_collection


# ============================================================
# SECURITY CONFIGURATION
# ============================================================

SECRET_KEY = os.getenv(
    "JWT_SECRET",
    "MEDXAI_LOCAL_DEVELOPMENT_SECRET_CHANGE_ME"
)

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_DAYS = 7


# ============================================================
# OAUTH2
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/login"
)


# ============================================================
# PASSWORD HASHING
# ============================================================

def get_password_hash(password: str) -> str:
    password_bytes = password.encode("utf-8")

    if len(password_bytes) > 72:
        raise ValueError(
            "Password cannot exceed 72 bytes"
        )

    hashed_password = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt()
    )

    return hashed_password.decode("utf-8")


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:
    password_bytes = plain_password.encode("utf-8")

    if len(password_bytes) > 72:
        return False

    try:
        return bcrypt.checkpw(
            password_bytes,
            hashed_password.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


# ============================================================
# CREATE JWT ACCESS TOKEN
# ============================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[datetime.timedelta] = None
):
    to_encode = data.copy()

    if expires_delta:
        expire = (
            datetime.datetime.utcnow()
            + expires_delta
        )
    else:
        expire = (
            datetime.datetime.utcnow()
            + datetime.timedelta(
                days=ACCESS_TOKEN_EXPIRE_DAYS
            )
        )

    to_encode.update({
        "exp": expire
    })

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return encoded_jwt


# ============================================================
# GET CURRENT USER (MONGODB ATLAS)
# ============================================================

def get_current_user(
    token: str = Depends(oauth2_scheme)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={
            "WWW-Authenticate": "Bearer"
        }
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = users_collection.find_one({"_id": user_id})

    if user is None:
        raise credentials_exception

    user["id"] = user["_id"]
    return user