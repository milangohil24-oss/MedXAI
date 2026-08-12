import datetime
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .database import get_db
from . import models


SECRET_KEY = "MEDXAI_SUPER_SECRET_KEY_EXPLAINABLE_MRI"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/login"
)


def verify_password(
    plain_password: str,
    hashed_password: str
):
    return pwd_context.verify(
        plain_password,
        hashed_password
    )


def get_password_hash(password: str):
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: Optional[datetime.timedelta] = None
):
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
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


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
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

    user = (
        db.query(models.User)
        .filter(models.User.id == user_id)
        .first()
    )

    if user is None:
        raise credentials_exception

    return user