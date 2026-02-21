from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from app.core.config import settings
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User
from app.services.api_key import APIKeyService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login/access-token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt 

def get_current_user(
    db: Session = Depends(get_db),
) -> User:
    user = db.query(User).first()
    if user is None:
        raise HTTPException(status_code=503, detail="No users in database")
    return user


def get_api_key_user(
    db: Session = Depends(get_db),
) -> User:
    user = db.query(User).first()
    if user is None:
        raise HTTPException(status_code=503, detail="No users in database")
    return user