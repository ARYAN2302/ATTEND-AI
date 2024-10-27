from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import jwt, JWTError
from database import get_user_by_email
import logging

SECRET_KEY = "your-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

logger = logging.getLogger(__name__)

def get_password_hash(password: str) -> str:
    """Generate a bcrypt hash of the password"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        return bcrypt.checkpw(
            plain_password.encode(),
            hashed_password.encode()
        )
    except Exception:
        return False

def authenticate_user(email: str, password: str):
    """Authenticate a user and return user data if successful"""
    try:
        user = get_user_by_email(email)
        if not user:
            logger.warning(f"No user found for email: {email}")
            return False
        if not verify_password(password, user["password"]):
            logger.warning(f"Invalid password for user: {email}")
            return False
        logger.info(f"User authenticated successfully: {email}")
        return {
            "id": user["id"],
            "email": user["email"],
            "role": user["role"],
            "department": user.get("department", "Unknown")
        }
    except Exception as e:
        logger.error(f"Error in authenticate_user: {str(e)}", exc_info=True)
        raise

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a new JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    logger.info(f"Creating token with payload: {to_encode}")  # Add this line
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str):
    """Decode and verify a JWT token"""
    try:
        logger.info(f"Attempting to decode token: {token[:10]}...")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info("Token decoded successfully")
        logger.info(f"Decoded payload: {payload}")
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return None
    except JWTError as e:
        logger.error(f"Invalid token: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {str(e)}", exc_info=True)
        return None
