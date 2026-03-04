"""JWT authentication utilities."""
import jwt
import time

SECRET = 'change-in-production'
ALGORITHM = 'HS256'

def create_token(user_id: str, expires_in: int = 3600) -> str:
    payload = {'sub': user_id, 'exp': time.time() + expires_in}
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)

def verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
