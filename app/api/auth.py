import secrets
import time
from collections import defaultdict
from threading import Lock

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

_bearer = HTTPBearer(auto_error=True)

# ── Простой in-memory rate limiter ────────────────────────────────────────────
# Максимум 60 запросов в минуту с одного IP
_RATE_LIMIT = 60
_RATE_WINDOW = 60  # секунд

_hits: dict[str, list[float]] = defaultdict(list)
_lock = Lock()


def _check_rate_limit(ip: str) -> None:
    now = time.monotonic()
    with _lock:
        window_start = now - _RATE_WINDOW
        _hits[ip] = [t for t in _hits[ip] if t > window_start]
        if len(_hits[ip]) >= _RATE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Слишком много запросов. Подождите минуту.",
            )
        _hits[ip].append(now)


def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> None:
    """
    FastAPI dependency: rate limiting + проверка Bearer-токена.

    TG-бот передаёт:
        headers={"Authorization": f"Bearer {API_KEY}"}
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    expected = get_settings().api_key.get_secret_value()
    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный API ключ",
            headers={"WWW-Authenticate": "Bearer"},
        )
