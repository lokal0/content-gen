from fastapi import Depends, HTTPException, Request

from app.core.config import settings


EXEMPT_PATHS = {"/health", "/docs", "/openapi.json"}


async def verify_bearer_token(request: Request):
    if request.url.path in EXEMPT_PATHS:
        return
    if not settings.api_bearer_token:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth[7:]
    if token != settings.api_bearer_token:
        raise HTTPException(status_code=401, detail="Invalid Bearer token")
