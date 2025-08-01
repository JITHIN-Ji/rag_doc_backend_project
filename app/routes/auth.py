import os
import time
from fastapi import APIRouter, Request, HTTPException
from starlette.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from jose import jwt


from app.core.config import settings
from . import email_auth

router = APIRouter(prefix="/auth", tags=["Authentication"])

oauth = OAuth()
oauth.register(
    name="google",
    
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile",
        "response_type": "code",
    },
)


router.include_router(email_auth.router)


@router.get("/login", tags=["Google Authentication"])
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback", tags=["Google Authentication"])
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not authorize access token: {e}")

    id_token = token.get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="Missing ID token in Google response")

    try:
        user = jwt.decode(
            id_token,
            key=None,
            algorithms=["RS256"],
            options={
                "verify_signature": False,
                "verify_at_hash": False
            },
            
            audience=settings.GOOGLE_CLIENT_ID
        )
    except jwt.JWTClaimsError as e:
        raise HTTPException(status_code=400, detail=f"Invalid token claims: {e}")

    if not user:
        raise HTTPException(status_code=400, detail="Google login failed")

    email = user["email"]

    
    app_token = jwt.encode(
        {"sub": email, "email": email, "exp": int(time.time()) + 3600},
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM, 
    )

    
    return RedirectResponse(url=f"http://localhost:3000/?token={app_token}")