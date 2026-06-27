import logging
import os
import urllib.parse
import httpx
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from bson import ObjectId

from src.orchestrator.chatbot_orchestrator import memory_manager
from src.utils.clerk_auth import verify_clerk_token
from src.utils.crypto_utils import (
    hash_password,
    verify_password,
    create_auth_token,
    verify_auth_token
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

class LoginRequest(BaseModel):
    email: str
    password: str

class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str
    plan: str = "Standard Workspace"

@router.post("/login")
def auth_login(request: LoginRequest):
    if not memory_manager or not memory_manager.is_available:
        raise HTTPException(status_code=503, detail="Database connection is unavailable")
        
    email = request.email.strip().lower()
    password = request.password
    
    try:
        users_col = memory_manager._db["users"]
        user_doc = users_col.find_one({"email": email})
        
        if not user_doc or not verify_password(user_doc.get("password_hash", ""), password):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
            
        payload = {
            "user": {
                "id": str(user_doc.get("_id")),
                "name": user_doc.get("name", "User"),
                "email": user_doc.get("email", email),
                "image": user_doc.get("image"),
                "plan": user_doc.get("plan", "Standard Workspace")
            }
        }
        token = create_auth_token(payload)
        return {"token": token, "session": payload}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Login failed due to database error")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

@router.post("/signup")
def auth_signup(request: SignUpRequest):
    if not memory_manager or not memory_manager.is_available:
        raise HTTPException(status_code=503, detail="Database connection is unavailable")
        
    email = request.email.strip().lower()
    password = request.password
    name = request.name.strip()
    plan = request.plan.strip()
    
    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="Missing required signup fields")
        
    try:
        users_col = memory_manager._db["users"]
        existing = users_col.find_one({"email": email})
        if existing:
            raise HTTPException(status_code=400, detail="An account with this email already exists.")
            
        pw_hash = hash_password(password)
        new_user = {
            "email": email,
            "name": name,
            "password_hash": pw_hash,
            "image": None,
            "plan": plan,
            "created_at": datetime.now(timezone.utc)
        }
        res = users_col.insert_one(new_user)
        
        payload = {
            "user": {
                "id": str(res.inserted_id),
                "name": name,
                "email": email,
                "image": None,
                "plan": plan
            }
        }
        token = create_auth_token(payload)
        return {"token": token, "session": payload}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Sign up failed due to database error")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

@router.get("/session")
def auth_session(request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"session": None}
    token = auth_header.split(" ")[1]
    payload = verify_clerk_token(token) or verify_auth_token(token)
    if not payload:
        return {"session": None}
    return {"session": payload}

@router.post("/logout")
def auth_logout():
    return {"status": "ok"}

@router.get("/oauth/login/{provider}")
def oauth_login(provider: str):
    provider = provider.strip().lower()
    
    if provider == "google":
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        redirect_uri = os.getenv("OAUTH_REDIRECT_URI_GOOGLE", "http://localhost:8000/api/auth/oauth/callback/google")
        if not client_id:
            logger.info("GOOGLE_CLIENT_ID not set. Redirecting to mock callback.")
            return RedirectResponse(f"{redirect_uri}?code=mock_google_code&state=mock_state")
            
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": "mock_state"
        }
        url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
        return RedirectResponse(url)
        
    elif provider == "github":
        client_id = os.getenv("GITHUB_CLIENT_ID")
        redirect_uri = os.getenv("OAUTH_REDIRECT_URI_GITHUB", "http://localhost:8000/api/auth/oauth/callback/github")
        if not client_id:
            logger.info("GITHUB_CLIENT_ID not set. Redirecting to mock callback.")
            return RedirectResponse(f"{redirect_uri}?code=mock_github_code&state=mock_state")
            
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": "user:email read:user",
            "state": "mock_state"
        }
        url = "https://github.com/login/oauth/authorize?" + urllib.parse.urlencode(params)
        return RedirectResponse(url)
        
    raise HTTPException(status_code=400, detail="Invalid provider")

@router.get("/oauth/callback/{provider}")
async def oauth_callback(provider: str, code: str, state: str = None):
    provider = provider.strip().lower()
    email = None
    name = None
    image = None
    
    is_mock = code.startswith("mock_")
    
    if is_mock:
        if provider == "google":
            email = "google-demo@governance.ai"
            name = "Google Demo User"
            image = "https://lh3.googleusercontent.com/a/default-user=s96-c"
        else:
            email = "github-demo@governance.ai"
            name = "GitHub Demo User"
            image = "https://avatars.githubusercontent.com/u/9919"
    else:
        try:
            async with httpx.AsyncClient() as client:
                if provider == "google":
                    client_id = os.getenv("GOOGLE_CLIENT_ID")
                    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
                    redirect_uri = os.getenv("OAUTH_REDIRECT_URI_GOOGLE", "http://localhost:8000/api/auth/oauth/callback/google")
                    
                    token_url = "https://oauth2.googleapis.com/token"
                    data = {
                        "code": code,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "redirect_uri": redirect_uri,
                        "grant_type": "authorization_code"
                    }
                    res = await client.post(token_url, data=data)
                    res.raise_for_status()
                    access_token = res.json().get("access_token")
                    
                    user_info_res = await client.get(
                        "https://www.googleapis.com/oauth2/v2/userinfo",
                        headers={"Authorization": f"Bearer {access_token}"}
                    )
                    user_info_res.raise_for_status()
                    user_data = user_info_res.json()
                    email = user_data.get("email")
                    name = user_data.get("name")
                    image = user_data.get("picture")

                elif provider == "github":
                    client_id = os.getenv("GITHUB_CLIENT_ID")
                    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
                    redirect_uri = os.getenv("OAUTH_REDIRECT_URI_GITHUB", "http://localhost:8000/api/auth/oauth/callback/github")
                    
                    token_url = "https://github.com/login/oauth/access_token"
                    headers = {"Accept": "application/json"}
                    data = {
                        "code": code,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "redirect_uri": redirect_uri
                    }
                    res = await client.post(token_url, data=data, headers=headers)
                    res.raise_for_status()
                    access_token = res.json().get("access_token")
                    
                    user_res = await client.get(
                        "https://api.github.com/user",
                        headers={"Authorization": f"Bearer {access_token}"}
                    )
                    user_res.raise_for_status()
                    user_data = user_res.json()
                    name = user_data.get("name") or user_data.get("login")
                    image = user_data.get("avatar_url")
                    
                    email_res = await client.get(
                        "https://api.github.com/user/emails",
                        headers={"Authorization": f"Bearer {access_token}"}
                    )
                    email_res.raise_for_status()
                    email = next((e["email"] for e in email_res.json() if e["primary"]), None)
                else:
                    raise HTTPException(status_code=400, detail="Invalid provider")
        except Exception as exc:
            logger.exception("OAuth code exchange failed")
            raise HTTPException(status_code=500, detail=f"OAuth exchange error: {exc}")

    if not email:
        raise HTTPException(status_code=400, detail="Could not retrieve email from provider")

    if not memory_manager or not memory_manager.is_available:
        raise HTTPException(status_code=503, detail="Database connection is unavailable")
        
    try:
        users_col = memory_manager._db["users"]
        user_doc = users_col.find_one({"email": email})
        
        plan = "Standard Workspace"
        if user_doc:
            user_id = str(user_doc["_id"])
            plan = user_doc.get("plan", "Standard Workspace")
            users_col.update_one(
                {"_id": user_doc["_id"]},
                {"$set": {"name": name, "image": image}}
            )
        else:
            new_user = {
                "email": email,
                "name": name,
                "password_hash": None,
                "image": image,
                "plan": plan,
                "created_at": datetime.now(timezone.utc)
            }
            res_db = users_col.insert_one(new_user)
            user_id = str(res_db.inserted_id)
            
        payload = {
            "user": {
                "id": user_id,
                "name": name,
                "email": email,
                "image": image,
                "plan": plan
            }
        }
        token = create_auth_token(payload)
        
        frontend_base_url = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173").rstrip("/")
        return RedirectResponse(f"{frontend_base_url}/?token={token}")
    except Exception as exc:
        logger.exception("Failed to complete OAuth user database operations")
        raise HTTPException(status_code=500, detail=f"Database error during OAuth callback: {exc}")
