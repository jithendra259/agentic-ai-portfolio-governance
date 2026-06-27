import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


def _get_clerk_issuer() -> str | None:
    issuer = os.getenv("CLERK_ISSUER") or os.getenv("CLERK_JWT_ISSUER")
    if issuer:
        return issuer.rstrip("/")
    return None


def _get_clerk_jwks_url() -> str | None:
    jwks_url = os.getenv("CLERK_JWKS_URL")
    if jwks_url:
        return jwks_url
    issuer = _get_clerk_issuer()
    if issuer:
        return f"{issuer}/.well-known/jwks.json"
    return None


@lru_cache(maxsize=1)
def _get_jwks_client(jwks_url: str):
    import jwt

    return jwt.PyJWKClient(jwks_url)


def _session_from_clerk_claims(claims: dict) -> dict | None:
    user_id = claims.get("sub")
    if not user_id:
        return None

    email = (
        claims.get("email")
        or claims.get("email_address")
        or claims.get("primary_email_address")
    )
    name = (
        claims.get("name")
        or claims.get("full_name")
        or claims.get("given_name")
        or email
        or "User"
    )

    return {
        "user": {
            "id": user_id,
            "name": name,
            "email": email,
            "image": claims.get("picture") or claims.get("image_url"),
            "plan": claims.get("plan") or "Advisory workspace",
        }
    }


def verify_clerk_token(token: str) -> dict | None:
    jwks_url = _get_clerk_jwks_url()
    if not jwks_url:
        return None

    try:
        import jwt

        signing_key = _get_jwks_client(jwks_url).get_signing_key_from_jwt(token)
        decode_kwargs = {
            "algorithms": ["RS256"],
            "options": {"verify_aud": False},
        }
        issuer = _get_clerk_issuer()
        if issuer:
            decode_kwargs["issuer"] = issuer
        claims = jwt.decode(token, signing_key.key, **decode_kwargs)
        return _session_from_clerk_claims(claims)
    except Exception as exc:
        logger.debug("Clerk token verification failed: %s", exc)
        return None
