import hashlib
import hmac
import binascii
import os
import time
import base64
import json

SECRET_KEY = os.environ.get("JWT_SECRET", "portfolio-governance-secret-key-change-in-prod")

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return binascii.hexlify(salt).decode('utf-8') + ":" + binascii.hexlify(pw_hash).decode('utf-8')

def verify_password(stored_password_hash: str, password: str) -> bool:
    try:
        salt_hex, hash_hex = stored_password_hash.split(":")
        salt = binascii.unhexlify(salt_hex)
        pw_hash = binascii.unhexlify(hash_hex)
        new_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return hmac.compare_digest(pw_hash, new_hash)
    except Exception:
        return False

def create_auth_token(payload: dict) -> str:
    token_data = {
        "payload": payload,
        "exp": time.time() + 86400
    }
    raw_bytes = json.dumps(token_data).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(raw_bytes).decode("utf-8").rstrip("=")
    sig = hmac.new(SECRET_KEY.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).digest()
    b64_sig = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    return f"{b64_payload}.{b64_sig}"

def verify_auth_token(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        b64_payload, b64_sig = parts
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).digest()
        expected_b64_sig = base64.urlsafe_b64encode(expected_sig).decode("utf-8").rstrip("=")
        if not hmac.compare_digest(b64_sig, expected_b64_sig):
            return None
        padding = 4 - (len(b64_payload) % 4)
        if padding < 4:
            b64_payload += "=" * padding
        raw_bytes = base64.urlsafe_b64decode(b64_payload.encode("utf-8"))
        token_data = json.loads(raw_bytes.decode("utf-8"))
        if time.time() > token_data.get("exp", 0):
            return None
        return token_data.get("payload")
    except Exception:
        return None
