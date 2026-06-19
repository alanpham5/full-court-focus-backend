import os
import random
import time
import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from firebase_config import db, auth
from firebase_admin.auth import UserNotFoundError

router = APIRouter(prefix="/api/auth", tags=["auth"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
GOOGLE_CLIENT_ID = os.getenv("REACT_APP_GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
HAS_SERVICE_ACCOUNT = bool(os.getenv("FIREBASE_SERVICE_ACCOUNT_B64"))

class SendVerificationRequest(BaseModel):
    email: EmailStr
    password: str
    displayName: str = None

class VerifyRegistrationRequest(BaseModel):
    email: EmailStr
    code: str

class SendResetRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str
    password: str

@router.post("/send-verification")
def send_verification(body: SendVerificationRequest):
    email_clean = body.email.strip().lower()
    try:
        auth.get_user_by_email(email_clean)
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    except UserNotFoundError:
        pass

    code = f"{random.randint(100000, 999999)}"
    expires_at = time.time() + 900

    db.collection("signup_verifications").document(email_clean).set({
        "password": body.password,
        "displayName": body.displayName,
        "code": code,
        "expires_at": expires_at
    })

    simulated_code = None
    if HAS_SERVICE_ACCOUNT:
        db.collection("mail").add({
            "to": email_clean,
            "message": {
                "subject": "Verify your Full Court Focus email",
                "html": f"<p>Welcome to Full Court Focus! Use this code to verify your email address:</p><p style='font-size:28px;font-weight:bold;letter-spacing:6px;'>{code}</p><p>This code expires in 15 minutes. If you didn't sign up, you can ignore this email.</p>"
            }
        })
    else:
        simulated_code = code

    return {"status": "code_sent", "email": email_clean, "simulatedCode": simulated_code}

@router.post("/verify-registration")
def verify_registration(body: VerifyRegistrationRequest):
    email_clean = body.email.strip().lower()
    doc_ref = db.collection("signup_verifications").document(email_clean)
    doc_snap = doc_ref.get()

    if not doc_snap.exists:
        raise HTTPException(status_code=400, detail="Invalid verification request details.")

    data = doc_snap.to_dict()
    if time.time() > data.get("expires_at", 0):
        doc_ref.delete()
        raise HTTPException(status_code=400, detail="Verification code expired. Please sign up again.")

    if data.get("code") != body.code.strip():
        raise HTTPException(status_code=400, detail="Incorrect verification code.")

    try:
        user = auth.create_user(
            email=email_clean,
            password=data.get("password"),
            display_name=data.get("displayName"),
            email_verified=True
        )
        custom_token = auth.create_custom_token(user.uid).decode("utf-8")
        doc_ref.delete()
        return {"customToken": custom_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-reset-code")
def send_reset_code(body: SendResetRequest):
    email_clean = body.email.strip().lower()
    try:
        auth.get_user_by_email(email_clean)
    except UserNotFoundError:
        raise HTTPException(status_code=404, detail="No account found with that email address.")

    code = f"{random.randint(100000, 999999)}"
    expires_at = time.time() + 900

    db.collection("password_resets").document(email_clean).set({
        "code": code,
        "expires_at": expires_at
    })

    simulated_code = None
    if HAS_SERVICE_ACCOUNT:
        db.collection("mail").add({
            "to": email_clean,
            "message": {
                "subject": "Reset your Full Court Focus password",
                "html": f"<p>You requested a password reset for your Full Court Focus account.</p><p>Use this code to reset your password:</p><p style='font-size:28px;font-weight:bold;letter-spacing:6px;'>{code}</p><p>This code expires in 15 minutes. If you did not make this request, you can ignore this email.</p>"
            }
        })
    else:
        simulated_code = code

    return {"status": "code_sent", "email": email_clean, "simulatedCode": simulated_code}

@router.post("/reset-password")
def reset_password(body: ResetPasswordRequest):
    email_clean = body.email.strip().lower()
    doc_ref = db.collection("password_resets").document(email_clean)
    doc_snap = doc_ref.get()

    if not doc_snap.exists:
        raise HTTPException(status_code=400, detail="Invalid password reset request details.")

    data = doc_snap.to_dict()
    if time.time() > data.get("expires_at", 0):
        doc_ref.delete()
        raise HTTPException(status_code=400, detail="Password reset code expired. Please request a new one.")

    if data.get("code") != body.code.strip():
        raise HTTPException(status_code=400, detail="Incorrect verification code.")

    try:
        user = auth.get_user_by_email(email_clean)
        auth.update_user(user.uid, password=body.password)
        doc_ref.delete()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/google/callback")
def google_callback(code: str = None, state: str = None, error: str = None):
    if error:
        return RedirectResponse(f"{FRONTEND_URL}/auth/google/callback?error={requests.utils.quote(error)}")
    if not code:
        return RedirectResponse(f"{FRONTEND_URL}/auth/google/callback?error=missing_code")

    try:
        redirect_uri = f"{os.getenv('BACKEND_URL', 'http://localhost:8000')}/api/auth/google/callback"
        token_res = requests.post("https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        token_data = token_res.json()
        if not token_res.ok or "error" in token_data:
            raise HTTPException(status_code=400, detail=token_data.get("error_description", "Token exchange failed."))

        user_info_res = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={
            "Authorization": f"Bearer {token_data.get('access_token')}"
        })
        user_info = user_info_res.json()
        email = user_info.get("email")
        name = user_info.get("name")
        picture = user_info.get("picture")

        try:
            existing = auth.get_user_by_email(email)
            uid = existing.uid
            auth.update_user(uid, display_name=existing.display_name or name, photo_url=existing.photo_url or picture, email_verified=True)
        except UserNotFoundError:
            created = auth.create_user(email=email, display_name=name, photo_url=picture, email_verified=True)
            uid = created.uid

        custom_token = auth.create_custom_token(uid).decode("utf-8")
        return RedirectResponse(f"{FRONTEND_URL}/auth/google/callback?customToken={custom_token}&state={state or ''}")
    except Exception as e:
        return RedirectResponse(f"{FRONTEND_URL}/auth/google/callback?error={requests.utils.quote(str(e))}")
