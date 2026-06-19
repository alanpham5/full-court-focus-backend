import base64
import json
import os
import firebase_admin
from firebase_admin import credentials, auth, firestore

db = None

try:
    service_account_b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_B64")
    if service_account_b64:
        service_account_json = base64.b64decode(service_account_b64).decode("utf-8")
        service_account_info = json.loads(service_account_json)
        cred = credentials.Certificate(service_account_info)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
    else:
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
    db = firestore.client()
except Exception as e:
    print(f"Warning: Firebase Admin SDK not initialized: {e}")
