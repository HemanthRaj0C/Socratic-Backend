# firebase_config.py
import firebase_admin
from firebase_admin import credentials, auth, firestore

# IMPORTANT: Make sure the path to your service account key is correct
SERVICE_ACCOUNT_KEY_PATH = "socratic-ai-tutor-backend-firebase.json"

try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    print("✅ Firebase Admin SDK initialized successfully.")
    # Initialize Firestore client
    db = firestore.client()
except Exception as e:
    print(f"❌ Failed to initialize Firebase Admin SDK: {e}")
    db = None