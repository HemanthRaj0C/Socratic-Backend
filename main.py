# main.py (to run on your local machine)
import os
import json
import requests
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

# Import our custom modules
import firebase_config as fb
import redis_cache

# Load environment variables (for NGROK_URL)
load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:3000",  # The default Next.js port
    "http://localhost:3001",  # Your current Next.js port
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # The list of allowed origins
    allow_credentials=True,      # Allows cookies to be included
    allow_methods=["*"],         # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],         # Allows all headers (including Authorization and Content-Type)
)

# --- GET THE NGROK URL FROM YOUR .env FILE ---
AI_MODEL_API_URL = os.getenv("AI_MODEL_API_URL")
if not AI_MODEL_API_URL:
    raise ValueError("AI_MODEL_API_URL not found in .env file. Please set it to your Ngrok URL.")

# Pydantic Models (same as before)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# Firebase Auth Dependency (same as before)
async def get_current_user(authorization: str = Header(...)):
    # ... (same code as before to verify token)
    token = authorization.split("Bearer ")[1]
    decoded_token = fb.auth.verify_id_token(token)
    return decoded_token['uid']


# Helper function to call the AI model on Colab
def get_ai_reply_from_colab(history: List[Dict[str, str]]) -> str:
    """Calls the separate AI model API running on Colab via Ngrok."""
    print(f"🔄 Calling AI model at: {AI_MODEL_API_URL}/generate")
    try:
        # The endpoint on our Colab server will be named /generate
        response = requests.post(
            f"{AI_MODEL_API_URL}/generate",
            json={"history": history},
            timeout=120  # Set a long timeout, as the model can be slow
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        response_data = response.json()
        reply = response_data["reply"]
        print(f"✅ AI model responded successfully")
        print(f"📝 Response data: {response_data}")
        print(f"💬 Reply: {reply[:100]}..." if len(reply) > 100 else f"💬 Reply: {reply}")
        return reply
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling AI model API: {e}")
        raise HTTPException(status_code=503, detail="The AI model service is unavailable.")


@app.post("/chat")
async def handle_chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """The main chat endpoint that orchestrates everything."""
    
    print(f"💬 Chat request received for user: {user_id}")
    print(f"🔗 AI Model URL configured as: {AI_MODEL_API_URL}")
    
    # 1. Get History (Cache-Aside Pattern)
    history = redis_cache.get_chat_history(user_id)
    if not history:
        print(f"📂 No cache found, checking Firestore for user: {user_id}")
        doc = fb.db.collection('chats').document(user_id).get()
        history = doc.to_dict().get('messages', []) if doc.exists else [{"role": "system", "content": "You are a helpful Socratic tutor."}]
        print(f"📚 Loaded {len(history)} messages from Firestore")
    else:
        print(f"⚡ Loaded {len(history)} messages from Redis cache")

    # 2. Update history with new message(s)
    for msg in request.messages:
        history.append(msg.dict())
        print(f"➕ Added message: {msg.role} - {msg.content[:50]}...")
    
    print(f"📝 Sending history to AI: {len(history)} messages")

    # 3. Call the AI Model
    assistant_reply = get_ai_reply_from_colab(history)

    # 4. Update history with AI reply
    history.append({"role": "assistant", "content": assistant_reply})
    print(f"🤖 Added AI reply to history")

    # 5. Update both storage systems
    try:
        # Update Redis cache
        redis_cache.set_chat_history(user_id, history)
        print(f"✅ Updated Redis cache with {len(history)} messages")
        
        # Update Firestore database
        fb.db.collection('chats').document(user_id).set({'messages': history})
        print(f"✅ Updated Firestore with {len(history)} messages")
    except Exception as e:
        print(f"⚠️ Error updating storage: {e}")
        # Continue anyway - the response can still be sent

    print(f"✅ Chat response sent successfully")
    return {"reply": assistant_reply}

@app.get("/debug/redis/{user_id}")
def debug_redis_data(user_id: str):
    """Debug endpoint to check what's stored in Redis for a user."""
    try:
        history = redis_cache.get_chat_history(user_id)
        return {
            "user_id": user_id,
            "redis_data_exists": history is not None,
            "message_count": len(history) if history else 0,
            "messages": history if history else "No data found in Redis"
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "error": str(e),
            "redis_data_exists": False
        }

@app.get("/debug/redis-keys")
def debug_redis_keys():
    """Debug endpoint to see Redis connection and keys."""
    try:
        # Check if Upstash Redis is configured
        upstash_url = os.getenv("UPSTASH_REDIS_URL")
        upstash_token = os.getenv("UPSTASH_REDIS_TOKEN")
        
        return {
            "redis_type": "Upstash Redis (Cloud)",
            "upstash_url_configured": upstash_url is not None,
            "upstash_token_configured": upstash_token is not None,
            "upstash_url": upstash_url[:50] + "..." if upstash_url else None,
            "redis_module_available": hasattr(redis_cache, 'get_chat_history'),
            "note": "Use /debug/redis/{user_id} to check specific user data"
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Redis connection issue"
        }

@app.get("/debug/ai-url")
def debug_ai_url():
    """Debug endpoint to check AI model URL configuration."""
    return {
        "AI_MODEL_API_URL": AI_MODEL_API_URL,
        "constructed_url": f"{AI_MODEL_API_URL}/generate",
        "url_ends_with_slash": AI_MODEL_API_URL.endswith('/') if AI_MODEL_API_URL else False
    }

@app.get("/")
def read_root():
    return {"message": "Socratic Tutor Logic Backend is running!"}