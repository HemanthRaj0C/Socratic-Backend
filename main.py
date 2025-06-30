# main.py (to run on your local machine)
import os
import json
import requests
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
from typing import Optional
from uuid import uuid4 # To generate unique IDs for new conversations

from fastapi.middleware.cors import CORSMiddleware

# Import our custom modules
import firebase_config as fb
import redis_cache

from firebase_admin import firestore

# Load environment variables (for NGROK_URL)
load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:3000",  # The default Next.js port
    "https://socratic-ai-tutor.vercel.app/" # deployed frontend URL
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

# --- NEW Pydantic Models ---
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None # Frontend will tell us which chat to continue

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

# NEW ENDPOINT: Get a list of all conversations for a user
@app.get("/conversations")
async def get_conversations(user_id: str = Depends(get_current_user)):
    """Fetches a list of all conversation titles and IDs for the logged-in user."""
    try:
        convos_ref = fb.db.collection('users').document(user_id).collection('conversations').order_by("createdAt", direction=firestore.Query.DESCENDING).stream()
        
        conversations = []
        for convo in convos_ref:
            convo_data = convo.to_dict()
            conversations.append({
                "id": convo.id,
                "title": convo_data.get("title", "Untitled Chat"),
                "createdAt": convo_data.get("createdAt")
            })
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NEW ENDPOINT: Get the messages for a specific conversation
@app.get("/conversations/{conversation_id}")
async def get_conversation_by_id(conversation_id: str, user_id: str = Depends(get_current_user)):
    """Fetches the full message history for a single conversation."""
    try:
        doc_ref = fb.db.collection('users').document(user_id).collection('conversations').document(conversation_id)
        doc = doc_ref.get()
        if doc.exists:
            return {"messages": doc.to_dict().get("messages", [])}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# MODIFIED ENDPOINT: The main chat logic
@app.post("/chat")
async def handle_chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """Handles an incoming message, either continuing an existing chat or starting a new one."""
    
    conversation_id = request.conversation_id
    new_message = request.messages[0].dict() # Assuming frontend sends only the new message

    # --- 1. Handle New vs. Existing Conversation ---
    if not conversation_id:
        # This is a new chat. Create a new conversation document.
        conversation_id = str(uuid4())
        history = [{"role": "system", "content": "You are a helpful Socratic tutor."}]
        # Let's generate a title from the first message
        first_message_content = new_message['content']
        # You could even use a quick LLM call here to generate a better title
        title = (first_message_content[:30] + '...') if len(first_message_content) > 30 else first_message_content
    else:
        # This is an existing chat. Load its history.
        # We'll use a simplified cache key for this example
        history = redis_cache.get_chat_history(f"{user_id}:{conversation_id}")
        if not history:
            doc_ref = fb.db.collection('users').document(user_id).collection('conversations').document(conversation_id)
            doc = doc_ref.get()
            history = doc.to_dict().get('messages', []) if doc.exists else []

    # 2. Add new user message and call AI
    history.append(new_message)
    assistant_reply = get_ai_reply_from_colab(history)
    history.append({"role": "assistant", "content": assistant_reply})
    
    # 3. Update databases
    convo_doc_ref = fb.db.collection('users').document(user_id).collection('conversations').document(conversation_id)
    
    if not request.conversation_id: # If it was a new chat, set title and timestamp
        convo_doc_ref.set({
            'title': title,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'messages': history
        })
    else: # Otherwise, just update the messages
        convo_doc_ref.update({'messages': history})
    
    # Update cache
    redis_cache.set_chat_history(f"{user_id}:{conversation_id}", history)
    
    # Return the reply AND the new conversation ID if it was created
    return {"reply": assistant_reply, "conversation_id": conversation_id}

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