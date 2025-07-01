# main.py
import os
import requests
import asyncio
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import firestore
from uuid import uuid4

# Import our custom modules
import firebase_config as fb
import redis_cache

# Load environment variables from .env file
load_dotenv()
app = FastAPI()

# --- CORS Middleware Configuration ---
# Add your deployed Vercel URL and local development URLs
origins = [
    "http://localhost:3000",
    "https://socratic-ai-tutor.vercel.app" # Your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load API URLs from Environment Variables ---
COLAB_API_URL = os.getenv("COLAB_AI_API_URL")
HF_API_URL = os.getenv("HF_AI_API_URL")

# --- Pydantic Models for Data Validation ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None

# --- Firebase Authentication Dependency ---
async def get_current_user(authorization: str = Header(...)):
    """Verifies the Firebase ID token from the request header and returns the user's UID."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
    token = authorization.split("Bearer ")[1]
    try:
        decoded_token = fb.auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Firebase token: {e}")

# --- Health Check and Failover Logic ---
async def check_service_health(url: str) -> bool:
    """Quickly checks if a service is responsive by hitting its root endpoint."""
    if not url:
        print(f"❌ No URL provided for health check")
        return False
    try:
        print(f"🔍 Checking health for: {url}")
        loop = asyncio.get_event_loop()
        # Try the root endpoint first (fast health check)
        response = await loop.run_in_executor(
            None, 
            lambda: requests.get(url, timeout=5)
        )
        print(f"📊 Health check response for {url}: Status {response.status_code}")
        if response.status_code != 200:
            print(f"📄 Response content: {response.text[:200]}...")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed for {url}: {e}")
        return False

async def get_ai_reply_with_failover(history: List[Dict[str, str]]) -> Dict[str, str]:
    """Tries Colab first, then fails over to Hugging Face Spaces."""
    loop = asyncio.get_event_loop()

    # Helper to run blocking requests in a separate thread
    async def call_api(url: str):
        return await loop.run_in_executor(
            None,
            lambda: requests.post(f"{url}/generate", json={"history": history}, timeout=600)
        )

    # 1. Try Colab (Fast GPU)
    if await check_service_health(COLAB_API_URL):
        print("✅ Using FAST service: Google Colab (GPU)")
        try:
            response = await call_api(COLAB_API_URL)
            response.raise_for_status()
            return {"reply": response.json()["reply"], "source": "colab_gpu"}
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Colab service failed mid-request: {e}")

    # 2. Try Hugging Face (Slower CPU)
    if await check_service_health(HF_API_URL):
        print("🟡 Using SLOW service: Hugging Face Spaces (CPU)")
        try:
            response = await call_api(HF_API_URL)
            response.raise_for_status()
            return {"reply": response.json()["reply"], "source": "hf_cpu_slow"}
        except requests.exceptions.RequestException as e:
            print(f"❌ Hugging Face service failed mid-request: {e}")

    # 3. Both services failed
    print("❌ CRITICAL: Both AI model services are offline.")
    raise HTTPException(status_code=503, detail="All AI model services are currently unavailable.")


# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Checks the status of the AI services for the frontend to display."""
    print(f"🏥 Health check requested - COLAB_URL: {COLAB_API_URL}, HF_URL: {HF_API_URL}")
    
    # Check both services
    is_colab_up = await check_service_health(COLAB_API_URL)
    is_hf_up = await check_service_health(HF_API_URL)
    
    # Determine service statuses
    colab_status = "not_configured" if not COLAB_API_URL else ("online" if is_colab_up else "offline")
    hf_status = "not_configured" if not HF_API_URL else ("online" if is_hf_up else "offline")
    
    # Determine overall status and primary service
    if is_colab_up:
        print("✅ Colab service is UP - using as primary")
        primary_service = "colab_gpu"
        overall_status = "online"
        chat_enabled = True
    elif is_hf_up:
        print("🟡 HF service is UP - using as fallback")
        primary_service = "hf_cpu_slow"
        overall_status = "slow"
        chat_enabled = True
    else:
        print("❌ Both services are DOWN")
        primary_service = "none"
        overall_status = "offline"
        chat_enabled = False
    
    return {
        "status": overall_status,
        "service": primary_service,
        "chat_enabled": chat_enabled,
        "services": {
            "colab": colab_status,
            "huggingface": hf_status
        }
    }

@app.get("/conversations")
async def get_conversations(user_id: str = Depends(get_current_user)):
    """Fetches a list of all conversation titles and IDs for the logged-in user."""
    try:
        convos_ref = fb.db.collection('users').document(user_id).collection('conversations').order_by("createdAt", direction=firestore.Query.DESCENDING).stream()
        conversations = [{"id": convo.id, "title": convo.to_dict().get("title", "Untitled Chat")} for convo in convos_ref]
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/chat")
async def handle_chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """Handles an incoming message, either continuing an existing chat or starting a new one."""
    
    # Check if any AI service is available before processing the message
    is_colab_up = await check_service_health(COLAB_API_URL)
    is_hf_up = await check_service_health(HF_API_URL)
    
    if not is_colab_up and not is_hf_up:
        raise HTTPException(
            status_code=503, 
            detail="All AI services are currently unavailable. Please try again later."
        )
    
    conversation_id = request.conversation_id
    new_message = request.messages[0].dict()

    if not conversation_id:
        conversation_id = str(uuid4())
        history = [{"role": "system", "content": "You are a helpful Socratic tutor."}]
        title = (new_message['content'][:35] + '...') if len(new_message['content']) > 35 else new_message['content']
    else:
        history = redis_cache.get_chat_history(f"{user_id}:{conversation_id}")
        if not history:
            doc = fb.db.collection('users').document(user_id).collection('conversations').document(conversation_id).get()
            history = doc.to_dict().get('messages', []) if doc.exists else []

    history.append(new_message)
    ai_response = await get_ai_reply_with_failover(history)

    assistant_reply_content = ai_response["reply"]
    history.append({"role": "assistant", "content": assistant_reply_content})

    convo_doc_ref = fb.db.collection('users').document(user_id).collection('conversations').document(conversation_id)

    if not request.conversation_id:
        convo_doc_ref.set({'title': title, 'createdAt': firestore.SERVER_TIMESTAMP, 'messages': history})
    else:
        convo_doc_ref.update({'messages': history})

    redis_cache.set_chat_history(f"{user_id}:{conversation_id}", history)

    return {"reply": assistant_reply_content, "source": ai_response["source"], "conversation_id": conversation_id}

@app.get("/")
def read_root():
    return {"message": "Socratic Tutor Logic Backend is running!"}