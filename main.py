# main.py
import os
import requests
import asyncio
import hmac
import hashlib
import json
import time
import random
import string
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import firestore
from uuid import uuid4
import razorpay
# --- NEW IMPORTS ---
from redis import Redis
from rq import Queue
import embedding # Our new module
import vector_store # Our new module

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

# --- Razorpay Configuration ---
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

# Initialize Razorpay client if credentials are available
razorpay_client = None
if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    try:
        razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    except Exception as e:
        print(f"Warning: Could not initialize Razorpay client: {e}")

try:
    redis_conn = Redis.from_url(os.getenv("UPSTASH_REDIS_URL_FOR_RQ", os.getenv("UPSTASH_REDIS_URL")))
    q = Queue('socratic-memories', connection=redis_conn)
    print("✅ Redis Queue (RQ) connected successfully.")
except Exception as e:
    print(f"❌ Could not connect to Redis Queue: {e}")
    q = None

# --- Pydantic Models for Data Validation ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None

# --- Payment Models ---
class CreateOrderRequest(BaseModel):
    amount: float  # Amount in INR
    currency: str = "INR"
    notes: Optional[Dict[str, str]] = None

class WebhookPayload(BaseModel):
    event: str
    payload: Dict

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
            lambda: requests.post(f"{url}/generate", json={"history": history}, timeout=1000)
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

    # --- NEW: Enqueue a background job to store this memory ---
    if q:
        new_message_content = request.messages[0].dict()['content']
        memory_chunk = f"User: {new_message_content}\nAssistant: {assistant_reply_content}"
        
        # This is a non-blocking call. It adds the job and returns instantly.
        q.enqueue(vector_store.add_memory_chunk, user_id, memory_chunk, conversation_id)
        print(f"📩 Enqueued memory storage job for user {user_id}")

    # Return the response to the user immediately
    return {"reply": assistant_reply_content, "source": ai_response["source"], "conversation_id": conversation_id}

# --- NEW /suggest-holistic-project Endpoint ---
@app.post("/suggest-holistic-project")
async def suggest_project(user_id: str = Depends(get_current_user)):
    """Analyzes all user history to suggest a single, cohesive project."""
    
    # 1. Get the user's entire history from Firestore to find key topics
    # (A simpler approach for now instead of multiple AI calls)
    all_messages = []
    convos_ref = fb.db.collection('users').document(user_id).collection('conversations').stream()
    for convo in convos_ref:
        all_messages.extend(convo.to_dict().get('messages', []))

    if not all_messages:
        raise HTTPException(status_code=404, detail="No conversation history found to generate a project from.")

    # We'll use the last ~2000 characters as context to find key topics
    recent_history_text = " ".join([msg['content'] for msg in all_messages[-20:]])

    # 2. Use the AI to synthesize key topics from recent history
    topic_synthesis_prompt = f"""
    Analyze the following conversation text and list the top 3-5 main technical concepts or topics being discussed.
    Your response should be a single JSON object with a key "topics" containing a list of strings.
    
    CONVERSATION TEXT:
    "{recent_history_text[:2000]}"
    """
    
    # We call the AI just to get the topics
    try:
        topic_response = await get_ai_reply_with_failover([{"role": "user", "content": topic_synthesis_prompt}])
        topics = json.loads(topic_response['reply']).get('topics', [])
    except Exception as e:
        print(f"⚠️ Could not synthesize topics, creating a generic project. Error: {e}")
        topics = ["General Problem Solving"]

    if not topics:
        topics = ["General Problem Solving"]

    print(f"Identified user topics: {topics}")

    # 3. For each topic, retrieve relevant memories from the Vector DB
    all_relevant_memories = []
    for topic in topics:
        memories = vector_store.find_relevant_memories(user_id, topic, top_k=3)
        all_relevant_memories.extend(memories)
    
    # Remove duplicates
    unique_memories_text = "\n".join(list(set(all_relevant_memories)))

    # 4. Construct the final "Mega-Prompt" to generate the project
    mega_prompt = f"""
    You are an expert career mentor. Based on the user's key interests and excerpts from their learning history,
    design a single, cohesive 'capstone' project that combines these skills.
    
    KEY INTERESTS: {', '.join(topics)}
    
    RELEVANT EXCERPTS FROM USER'S HISTORY:
    {unique_memories_text[:3000]}
    
    YOUR TASK:
    Generate a detailed project plan. The output must be a single JSON object with the keys:
    "project_title", "project_description", "key_features" (a list of strings), 
    and "skills_reinforced" (a list of strings).
    """

    final_project_response = await get_ai_reply_with_failover([{"role": "user", "content": mega_prompt}])

    try:
        # Return the final JSON project plan
        return json.loads(final_project_response['reply'])
    except:
        # Fallback if the AI didn't return perfect JSON
        return {"project_title": "Custom Project Idea", "project_description": final_project_response['reply'], "key_features": [], "skills_reinforced": topics}

# --- Payment Endpoints ---
# ToDo - Implement Razorpay payment endpoints for creating orders, verifying payments, and handling webhooks.

@app.get("/")
def read_root():
    return {"message": "Socratic Tutor Logic Backend is running!"}