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

    return {"reply": assistant_reply_content, "source": ai_response["source"], "conversation_id": conversation_id}

# --- Payment Endpoints ---

@app.post("/create-order")
async def create_payment_order(request: CreateOrderRequest, user_id: str = Depends(get_current_user)):
    """Creates a Razorpay order for payment processing."""
    try:
        if not razorpay_client:
            # Mock order creation for development/testing
            order_id = f"order_{int(time.time())}_{random.randint(1000, 9999)}"
            order_data = {
                "id": order_id,
                "amount": int(request.amount * 100),  # Convert to paise
                "currency": request.currency,
                "receipt": f"receipt_{int(time.time())}",
                "status": "created",
                "created_at": int(time.time()),
                "notes": request.notes or {"purpose": "Socratic AI Payment"}
            }
            print(f"🧪 Mock order created: {order_id}")
        else:
            # Real Razorpay order creation
            order_data = {
                "amount": int(request.amount * 100),  # Convert to paise
                "currency": request.currency,
                "receipt": f"receipt_{int(time.time())}",
                "notes": request.notes or {"purpose": "Socratic AI Payment"}
            }
            order = razorpay_client.order.create(data=order_data)
            print(f"💳 Razorpay order created: {order['id']}")
            order_data = order

        # Store order in Firebase for the user
        order_doc_ref = fb.db.collection('users').document(user_id).collection('orders').document(order_data['id'])
        order_record = {
            "orderId": order_data['id'],
            "amount": request.amount,
            "currency": request.currency,
            "status": "created",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "userId": user_id,
            "notes": request.notes or {"purpose": "Socratic AI Payment"}
        }
        order_doc_ref.set(order_record)
        
        return order_data
        
    except Exception as e:
        print(f"❌ Order creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create order: {str(e)}")
    
@app.post("/verify-payment")
async def verify_payment(request: dict, user_id: str = Depends(get_current_user)):
    """Manually verify a payment and update order status."""
    try:
        payment_id = request.get('payment_id')
        order_id = request.get('order_id')
        
        if not payment_id or not order_id:
            raise HTTPException(status_code=400, detail="Payment ID and Order ID required")
        
        # Verify payment with Razorpay
        if razorpay_client:
            try:
                payment = razorpay_client.payment.fetch(payment_id)
                if payment['status'] == 'captured' and payment['order_id'] == order_id:
                    # Update order status in Firebase
                    order_ref = fb.db.collection('users').document(user_id).collection('orders').document(order_id)
                    order_doc = order_ref.get()
                    
                    if order_doc.exists:
                        order_ref.update({
                            "status": "paid",
                            "paymentId": payment_id,
                            "paidAt": firestore.SERVER_TIMESTAMP,
                            "paymentDetails": payment
                        })
                        
                        # Create payment record
                        payment_ref = fb.db.collection('users').document(user_id).collection('payments').document(payment_id)
                        payment_record = {
                            "paymentId": payment_id,
                            "orderId": order_id,
                            "amount": payment.get('amount', 0) / 100,
                            "status": "captured",
                            "method": payment.get('method'),
                            "capturedAt": firestore.SERVER_TIMESTAMP,
                            "razorpayDetails": payment
                        }
                        payment_ref.set(payment_record)
                        
                        return {"success": True, "message": "Payment verified and order updated"}
                    else:
                        raise HTTPException(status_code=404, detail="Order not found")
                else:
                    raise HTTPException(status_code=400, detail="Payment not captured or order mismatch")
            except Exception as e:
                print(f"Razorpay verification error: {e}")
                raise HTTPException(status_code=500, detail="Payment verification failed")
        else:
            # Mock verification for development
            order_ref = fb.db.collection('users').document(user_id).collection('orders').document(order_id)
            order_ref.update({
                "status": "paid",
                "paymentId": payment_id,
                "paidAt": firestore.SERVER_TIMESTAMP
            })
            return {"success": True, "message": "Payment verified (mock mode)"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/razorpay")
async def handle_razorpay_webhook(request: Request):
    """Handles Razorpay webhook events to update payment status."""
    try:
        body = await request.body()
        signature = request.headers.get("x-razorpay-signature")
        
        if not signature:
            raise HTTPException(status_code=400, detail="Missing signature")
        
        # Verify webhook signature if webhook secret is configured
        if RAZORPAY_WEBHOOK_SECRET:
            expected_signature = hmac.new(
                RAZORPAY_WEBHOOK_SECRET.encode('utf-8'),
                body,
                hashlib.sha256
            ).hexdigest()
            
            if signature != expected_signature:
                print(f"❌ Invalid webhook signature")
                raise HTTPException(status_code=400, detail="Invalid signature")
        
        event = json.loads(body.decode('utf-8'))
        print(f"📩 Webhook received: {event.get('event', 'unknown')}")
        
        # Handle different webhook events
        event_type = event.get('event')
        payload = event.get('payload', {})
        
        if event_type == "payment.captured":
            await handle_payment_captured(payload)
        elif event_type == "payment.failed":
            await handle_payment_failed(payload)
        elif event_type == "order.paid":
            await handle_order_paid(payload)
        else:
            print(f"🤷 Unhandled webhook event: {event_type}")
        
        return {"received": True}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        print(f"❌ Webhook processing error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

async def handle_payment_captured(payload):
    """Handles successful payment capture."""
    try:
        payment = payload.get('payment', {}).get('entity', {})
        order_id = payment.get('order_id')
        payment_id = payment.get('id')
        amount = payment.get('amount', 0) / 100  # Convert from paise to INR
        
        print(f"✅ Payment captured: {payment_id} for order: {order_id}")
        
        # Find the order in Firebase and update it
        if order_id:
            # Query across all users to find the order (you might want to optimize this)
            users_ref = fb.db.collection('users')
            for user_doc in users_ref.stream():
                order_ref = user_doc.reference.collection('orders').document(order_id)
                order_doc = order_ref.get()
                
                if order_doc.exists:
                    # Update order status
                    order_ref.update({
                        "status": "paid",
                        "paymentId": payment_id,
                        "paidAt": firestore.SERVER_TIMESTAMP,
                        "paymentDetails": payment
                    })
                    
                    # Create a payment record
                    payment_ref = user_doc.reference.collection('payments').document(payment_id)
                    payment_record = {
                        "paymentId": payment_id,
                        "orderId": order_id,
                        "amount": amount,
                        "status": "captured",
                        "method": payment.get('method'),
                        "capturedAt": firestore.SERVER_TIMESTAMP,
                        "razorpayDetails": payment
                    }
                    payment_ref.set(payment_record)
                    
                    print(f"💾 Updated order {order_id} and created payment record {payment_id}")
                    break
                    
    except Exception as e:
        print(f"❌ Error handling payment captured: {e}")

async def handle_payment_failed(payload):
    """Handles failed payments."""
    try:
        payment = payload.get('payment', {}).get('entity', {})
        order_id = payment.get('order_id')
        payment_id = payment.get('id')
        
        print(f"❌ Payment failed: {payment_id} for order: {order_id}")
        
        # Find and update the order
        if order_id:
            users_ref = fb.db.collection('users')
            for user_doc in users_ref.stream():
                order_ref = user_doc.reference.collection('orders').document(order_id)
                order_doc = order_ref.get()
                
                if order_doc.exists:
                    order_ref.update({
                        "status": "failed",
                        "paymentId": payment_id,
                        "failedAt": firestore.SERVER_TIMESTAMP,
                        "failureReason": payment.get('error_description', 'Payment failed')
                    })
                    print(f"💾 Updated failed order {order_id}")
                    break
                    
    except Exception as e:
        print(f"❌ Error handling payment failed: {e}")

async def handle_order_paid(payload):
    """Handles order completion."""
    try:
        order = payload.get('order', {}).get('entity', {})
        order_id = order.get('id')
        
        print(f"✅ Order paid: {order_id}")
        
        # Additional order completion logic can go here
        # e.g., activate premium features, send confirmation email, etc.
        
    except Exception as e:
        print(f"❌ Error handling order paid: {e}")

@app.get("/orders")
async def get_user_orders(user_id: str = Depends(get_current_user)):
    """Fetches all orders for the authenticated user."""
    try:
        orders_ref = fb.db.collection('users').document(user_id).collection('orders').order_by("createdAt", direction=firestore.Query.DESCENDING).stream()
        orders = []
        for order_doc in orders_ref:
            order_data = order_doc.to_dict()
            order_data['id'] = order_doc.id
            orders.append(order_data)
        return orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/payments")
async def get_user_payments(user_id: str = Depends(get_current_user)):
    """Fetches all payments for the authenticated user."""
    try:
        payments_ref = fb.db.collection('users').document(user_id).collection('payments').order_by("capturedAt", direction=firestore.Query.DESCENDING).stream()
        payments = []
        for payment_doc in payments_ref:
            payment_data = payment_doc.to_dict()
            payment_data['id'] = payment_doc.id
            payments.append(payment_data)
        return payments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Socratic Tutor Logic Backend is running!"}