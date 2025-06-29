# redis_cache.py
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv() # Loads variables from .env file

UPSTASH_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")

headers = {
    "Authorization": f"Bearer {UPSTASH_TOKEN}"
}

def get_chat_history(user_id: str):
    """Fetches chat history from Redis cache."""
    try:
        response = requests.post(f"{UPSTASH_URL}/get/{user_id}", headers=headers)
        if response.status_code == 200 and response.json().get("result"):
            # The result is a JSON string, so we parse it back into a list
            return json.loads(response.json()["result"])
    except Exception as e:
        print(f"Redis get failed: {e}")
    return None # Return None if not found or error

def set_chat_history(user_id: str, history: list):
    """Saves chat history to Redis cache. TTL sets it to expire in 24 hours."""
    try:
        # Convert the list to a JSON string before saving
        payload = json.dumps(history)
        # The 'ex' command sets an expiration time in seconds (e.g., 86400 for 24 hours)
        requests.post(f"{UPSTASH_URL}/set/{user_id}?ex=86400", headers=headers, data=payload)
    except Exception as e:
        print(f"Redis set failed: {e}")