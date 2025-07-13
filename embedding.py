# embedding.py (in your local socratic-tutor-backend project)
import os
import requests

# This line reads the new URL from your .env file
HF_EMBEDDER_API_URL = os.getenv("HF_EMBEDDER_API_URL")

def create_embedding(text: str):
    """
    Creates a vector embedding by calling our dedicated /embed endpoint
    on the Hugging Face Space.
    """
    if not text or not isinstance(text, str) or not HF_EMBEDDER_API_URL:
        print("⚠️ create_embedding called but HF_EMBEDDER_API_URL is not configured.")
        return None
    
    # It constructs the full endpoint URL correctly
    embed_endpoint = f"{HF_EMBEDDER_API_URL}/embed"
    
    try:
        print(f"🧠 Calling Embedding Server at {embed_endpoint}...")
        response = requests.post(embed_endpoint, json={"text_chunk": text}, timeout=30)
        response.raise_for_status()
        
        embedding = response.json().get("embedding")
        if embedding:
            print("✅ Embedding created successfully via API.")
            return embedding
        else:
            print(f"❌ API returned success but no embedding. Response: {response.json()}")
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Calling our /embed endpoint failed: {e}")
        return None