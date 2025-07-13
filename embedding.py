# embedding.py
from sentence_transformers import SentenceTransformer
import torch

# This function loads the model ONCE and re-uses it.
# It's a small model, so it can run on the CPU.
def load_embedding_model():
    """Loads the SentenceTransformer model."""
    print("🚀 Loading embedding model (sentence-transformers)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"✅ Embedding model loaded successfully on '{device}'.")
    return model

# Global variable to hold our loaded model
EMBEDDING_MODEL = load_embedding_model()

def create_embedding(text: str):
    """Creates a vector embedding for a given text chunk."""
    if not text or not isinstance(text, str):
        return None
    # The .encode() method creates the embedding
    return EMBEDDING_MODEL.encode(text).tolist()