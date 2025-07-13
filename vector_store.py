# vector_store.py
import os
from pinecone import Pinecone, ServerlessSpec
from embedding import create_embedding
from dotenv import load_dotenv

load_dotenv()

# Load Pinecone credentials from .env file
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "socratic-tutor-memories"

pc = None
index = None

# Initialize Pinecone
try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Check if the index exists, create it if not
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384, # The dimension of the all-MiniLM-L6-v2 model
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        index = pc.Index(PINECONE_INDEX_NAME)
        print("✅ Pinecone initialized successfully.")
    else:
        print("⚠️ Pinecone API key not found. Vector store functionality will be disabled.")
except Exception as e:
    print(f"❌ Failed to initialize Pinecone: {e}")

# --- Core Functions ---

def add_memory_chunk(user_id: str, text_chunk: str, conversation_id: str):
    """Creates an embedding and upserts it into the Pinecone index."""
    if not index or not text_chunk:
        return
    
    print(f"🧠 Creating embedding for memory chunk for user: {user_id}")
    vector = create_embedding(text_chunk)
    if not vector:
        return
        
    # We use a unique ID for each vector, combining user and conversation ID
    # A timestamp or hash could also be used for more granularity.
    vector_id = f"{user_id}-{conversation_id}-{hash(text_chunk)}"

    # Upsert the vector into Pinecone with metadata
    index.upsert(
        vectors=[{
            "id": vector_id,
            "values": vector,
            "metadata": {"user_id": user_id, "text": text_chunk}
        }]
    )
    print(f"✅ Memory chunk for user {user_id} added to Pinecone.")

def find_relevant_memories(user_id: str, query_text: str, top_k: int = 5):
    """Finds the most relevant memory chunks for a user based on a query."""
    if not index or not query_text:
        return []

    print(f"🔍 Searching for memories for user {user_id} with query: '{query_text}'")
    query_vector = create_embedding(query_text)
    if not query_vector:
        return []

    # Query Pinecone, filtering by the specific user_id
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        filter={"user_id": {"$eq": user_id}},
        include_metadata=True
    )
    
    # Extract just the text from the results
    memories = [match['metadata']['text'] for match in results['matches']]
    print(f"✅ Found {len(memories)} relevant memories.")
    return memories