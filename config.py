from pathlib import Path

BASE_DIR = Path(__file__).parent

# Documents and database
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "schemes"

# Chunking: each PDF page is split into pieces of about CHUNK_SIZE characters,
# with CHUNK_OVERLAP characters shared between neighbouring pieces.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval: how many pieces to give the AI for each question
TOP_K = 5

# AI model (Groq)
MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"
