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

# Retrieval: how many pieces to give the AI for each question.
# 8 beat 5 in evaluation (eval/RESULTS.md): 93.6% vs 85.9% correct on gpt-oss-120b.
TOP_K = 8

# AI model (Groq)
MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"

# Providers with OpenAI-compatible APIs. The website uses Groq; NVIDIA is used only for evaluation.
PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY"},
}

# Evaluation. The judge is from a different company (Google) than the answer model (OpenAI gpt-oss).
# Runs before 25 Sep 2026 evening were first graded by "qwen/qwen3.8-27b" on Groq.
JUDGE_PROVIDER = "nvidia"
JUDGE_MODEL = "google/gemma-4-31b-it"
# Independent model that double-checks the key-fact lists in eval/questions.json.
KEY_FACTS_CHECK_MODEL = "mistralai/mistral-nemotron"
# Groq's published prices in USD per 1M tokens (input, output), used for cost estimates only.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}
