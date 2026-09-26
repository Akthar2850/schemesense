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

# Scheme-aware search: if a question names a scheme, prefer pieces from that scheme's documents
# (fewer mix-ups between similar schemes). Names are matched as whole words, ignoring case.
SCHEME_AWARE = True
SCHEME_NAMES = {
    "pm-kisan-guidelines.pdf": ["pm-kisan", "pm kisan", "pmkisan", "kisan samman"],
    "pm-jay-pib-2024.pdf": ["pm-jay", "pmjay", "pm jay", "ayushman", "jan arogya"],
    "atal-pension-yojana.pdf": ["atal pension", "apy"],
    "sukanya-samriddhi-scheme-2019.pdf": ["sukanya", "ssy"],
    "pmay-g-pib-2024.pdf": ["pmay-g", "pmayg", "pmay", "pmay-gramin", "pmay gramin", "awas yojana",
                            "awaas yojana", "pm awas", "pm awaas"],
    "pm-ujjwala-pib-2024.pdf": ["ujjwala", "pmuy"],
    "pm-jan-dhan-pib-2024.pdf": ["jan dhan", "jan-dhan", "pmjdy"],
    "pm-mudra-pib-2026.pdf": ["mudra", "pmmy"],
    "jan-suraksha-pib-2026.txt": ["jeevan jyoti", "pmjjby", "suraksha bima", "pmsby", "jan suraksha",
                                  "atal pension", "apy"],
    "stand-up-india-faq-2022.pdf": ["stand-up india", "stand up india", "standup india"],
    "pm-vishwakarma-pib-2023.pdf": ["vishwakarma"],
    "pm-svanidhi-guidelines-2026.pdf": ["svanidhi", "street vendor", "street vendors"],
    "nps-all-citizen-faq.pdf": ["national pension system", "nps"],
    "pmfby-summary.pdf": ["fasal bima", "pmfby", "crop insurance"],
}

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
