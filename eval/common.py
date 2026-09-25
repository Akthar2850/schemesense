"""Shared helpers for the evaluation scripts."""

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))  # so "import config, ingest, rag" works

import chromadb  # noqa: E402

import config  # noqa: E402
import ingest  # noqa: E402

RESULTS_DIR = EVAL_DIR / "results"
REFUSAL = "couldn't find this in the scheme documents"


def load_questions():
    return json.loads((EVAL_DIR / "questions.json").read_text())


def collection_for(chunk_size):
    """Collection name for a chunk size; builds it from data/ if it doesn't exist yet (local, no tokens)."""
    if chunk_size == config.CHUNK_SIZE:
        name = config.COLLECTION_NAME
    else:
        name = f"{config.COLLECTION_NAME}_c{chunk_size}"
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    if name not in [c.name for c in client.list_collections()]:
        ingest.main(chunk_size, chunk_size // 5, name)  # overlap = 20% of the chunk size
    return name


def is_hit(sources, expected):
    """True if any retrieved chunk comes from an expected file and page."""
    wanted = {(e["file"], page) for e in expected for page in e["pages"]}
    return any((s["source"], s["page"]) in wanted for s in sources)


def is_refusal(answer):
    return REFUSAL in answer.lower().replace("’", "'")
