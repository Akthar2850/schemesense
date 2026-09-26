"""Core RAG logic: find the relevant passages, then ask the AI to answer only from them."""

import os
import re
import time

import chromadb
from groq import Groq

import config
import ingest

SYSTEM_PROMPT = """You are SchemeSense, an assistant for Indian government schemes.
Answer ONLY using the numbered context passages you are given. Do not use outside knowledge.
Cite the passages that support each fact, like [1] or [2][3].
Use only plain square-bracket numbers for citations, never any other citation format.
If the passages do not contain the answer, reply exactly: "I couldn't find this in the scheme documents."
Keep the answer short and clear."""

# Used only by the evaluation's "no documents" baseline (a plain AI answering from memory).
NO_DOCUMENTS_PROMPT = "You are an assistant for Indian government schemes. Answer the question briefly."


def document_files():
    return {p.name for p in config.DATA_DIR.iterdir() if p.suffix in (".pdf", ".txt")}


_checked = False  # the "has data/ changed?" check runs once per process


def get_collection(collection_name=config.COLLECTION_NAME):
    global _checked
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    exists = collection_name in [c.name for c in client.list_collections()]
    if collection_name != config.COLLECTION_NAME:
        if not exists:
            raise ValueError(f"Collection '{collection_name}' not found. Build it with ingest.main().")
        return client.get_collection(collection_name)
    if exists and not _checked:
        stored = client.get_collection(collection_name).get(include=["metadatas"])["metadatas"]
        exists = {m["source"] for m in stored} == document_files()  # rebuild if data/ changed
    if not exists:
        ingest.main()  # first run (e.g. on the cloud server), or documents were added/removed
    _checked = True
    return client.get_collection(collection_name)


def schemes_named_in(question):
    """Documents whose scheme is named in the question (whole-word match, ignoring case)."""
    text = question.lower()
    return sorted(
        doc for doc, names in config.SCHEME_NAMES.items()
        if any(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) for name in names)
    )


def retrieve(question, top_k=config.TOP_K, collection_name=config.COLLECTION_NAME,
             scheme_aware=config.SCHEME_AWARE):
    collection = get_collection(collection_name)

    def query(n, where=None):
        results = collection.query(query_texts=[question], n_results=n, where=where)
        return [
            {"text": text, "source": meta["source"], "page": meta["page"]}
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    docs = schemes_named_in(question) if scheme_aware else []
    if not docs:
        return query(top_k)
    # The named scheme's pieces first; if it has fewer than top_k pieces, fill up from everything else.
    preferred = query(top_k, where={"source": {"$in": docs}})
    if len(preferred) >= top_k:
        return preferred
    seen = {(p["source"], p["page"], p["text"]) for p in preferred}
    others = [p for p in query(top_k * 2) if (p["source"], p["page"], p["text"]) not in seen]
    return preferred + others[: top_k - len(preferred)]


def chat_client(provider="groq"):
    if provider == "groq":
        return Groq()
    from openai import OpenAI  # only needed for evaluation (eval/requirements.txt)

    settings = config.PROVIDERS[provider]
    # Some free-tier requests never get a reply; give up after 60 s so the caller can retry.
    return OpenAI(base_url=settings["base_url"], api_key=os.environ[settings["key_env"]],
                  timeout=60, max_retries=0)


def answer(question, top_k=config.TOP_K, collection_name=config.COLLECTION_NAME,
           model=config.MODEL, provider="groq", use_documents=True,
           scheme_aware=config.SCHEME_AWARE):
    start = time.perf_counter()
    if use_documents:
        sources = retrieve(question, top_k, collection_name, scheme_aware)
        context = "\n\n".join(
            f"[{n}] ({s['source']}, page {s['page']})\n{s['text']}"
            for n, s in enumerate(sources, start=1)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
    else:
        sources = []
        messages = [
            {"role": "system", "content": NO_DOCUMENTS_PROMPT},
            {"role": "user", "content": question},
        ]

    response = chat_client(provider).chat.completions.create(
        model=model,
        reasoning_effort=config.REASONING_EFFORT,
        messages=messages,
    )

    text = response.choices[0].message.content or ""
    # gpt-oss sometimes writes citations as 【1†L2-L3】; turn them into [1].
    text = re.sub(r"【(\d+)[^】]*】", r"[\1]", text)

    return {
        "answer": text,
        "sources": sources,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "seconds": round(time.perf_counter() - start, 1),
    }
