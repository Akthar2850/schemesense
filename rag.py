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


def get_collection(collection_name=config.COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    if collection_name not in [c.name for c in client.list_collections()]:
        if collection_name != config.COLLECTION_NAME:
            raise ValueError(f"Collection '{collection_name}' not found. Build it with ingest.main().")
        ingest.main()  # first run (e.g. on the cloud server): build the database from data/
    return client.get_collection(collection_name)


def retrieve(question, top_k=config.TOP_K, collection_name=config.COLLECTION_NAME):
    results = get_collection(collection_name).query(query_texts=[question], n_results=top_k)
    return [
        {"text": text, "source": meta["source"], "page": meta["page"]}
        for text, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def chat_client(provider="groq"):
    if provider == "groq":
        return Groq()
    from openai import OpenAI  # only needed for evaluation (eval/requirements.txt)

    settings = config.PROVIDERS[provider]
    # Some free-tier requests never get a reply; give up after 60 s so the caller can retry.
    return OpenAI(base_url=settings["base_url"], api_key=os.environ[settings["key_env"]],
                  timeout=60, max_retries=0)


def answer(question, top_k=config.TOP_K, collection_name=config.COLLECTION_NAME,
           model=config.MODEL, provider="groq", use_documents=True):
    start = time.perf_counter()
    if use_documents:
        sources = retrieve(question, top_k, collection_name)
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
