"""Core RAG logic: find the relevant passages, then ask the AI to answer only from them."""

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


def get_collection():
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    if config.COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        ingest.main()  # first run (e.g. on the cloud server): build the database from data/
    return client.get_collection(config.COLLECTION_NAME)


def retrieve(question):
    results = get_collection().query(query_texts=[question], n_results=config.TOP_K)
    return [
        {"text": text, "source": meta["source"], "page": meta["page"]}
        for text, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def answer(question):
    start = time.perf_counter()
    sources = retrieve(question)
    context = "\n\n".join(
        f"[{n}] ({s['source']}, page {s['page']})\n{s['text']}"
        for n, s in enumerate(sources, start=1)
    )

    response = Groq().chat.completions.create(
        model=config.MODEL,
        reasoning_effort=config.REASONING_EFFORT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    text = response.choices[0].message.content
    # gpt-oss sometimes writes citations as 【1†L2-L3】; turn them into [1].
    text = re.sub(r"【(\d+)[^】]*】", r"[\1]", text)

    return {
        "answer": text,
        "sources": sources,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "seconds": round(time.perf_counter() - start, 1),
    }
