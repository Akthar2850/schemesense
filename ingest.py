"""Read the PDFs in data/, split them into chunks, and store them in Chroma."""

import hashlib
import logging
import re

import chromadb
from pypdf import PdfReader

import config

logging.getLogger("pypdf").setLevel(logging.ERROR)  # hide font warnings


def clean(text):
    # Hindi text in some PDFs comes out as codes like /uni0915 or /g7407; drop them.
    text = re.sub(r"/uni[0-9A-Fa-f]{4}|/g\d+", " ", text)
    return " ".join(text.split())


def split_into_chunks(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # End at a space so words aren't cut in half.
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def fingerprint():
    """A hash of every document's name and contents: changes whenever data/ changes."""
    digest = hashlib.sha256()
    for doc in sorted(config.DATA_DIR.glob("*.pdf")) + sorted(config.DATA_DIR.glob("*.txt")):
        digest.update(doc.name.encode() + doc.read_bytes())
    return digest.hexdigest()


def main(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP, collection_name=config.COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)  # rebuild from scratch
    collection = client.create_collection(collection_name, metadata={"fingerprint": fingerprint()})

    ids, documents, metadatas = [], [], []
    for doc in sorted(config.DATA_DIR.glob("*.pdf")) + sorted(config.DATA_DIR.glob("*.txt")):
        if doc.suffix == ".pdf":
            pages = [page.extract_text() or "" for page in PdfReader(doc).pages]
        else:
            # A saved web page: sections separated by a form feed (\f) count as pages.
            pages = doc.read_text(encoding="utf-8").split("\f")
        file_chunks = 0
        for page_number, page_text in enumerate(pages, start=1):
            text = clean(page_text)
            for i, chunk in enumerate(split_into_chunks(text, chunk_size, overlap)):
                ids.append(f"{doc.name}-p{page_number}-{i}")
                documents.append(chunk)
                metadatas.append({"source": doc.name, "page": page_number})
                file_chunks += 1
        print(f"{doc.name}: {len(pages)} pages, {file_chunks} chunks")

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Stored {len(ids)} chunks in {config.DB_DIR.name}/ (collection '{collection_name}')")


if __name__ == "__main__":
    main()
