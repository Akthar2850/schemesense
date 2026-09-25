"""Read the PDFs in data/, split them into chunks, and store them in Chroma."""

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


def main(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP, collection_name=config.COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)  # rebuild from scratch
    collection = client.create_collection(collection_name)

    ids, documents, metadatas = [], [], []
    for pdf in sorted(config.DATA_DIR.glob("*.pdf")):
        reader = PdfReader(pdf)
        file_chunks = 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = clean(page.extract_text() or "")
            for i, chunk in enumerate(split_into_chunks(text, chunk_size, overlap)):
                ids.append(f"{pdf.name}-p{page_number}-{i}")
                documents.append(chunk)
                metadatas.append({"source": pdf.name, "page": page_number})
                file_chunks += 1
        print(f"{pdf.name}: {len(reader.pages)} pages, {file_chunks} chunks")

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Stored {len(ids)} chunks in {config.DB_DIR.name}/ (collection '{collection_name}')")


if __name__ == "__main__":
    main()
