# SchemeSense

An AI assistant that answers questions about Indian government schemes using only official
government documents, and shows the file and page every answer came from.

**Live demo:** https://schemesense.streamlit.app

> Status: Phase 3. Website built with Streamlit. Automated evaluation scores coming next.

Schemes covered: PM-Kisan, Ayushman Bharat PM-JAY, Atal Pension Yojana, Sukanya Samriddhi,
PM Awas Yojana (Gramin). See [data/SOURCES.md](data/SOURCES.md) for the documents used.

## Why it's needed

A plain AI model answering from memory got PM-Kisan wrong:

| | Plain AI (no documents) | SchemeSense |
|---|---|---|
| Benefit per year | ₹5,000 ❌ | ₹6,000 ✅ `pm-kisan-guidelines.pdf`, p.2 |
| Instalments | 3 × ₹1,666.67 ❌ | 3 × ₹2,000 ✅ `pm-kisan-guidelines.pdf`, p.10 |
| Out-of-scope question | Answers anyway | "I couldn't find this in the scheme documents." |

## How it works

1. **Ingest** (`ingest.py`): read each PDF page by page, split it into overlapping ~1,000-character
   chunks, and store them in a Chroma vector database with their file name and page number.
2. **Retrieve** (`rag.py`): turn the question into an embedding and find the 5 most similar chunks.
3. **Generate** (`rag.py`): send the question plus those chunks to an LLM (`openai/gpt-oss-120b`
   on Groq), instructed to answer only from them and cite them as [1], [2]…
4. **Show** (`app.py`): the answer, the cited passages, response time and token counts.

Production touches: a per-session question limit, friendly errors when the AI service is busy,
citation clean-up, and all settings in one place (`config.py`).

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt        # Linux / Mac
   .venv\Scripts\python -m pip install -r requirements.txt    # Windows
   ```
2. Get a free API key from [console.groq.com](https://console.groq.com).
3. Copy `.env.example` to `.env` and paste your key into it.

## Run

Website:
```
.venv/bin/streamlit run app.py        # Linux / Mac
.venv\Scripts\streamlit run app.py    # Windows
```

Command line:
```
.venv/bin/python ask.py "How much money does a farmer get per year under PM-Kisan?"
```

The search database is built automatically on first run. To rebuild it after changing `data/`:
```
.venv/bin/python ingest.py
```

`hello_ai.py` is the Phase 1 example: a plain AI call with no documents.
