# SchemeSense

[![Search quality check](https://github.com/Akthar2850/schemesense/actions/workflows/tests.yml/badge.svg)](https://github.com/Akthar2850/schemesense/actions/workflows/tests.yml)

An AI assistant that answers questions about Indian government schemes using only official
government documents, and shows the file and page every answer came from.

**Live demo:** https://schemesense.streamlit.app

> Status: Phase 4b. 15 schemes, evaluated on 93 test questions: 93.7% correct, 100% of off-topic questions refused.

Schemes covered (15): PM-Kisan, Ayushman Bharat PM-JAY, Atal Pension Yojana, Sukanya Samriddhi,
PM Awas Yojana (Gramin), PM Ujjwala, PM Jan Dhan, PM Mudra, PM Jeevan Jyoti Bima, PM Suraksha Bima,
Stand-Up India, PM Vishwakarma, PM SVANidhi, National Pension System, PM Fasal Bima.
See [data/SOURCES.md](data/SOURCES.md) for the documents used.

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
2. **Retrieve** (`rag.py`): turn the question into an embedding and find the 8 most similar chunks.
   If the question names a scheme, search that scheme's documents first (scheme-aware search), so
   similar schemes don't crowd out the right page.
3. **Generate** (`rag.py`): send the question plus those chunks to an LLM (`openai/gpt-oss-120b`
   on Groq), instructed to answer only from them and cite them as [1], [2]…
4. **Show** (`app.py`): the answer, the cited passages, response time and token counts.

Production touches: a per-session question limit, friendly errors when the AI service is busy,
citation clean-up, one JSON log line per question (sources used, speed, tokens), a search-quality check
that runs on every push (GitHub Actions, `eval/ci_check.py`), and all settings in one place (`config.py`).

## Evaluation

93 test questions written from the official documents (87 answerable, 6 that must be refused), graded
by an independent AI judge that was checked against a human reviewer (10 of 10 agreed).

| gpt-oss-120b | Correct | Off-topic refused |
|---|---|---|
| **SchemeSense, 15 schemes (live)** | **93.7%** | 100% |
| SchemeSense, 5 schemes, 8 pieces | 97.4% | 100% |
| SchemeSense, 5 schemes, 5 pieces | 89.7% | 100% |
| Same AI without documents (5 schemes) | 41.0% | 0% |

On the original 39 questions, going from 5 to 15 schemes kept accuracy at 98.7% thanks to scheme-aware search.

Details, experiments and limitations: [eval/RESULTS.md](eval/RESULTS.md).

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
