"""SchemeSense website. Run locally with: streamlit run app.py"""

import json
import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, RateLimitError

import rag

MAX_QUESTIONS_PER_SESSION = 10
EXAMPLE_QUESTIONS = [
    "How much money does a farmer get per year under PM-Kisan?",
    "What is the age limit to join Atal Pension Yojana?",
    "How much health cover does Ayushman Bharat PM-JAY give per family?",
    "How much assistance is given to build a house under PMAY-G?",
    "How much can a street vendor borrow under PM SVANidhi?",
    "What is the age limit for PMJJBY and PMSBY?",
]

# One JSON line per question in the app's logs (Streamlit Cloud: Manage app -> Logs).
log = logging.getLogger("schemesense")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def log_event(**fields):
    log.info(json.dumps({"event": "question", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields},
                        ensure_ascii=False))


load_dotenv()  # locally: reads .env. On Streamlit Cloud the key comes from the app's Secrets.

st.set_page_config(page_title="SchemeSense", page_icon="📄")
st.title("SchemeSense")
st.write(
    "Ask about Indian government schemes. Answers come only from official government "
    "documents, with the file and page they came from."
)
st.caption(
    "Covers 15 schemes: PM-Kisan · Ayushman Bharat PM-JAY · Atal Pension Yojana · Sukanya Samriddhi · "
    "PM Awas Yojana (Gramin) · PM Ujjwala · PM Jan Dhan · PM Mudra · PM Jeevan Jyoti Bima · "
    "PM Suraksha Bima · Stand-Up India · PM Vishwakarma · PM SVANidhi · National Pension System · "
    "PM Fasal Bima"
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to .env (locally) or to the app's Secrets (Streamlit Cloud).")
    st.stop()


@st.cache_resource(show_spinner="Loading the scheme documents…")
def load_documents():
    rag.get_collection()  # builds the search database on first run


load_documents()
st.session_state.setdefault("questions_asked", 0)


def use_example(question):
    st.session_state.question = question


st.write("**Try an example:**")
columns = st.columns(2)
for i, example in enumerate(EXAMPLE_QUESTIONS):
    columns[i % 2].button(example, key=f"example_{i}", on_click=use_example, args=(example,))

question = st.text_input("Your question", key="question", placeholder="e.g. Who is eligible for PM-Kisan?")
st.caption("Please don't include personal details (name, phone, Aadhaar) in your question; questions are logged.")

if st.button("Ask", key="ask", type="primary") and question.strip():
    if st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning(
            f"You've reached the limit of {MAX_QUESTIONS_PER_SESSION} questions for this session. "
            "Please come back later."
        )
        st.stop()
    st.session_state.questions_asked += 1

    try:
        with st.spinner("Searching the documents…"):
            result = rag.answer(question)
    except RateLimitError:
        log_event(question=question[:300], status="rate_limited")
        st.warning("The AI is busy, or today's free usage limit has been reached. "
                   "Please try again in a few minutes, or tomorrow.")
        st.stop()
    except (APIConnectionError, APIStatusError) as error:
        log_event(question=question[:300], status="ai_error", error=type(error).__name__)
        st.error("Couldn't reach the AI service. Please try again in a minute.")
        st.stop()

    log_event(
        question=question[:300],
        status="answered",
        refused=rag.is_refusal(result["answer"]),
        documents=sorted({s["source"] for s in result["sources"]}),
        seconds=result["seconds"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )

    st.markdown(result["answer"])

    st.subheader("Sources")
    for n, source in enumerate(result["sources"], start=1):
        with st.expander(f"[{n}] {source['source']}, page {source['page']}"):
            st.write(source["text"])

    st.caption(
        f"Answered in {result['seconds']} s · "
        f"tokens: {result['input_tokens']} in, {result['output_tokens']} out"
    )

st.divider()
st.caption(
    "Answers come from official documents that may be out of date. "
    "Always verify on the official scheme website before acting on them."
)
