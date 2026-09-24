"""Answer a question from the command line: python ask.py "your question" """

import os
import sys

from dotenv import load_dotenv

import rag


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python ask.py "your question"')
    question = " ".join(sys.argv[1:])

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not found. Copy .env.example to .env and paste your key.")

    result = rag.answer(question)

    print("Answer:", result["answer"])
    print("\nSources:")
    for n, s in enumerate(result["sources"], start=1):
        print(f"  [{n}] {s['source']}, page {s['page']}")
    print(f"\nTokens: input {result['input_tokens']}, output {result['output_tokens']}")
    print(f"Time: {result['seconds']} s")


if __name__ == "__main__":
    main()
