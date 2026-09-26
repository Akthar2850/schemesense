"""Search-quality check for CI: does search find the right page for the test questions?

Uses the live settings (config.TOP_K pieces, scheme-aware search). Runs locally, no AI calls or API keys.
Exits with an error if the right-page rate drops below MIN_HIT_RATE.

Usage: python eval/ci_check.py
"""

import sys

from common import is_hit, load_questions

import config
import rag

MIN_HIT_RATE = 0.95


def main():
    questions = [q for q in load_questions() if q["sources"]]
    misses = [q["id"] for q in questions if not is_hit(rag.retrieve(q["question"]), q["sources"])]
    rate = 1 - len(misses) / len(questions)
    print(f"Right page found for {len(questions) - len(misses)} of {len(questions)} questions "
          f"({rate:.1%}) with {config.TOP_K} pieces, scheme-aware={config.SCHEME_AWARE}.")
    print(f"Missed: {', '.join(misses) or 'none'}")
    if rate < MIN_HIT_RATE:
        print(f"FAIL: below the minimum of {MIN_HIT_RATE:.0%}")
        sys.exit(1)
    print(f"PASS (minimum {MIN_HIT_RATE:.0%})")


if __name__ == "__main__":
    main()
