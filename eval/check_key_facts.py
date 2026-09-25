"""Ask an independent model whether each question's key facts cover exactly what the question asks.

Usage: python eval/check_key_facts.py [model]     (flags questions for a human to review)
Results are saved after every question; re-running skips questions already checked.
"""

import json
import sys

from common import RESULTS_DIR, load_questions

from dotenv import load_dotenv

import config
from run_eval import Judge

PROMPT = """You review a test set for an assistant about Indian government schemes.
Each item has a QUESTION, the full REFERENCE ANSWER, and KEY FACTS: the facts an answer must state to count
as correct. Key facts must cover everything the QUESTION explicitly asks for, and nothing it does not ask for.
Reply with JSON only: {"covers_question": true or false, "too_strict": true or false,
"comment": "one short sentence: what is missing, or what is required but not asked"}"""

OUTPUT = RESULTS_DIR / "key_facts_check.json"


def main():
    load_dotenv(config.BASE_DIR / ".env")
    model = sys.argv[1] if len(sys.argv) > 1 else config.KEY_FACTS_CHECK_MODEL
    checker = Judge(provider="nvidia", model=model)
    results = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else []
    done = {r["id"] for r in results}

    for q in load_questions():
        if not q["key_facts"] or q["id"] in done:
            continue
        user = (f"QUESTION: {q['question']}\nREFERENCE ANSWER: {q['expected_answer']}\n"
                f"KEY FACTS: {json.dumps(q['key_facts'], ensure_ascii=False)}")
        reply = checker.ask(PROMPT, user, 500)
        flagged = reply.get("covers_question") is not True or reply.get("too_strict") is True
        results.append({"id": q["id"], "key_facts": q["key_facts"], "flagged": flagged,
                        "checked_by": model, **reply})
        OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))  # save as we go
        print(f"{'FLAG' if flagged else 'ok  '} {q['id']}: {reply.get('comment') or reply.get('error')}")

    print(f"\n{sum(r['flagged'] for r in results)} of {len(results)} flagged for human review")


if __name__ == "__main__":
    main()
