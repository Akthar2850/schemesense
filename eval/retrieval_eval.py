"""Search accuracy (hit rate@k) for different settings. Runs locally and uses no AI tokens.

Usage: python eval/retrieval_eval.py [output-name]     (default: retrieval)
"""

import json
import sys

from common import RESULTS_DIR, collection_for, is_hit, load_questions

import rag

CHUNK_SIZES = [500, 1000, 1500]
TOP_KS = [3, 5, 8, 10]


def main():
    questions = [q for q in load_questions() if q["sources"]]  # only questions with a known page
    results = {}
    for chunk_size in CHUNK_SIZES:
        name = collection_for(chunk_size)
        hits = {k: 0 for k in TOP_KS}
        misses = []
        for q in questions:
            sources = rag.retrieve(q["question"], max(TOP_KS), name)
            for k in TOP_KS:
                hits[k] += is_hit(sources[:k], q["sources"])
            if not is_hit(sources[:5], q["sources"]):
                misses.append(q["id"])
        results[chunk_size] = {
            "hit_rate": {k: round(hits[k] / len(questions), 3) for k in TOP_KS},
            "missed_at_5": misses,
        }

    print(f"\nSearch accuracy on {len(questions)} questions (right page among the top k pieces)\n")
    print("chunk size | " + " | ".join(f"hit@{k}" for k in TOP_KS))
    for chunk_size, r in results.items():
        print(f"{chunk_size:>10} | " + " | ".join(f"{r['hit_rate'][k]:>5.0%}" for k in TOP_KS))
    for chunk_size, r in results.items():
        print(f"missed at k=5, chunk {chunk_size}: {', '.join(r['missed_at_5']) or 'none'}")

    RESULTS_DIR.mkdir(exist_ok=True)
    name = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
