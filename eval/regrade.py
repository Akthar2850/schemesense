"""Re-grade saved answers with the current judge (uses judge calls only, no new answers).

Usage: python eval/regrade.py baseline                         # re-grade every answer, overwrite the file
       python eval/regrade.py baseline --missing-only          # only answers not graded yet
       python eval/regrade.py baseline --save-as baseline-v2   # keep the original, write a new file
Progress is saved after every answer; re-running with --save-as continues where it stopped.
"""

import argparse
import json

from common import RESULTS_DIR, load_questions

from dotenv import load_dotenv

import config
from run_eval import Judge, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--save-as")
    args = parser.parse_args()
    load_dotenv(config.BASE_DIR / ".env")

    out_path = RESULTS_DIR / f"{args.save_as or args.name}.json"
    data = json.loads((RESULTS_DIR / f"{args.name}.json").read_text())
    judge = Judge()
    if args.save_as and out_path.exists():  # resume an interrupted re-grade
        data = json.loads(out_path.read_text())
    else:
        for r in data["records"]:
            if not args.missing_only:
                r["regraded_by"] = None
    data["config"].update({"judge_model": judge.model, "judge_provider": judge.provider})
    if args.save_as:
        data["name"] = args.save_as
    questions = {q["id"]: q for q in load_questions()}

    def save():
        summary = summarize(data["records"], data["config"]["use_documents"], data["config"]["model"])
        summary["answer_tokens_used"] = data["summary"].get("answer_tokens_used")
        data["summary"] = summary
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    for r in data["records"]:
        if r["type"] in ("out_of_scope", "known_gap") or r["refused"]:
            continue
        if args.missing_only and r.get("verdict") is not None:
            continue
        if r.get("regraded_by") == judge.model:
            continue  # already done in an earlier, interrupted run
        q = questions[r["id"]]
        r["verdict"], r["score"], r["judge_reason"] = judge.correctness(
            q["question"], q["key_facts"], q["expected_answer"], r["answer"])
        r["regraded_by"] = judge.model
        r.pop("correct", None)  # from the older right/wrong grading
        save()
        print(f"{r['id']}: {r['verdict']} ({r['judge_reason']})")

    save()
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
