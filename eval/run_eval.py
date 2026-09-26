"""Run the test questions through SchemeSense and score the answers with an AI judge.

Usage examples:
  python eval/run_eval.py --name baseline --provider nvidia
  python eval/run_eval.py --name top-k-8 --provider nvidia --top-k 8
  python eval/run_eval.py --name no-rag --provider nvidia --no-rag
  python eval/run_eval.py --name groq-20b-baseline --provider groq --model openai/gpt-oss-20b
  python eval/run_eval.py --name smoke --provider nvidia --limit 5
"""

import argparse
import json
import statistics
import time
from collections import deque

from common import RESULTS_DIR, collection_for, is_hit, is_refusal, load_questions

from dotenv import load_dotenv

import config
import rag

CORRECTNESS_PROMPT = """You strictly grade answers from an assistant about Indian government schemes.
You get the QUESTION, the KEY FACTS (exactly what the question asks for), the full REFERENCE ANSWER, and the
ASSISTANT ANSWER.
1. Check each KEY FACT: it must be STATED EXPLICITLY in the assistant answer. Different wording is fine.
   Implied, vague or partial mentions do NOT count. Only the KEY FACTS are required.
2. Check for contradictions: if anything in the assistant answer contradicts the REFERENCE ANSWER (a wrong
   number, condition, reason or name), the answer is incorrect.
- "correct": every key fact is stated explicitly and nothing contradicts the reference.
- "partial": at least one key fact is stated but at least one is missing or vague, and nothing contradicts it.
- "incorrect": no key fact is stated, or something contradicts the reference. A refusal is incorrect.
Reply with JSON only: {"verdict": "correct" or "partial" or "incorrect", "missing_key_facts": [..],
"reason": "one short sentence"}"""

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

FAITHFULNESS_PROMPT = """You check whether an assistant's answer sticks to the provided passages.
Mark it faithful if every factual claim in the ANSWER is supported by the PASSAGES.
Citation numbers like [1] are fine. Ignore style. A refusal is faithful.
Reply with JSON only: {"faithful": true or false, "reason": "one short sentence"}"""


class RateGate:
    """Keeps calls under a tokens-per-minute and requests-per-minute budget."""

    def __init__(self, tokens_per_minute, requests_per_minute):
        self.tpm, self.rpm = tokens_per_minute, requests_per_minute
        self.calls = deque()  # (time, tokens)

    def wait(self, estimated_tokens):
        while True:
            now = time.time()
            while self.calls and now - self.calls[0][0] > 60:
                self.calls.popleft()
            used = sum(t for _, t in self.calls)
            if len(self.calls) < self.rpm and used + estimated_tokens <= self.tpm:
                return
            time.sleep(1)

    def record(self, tokens):
        self.calls.append((time.time(), tokens))


def with_retries(call):
    """Retry on rate limits (429) and temporary server errors, waiting as long as the API asks."""
    for attempt in range(8):
        try:
            return call()
        except Exception as error:
            status = getattr(error, "status_code", None)
            if type(error).__name__ in ("APITimeoutError", "APIConnectionError"):
                status = "timeout"  # no reply in time; try again
            if status not in (429, 500, 502, 503, 504, 529, "timeout") or attempt == 7:
                raise
            headers = getattr(getattr(error, "response", None), "headers", {}) or {}
            wait = float(headers.get("retry-after", 0) or 0) or min(10 * (attempt + 1), 60)
            print(f"    (API busy, status {status}; waiting {wait:.0f}s)")
            time.sleep(wait)


def parse_json(text):
    """Read a JSON reply, allowing a ```json ... ``` wrapper around it."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": text}


class Judge:
    def __init__(self, provider=config.JUDGE_PROVIDER, model=config.JUDGE_MODEL):
        self.provider, self.model = provider, model
        self.client = rag.chat_client(provider)
        # Groq free tier: 8K tokens/min per model; NVIDIA free tier: about 40 requests/min.
        self.gate = RateGate(6500, 25) if provider == "groq" else RateGate(10**9, 35)
        self.tokens = 0

    def ask(self, system, user, estimated_tokens):
        self.gate.wait(estimated_tokens)
        extra = {}
        if self.provider == "groq":  # Qwen on Groq: instruct mode (no thinking tokens), JSON mode
            extra = {"reasoning_effort": "none", "response_format": {"type": "json_object"}}
        response = with_retries(lambda: self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=400,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        ))
        used = response.usage.total_tokens
        self.gate.record(used)
        self.tokens += used
        return parse_json(response.choices[0].message.content)

    def correctness(self, question, key_facts, expected_answer, answer):
        """Returns (verdict, score, reason); verdict is None if the reply couldn't be read."""
        user = (f"QUESTION: {question}\nKEY FACTS: {json.dumps(key_facts, ensure_ascii=False)}\n"
                f"REFERENCE ANSWER: {expected_answer}\nASSISTANT ANSWER: {answer}")
        grade = self.ask(CORRECTNESS_PROMPT, user, 700)
        verdict = grade.get("verdict")
        if verdict not in SCORES:
            return None, None, grade.get("error") or f"unreadable verdict: {grade}"
        reason = grade.get("reason") or ""
        if grade.get("missing_key_facts"):
            reason += f" (missing: {', '.join(map(str, grade['missing_key_facts']))})"
        return verdict, SCORES[verdict], reason

    def faithfulness(self, answer, sources):
        passages = "\n\n".join(f"[{n}] {s['text']}" for n, s in enumerate(sources, start=1))
        return self.ask(FAITHFULNESS_PROMPT, f"PASSAGES:\n{passages}\n\nANSWER: {answer}", 2000)


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def summarize(records, use_documents, model):
    answerable = [r for r in records if r["type"] not in ("out_of_scope", "known_gap")]
    refusals = [r for r in records if r["type"] in ("out_of_scope", "known_gap")]
    graded = [r for r in answerable if r.get("score") is not None]
    summary = {
        "questions": len(records),
        # correctness = average score (correct 1, partial 0.5, incorrect 0)
        "correctness": round(sum(r["score"] for r in graded) / len(graded), 3) if graded else None,
        "verdicts": {v: sum(r.get("verdict") == v for r in graded) for v in SCORES},
        "refusal_accuracy": round(sum(r["refused"] for r in refusals) / len(refusals), 3) if refusals else None,
        "latency_p50_s": percentile([r["seconds"] for r in records], 50),
        "latency_p95_s": percentile([r["seconds"] for r in records], 95),
        "avg_input_tokens": round(statistics.mean(r["input_tokens"] for r in records)),
        "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in records)),
    }
    if use_documents:
        hits = [r["hit"] for r in answerable]
        summary["hit_rate"] = round(sum(hits) / len(hits), 3)
    faithful = [r for r in answerable if r.get("faithful") is not None]
    if faithful:
        summary["faithfulness"] = round(sum(r["faithful"] for r in faithful) / len(faithful), 3)
    if model in config.PRICES:
        price_in, price_out = config.PRICES[model]
        cost = (summary["avg_input_tokens"] * price_in + summary["avg_output_tokens"] * price_out) / 1e6
        summary["est_cost_per_1000_questions_usd"] = round(cost * 1000, 3)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--provider", choices=sorted(config.PROVIDERS), default="groq")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--no-rag", action="store_true", help="plain AI with no documents")
    parser.add_argument("--faithfulness", action="store_true", help="also judge faithfulness")
    parser.add_argument("--no-scheme-aware", action="store_true", help="plain search, ignore scheme names")
    parser.add_argument("--limit", type=int, help="only the first N questions (for a smoke test)")
    parser.add_argument("--grade-later", action="store_true",
                        help="only collect answers now; grade them later with eval/regrade.py --missing-only")
    parser.add_argument("--fresh", action="store_true",
                        help="start over instead of continuing an interrupted run with the same name")
    parser.add_argument("--max-tokens", type=int, default=150_000,
                        help="stop before the answer model uses more than this many tokens")
    args = parser.parse_args()
    load_dotenv(config.BASE_DIR / ".env")

    questions = load_questions()[: args.limit] if args.limit else load_questions()
    collection = None if args.no_rag else collection_for(args.chunk_size)
    # Groq free tier: 8K tokens/min and 30 req/min per model. NVIDIA free tier: about 40 req/min.
    gate = RateGate(6500, 25) if args.provider == "groq" else RateGate(10**9, 35)
    judge = Judge()
    records, answer_tokens = [], 0
    previous = RESULTS_DIR / f"{args.name}.json"
    if previous.exists() and not args.fresh:  # continue an interrupted run with the same settings
        saved = json.loads(previous.read_text())
        if saved.get("config") == run_config(args):
            records = saved["records"]
            answer_tokens = saved["summary"].get("answer_tokens_used") or 0
            judge.tokens = saved["summary"].get("judge_tokens_used") or 0
            print(f"Resuming '{args.name}': {len(records)} questions already done.")
    done = {r["id"] for r in records}

    for i, q in enumerate(questions, start=1):
        if q["id"] in done:
            continue
        if answer_tokens >= args.max_tokens:
            print(f"Stopping early: reached --max-tokens ({args.max_tokens}).")
            break
        gate.wait(1500)
        result = with_retries(lambda: rag.answer(
            q["question"], args.top_k, collection or config.COLLECTION_NAME,
            args.model, args.provider, use_documents=not args.no_rag,
            scheme_aware=not args.no_scheme_aware))
        used = result["input_tokens"] + result["output_tokens"]
        gate.record(used)
        answer_tokens += used

        record = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "expected_answer": q["expected_answer"], "answer": result["answer"],
            "sources": [{"source": s["source"], "page": s["page"]} for s in result["sources"]],
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            "seconds": result["seconds"], "refused": is_refusal(result["answer"]),
        }
        if q["type"] in ("out_of_scope", "known_gap"):
            verdict = "refused" if record["refused"] else "answered (should refuse)"
        else:
            record["hit"] = is_hit(result["sources"], q["sources"]) if not args.no_rag else None
            if record["refused"]:
                record["verdict"], record["score"], record["judge_reason"] = "incorrect", 0.0, "refused to answer"
            else:
                if args.grade_later:
                    record["verdict"], record["score"] = None, None
                    record["judge_reason"] = "not graded yet; run eval/regrade.py --missing-only"
                else:
                  try:
                    record["verdict"], record["score"], record["judge_reason"] = judge.correctness(
                        q["question"], q["key_facts"], q["expected_answer"], result["answer"])
                  except Exception as error:  # e.g. judge's daily limit reached; keep the answer, grade later
                    record["verdict"], record["score"] = None, None
                    record["judge_reason"] = f"not graded yet ({type(error).__name__}); run eval/regrade.py"
            if args.faithfulness and not args.no_rag:
                check = judge.faithfulness(result["answer"], result["sources"])
                record["faithful"] = bool(check.get("faithful")) if "error" not in check else None
                record["faithful_reason"] = check.get("reason") or check.get("error")
            verdict = record["verdict"] or "not graded yet"
        print(f"[{i}/{len(questions)}] {q['id']}: {verdict}")
        records.append(record)
        output = save(args, records, answer_tokens, judge.tokens)  # after every question, so nothing is lost

    print("\n" + json.dumps(output["config"]))
    print(json.dumps(output["summary"], indent=2))


def run_config(args):
    return {"provider": args.provider, "model": args.model, "top_k": args.top_k,
            "chunk_size": args.chunk_size, "use_documents": not args.no_rag,
            "scheme_aware": not args.no_scheme_aware,
            "judge_model": config.JUDGE_MODEL, "judge_provider": config.JUDGE_PROVIDER}


def save(args, records, answer_tokens, judge_tokens):
    summary = summarize(records, not args.no_rag, args.model)
    summary.update({"answer_tokens_used": answer_tokens, "judge_tokens_used": judge_tokens})
    output = {
        "name": args.name,
        "config": run_config(args),
        "summary": summary,
        "records": records,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    main()
