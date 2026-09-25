# Evaluation results

How well does SchemeSense answer questions about the 5 schemes, and which settings work best?

## Test set
- 45 questions written from the official PDFs (`eval/questions.json`) and reviewed by a human:
  39 answerable (facts, numbers, eligibility, tricky real-life wording) and 6 that must be refused
  (5 off-topic, 1 whose answer is only in an image).
- Each answerable question lists its **key facts**: exactly what the question asks for.

## How answers are scored
- Judge: `google/gemma-4-31b-it` (on NVIDIA), from a different company than the answer models.
- An answer is **correct** if it states every key fact and contradicts nothing in the reference answer,
  **partial** if some key facts are missing, **incorrect** otherwise. Correctness = average score
  (correct 1, partial 0.5, incorrect 0). A refusal on an answerable question counts as incorrect.
- **Right page found**: the page holding the answer is among the retrieved pieces (measured locally, no AI).

## Results on the website's model (gpt-oss-120b on Groq)
| Setting | Correct | Right page found | Off-topic refused | Speed (p50 / p95) | Cost if paid* |
|---|---|---|---|---|---|
| **8 pieces × 1,000 chars (live setting)** | **97.4%** | 97% | 100% | 1.5 s / 1.8 s | $0.33 |
| 5 pieces × 1,000 chars (previous) | 89.7% | 92% | 100% | 0.8 s / 1.1 s | $0.23 |
| No documents (plain AI) | 41.0% | – | 0% | 0.7 s / 1.6 s | $0.14 |

\*USD per 1,000 questions at Groq's published prices; the project runs on the free tier.

With 8 pieces, 38 of 39 answers were fully correct; the other was a refusal because search missed the page.
Faithfulness (every claim supported by the retrieved passages), checked on the 5-piece run by the first
judge (Qwen): 97%.

## Settings experiments (gpt-oss-20b on NVIDIA, same model throughout)
| Setting | Correct | Right page found |
|---|---|---|
| 5 pieces × 1,000 chars (baseline) | 79.5% | 92% |
| 3 pieces | 79.5% | 85% |
| 8 pieces | 85.9% | 97% |
| 5 pieces × 500 chars | 73.1% | 82% |
| 5 pieces × 1,500 chars | 88.5% | 95% |
| No documents | 11.5% | – |

More (or larger) pieces help; smaller pieces hurt. 8 pieces was then confirmed on 120b (+7.7 points) and is
the live setting. 1,500-char pieces look as good on 20b but have not been tested on 120b.

## Other comparisons (5 pieces × 1,000 chars)
| Comparison | Result |
|---|---|
| Model size: gpt-oss-120b vs gpt-oss-20b (both on Groq) | 89.7% vs 82.1% |
| Provider: gpt-oss-20b on Groq vs on NVIDIA | 82.1% vs 79.5% (about 1 question apart) |

## Checking the judge
- **Human check:** a person checked which facts 10 answers contained. The first judge (`qwen/qwen3.8-27b` on
  Groq, without key facts) matched the fair mark on 7 of 10; it was too strict about missing background
  details the questions did not ask for. With key facts and Gemma: **10 of 10**. (`eval/human_check.json`)
- **Leniency check:** on the no-documents answers, many of which are wrong, Gemma marked 21 of 39 incorrect
  with specific reasons (wrong interest rate, wrong helpline number, and so on), so it does catch mistakes.
- **Key-fact lists** were double-checked by a second model (15 by Mistral-Nemotron, 24 by Gemma when
  Mistral-Nemotron stopped responding): 36 of 39 confirmed. The 3 flagged were deliberate choices
  (only what the question asks; the reference answer still catches contradictions).
- Both judges agree on the main conclusions; Gemma scores 0–6 points higher than the stricter Qwen.
  Original Qwen scores are kept in `eval/results/<run>.json`, Gemma scores in `<run>-gemma.json`.

## Limitations
- 45 questions on 5 schemes: one question is 2.6 points, so small differences are not meaningful.
- Search depends on wording: "not covered" missed a page that "not operational" found. Hybrid
  (keyword + meaning) search or query rewriting could help.
- Some documents are summaries or older rules (see `data/SOURCES.md`); answers reflect the documents.
- NVIDIA retired gpt-oss-120b during the project, so the setting experiments ran on 20b and the winner was
  confirmed on 120b.

## Reproduce
```
pip install -r eval/requirements.txt        # and put GROQ_API_KEY and NVIDIA_API_KEY in .env
python eval/retrieval_eval.py               # search accuracy only, no AI calls
python eval/run_eval.py --name top-k-8 --provider groq --top-k 8    # answers + judge
python eval/regrade.py top-k-8              # re-grade saved answers with the current judge
```
