# FinTutor eval (40 in-domain paraphrases, 20 out-of-domain questions)

## Retrieval (top 3)

| index | hit@1 | hit@3 | MRR |
|---|---|---|---|
| raw rows, instruction prefix kept (before) | 82% | 82% | 0.82 |
| cleaned + deduplicated (after) | 88% | 98% | 0.92 |

## Refusal threshold sweep (cosine similarity of the top match)

| threshold | in-domain answered | out-of-domain refused |
|---|---|---|
| 0.40 | 100% | 80% |
| 0.45 | 100% | 85% |
| 0.50 ← current | 100% | 100% |
| 0.55 | 100% | 100% |
| 0.60 | 98% | 100% |
| 0.65 | 95% | 100% |

Top-1 similarity, in-domain: min 0.55, median 0.80. Out-of-domain: median 0.30, max 0.48.

## End to end (openai/gpt-oss-20b, threshold 0.5)

| metric | value |
|---|---|
| in-domain questions answered | 40/40 |
| answers judged fully supported by sources | 33/40 |
| out-of-domain refused (no LLM call made) | 20/20 |
| errors | 0 |
| latency p50 / p95 (answered) | 10100 ms / 15136 ms (inflated: includes the SDK silently retrying free-tier rate limits; a spaced spot check of 6 calls took 0.6 to 1.7 s) |
| avg tokens in / out | 1267 / 183 |
| avg cost per answered question | $0.00015 |
| cost per 1,000 questions at this mix | $0.10 |
