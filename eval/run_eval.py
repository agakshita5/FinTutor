"""Evaluate FinTutor's retrieval and answers against eval/eval_set.json.

    python eval/run_eval.py          # retrieval + threshold sweep (free, no LLM calls)
    python eval/run_eval.py --e2e    # also run every question through the LLM (~40 calls + judge)

Writes the tables to eval/results.md.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from corpus import strip_prefix  # noqa: E402
from finance_ai_chatbot import SIM_THRESHOLD, FinanceAIChatbot  # noqa: E402

EVAL = json.loads(Path("eval/eval_set.json").read_text())
K = 3
out = []
GAP_S = 0.0  # seconds between LLM calls; set by --gap for rate-limited (free tier) keys
_last_call = [0.0]


def pace():
    """Sleep outside the timed section so rate-limit pacing doesn't count as latency."""
    wait = GAP_S - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def emit(line=""):
    print(line)
    out.append(line)


def retrieval_metrics(bot, collection):
    bot.collection = bot.client_db.get_collection(collection)
    hit1 = hit3 = rr = 0
    for case in EVAL["in_domain"]:
        hits = [strip_prefix(h["question"]) for h in bot.retrieve(case["query"], k=K)]
        rank = hits.index(case["target"]) + 1 if case["target"] in hits else None
        hit1 += rank == 1
        hit3 += rank is not None
        rr += 1 / rank if rank else 0
    n = len(EVAL["in_domain"])
    return hit1 / n, hit3 / n, rr / n


def threshold_sweep(bot):
    bot.collection = bot.client_db.get_collection("faqs")
    top_in = [bot.retrieve(c["query"], k=1)[0]["similarity"] for c in EVAL["in_domain"]]
    top_out = [bot.retrieve(q, k=1)[0]["similarity"] for q in EVAL["out_of_domain"]]
    emit("| threshold | in-domain answered | out-of-domain refused |")
    emit("|---|---|---|")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        ans = sum(s >= t for s in top_in) / len(top_in)
        ref = sum(s < t for s in top_out) / len(top_out)
        mark = " ← current" if abs(t - SIM_THRESHOLD) < 1e-9 else ""
        emit(f"| {t:.2f}{mark} | {ans:.0%} | {ref:.0%} |")
    emit()
    emit(f"Top-1 similarity, in-domain: min {min(top_in):.2f}, median {statistics.median(top_in):.2f}. "
         f"Out-of-domain: median {statistics.median(top_out):.2f}, max {max(top_out):.2f}.")


JUDGE_SYSTEM = ("You grade answers from a retrieval-augmented chatbot. Decide whether every factual "
                "claim in the answer is supported by the sources. Reply with exactly one word: "
                "SUPPORTED or UNSUPPORTED.")


def judge(bot, query, answer):
    pace()
    hits = [h for h in bot.retrieve(query) if h["similarity"] >= bot.threshold]
    sources = "\n\n".join(f"[{i}] {h['question']}\n{h['answers']}" for i, h in enumerate(hits, 1))
    verdict, _, _ = bot.complete(JUDGE_SYSTEM, f"Sources:\n{sources}\n\nAnswer:\n{answer}",
                                 client=bot.judge_client)
    return verdict.strip().upper().startswith("SUPPORTED")


def end_to_end(bot):
    bot.collection = bot.client_db.get_collection("faqs")
    rows = []
    for case in EVAL["in_domain"]:
        pace()
        rows.append(("in", case["query"], bot.get_response(case["query"])))
    for q in EVAL["out_of_domain"]:
        rows.append(("out", q, bot.get_response(q)))

    errors = [r for _, _, r in rows if r["status"] == "error"]
    called = [r for _, _, r in rows if r["status"] == "answered"]
    in_answered = [(q, r) for kind, q, r in rows if kind == "in" and r["status"] == "answered"]
    out_refused = sum(r["status"] == "no_match" for kind, _, r in rows if kind == "out")
    lat = sorted(r["latency_ms"] for r in called)
    verdicts = []
    for q, r in in_answered:
        try:
            verdicts.append(judge(bot, q, r["answer"]))
        except Exception as e:  # e.g. daily token limit: keep the results gathered so far
            print(f"Judge failed: {e}")
    supported = sum(verdicts)

    emit("| metric | value |")
    emit("|---|---|")
    emit(f"| in-domain questions answered | {len(in_answered)}/{len(EVAL['in_domain'])} |")
    emit(f"| answers judged fully supported by sources | {supported}/{len(verdicts)} judged |")
    emit(f"| out-of-domain refused (no LLM call made) | {out_refused}/{len(EVAL['out_of_domain'])} |")
    emit(f"| errors | {len(errors)} |")
    emit(f"| latency p50 / p95 (answered) | {lat[len(lat) // 2]} ms / {lat[int(len(lat) * 0.95) - 1]} ms |")
    emit(f"| avg tokens in / out | {statistics.mean(r['input_tokens'] for r in called):.0f} / "
         f"{statistics.mean(r['output_tokens'] for r in called):.0f} |")
    emit(f"| avg cost per answered question | ${statistics.mean(r['cost_usd'] for r in called):.5f} |")
    emit(f"| cost per 1,000 questions at this mix | "
         f"${sum(r['cost_usd'] for _, _, r in rows) / len(rows) * 1000:.2f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true", help="also call the LLM for every question")
    parser.add_argument("--gap", type=float, default=0.0,
                        help="seconds between LLM calls, to stay under a free-tier rate limit")
    args = parser.parse_args()
    GAP_S = args.gap

    import chromadb
    from build_index import INDEX_DIR

    bot = FinanceAIChatbot(os.getenv("GROQ_API_KEY"))
    # The SDK silently retries 429s with backoff, which would show up here as latency.
    # Off for the answers we time, so rate limiting is counted as errors instead;
    # the judge keeps retries because its latency isn't measured.
    bot.judge_client = bot.client.with_options(max_retries=5)
    bot.client = bot.client.with_options(max_retries=0)
    bot.client_db = chromadb.PersistentClient(path=INDEX_DIR)

    n_in, n_out = len(EVAL["in_domain"]), len(EVAL["out_of_domain"])
    emit(f"# FinTutor eval ({n_in} in-domain paraphrases, {n_out} out-of-domain questions)\n")
    emit(f"## Retrieval (top {K})\n")
    emit("| index | hit@1 | hit@3 | MRR |")
    emit("|---|---|---|---|")
    for name, label in [("faqs_raw", "raw rows, instruction prefix kept (before)"),
                        ("faqs", "cleaned + deduplicated (after)")]:
        try:
            h1, h3, mrr = retrieval_metrics(bot, name)
            emit(f"| {label} | {h1:.0%} | {h3:.0%} | {mrr:.2f} |")
        except Exception:
            emit(f"| {label} | not built (run build_index.py{' --raw' if name == 'faqs_raw' else ''}) | | |")
    emit("\n## Refusal threshold sweep (cosine similarity of the top match)\n")
    threshold_sweep(bot)
    if args.e2e:
        emit(f"\n## End to end ({bot.model_name}, threshold {bot.threshold})\n")
        end_to_end(bot)

    Path("eval/results.md").write_text("\n".join(out) + "\n")
