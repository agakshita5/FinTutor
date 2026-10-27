import os
import re
import time
from datetime import datetime, timedelta

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from build_index import EMBED_MODEL, INDEX_DIR

load_dotenv()

MODEL = "openai/gpt-oss-20b"
# USD per 1M tokens on Groq's paid tier. Used only for cost estimates.
PRICE_IN, PRICE_OUT = 0.075, 0.30

# Below this cosine similarity the closest FAQ is not about the user's question,
# so we refuse instead of letting the model answer from unrelated context.
# Picked from the threshold sweep in run_eval.py.
SIM_THRESHOLD = 0.50
TOP_K = 3

NO_MATCH = ("I couldn't find anything in my finance knowledge base that matches this question, "
            "so I'd rather not guess. Try rephrasing, or ask about budgeting, investing, taxes, "
            "loans or credit.")

SYSTEM_PROMPT = """You are FinTutor, a personal-finance assistant.
Answer ONLY from the numbered sources provided. They are community answers to similar questions,
so they can be informal, US-centric or out of date: say so when it matters.
If the sources do not answer the question, say that plainly instead of using outside knowledge.
Cite sources inline as [1], [2]. Keep the answer under 150 words and end with one practical tip
if the sources support one."""


class FinanceAIChatbot:
    def __init__(self, api_key: str, collection: str = "faqs", threshold: float = SIM_THRESHOLD):
        if not api_key:
            raise ValueError("API key is required!")
        self.client = Groq(api_key=api_key)
        self.model_name = MODEL
        self.threshold = threshold

        self.embedder = SentenceTransformer(EMBED_MODEL)
        db = chromadb.PersistentClient(path=INDEX_DIR)
        try:
            self.collection = db.get_collection(collection)
        except Exception:
            raise RuntimeError(f"Index '{collection}' not found. Run: python build_index.py")

        self.session_start = datetime.now()
        self.total_queries = 0
        self.response_cache = {}
        self.cache_duration = timedelta(minutes=10)

    def _normalize(self, text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=emb, n_results=k)
        return [
            {"question": q, "answers": m["answers"], "similarity": round(1 - d, 3)}
            for q, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def _build_prompt(self, query: str, hits: list[dict]) -> str:
        sources = "\n\n".join(
            f"[{i}] Q: {h['question']}\nA: {h['answers']}" for i, h in enumerate(hits, 1)
        )
        return f"Sources:\n{sources}\n\nUser question: {query}"

    def complete(self, system: str, user: str, max_tokens: int = 600,
                 client: Groq | None = None) -> tuple[str, int, int]:
        """One chat completion. Returns (text, input tokens, output tokens)."""
        resp = (client or self.client).chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            # gpt-oss is a reasoning model; reasoning tokens count against max_tokens and add
            # latency, and summarising three FAQs needs very little of it.
            reasoning_effort="low",
            max_tokens=max_tokens,
        )
        usage = resp.usage
        return (resp.choices[0].message.content or "").strip(), usage.prompt_tokens, usage.completion_tokens

    def get_response(self, query: str) -> dict:
        """Returns answer, sources, status (answered | no_match | error) and cost/latency stats."""
        t0 = time.perf_counter()
        key = self._normalize(query)
        cached = self.response_cache.get(key)
        if cached and datetime.now() - cached[1] < self.cache_duration:
            return {**cached[0], "cached": True, "latency_ms": round((time.perf_counter() - t0) * 1000)}

        self.total_queries += 1
        hits = [h for h in self.retrieve(query) if h["similarity"] >= self.threshold]
        if not hits:
            result = {"answer": NO_MATCH, "status": "no_match", "sources": [],
                      "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        else:
            try:
                text, tin, tout = self.complete(SYSTEM_PROMPT, self._build_prompt(query, hits))
            except Exception as e:
                # Logged, not shown: provider errors can leak request details.
                # Not cached, so the next attempt retries instead of replaying the error.
                print(f"LLM call failed: {e}")
                return {"answer": "The AI service is unavailable right now. Please try again.",
                        "status": "error", "sources": [],
                        "cached": False, "latency_ms": round((time.perf_counter() - t0) * 1000)}
            result = {
                "answer": text or "No answer generated.",
                "status": "answered",
                "sources": [{"question": h["question"], "similarity": h["similarity"]} for h in hits],
                "input_tokens": tin,
                "output_tokens": tout,
                "cost_usd": (tin * PRICE_IN + tout * PRICE_OUT) / 1_000_000,
            }

        self.response_cache[key] = (result, datetime.now())
        return {**result, "cached": False, "latency_ms": round((time.perf_counter() - t0) * 1000)}


if __name__ == "__main__":
    bot = FinanceAIChatbot(os.getenv("GROQ_API_KEY"))
    print("FinTutor ready. Type 'exit' to quit.\n")
    while (q := input("You: ").strip()).lower() not in {"exit", "quit", "bye"}:
        if q:
            r = bot.get_response(q)
            print(f"\nBot: {r['answer']}\n   [{r['status']}, {r['latency_ms']} ms]\n")
