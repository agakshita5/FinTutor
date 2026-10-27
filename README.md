# FinTutor: personal-finance Q&A with retrieval-augmented generation

Ask a personal-finance question; FinTutor finds the most similar questions in a knowledge base of
7,305 community-answered finance questions, and an LLM (gpt-oss-20b on Groq) writes a short answer **from those sources only**,
with inline citations. Questions the knowledge base can't answer are refused before any LLM call.

**[Live demo](https://agakshita-fintutor.hf.space)**

## How it works

```
question ─► embed (all-MiniLM-L6-v2) ─► ChromaDB top-3 (cosine)
                                           │
                     best similarity < 0.50? ── yes ─► "I couldn't find this" (no LLM call, $0)
                                           │ no
                                           ▼
              gpt-oss-20b via Groq: answer only from the 3 sources, cite [1] [2] [3]
                                           │
                                           ▼
                answer + sources + latency/tokens/cost   (10-minute in-memory cache)
```

- **Index built once, at build time** (`build_index.py`), persisted to `chroma_db/`. The app loads it
  in seconds instead of re-embedding the corpus on every start.
- **Corpus cleaning** (`corpus.py`): every question in the source dataset began with one of five
  instruction-tuning templates ("based on your financial expertise, provide your response…"). Embedded
  as-is, the template dominated the vectors. Stripping it and deduplicating took 18,050 rows down to
  7,305 unique questions (16,082 answers) and raised retrieval hit@3 from 82% to 98% (see below).
- **Refusal threshold**: chosen from a sweep on the eval set: in-domain questions' best match never
  scored below 0.55, out-of-domain never above 0.48, so the cutoff sits between them at 0.50.
- **Free-tier limits**: Groq's free tier allows roughly 200k tokens a day (about 135 answers at
  ~1,450 tokens each). The 10-minute cache and refusing off-topic questions before the LLM call
  both stretch that budget.
- **Frontend**: one HTML page, vanilla JS, voice input via the Web Speech API. Chat history is kept in
  `sessionStorage` (per tab); the server stores nothing about users.

## Evaluation

`eval/eval_set.json`: 40 hand-written paraphrases of randomly sampled corpus questions (the correct
retrieval target is known) and 20 off-topic questions that should be refused.

```bash
python eval/run_eval.py             # retrieval + threshold sweep, no API calls
python eval/run_eval.py --e2e       # plus answers, LLM-judged grounding, latency and cost
```

Results (full output in [`eval/results.md`](eval/results.md)):

| retrieval (top 3) | hit@1 | hit@3 | MRR |
|---|---|---|---|
| before: raw rows, template prefix kept | 82% | 82% | 0.82 |
| after: cleaned + deduplicated | 88% | 98% | 0.92 |

| end to end (openai/gpt-oss-20b) | |
|---|---|
| in-domain questions answered | 40/40 |
| answers judged fully supported by their sources | 33/40 |
| off-topic questions refused, with no LLM call | 20/20 |
| avg tokens in / out | 1,267 / 183 |
| avg cost per answered question (Groq paid-tier prices) | $0.00015 |

Latency was not measured cleanly yet: the eval run on the free tier was throttled by Groq's rate
limits. A spot check of 6 spaced-out calls took 0.6 to 1.7 s. The 7 unsupported answers mostly add
general advice that isn't in the sources, which is the next thing to fix.

Caveats: 60 questions is small, and the threshold was tuned on the same set, so treat the refusal
numbers as optimistic. Grounding is judged by the same model family that wrote the answer.

## Data

The knowledge base is a public finance Q&A dataset (community answers, largely US-centric and
from 2010s forums). Answers can be informal or out of date; the prompt tells the model to say so,
and every answer shows its sources.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add your Groq key (and the dataset's Drive file id)
python build_index.py             # downloads the dataset if needed, builds chroma_db/ (~20 s)
python app.py                     # http://localhost:5000
```

## Deploy (Hugging Face Spaces, Docker)

The `Dockerfile` installs CPU-only PyTorch, builds the index during the image build and serves with
gunicorn on port 7860. In the Space settings add `GROQ_API_KEY` as a secret, and either commit
`datasets/final_combined.csv` to the Space or add `DATASET_FILE_ID` as a secret for the build.

## API

| method | path | |
|---|---|---|
| POST | `/chat` | `{"message": "..."}` → `{response, status: answered\|no_match\|error, sources, cached, latency_ms}` |
| GET | `/health` | readiness |
| GET | `/stats` | queries served, cache size, uptime |

## Tech

Python, Flask, gunicorn, ChromaDB, sentence-transformers, Groq (gpt-oss-20b), Docker,
vanilla HTML/CSS/JS.
