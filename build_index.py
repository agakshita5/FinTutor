"""Embed the FAQ corpus once and save it to disk, so the app starts in seconds
instead of re-embedding everything on every boot.

    python build_index.py          # cleaned corpus -> collection "faqs"
    python build_index.py --raw    # original prefixed rows -> "faqs_raw" (baseline for eval)
"""
import argparse
import time

import chromadb
from sentence_transformers import SentenceTransformer

from corpus import load_corpus
from download_ds import DATASET_PATH, download_dataset

INDEX_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_ANSWERS = 3          # answers kept per question
MAX_ANSWER_CHARS = 1500  # some answers are 16k chars; the prompt doesn't need all of that
BATCH = 1000


def build(raw: bool = False):
    if not download_dataset():
        raise SystemExit("Dataset missing and could not be downloaded.")

    if raw:
        # One document per CSV row, instruction prefix left in: what the app did before.
        df = load_corpus(DATASET_PATH, clean=False)
        name = "faqs_raw"
        questions = df["question"].tolist()
        answers = [[a] for a in df["answer"]]
    else:
        df = load_corpus(DATASET_PATH)
        name = "faqs"
        grouped = df.groupby("question", sort=False)["answer"].apply(list)
        questions = grouped.index.tolist()
        answers = grouped.tolist()

    client = chromadb.PersistentClient(path=INDEX_DIR)
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    # Cosine distance, so similarity = 1 - distance is in a range we can threshold on.
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    t0 = time.time()
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(questions, batch_size=128, show_progress_bar=True,
                                 normalize_embeddings=True)

    for i in range(0, len(questions), BATCH):
        collection.add(
            ids=[str(j) for j in range(i, min(i + BATCH, len(questions)))],
            documents=questions[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH].tolist(),
            metadatas=[
                {"answers": "\n---\n".join(a[:MAX_ANSWER_CHARS] for a in ans[:MAX_ANSWERS])}
                for ans in answers[i:i + BATCH]
            ],
        )
    print(f"Indexed {len(questions)} questions into '{name}' in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="build the uncleaned baseline index")
    build(parser.parse_args().raw)