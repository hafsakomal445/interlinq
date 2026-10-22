"""Interlinq indexer: embed pages, build a FAISS index, find related pages.

Run after crawler.py has created pages.json:
    python indexer.py
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, runs on your CPU
TEXT_CHARS = 1500                # how much of each page to embed
TOP_K = 3                        # related pages to show per page


def load_pages(path="pages.json"):
    file = Path(path)
    if not file.exists():
        raise SystemExit(f"{path} not found. Run crawler.py first.")
    pages = json.loads(file.read_text(encoding="utf-8"))
    if len(pages) < 2:
        raise SystemExit("Need at least 2 pages to find related pages.")
    return pages


def build_index(pages):
    model = SentenceTransformer(MODEL_NAME)
    docs = [f"{p['title']}. {p['text'][:TEXT_CHARS]}" for p in pages]
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine on normalized vectors
    index.add(emb)
    return index, emb


def related_pages(index, emb, pages, i, k=TOP_K):
    """Return the k pages most similar to page i, excluding itself."""
    scores, ids = index.search(emb[i : i + 1], k + 1)
    results = []
    for score, j in zip(scores[0], ids[0]):
        if j == i or j == -1:
            continue
        results.append((pages[j]["title"], pages[j]["url"], float(score)))
    return results[:k]


if __name__ == "__main__":
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")
    index, emb = build_index(pages)
    faiss.write_index(index, "index.faiss")

    for i, page in enumerate(pages):
        print(f"\n{page['title']}")
        for title, url, score in related_pages(index, emb, pages, i):
            print(f"   -> {title}  (similarity {score:.2f})")