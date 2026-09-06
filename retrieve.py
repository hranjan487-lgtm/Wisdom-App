"""
Osho corpus retrieval — Step 2: query -> top relevant chunks.

Uses TF-IDF (keyword-weighted search) as a placeholder for real semantic
embeddings, since this environment can't reach embedding provider APIs or
model hubs. Swap `TfidfVectorizer` for a real embedding model + vector DB
call once this runs somewhere with full internet access — everything else
(chunk loading, top-k retrieval, printing results) stays the same shape.

Usage:
    python retrieve.py "I am jealous of my colleague" --corpus osho_corpus_master.jsonl --top-k 5
"""

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_corpus(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def retrieve(query, chunks, top_k=5):
    texts = [c["chunk_text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(texts + [query])
    query_vec = matrix[-1]
    chunk_vecs = matrix[:-1]
    scores = cosine_similarity(query_vec, chunk_vecs)[0]
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return ranked[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str)
    ap.add_argument("--corpus", type=str, default="osho_corpus_master.jsonl")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    chunks = load_corpus(args.corpus)
    print(f"Loaded {len(chunks)} chunks from {args.corpus}\n")

    results = retrieve(args.query, chunks, args.top_k)

    print(f'Query: "{args.query}"\n')
    print(f"Top {args.top_k} results:\n")
    for i, (score, chunk) in enumerate(results, 1):
        print(f"[{i}] score={score:.3f} | {chunk['book_title']} | {chunk.get('chapter_title') or 'no chapter'}")
        print(f"    {chunk['chunk_text'][:300]}...")
        print()


if __name__ == "__main__":
    main()
