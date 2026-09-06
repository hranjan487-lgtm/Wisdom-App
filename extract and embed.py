"""
Osho corpus pipeline — full run: PDFs -> cleaned chunks -> Voyage embeddings.

Meant to run inside GitHub Actions (or any environment with real internet
access), reading all PDFs from a folder and producing one JSONL file with
every chunk plus its embedding vector.

Usage:
    python extract_and_embed.py --pdf-dir pdfs --out corpus_embeddings.jsonl

Requires the VOYAGE_API_KEY environment variable to be set (GitHub Actions
secret, passed in via the workflow file).
"""

import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import voyageai

CHAPTER_PATTERN = re.compile(r"^(CHAPTER|DISCOURSE)\s*\d+\.?\s*(.*)$", re.IGNORECASE)
FUSION_PATTERN = re.compile(r"\d[a-zA-Z]{3,}")

EMBEDDING_MODEL = "voyage-4-lite"  # good quality/cost balance for this corpus size
EMBED_BATCH_SIZE = 128  # chunks per API call


# ---------- extraction (same logic as osho_pdf_extractor.py) ----------

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append((i + 1, text))
    doc.close()
    return pages


def find_repeated_lines(pages, min_repeat_ratio=0.4):
    line_counts = Counter()
    total_pages = len(pages)
    for _, text in pages:
        lines = {ln.strip() for ln in text.split("\n") if ln.strip()}
        for ln in lines:
            if re.fullmatch(r"\d+", ln):
                continue
            if CHAPTER_PATTERN.match(ln):
                continue
            line_counts[ln] += 1
    return {
        ln for ln, count in line_counts.items()
        if count >= max(3, int(total_pages * min_repeat_ratio))
    }


def clean_page_text(text, repeated_lines):
    text = unicodedata.normalize("NFKC", text)
    lines = text.split("\n")
    cleaned_lines = []
    chapter_titles_found = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        m = CHAPTER_PATTERN.match(stripped)
        if m:
            chapter_titles_found.append((stripped, m.group(2).strip()))
            continue
        if stripped in repeated_lines:
            continue
        if re.fullmatch(r"\d+", stripped):
            continue
        cleaned_lines.append(stripped)
    joined = " ".join(cleaned_lines)
    joined = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined, chapter_titles_found


def chunk_words(words, min_words=150, max_words=400):
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + max_words, n)
        if end < n:
            extra = 0
            while end + extra < n and extra < 60 and not words[end + extra - 1].endswith((".", "!", "?", '."', '?"')):
                extra += 1
            end += extra
        chunks.append(words[start:end])
        start = end
    if len(chunks) > 1 and len(chunks[-1]) < min_words // 2:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()
    return chunks


def flag_fusions(text):
    return FUSION_PATTERN.findall(text)


def process_book(pdf_path, book_title):
    """Extract and chunk a single book. Returns a list of chunk dicts
    (without embeddings yet)."""
    pages = extract_pages(str(pdf_path))
    repeated_lines = find_repeated_lines(pages)

    all_words = []
    word_page = []
    word_chapter = []
    current_chapter = None

    for page_num, raw_text in pages:
        cleaned, chapter_titles = clean_page_text(raw_text, repeated_lines)
        for full_line, title in chapter_titles:
            title = title.rstrip(".")
            if title:  # ignore bare running-header markers with no title text
                number_match = re.search(r"\d+", full_line)
                number = number_match.group(0) if number_match else ""
                normalized = f"CHAPTER {number} {title}".strip()
                if normalized != current_chapter:
                    current_chapter = normalized
        if not cleaned:
            continue
        for w in cleaned.split(" "):
            if w:
                all_words.append(w)
                word_page.append(page_num)
                word_chapter.append(current_chapter)

    word_chunks = chunk_words(all_words)

    chunks = []
    word_index = 0
    for i, wc in enumerate(word_chunks, 1):
        start_page = word_page[word_index] if word_index < len(word_page) else None
        end_page = word_page[word_index + len(wc) - 1] if word_index + len(wc) - 1 < len(word_page) else start_page
        chapter_title = word_chapter[word_index] if word_index < len(word_chapter) else None
        chunk_text_str = " ".join(wc)
        fusions = flag_fusions(chunk_text_str)
        chunks.append({
            "chunk_id": f"{pdf_path.stem}_{i:04d}",
            "book_title": book_title,
            "chapter_title": chapter_title,
            "page_start": start_page,
            "page_end": end_page,
            "chunk_text": chunk_text_str,
            "word_count": len(wc),
            "needs_review": bool(fusions),
            "flagged_tokens": fusions,
        })
        word_index += len(wc)

    return chunks


# ---------- embedding ----------

def embed_chunks(chunks, client):
    """Embed all chunks in batches, adding an 'embedding' field to each."""
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]
        for attempt in range(3):
            try:
                result = client.embed(texts, model=EMBEDDING_MODEL, input_type="document")
                for c, vec in zip(batch, result.embeddings):
                    c["embedding"] = vec
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  WARNING: embedding failed for batch at {batch_start}: {e}")
                    for c in batch:
                        c["embedding"] = None
                else:
                    time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=str, default="pdfs")
    ap.add_argument("--out", type=str, default="corpus_embeddings.jsonl")
    args = ap.parse_args()

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit("VOYAGE_API_KEY environment variable not set")
    client = voyageai.Client(api_key=api_key)

    pdf_dir = Path(args.pdf_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs in {pdf_dir}")

    out_path = Path(args.out)
    total_chunks = 0
    total_flagged = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, pdf_path in enumerate(pdf_paths, 1):
            book_title = pdf_path.stem.replace("_", " ")
            print(f"[{i}/{len(pdf_paths)}] {book_title}")
            try:
                chunks = process_book(pdf_path, book_title)
            except Exception as e:
                print(f"  ERROR extracting {pdf_path.name}: {e}")
                continue

            embed_chunks(chunks, client)

            for c in chunks:
                total_chunks += 1
                if c["needs_review"]:
                    total_flagged += 1
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

            print(f"  {len(chunks)} chunks")

    print(f"\nDone. {total_chunks} total chunks across {len(pdf_paths)} books.")
    print(f"{total_flagged} chunks flagged for possible character-loss review.")


if __name__ == "__main__":
    main()
