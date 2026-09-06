"""
Osho corpus extraction pipeline — Step 1: PDF -> cleaned, chunked text with metadata.

Usage:
    python osho_pdf_extractor.py path/to/book.pdf --book-title "Book Title" --out chunks.jsonl

What it does:
    1. Extracts text page by page (keeps page numbers for source citation)
    2. Detects chapter/section markers (e.g. "CHAPTER 1") and keeps them as metadata
       instead of discarding them
    3. Detects and strips repeated headers/footers (book title, running footer, etc.)
    4. Normalizes ligature characters (fl, fi, etc.) and rejoins hyphenated line-breaks
    5. Merges text ACROSS page boundaries before chunking, so chunks aren't
       artificially cut off at page edges
    6. Chunks into paragraphs of ~150-400 words
    7. Writes one JSON object per line (JSONL) with: book_title, chapter_title,
       page_number (starting page of the chunk), chunk_id, chunk_text

Run this on ONE book first and inspect the output before doing the full corpus.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

CHAPTER_PATTERN = re.compile(r"^(CHAPTER|DISCOURSE|CHAPTER\s+\d+|DISCOURSE\s+\d+)\b.*", re.IGNORECASE)


def extract_pages(pdf_path):
    """Return list of (page_number, raw_text, list_of_text_spans_with_size)."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        spans = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    spans.append((span["text"].strip(), round(span["size"], 1)))
        pages.append((i + 1, text, spans))
    doc.close()
    return pages


def find_repeated_lines(pages, min_repeat_ratio=0.4):
    """Find lines (e.g. running headers/footers) that repeat across many pages.
    Chapter/discourse markers are excluded even if they repeat in style,
    since each occurrence usually differs (CHAPTER 1, CHAPTER 2, ...) —
    but as a safety net we also never strip anything matching CHAPTER_PATTERN."""
    line_counts = Counter()
    total_pages = len(pages)
    for _, text, _ in pages:
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


def normalize_text(text):
    """Fix ligatures (ﬂ -> fl, ﬁ -> fi, etc.) and other Unicode quirks."""
    return unicodedata.normalize("NFKC", text)


def clean_page_text(text, repeated_lines):
    """Clean a page's text, but keep chapter markers as separate flagged lines
    rather than folding them into body text or discarding them."""
    text = normalize_text(text)
    lines = text.split("\n")
    cleaned_lines = []
    chapter_titles_found = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if CHAPTER_PATTERN.match(stripped):
            chapter_titles_found.append(stripped)
            continue  # don't fold chapter markers into body text
        if stripped in repeated_lines:
            continue
        if re.fullmatch(r"\d+", stripped):
            continue
        cleaned_lines.append(stripped)
    joined = " ".join(cleaned_lines)
    joined = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", joined)  # rejoin hyphenated breaks
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined, chapter_titles_found


def chunk_words(words, min_words=150, max_words=400):
    """Split a flat list of words into chunks, extending to the next sentence
    boundary near max_words rather than cutting mid-sentence."""
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
        chunk_words_slice = words[start:end]
        chunks.append(chunk_words_slice)
        start = end
    if len(chunks) > 1 and len(chunks[-1]) < min_words // 2:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()
    return chunks


FUSION_PATTERN = re.compile(r"\d[a-zA-Z]{3,}")  # e.g. "100,000andawell" from a missing $ glyph


def flag_fusions(chunk_text):
    """Detect likely character-loss fusions (e.g. a '$' glyph that didn't
    extract, causing '$100,000 and a...' to become '100,000andawell...').
    Returns the list of suspicious tokens found, or an empty list if none."""
    return FUSION_PATTERN.findall(chunk_text)


def report_font_sizes(pages, top_n=10):
    size_counts = Counter()
    for _, _, spans in pages:
        for text, size in spans:
            if text:
                size_counts[size] += 1
    print("\nFont size frequency (helps spot title sizes vs body text):")
    for size, count in size_counts.most_common(top_n):
        print(f"  size {size}: {count} spans")


def main():
    ap = argparse.ArgumentParser(description="Extract and chunk an Osho book PDF.")
    ap.add_argument("pdf_path", type=str)
    ap.add_argument("--book-title", type=str, required=True)
    ap.add_argument("--out", type=str, default="chunks.jsonl")
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=400)
    args = ap.parse_args()

    pdf_path = Path(args.pdf_path)
    print(f"Extracting pages from {pdf_path.name} ...")
    pages = extract_pages(str(pdf_path))
    print(f"  {len(pages)} pages found")

    repeated_lines = find_repeated_lines(pages)
    print(f"  Detected {len(repeated_lines)} repeated header/footer lines to strip")
    if repeated_lines:
        print("  Examples:", list(repeated_lines)[:5])

    # Build one continuous word stream across the whole book, but remember
    # which page and which chapter each word came from, so chunks can span
    # page boundaries without losing that metadata.
    all_words = []          # flat list of words for the whole book
    word_page = []          # word_page[i] = page number that word i came from
    word_chapter = []       # word_chapter[i] = current chapter title at word i
    current_chapter = None
    chapters_seen = 0

    for page_num, raw_text, _ in pages:
        cleaned, chapter_titles = clean_page_text(raw_text, repeated_lines)
        for title in chapter_titles:
            # Only count it as a NEW chapter if the text actually changed —
            # otherwise a repeating running header (e.g. "CHAPTER 1" on every
            # page of a single-chapter book) would be miscounted as dozens
            # of distinct chapters.
            normalized_title = title.rstrip(".").strip()
            if normalized_title != current_chapter:
                current_chapter = normalized_title
                chapters_seen += 1
        if not cleaned:
            continue
        for w in cleaned.split(" "):
            if not w:
                continue
            all_words.append(w)
            word_page.append(page_num)
            word_chapter.append(current_chapter)

    print(f"  Detected {chapters_seen} chapter/discourse markers")

    word_chunks = chunk_words(all_words, args.min_words, args.max_words)

    chunk_id = 0
    out_path = Path(args.out)
    word_index = 0
    with out_path.open("w", encoding="utf-8") as f:
        for wc in word_chunks:
            chunk_id += 1
            start_page = word_page[word_index] if word_index < len(word_page) else None
            end_page = word_page[word_index + len(wc) - 1] if word_index + len(wc) - 1 < len(word_page) else start_page
            chapter_title = word_chapter[word_index] if word_index < len(word_chapter) else None
            chunk_text_str = " ".join(wc)
            fusions = flag_fusions(chunk_text_str)
            record = {
                "chunk_id": f"{pdf_path.stem}_{chunk_id:04d}",
                "book_title": args.book_title,
                "chapter_title": chapter_title,
                "page_start": start_page,
                "page_end": end_page,
                "chunk_text": chunk_text_str,
                "word_count": len(wc),
                "needs_review": bool(fusions),
                "flagged_tokens": fusions,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            word_index += len(wc)

    flagged_count = sum(1 for wc in word_chunks if flag_fusions(" ".join(wc)))
    print(f"\nWrote {chunk_id} chunks to {out_path}")
    print(f"  {flagged_count} chunks flagged for possible character-loss fusion (needs_review=true)")
    report_font_sizes(pages)
    print(
        "\nNext: open the .jsonl file and skim ~20 random chunks. Check for:"
        "\n  - leftover junk (running headers not caught, OCR-like garbling)"
        "\n  - chunks that cut off mid-thought"
        "\n  - whether chapter_title is populating correctly"
        "\n  - whether page_start/page_end will be useful for citing back to source"
    )


if __name__ == "__main__":
    main()
