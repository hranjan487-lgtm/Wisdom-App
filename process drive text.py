"""
Osho corpus extraction pipeline — Step 1b: process Drive-extracted text -> chunks.

This works on the JSON result saved by Google Drive:read_file_content when a
book's text is too large for chat context. It never loads the full book text
into the conversation — only this script (running via bash) touches it.

Usage:
    python process_drive_text.py path/to/saved_result.json --book-title "Book Title" --out chunks.jsonl

Differs from osho_pdf_extractor.py (which works on raw PDF bytes via PyMuPDF)
because Drive's text extraction:
  - has no page numbers or font-size info (so page_start/page_end and the
    font-size report aren't available here — chapter_title is still tracked)
  - already breaks into paragraph-ish blocks separated by blank lines
  - has its own footer-junk pattern: a repeated line (e.g. a website URL)
    hammered many times in a row at a chapter/book boundary, rather than
    once per page
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

CHAPTER_PATTERN = re.compile(r"^(CHAPTER|DISCOURSE)\s*\d+\.?\s*(.*)$", re.IGNORECASE)
FUSION_PATTERN = re.compile(r"\d[a-zA-Z]{3,}")


def load_drive_text(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # result is a list of content blocks; find the text block and parse its
    # embedded JSON (Drive wraps the actual extracted text in a "fileContent" field)
    for block in data:
        if isinstance(block, dict) and "text" in block:
            inner = block["text"]
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict) and "fileContent" in parsed:
                    return parsed["fileContent"]
            except json.JSONDecodeError:
                pass
            return inner
    raise ValueError("Could not find text content in the saved result file")


def strip_repeated_junk_runs(paragraphs, min_run=4):
    """Remove a paragraph that repeats itself many times in a row
    (e.g. a footer URL hammered dozens of times at a boundary)."""
    cleaned = []
    i = 0
    n = len(paragraphs)
    while i < n:
        j = i
        while j < n and paragraphs[j] == paragraphs[i]:
            j += 1
        run_length = j - i
        if run_length < min_run:
            cleaned.extend(paragraphs[i:j])
        # else: drop the whole repeated run (keep none of it)
        i = j
    return cleaned


def strip_internal_repeat_spam(paragraphs, min_lines=4, max_unique_ratio=0.3):
    """Drop any paragraph block that is itself mostly the same line repeated
    many times internally (e.g. a single block containing a URL line
    hammered 70 times, joined by single newlines rather than blank lines)."""
    cleaned = []
    for p in paragraphs:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if len(lines) >= min_lines:
            unique_ratio = len(set(lines)) / len(lines)
            if unique_ratio <= max_unique_ratio:
                continue  # drop the whole block as repeat-spam
        cleaned.append(p)
    return cleaned


def clean_paragraph(p):
    p = unicodedata.normalize("NFKC", p)
    p = re.sub(r"<[^>]+>", "", p)  # strip stray HTML-ish tags e.g. <http://...>
    p = re.sub(r"\s+", " ", p).strip()
    return p


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=str)
    ap.add_argument("--book-title", type=str, required=True)
    ap.add_argument("--out", type=str, default="chunks.jsonl")
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=400)
    args = ap.parse_args()

    raw_text = load_drive_text(args.json_path)
    raw_paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    print(f"{len(raw_paragraphs)} raw paragraph blocks")

    paragraphs = strip_repeated_junk_runs(raw_paragraphs)
    paragraphs = strip_internal_repeat_spam(paragraphs)
    removed = len(raw_paragraphs) - len(paragraphs)
    print(f"Removed {removed} paragraphs as repeated-junk (cross-block or internal-repeat spam)")

    all_words = []
    word_chapter = []
    current_chapter = None
    chapters_seen = 0

    for p in paragraphs:
        m = CHAPTER_PATTERN.match(p.strip())
        if m:
            title = m.group(2).strip().rstrip(".")
            if title:  # only count headers that carry an actual title;
                # bare markers like "CHAPTER1" with no title text are running-header
                # noise and should neither reset nor count as a new chapter
                number_match = re.search(r"\d+", p)
                number = number_match.group(0) if number_match else ""
                normalized = f"CHAPTER {number} {title}".strip()
                if normalized != current_chapter:
                    current_chapter = normalized
                    chapters_seen += 1
            continue  # don't fold chapter headers (titled or bare) into body text
        cleaned = clean_paragraph(p)
        if not cleaned:
            continue
        for w in cleaned.split(" "):
            if w:
                all_words.append(w)
                word_chapter.append(current_chapter)

    print(f"Detected {chapters_seen} distinct chapter/discourse markers")

    word_chunks = chunk_words(all_words, args.min_words, args.max_words)

    chunk_id = 0
    word_index = 0
    flagged = 0
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for wc in word_chunks:
            chunk_id += 1
            chapter_title = word_chapter[word_index] if word_index < len(word_chapter) else None
            chunk_text_str = " ".join(wc)
            fusions = flag_fusions(chunk_text_str)
            if fusions:
                flagged += 1
            record = {
                "chunk_id": f"{Path(args.json_path).stem}_{chunk_id:04d}",
                "book_title": args.book_title,
                "chapter_title": chapter_title,
                "chunk_text": chunk_text_str,
                "word_count": len(wc),
                "needs_review": bool(fusions),
                "flagged_tokens": fusions,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            word_index += len(wc)

    print(f"\nWrote {chunk_id} chunks to {out_path}")
    print(f"{flagged} chunks flagged for possible character-loss fusion")


if __name__ == "__main__":
    main()
