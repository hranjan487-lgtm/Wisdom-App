# Osho Wisdom Corpus Pipeline

Extraction and retrieval pipeline for building an AI-guided reflection app
sourced from an Osho book corpus (208 PDFs).

## Status

- Extraction pipeline validated on 8 books (775 chunks) across three formats:
  letters, jokes/discourse-with-stories, and pure discourse transcripts.
- Retrieval currently uses TF-IDF (keyword matching) as a placeholder —
  see "Known limitation" below. This needs to be swapped for real semantic
  embeddings before the app is usable for its actual purpose.

## Files

- `osho_pdf_extractor.py` — extracts/cleans/chunks a book from a raw PDF file
  (via PyMuPDF). Use this if you have direct PDF bytes to process (e.g. in a
  GitHub Actions runner or any environment with normal file access).
- `process_drive_text.py` — same job, but for text already extracted by
  Google Drive's `read_file_content` (used because the working environment
  this was built in couldn't handle raw PDF bytes at scale). You likely
  won't need this version if running from GitHub Actions — use
  `osho_pdf_extractor.py` directly on the PDFs instead.
- `retrieve.py` — loads a corpus JSONL file and retrieves top-k chunks for a
  query. Currently TF-IDF based.
- `osho_corpus_master.jsonl` — the 8-book sample corpus processed so far.
  Each line is one chunk: `chunk_id`, `book_title`, `chapter_title`,
  `chunk_text`, `word_count`, `needs_review`, `flagged_tokens`.

## Known limitation: TF-IDF retrieval is not good enough for production

Tested query: "I am jealous of my colleague" against the 8-book corpus.
The single most relevant passage (an actual Osho teaching on jealousy and
meditation, in "Walking in Zen, Sitting in Zen") ranked **4th**, below two
unrelated jokes about laughter — because TF-IDF only matches shared words,
not meaning. A user describing "my colleague getting credit for my work"
won't share vocabulary with Osho's actual teaching on ego, so keyword
search will frequently miss or bury the right passage.

**Fix:** replace TF-IDF in `retrieve.py` with real embeddings:
1. Embed every chunk once (e.g. via Voyage AI or OpenAI's embeddings API —
   both work well paired with Claude for the response-generation side) and
   store the vectors (a vector DB like Pinecone/Weaviate, or even a local
   file for this corpus size).
2. At query time, embed the user's message the same way, retrieve by
   cosine similarity, same as `retrieve.py` does now — just swap the
   scoring function.
3. Add a lightweight reranking pass (an LLM call asking "which 2-3 of
   these actually address this problem") before generating the response —
   raw similarity alone still surfaces topically-close-but-emotionally-off
   passages.

This requires real internet access to reach an embeddings API or download
a model — which is why it wasn't done in the environment this was built in.
GitHub Actions (with secrets for API keys) or any normal dev environment
will work fine.

## Remaining extraction work

200 of 208 books still need processing. Once you have PDF access from a
normal environment (not chat), run `osho_pdf_extractor.py` in a loop over
the full PDF folder — each book takes seconds, so all 208 should take
minutes total, versus the batch-by-batch approach used here.

Known PDF quirk to watch for: some books have a font-encoding issue where
certain glyphs (seen so far: `$`) don't extract as characters, fusing
adjacent text together ("$100,000 and..." becomes "100,000andawell...").
Both extractor scripts flag chunks where this is detected
(`needs_review: true`) — review those chunks per-book rather than assuming
it's uniform across the corpus.

## Next steps after extraction + real embeddings work

- Response-generation prompt: retrieved passages -> Claude -> user-facing
  answer, with a "Source Confidence" framing (direct source / strongly
  related / interpretation) and a hard rule to never fabricate an
  attribution to Osho beyond what's in the retrieved corpus.
- Safety handling for high-risk disclosures (self-harm, "should I leave my
  marriage" type questions) — the app should recognize these and respond
  appropriately rather than retrieving a philosophical passage.
- MVP UI: single input -> insight + retrieved passage + source citation.
  Journal/daily-reflection/voice features are deliberately out of scope
  until the core retrieval loop is proven.
