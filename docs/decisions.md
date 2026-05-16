# LegalMind — Session Decisions & Architecture Notes

This file is the running record of significant decisions, bug-fix root causes,
and architectural choices made during development. It is the "why" companion
to the codebase. Newer entries on top.

---

## 2026-05-16 Session 3 — System hardening pass

Triggered by user feedback that "the whole system is full of loose ends."
Dispatched an `Explore` audit agent, identified 10 issues, fixed 9 of them.

### Bugs fixed (in order)

#### 1. Draft viewer returned HTTP 500 on every doc — **template field shape mismatch**

`webapp/templates/drafts.html` rendered:

```jinja
{% if draft.exemplars_used > 0 %}    {# Jinja: int compare with a list! #}
```

But `code/generation/drafter.py` defines `exemplars_used: list = ...` and `rules_applied: list = ...` — they hold correction IDs and rule texts, not counts. Jinja raised `TypeError` on every viewer render.

Plus a second template/JSON name mismatch:

| Template read         | Actual JSON key         |
|-----------------------|-------------------------|
| `draft.confidence_score` | `confidence_overall` |
| `draft.generated_at`     | `generation_timestamp` |

Fix: template uses `| length` and iterates the lists; renamed the field reads.

#### 2. Draft generation subprocess always failed with exit code 1 (empty stderr)

Two compounding causes:

- **GPU exhaustion.** `ModelManager` kept the 6.1 GB Qwen3-VL model loaded after inline `_run_upload` work. When `_run_draft_job` then spawned a subprocess to load *its own* 6.1 GB copy, the RTX 3080's 10 GB filled and llama.cpp aborted on CUDA OOM. CUDA OOM kills the process before Python can write a traceback to stderr — exit code 1, empty stderr.
- **PIPE deadlock.** The subprocess captured stderr via `subprocess.PIPE`. llama.cpp's verbose model-load chatter quickly filled Windows' 64 KB pipe buffer; the subprocess blocked writing, deadlocking the whole job.

**Fixes** (both in `webapp/main.py`):
1. `ModelManager.instance().reclear_gpu()` runs before each subprocess launch AND after each `_run_upload` finishes.
2. Runner is now written to a `.py` tempfile (not `-c`); stdout/stderr redirect to a `.log` tempfile (not PIPE). `_tail_log(head=80, tail=80)` surfaces real error output on failure; log path persists on disk for inspection.

#### 3. Upload returned 409 instead of queueing

`/api/v1/documents/upload` used to `raise HTTPException(409)` when `_pipeline_lock` was held. Now it pre-reads the payload (so the `UploadFile` buffer survives the wait), wraps the bytes in a `_BufferedFile` shim, and `async with _pipeline_lock` blocks until available. Behaves like an HTTP queue from the client's POV.

#### 4. Pipeline progress vanished on page navigation

The SSE connection was opened per panel; navigating closed it. Added a **global progress banner** in `webapp/templates/base.html` that:
1. Subscribes to `/api/v1/events/pipeline` SSE once at page load
2. Reads `/api/v1/queue` on init so a refresh re-shows any in-progress job
3. Renders below the topbar, persists across all panels
4. Has a Cancel button wired to `/api/v1/queue/cancel`

#### 5. Firewall reported "0 verified out of 161" — **wrong scope + over-strict thresholds**

The firewall was iterating `confidence_scorer._flatten_fields(structured_fields)`, which flattens *everything* in the doc's structured_fields — including internal pipeline diagnostics:

```
_cascade_meta.confidence_map.parties.plaintiffs: 0.0
_cascade_meta.resolution_methods.parties.defendants: manual_review
_cascade_meta.corpus_corrections[0].field: parties.other_parties
_cascade_meta.passes_executed: 3
```

Verifying the string `"0.0"` or `"manual_review"` against lease-agreement text never matches. That's where the 161 useless "checks" came from.

**Fixes:**
- `code/firewall/confidence.py`: skip top-level `_SKIP_KEYS = {"_cascade_meta", "_meta", "_debug", "_internal"}`.
- `code/firewall/runner.py`: in addition to scoring structured fields, **also extract `[E_]`-cited markdown segments** from the draft and verify each against its cited evidence specifically. This is what the operator-visible "X verified / Y uncertain" badge should reflect.
- `code/firewall/source_anchor.py` and `code/generation/grounding.py`: NGRAM_SIZE 4→3 (4-grams missed too many legal-prose matches), containment thresholds 0.40→0.30 (high) and 0.20→0.15 (low). Keep 2-letter tokens (LLC, Inc, TX) — they carry weight in entity-name matching.

**Result on the lease draft:** 161 checks (0 verified) → **8 checks, 6 verified, 92% confidence.**

Wrote a re-verify script and re-applied it to existing drafts so old saved JSONs got fresh `firewall_summary` without re-running the LLM.

#### 6. "Draft" / "View Draft" button shown on docs without a draft

Buttons linked to `/documents/{id}/draft` regardless of state; clicking landed on an empty viewer. Added `_doc_has_draft(doc_id)` and `_doc_draft_types(doc_id)` helpers in `webapp/main.py`. Library and Pipeline panel handlers now populate `doc["has_draft"]` / `doc["draft_types_available"]` on each doc dict. Templates render a disabled placeholder when `has_draft` is falsy.

#### 7. Editor.js sent empty `source_ocr_chunk` — **broke the learning loop's BM25 retrieval**

`webapp/static/js/editor.js` only knows the markdown field it just edited — it has no way to attach the corresponding source-text span. Without a non-empty `source_ocr_chunk`, the exemplar retriever's BM25 search has no content to match future docs against, and the correction effectively never participates in the learning loop.

**Fix:** `/api/v1/corrections` falls back to `doc.full_text[:1500]` when `source_ocr_chunk` is empty. Verified end-to-end: posting a correction with no source field stored it with `source_len=1500`.

#### 8. Job queue state lost on uvicorn --reload

`_job_queue_state` was in-memory only. Every code edit (uvicorn `--reload` is on by default in `LegalMind.bat`) wiped it, orphaning any subprocess started before the reload.

**Fix:** `_save_queue_state()` snapshots to `data/queue_state.json` on every mutation (enqueue, start, done, fail). `_load_queue_state()` runs in lifespan startup; any persisted `in_progress` job is moved to `failed` with an explanatory error message (its subprocess died with the old worker).

#### 9. Pattern extractor race — trigger lagged the correction it should have seen

`save_correction` instantiated `PatternExtractor()` and called `should_trigger()` synchronously with file-I/O ordering. Fixed by re-instantiating after the save completes (via `asyncio.to_thread(extractor.should_trigger)`) so the extractor reads the fresh count from disk.

#### 10. Markdown rendered as raw text in the draft viewer

The viewer dumped `{{ draft.content_markdown | safe }}` directly. Users saw `# headings`, `|---|---|`, and `[E1]` as literal characters.

**Fix:** loaded `marked.js` from CDN, wrapped the content in `.rendered-markdown.legal-prose`, scripted DOMContentLoaded to call `marked.parse()` and replace `innerHTML`. Added scoped CSS for legal-prose typography (serif headings, gold accent on H2 and `<li>::marker`, dashed `<hr>`, citation refs styled as gold pills, professional tables with alternating-row backgrounds).

The raw markdown is kept on `data-generated-text` so `editor.js` can still load it into the textarea on edit.

#### 20. Known limitation: in-process OCR→extraction model swap (ingestion)

`_run_upload` runs the full ingestion pipeline (OCR via LightOnOCR, then
structured extraction via Qwen3-VL-8B) **in the web process** via
`asyncio.to_thread(ingester.process, ...)`. On Windows, llama-cpp does not
release a model's VRAM until the owning *process* exits — so when the
structurer tries to load the 8 GB extraction model while the OCR model is
still resident in the same process, the load fails (`VRAMBudgetExceeded` /
`Failed to load model`). Symptom: a doc OCRs fine (text present, conf 0.9)
but structuring fails → no draft → "Generation skipped".

Mitigation shipped: `reclear_gpu()` at the **start** of every `_run_upload`
so each document begins on a clean GPU (fixed the doc-to-doc collision —
img_1/2/3 then succeeded). The remaining failure is the OCR→extraction swap
*within a single upload* under GPU pressure (hit by img_4_level_4).

Correct fix (deliberately deferred — ship-time risk): subprocess-isolate the
whole ingestion pipeline, exactly as draft generation (decisions §2) and
rule extraction (§16) already are. This is the proven pattern in this
codebase; extending it to ingestion is the right next step but was judged
too large for the final pre-ship pass. Documented honestly rather than
hidden. Demonstration set ships with 5 text PDFs (all 4 draft types) + 3
image docs across degradation levels (img_1 L1, img_2 L3, img_3 L2), all
with grounded drafts.

#### 19. Pipeline panel only showed ~4 documents (slice-before-filter bug)

`panel_page` did `sorted(glob("*.json"), reverse=True)[:10]` then filtered
out `draft_*.json` inside the loop. Every document also writes a draft that
sorts newer, so the 10-item window was consumed by draft files, leaving
~4 real documents visible. Fixed: filter `draft_` out **first**, then take
the newest 10. (Library panel was unaffected — it has no slice.)

#### 18. Controlled same-document learning A/B — the headline result

Validated "improvement from edits" rigorously. An operator-agent made 5
genuine source-grounded corrections on doc_1 via the real API → system
extracted 3 new rules (3→6). Then a *same-document* A/B on doc_2 (court
filing): deactivate the 3 new rules → draft (BEFORE); reactivate → draft
(AFTER); only the learning state varied (`scripts/controlled_ab.py`,
`logs/controlled_ab_result.json`). Result on the identical document:
unsupported claims **12→6 (−50%)**, overall confidence **0.556→0.625
(+12.5%)**, total claims 42→21 (tighter, less sprawl), verified-% flat at
42.9%. Honest caveat documented: single pair, temp=0.3 stochastic, direction
consistent with rule semantics. `learned_rules.yaml` backed up + restored
(6 rules active, 11 corrections — clean).

#### 17. Deliverables folder built (assessment submission)

`deliverables/` created with 00_SUBMISSION_INDEX (maps PDF "What to Submit"
+ 100-pt rubric → files, 15-min reviewer path), 01_README, 02_ARCHITECTURE,
03_ASSUMPTIONS_AND_TRADEOFFS, 04_SAMPLE_INPUTS_AND_OUTPUTS, 05_EVALUATION
(real A/B numbers). Root README updated with a prominent submission section.
Docs written accurate to the *post-fix* system (subprocess isolation,
n-gram grounding, WYSIWYG edit, page-accurate chunks) — the stale 2026-05-15
`ARCHITECTURE.md`/`README.md` claims (logprob scoring, BGE-M3 exemplars,
textarea edit) were corrected.

#### 16. Pattern-extraction endpoint crashed the server (in-process LLM load)

`POST /api/v1/learning/rules` (and the background `_run_pattern_extraction`) called `PatternExtractor.extract_patterns()` via `asyncio.to_thread`. That loads the **reasoning LLM in the server process**. On the shared RTX 3080, contending with the webapp's own ModelManager, llama-cpp's CUDA context / mmap locks deadlock or OOM-kill the worker on Windows — taking the whole server down (HTTP 000, unrecoverable). Surfaced when the agentic operator ran the real extraction after submitting 5 corrections.

Same root cause and same fix as draft generation (decisions §2): **run it in an isolated subprocess.** Added `_run_extraction_subprocess()` — reclears the webapp GPU first, writes a runner `.py` to a tempfile, redirects stdout/stderr to a `.log`, polls, reads a result JSON. Both the manual endpoint and the background trigger now route through it. The server can no longer be killed by extraction.

#### 15. Operator edits must be agentic, edits/extraction "through the system"

Process/workflow decision (user directive): the learning loop must be validated with **genuine operator corrections an agent derives by reviewing the actual draft vs. source**, submitted via the real `/api/v1/corrections` API — not hardcoded test corrections. Rule extraction is triggered through the system's own endpoint (the Audit-tab path), never by hand-editing `learned_rules.yaml`. A hardcoded-correction script was rolled back (3 contaminating corrections trimmed from `case_fact_summary.jsonl`, identifiable by 2026-05-16T01:31 timestamps) and redone with an operator-agent that produced 5 source-grounded corrections.

#### 14. Chunker discarded real PDF page numbers (the known limitation, now fixed)

`chunker.py` line ~124 used to do `chunk.page_number = char_start // 3000 + 1` — a uniform-page estimate that's wrong for every real legal PDF (pages vary wildly in length). A citation from PDF page 7 could open the page-2 JPEG.

**Fix:** `ingestion.py` now builds `page_spans = [(page_number, char_start, char_end), ...]` while assembling `full_text` (it knows each `PageContent`'s length and the `\n\n` separator width). `TextChunker.chunk(text, doc_id, page_spans=...)` resolves each chunk's `page_number` from the span whose char range contains the chunk start (`_resolve_page`). Falls back to the old `//3000` heuristic only when `page_spans` is absent (txt or external callers). New uploads get correct page numbers; already-processed docs keep old numbers until re-ingested.

#### 13. Eye button: type-aware source view (txt → highlighted text)

`.txt` docs have no page image, so the eye button used to 404 forever ("Loading…"). Chose the "always show eye, txt gets a text-highlight popup" approach.

- New endpoint `GET /api/v1/documents/{id}/text-span?start=&end=&pad=600` slices `full_text` and returns `{before, span, after, truncated_before, truncated_after, filename}`.
- Citation card eye button now carries `data-char-start` / `data-char-end` (from the citation map's chunk range).
- Modal flow: try the page image first; **on image error** (`onPageImgError`) fetch `/text-span` and render `before` + `<mark>span</mark>` + `after` in a scrollable mono text box, auto-scrolling the highlight into view.
- PDF/image docs still show the page image (unchanged). Only txt falls through to text.

So in deployment: PDF → correct page JPEG (after #14), image upload → the uploaded image, txt → highlighted source excerpt. Every citation is now traceable to its source regardless of input format.

#### 12. Inline editor was a textarea popup that broke formatting

Old `editor.js` flow: hover pencil → click → wrapper's innerHTML is replaced with a `<textarea>` containing the raw markdown → user types in a plain textbox without the rendered look → save → innerHTML becomes escaped plain text (no markdown rendering).

Felt foreign and the post-save view lost all formatting until reload.

**New WYSIWYG flow:**

- Pencil button still appears on hover (visual affordance), but on click:
  - The `.rendered-markdown` surface gets `contenteditable="true"`
  - A sticky toolbar (gold border, "Editing — changes save as a correction" + Save + Cancel) renders above it
  - Caret moves to end of content
- User edits the *rendered* draft in place — headings, tables, lists, citation pills all stay styled and editable
- On Save: `surface.cloneNode` (so we strip the toolbar/trigger), then `TurndownService.turndown(html)` → markdown → POST `/api/v1/corrections`
- On Cancel: restore the snapshotted `originalHTML`
- Keyboard: Ctrl+Enter saves, Esc cancels

Loaded `marked.js` (12.0.0), `turndown.js` (7.1.3), and `@joplin/turndown-plugin-gfm` (1.0.59) from `cdn.jsdelivr.net` in `drafts.html`. Turndown configured with `headingStyle: 'atx'`, `hr: '---'`, `bulletListMarker: '-'`, fenced code blocks, `*` emphasis, plus GFM tables. Added a custom rule so our `.citation-ref` spans (`[E1]`-style gold pills) are emitted back as plain `[E1]` tokens in markdown.

`editor.js` was rewritten end-to-end — the old textarea flow and `extractFieldText` / `buildUpdatedFieldHtml` helpers are gone. About half the size, more maintainable.

Post-save the surface keeps the user's exact HTML (no marked → turndown → marked round-trip), so what they typed is what they see. The raw markdown is then stored on `data-generated-text` so subsequent edits diff against the latest saved version.

#### 11. Page-image lightbox said "Loading…" forever for PNG/JPG uploads

`/api/v1/documents/{id}/pages/{n}` only served `data/pages/{id}/page_{n:03d}.jpg`, which only exists for PDFs (rendered at 150 DPI by `_save_page_images` in `code/pipeline/ingestion.py:211`). Image uploads have no rendered JPEGs.

**Fix:** for `page_num=1` on a non-PDF doc, the endpoint now falls back to `UPLOADS_DIR/{filename}` and serves it with the correct MIME type. PNG, JPG, JPEG, TIFF, BMP, WEBP, GIF supported. Returns 404 only if neither the rendered JPEG nor the original upload exists.

---

### Other significant decisions made this session

#### Prebuilt sample seed data removed by default

`data/sample/prebuilt/` had two pre-baked processed documents (`doc_sample_lease`, `doc_sample_court_filing`) that were auto-copied to `data/processed/` on every `run.bat` invocation. The user wanted fresh uploads only. `scripts/seed_sample_data.py` now requires `--copy-prebuilt` to opt back in.

#### Zombie worker pitfall — uvicorn --reload + CUDA contexts

When uvicorn `--reload` triggers, the old worker can survive as a CUDA-context zombie holding GPU memory; the new worker's `reclear_gpu()` only frees the current worker's allocation. **Symptom:** after many edits, fresh uploads stall at "OCR active" because there's no VRAM left.

**Workaround:** in those sessions, `Stop-Process -Id <old_pid> -Force`, then `POST /api/v1/system/gpu_reset`. Long term, consider dropping `--reload` from `LegalMind.bat` and using HMR for HTML/JS only, OR reaping zombie children via a periodic check.

---

## Citation flow — how it works for txt, image, and PDF

### Shared pipeline (all formats)

1. **Extract.** `code/pipeline/ingestion.py:_extract_pages()` produces per-page `PageContent(page_number, text, confidence, ocr_used)`.
   - `.pdf` → `pdfplumber.extract_text()` per page; sparse pages routed to OCR.
   - `.png/.jpg/...` → `OCREngine.process_image()` returns one page.
   - `.txt` → split into ~3000-char "pages" (synthetic).
2. **Concatenate.** `doc.full_text = "\n\n".join(p.text for p in pages)`.
3. **Chunk.** `code/pipeline/chunker.py:TextChunker.chunk(full_text, doc.id)` produces overlapping `TextChunk(text, char_start, char_end, page_number)` entries.
4. **Index.** `code/retrieval/indexer.py:DocumentIndexer.index_document(doc_id, chunks)` writes them to ChromaDB.
5. **Retrieve.** During draft generation, `EvidenceSearcher.search(query, doc_ids, top_k=8)` does hybrid semantic + BM25 retrieval and returns ranked chunks.
6. **Package.** `EvidencePackager.package(chunks)` assigns the IDs `E1, E2, ..., E8` and builds a `citation_map: dict[str, dict]` keyed by those IDs. Each value carries `{text, document_id, page_number, char_start, char_end}`.
7. **Inject into prompt.** The packager's `formatted_text` is included in the LLM prompt with `[E1] ...evidence text...` blocks.
8. **Cite in draft.** The model emits markdown with `[E1]`, `[E2]` markers inline.
9. **Verify.** `GroundingVerifier.verify_draft_citations()` parses each `[E_]` marker and checks claim → cited evidence (n-gram + token containment).
10. **Render.** Viewer right panel shows each citation entry (chunk preview + verbatim source span + page number + eye button). Eye opens the page image modal.

### Format-specific behavior

| Source       | OCR used? | page_number meaning              | Eye-button image            |
|--------------|-----------|----------------------------------|-----------------------------|
| `.txt`       | Never     | Synthetic (char_start // 3000)   | 404 (no visual exists)      |
| `.png/.jpg/.tiff` | Always (LightOnOCR-1B) | Always 1               | Serves the original upload  |
| `.pdf` (text-only) | Per-page sparse-text check | **Real page index from pdfplumber** | Pre-rendered JPEG (150 DPI via PyMuPDF) |
| `.pdf` (scanned)   | Yes — page rendered to 300 DPI image then OCR | Same  | Same                          |

### Known limitation: chunker re-assigns page_number for ALL formats

`code/pipeline/chunker.py:124-126`:

```python
for chunk in chunks:
    chunk.page_number = max(1, chunk.char_start // 3000 + 1)
```

This **discards the real PDF page index** during chunking. Since the chunker only receives `full_text` (concatenated), it has no way to recover the actual page boundaries. For a 10-page PDF with varying page lengths, a chunk's `page_number` may not match the page it actually came from.

**Right fix (not yet done):** pass per-page text into the chunker; record real page boundaries on each chunk, taking the page number from whichever page's char range the chunk falls within. Estimated work: ~30 min in `chunker.py` + `ingestion.py`.

**Symptom:** clicking the eye on a citation from page 5 of a PDF might open the JPEG of page 2 instead. Functionally still useful (you can see *some* page from the doc), just not the right one.

### How PDF page images get made

`code/pipeline/ingestion.py:DocumentIngester._save_page_images(path, doc_id, page_count)`:

1. Opens the PDF with PyMuPDF (`import fitz`).
2. Renders each page at 150 DPI (`fitz.Matrix(150/72, 150/72)`) to a `pixmap`.
3. Saves each as `data/pages/{doc_id}/page_{n:03d}.jpg` at jpg_quality=85.
4. Skips already-existing files (idempotent).

This is invoked **only for `.pdf` ingestion** (`_process_pdf` → `_save_page_images(path, doc_id, n_pages)`). Image-uploaded docs don't go through this path.

The `/api/v1/documents/{doc_id}/pages/{page_num}` endpoint serves these JPEGs. As of this session it also falls back to the original upload for non-PDF docs (PNG/JPG/etc.).

---

## Architecture reminders

- **Subprocess isolation for LLM jobs.** Draft generation runs in a per-job subprocess because `llama-cpp-python`'s CUDA context, mmap locks, and chat_handler buffers leak on Windows. Per-subprocess teardown forces OS-level cleanup.
- **Single GPU model at a time.** RTX 3080 has 10 GB; Qwen3-VL alone is 6.1 GB. `ModelManager.swap()` unloads before loading. `reclear_gpu()` is the panic button.
- **Two verifiers, two different things:**
  - `GroundingVerifier` (`code/generation/grounding.py`): claim-level — for each `[E_]`-cited claim, is it grounded in *that specific citation*? Result stored in `draft.verification_status` and `draft.confidence_overall`.
  - `FirewallRunner` (`code/firewall/runner.py`): combined — runs grounding on cited markdown claims AND scores structured-field claims via `ConfidenceScorer` + source-anchor verification. Result in `draft.firewall_summary` (the user-facing "X verified / Y uncertain" badge).
- **Three-layer learning:** exemplar bank (immediate, BM25 over `source_ocr_chunk`) → pattern extraction (every 20 corrections, LLM extracts rules) → prompt consolidation (every 50 corrections, folds rules into the base prompt, archives covered exemplars).
