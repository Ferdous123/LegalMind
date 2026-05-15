# LegalMind: Design Assumptions and Tradeoffs

This document records the key architectural decisions made during development of LegalMind, including the rationale for each choice and the tradeoffs accepted.

---

## 1. All-Local Inference (No API Calls)

### Decision
All LLM inference runs on locally-hosted GGUF models using `llama-cpp-python`. No data is sent to any external API (OpenAI, Anthropic, Google, etc.) at any point.

### Rationale
Legal documents handled by Pearson Specter Litt contain privileged attorney-client communications, confidential client information, and strategically sensitive case details. Transmitting this material to a third-party API — even under a data processing agreement — introduces unacceptable privilege and confidentiality risk. Running locally eliminates the data-egress attack surface entirely.

### Tradeoffs Accepted
- **Capability ceiling**: Local GGUF models at practical quantization levels (Q4_K_M, Q8_0) underperform frontier API models on complex reasoning tasks. The system compensates via structured prompting, few-shot exemplar injection, and the verification firewall.
- **Hardware dependency**: The system requires a capable GPU and significant local storage for model files. It is not deployable on a standard office workstation without a discrete GPU.
- **Update lag**: Improving model quality requires manually downloading and swapping GGUF files. There is no automatic model update path.
- **Inference latency**: A single document can take 30-90 seconds to process end-to-end on an RTX 3080, versus 2-5 seconds with a frontier API.

---

## 2. Single Model at a Time (GPU Constraint)

### Decision
The `ModelManager` is a singleton that loads at most one model into GPU VRAM at a time. When a different model is needed, the current model is unloaded before the next is loaded.

### Rationale
The target hardware is an RTX 3080 with 10 GB VRAM. The models in use are:

| Model | Quant | Approx VRAM |
|---|---|---|
| LightOnOCR-1B | Q8_0 | ~1.5 GB |
| Qwen3-VL-8B | Q4_K_M | ~5.5 GB |
| Gemma-4-E4B | Q4_K_M | ~3.5 GB |

None of these can co-reside with another model of similar size within 10 GB of VRAM. Sequential loading keeps the system functional on the target hardware without requiring model quantization so aggressive that output quality degrades to unusability.

### Tradeoffs Accepted
- **Sequential pipeline**: Documents cannot be partially pipelined across model stages. The OCR model must finish all pages before the extraction model can start.
- **Swap overhead**: Loading and unloading models adds 5-15 seconds of overhead per stage transition. This is acceptable for batch processing but makes interactive latency worse.
- **No parallelism across documents**: Two documents cannot be processed concurrently because both would compete for the same model slot. A queue is used; throughput is one document at a time.

---

## 3. BM25 Keyword Retrieval (No Vector Store)

### Decision
Document retrieval uses a custom BM25 implementation over chunked document text. No embedding model, vector database, or external service is used.

### Rationale
The system is constrained to a single RTX 3080 with 10 GB VRAM, all of which is consumed by the inference models (OCR, extraction, reasoning). Running an embedding model (even a small one like BGE-M3 at ~1.1 GB) would require either: sharing the GPU and complicating the model-swap logic, or running on CPU with significant indexing latency.

BM25 is a proven retrieval algorithm that requires zero GPU, zero external dependencies, and zero model downloads. For the document volumes expected (hundreds of documents per matter, with 2-5 chunks per document), BM25 provides adequate retrieval quality — legal documents contain distinctive terminology (case numbers, party names, statutory references) that keyword search handles well.

### Tradeoffs Accepted
- **No semantic similarity**: BM25 cannot match paraphrases or synonyms. A search for "breach of fiduciary duty" will not retrieve a chunk that only mentions "violation of the duty of loyalty." In practice, legal documents are sufficiently formulaic that this limitation rarely causes missed evidence.
- **No pre-computed index**: Each query scans all chunks of the target document(s) and computes BM25 scores from scratch. This is acceptable for the expected corpus size (sub-second even with thousands of chunks) but would not scale to a firm-wide search across millions of documents.
- **No learned relevance**: The BM25 parameters (k1=1.5, b=0.75) are standard defaults. There is no relevance feedback loop or query expansion. A future enhancement could use the exemplar bank to learn query reformulations.

---

## 4. Three-Layer Learning System

### Decision
The system learns from operator corrections through three layers operating at different timescales: exemplar retrieval (immediate), pattern extraction (every 20 corrections), and prompt consolidation (every 50 corrections).

### Rationale

A purely fine-tuning-based improvement approach is impractical for a local deployment: fine-tuning a 8B parameter model requires significantly more VRAM than the target hardware provides, and each fine-tuning run would take hours. A purely rule-based approach would require operators to explicitly write rules, creating a maintenance burden.

The three-layer hybrid solves different problems:

- **Exemplars (Layer 1)** provide immediate effect with zero latency overhead: a correction made at 3pm influences the next generation at 3:01pm. They handle rare, case-specific patterns well. However, they do not generalize — the same correction for a different document does not benefit from an earlier correction on an unrelated document.
- **Pattern extraction (Layer 2)** provides generalization: when 20 operators have all corrected "always include the effective date field", a rule is extracted and applied universally, even for documents that have no similar exemplars. This is where cross-document learning emerges.
- **Prompt consolidation (Layer 3)** prevents unbounded growth of the exemplar bank and rule list. Without it, the injected few-shot context grows indefinitely, eventually crowding out the actual document content from the context window.

Together, the three layers approximate the behavior of fine-tuning (generalization) without requiring it.

### Tradeoffs Accepted
- **Pattern extraction quality depends on Gemma-4-E4B**: If the pattern extraction model produces low-quality or contradictory rules, the base system prompt can degrade over time. A rule review UI is provided in the audit dashboard to allow operators to delete bad rules.
- **20-correction and 50-correction thresholds are heuristic**: These values were chosen as reasonable defaults. A high-volume deployment may need to trigger consolidation more frequently; a low-volume deployment may rarely reach Layer 2.
- **Consolidation is irreversible within a version**: Once exemplars are archived by the consolidator, restoring them requires manually reverting the corrections JSONL and learned_rules.yaml to earlier states. There is no automated rollback.

---

## 5. BM25 for Exemplar Retrieval (No Embedding Model)

### Decision
The exemplar retriever (Layer 1 of the learning system) uses BM25 keyword matching over stored corrections, the same algorithm used for document retrieval. No embedding model is used for similarity search.

### Rationale
Exemplar retrieval must answer: "given this source text, which past corrections are most relevant?" The corrections store is small (tens to hundreds of entries), and legal corrections are highly specific — a correction about "Gerald P. Norwood" being omitted is relevant when the source text mentions "Norwood." BM25 keyword overlap identifies this relationship directly.

Using an embedding model would add complexity (model download, GPU/CPU allocation, index maintenance) for marginal benefit on a corpus of this size. The decision is consistent with the overall constraint of keeping the system dependency-free and GPU-focused on the inference models.

### Tradeoffs Accepted
- **No semantic matching for corrections**: If a correction teaches "always list all named defendants" on a document mentioning "Norwood," BM25 will not retrieve this correction for a different document mentioning "Harrison" even though the correction pattern is identical. Layer 2 (pattern extraction) compensates by extracting generalizable rules that apply regardless of specific entity names.
- **Cold start**: With fewer than 3 corrections, BM25 retrieval returns no meaningful results. The system degrades gracefully — no exemplars are injected, and the base prompt alone drives generation.

---

## 6. pdfplumber + pypdfium2 Combination

### Decision
PDF processing uses two libraries in sequence: `pdfplumber` for text extraction from text-layer PDFs, and `pypdfium2` to render pages as images for the OCR fallback path.

### Rationale
Legal PDFs arrive in two fundamentally different forms: those generated by word processors (which contain a text layer) and those that are scans of paper documents (which contain only images). A single library cannot handle both optimally.

`pdfplumber` excels at extracting text from text-layer PDFs, preserving layout, table structure, and column ordering — all critical for legal documents where the position of a clause matters. It is not designed to render images for OCR.

`pypdfium2` is a Python binding to PDFium, Google's production PDF renderer. It produces high-fidelity page images at configurable DPI (300 DPI is used for OCR inputs) with correct rendering of complex PDF features. It does not attempt text extraction.

**OCR trigger threshold**: If `pdfplumber` extracts fewer than 100 characters per page (averaged across the document), the document is rerouted to the OCR path. This threshold was calibrated to handle edge cases:
- A PDF with only a header and signature block on each page (text-layer, legitimately sparse) will still pass the threshold at approximately 150-200 characters per page.
- A scan of a dense legal brief will typically return 0-15 characters per page from pdfplumber (from embedded metadata artifacts) and correctly triggers OCR.
- The threshold can be adjusted in `config/models.yaml` under `ingestion.ocr_threshold`.

### Tradeoffs Accepted
- **Two-pass cost for borderline documents**: A document that fails the pdfplumber threshold must be re-rendered via pypdfium2 and processed through OCR, taking significantly longer than a clean text extraction.
- **No hybrid mode**: A PDF where some pages have a text layer and others are scans is currently treated as all-OCR if the average character count falls below threshold. A page-by-page routing decision would be more accurate but adds complexity.
- **pdfplumber table extraction is not used**: The structurer uses the raw text output, not pdfplumber's table objects. This is a simplification; tabular legal data (e.g., a fee schedule) may be better extracted by passing the table structure explicitly. This is a known improvement area.

---

## 7. JSONL for Corrections Storage

### Decision
Operator corrections are stored as newline-delimited JSON in `data/corrections/corrections.jsonl`.

### Rationale
Corrections are append-only by nature: a correction is never modified after it is recorded, only added to or (rarely) soft-deleted by flagging. JSONL is a natural fit for append-only logs:

- **Human-readable**: An operator or developer can open the file in any text editor and read its contents without tooling.
- **Zero migration risk**: Adding new fields to a correction record requires no schema migration — old records simply lack the new field, and code handles missing fields with defaults.
- **No database dependency**: Eliminating a database (SQLite, PostgreSQL) removes an operational dependency and simplifies the deployment footprint.
- **Trivially backupable**: A single file copy is a complete backup.
- **Streamable**: The learning system can process corrections in a single sequential pass without loading the entire file into memory.

### Tradeoffs Accepted
- **No efficient random access**: Looking up a specific correction by ID requires scanning the entire file. At expected correction volumes (hundreds to low thousands per matter), this is acceptable; at tens of thousands of corrections, an index or migration to SQLite should be considered.
- **No concurrent writes**: If two browser tabs submit corrections simultaneously, one write could corrupt the file. The current implementation uses a file lock (`filelock`) to serialize writes. This is adequate for single-firm use but would not scale to multi-user concurrent writes in a hosted deployment.
- **Soft deletes only**: Removing a correction requires adding a `{"deleted": true, "id": "..."}` entry rather than physically removing the original record. The file grows monotonically and must be compacted periodically.

---

## 8. FastAPI + Jinja2 + HTMX

### Decision
The web frontend uses server-side rendering via FastAPI's Jinja2 integration, with HTMX for dynamic interactions (field editing, status updates, SSE progress).

### Rationale
The assessment constraints and operational context favor a no-build-step frontend. A React or Vue SPA would require:
- A Node.js toolchain in the development environment
- A separate build step (which must either run in the Docker image or produce artifacts committed to the repository)
- Either a separate static file server or a bundler integration with FastAPI
- Frontend-specific testing infrastructure

HTMX with server-side rendering eliminates all of these. The entire frontend is plain HTML templates rendered by FastAPI on the server. Dynamic behavior (field editing, toast notifications, SSE progress bars) is handled by HTMX attributes in the template, requiring no client-side JavaScript to be written or maintained. The result is a fully functional, testable web application with the same backend-only toolchain used everywhere else in the project.

### Tradeoffs Accepted
- **Limited interactivity ceiling**: Highly interactive UI patterns (real-time collaborative editing, complex drag-and-drop) are difficult to implement cleanly with HTMX alone. If the UI requirements evolve toward a rich document editor, a JavaScript framework would become the right choice.
- **Full page re-renders for some operations**: Some HTMX interactions swap out larger portions of the DOM than a fine-grained React component would. This is not perceptible to users on LAN or localhost but would matter on a high-latency connection.
- **Jinja2 templates are not type-checked**: The template layer has no compile-time validation that it is passing the correct context variables. Errors surface at request time as TemplateSyntaxError or UndefinedError. The mitigation is thorough integration testing.
- **SSE for progress**: Server-sent events for pipeline progress require a persistent HTTP connection per document being processed. This is fine for single-user use; a multi-user deployment would need to ensure the SSE endpoint handles connection limits and back-pressure correctly.

---

## Additional Features

The following components were implemented beyond the base specification. They integrate cleanly with the base pipeline and do not alter its external interface or data contracts. All are loaded lazily to keep import overhead minimal.

### A. Cascading Multi-Pass Extraction (`code/pipeline/cascade.py`)

**What it does:** Instead of a single extraction pass, the cascade runs up to three passes on each document with progressively different strategies:
- Pass 1 (deterministic, temperature=0.1): Standard extraction with the full document context.
- Pass 2 (exploratory, temperature=0.4): Re-extraction with shifted chunk boundaries and legal corpus vocabulary hints injected into the prompt.
- Pass 3 (reasoning model): Targeted re-extraction of fields that remain uncertain after votes, using Gemma-4-E4B with candidate values from prior passes.

Fields are resolved using a priority hierarchy: majority vote (2+ passes agree) → corpus anchor (extracted value fuzzy-matches a known legal term) → source anchor (5-gram n-gram appears verbatim in source text) → manual review flag. Each resolved field carries a confidence score and a resolution method label.

**Why it does not break the base pipeline:** `DocumentStructurer.extract()` defaults to `use_cascade=True` but accepts `use_cascade=False` to restore the original single-pass behavior. If the cascade raises any unhandled exception, it falls back to single-pass automatically. The cascade result is merged into the same schema-shaped dict the rest of the pipeline expects, with an additional `_cascade_meta` key that downstream code may ignore.

**Tradeoff:** The cascade adds 2–3× latency compared to single-pass extraction. For scanned documents with significant OCR noise, the improvement in field accuracy justifies this cost. For clean text-layer PDFs, single-pass is recommended.

---

### B. Source Anchor Verifier (`code/pipeline/source_anchor.py`)

**What it does:** Provides grounding verification for extracted field values — ensuring that what the model extracted can actually be traced back to a specific span in the original document text. It is used both within the extraction cascade (to identify weak fields for re-extraction) and as a standalone verification tool.

Three signals are combined into a single confidence score:
1. 5-gram n-gram presence (strongest signal): a 5-word window from the extracted value appears verbatim in the source text.
2. 3-gram n-gram presence (moderate signal).
3. Best sliding-window span similarity using `difflib.SequenceMatcher`.

A field with confidence ≥ 0.5 is considered anchored. Fields below this threshold are flagged for manual review.

**Why it does not break the base pipeline:** `SourceAnchor` has no side effects; it reads from the source text string passed to it and returns scores. It is used as an internal quality signal within the cascade and does not alter any stored data.

---

### C. Vocabulary Learner (`code/corpus/vocabulary_learner.py`)

**What it does:** Expands the legal extraction lexicon automatically from operator corrections. When a correction is saved via `CorrectionStore.save_correction()`, the vocabulary learner is called as a non-critical hook. It extracts the delta between the AI-generated text and the operator's corrected text, filters for terms that appear to be legal terminology (based on suffix patterns, capitalization, and known legal keyword prefixes), and appends novel terms to `data/learned_vocabulary.jsonl`.

Learned terms are merged back into the `LegalLexicon` at initialization time. Terms must meet a minimum threshold (confidence ≥ 0.6 or frequency ≥ 2 across separate corrections) before being promoted into active use, preventing low-quality noise from entering the extraction vocabulary.

**Why it does not break the base pipeline:** The vocabulary learning call in `CorrectionStore.save_correction()` is wrapped in a bare `except Exception: pass` block. If the learner fails for any reason (import error, malformed correction, disk error), the correction save succeeds normally and the failure is silently absorbed. The learned vocabulary file (`data/learned_vocabulary.jsonl`) is an additive supplement to the static corpus; removing it or starting with an empty file restores the original static-only behavior with no other changes required.
