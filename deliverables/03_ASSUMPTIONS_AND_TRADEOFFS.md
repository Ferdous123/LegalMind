# Assumptions & Tradeoffs

This document covers the assumptions I made building LegalMind and the
choices I traded off. The goal was a system that grounds every claim,
learns measurably from operator corrections, and stays up under real
pressure on a single 10 GB GPU.

## Assumptions

1. **Synthetic inputs are acceptable for demonstration.** All sample
   documents in `data/sample/` are fictional. The 25 progressively-degraded
   scans simulate the messy-input requirement.
2. **One local GPU.** Target hardware is a single 10 GB GPU (developed on
   RTX 3080). Only one large model is resident at a time; models swap as
   the pipeline advances.
3. **Four bounded draft types** cover the realistic picks: case-fact
   summary, title-review summary, notice-related summary, document
   checklist. Each has a dedicated prompt and field schema.
4. **Operators edit localized fixes**, not wholesale rewrites — a wrong
   name, a missing date, an incomplete entity designation. This is what
   makes pattern extraction meaningful.
5. **Grounding, not legal correctness, is what's being optimized.** The
   verifier measures whether a claim is anchored in the source text. It
   does not judge whether the underlying legal conclusion is sound.

## Key tradeoffs

### Retrieval: hybrid BM25 + small embedding, not a large embedder

- **Chose:** ChromaDB with `bge-small-en` on CPU, fused with BM25 via
  Reciprocal Rank Fusion.
- **Why:** keeps all VRAM available for the generation model. BM25 alone
  is strong on legal text — entity names, statute numbers, case numbers
  are exact tokens. The small embedder adds semantic recall cheaply.
- **Cost:** lower semantic ceiling than BGE-M3 or a larger embedder.
  Acceptable because legal retrieval is keyword-heavy and the fusion
  recovers most of the gap.

### Grounding: lexical, not an LLM judge

- **Chose:** 3-gram verbatim match, with a token-containment fallback at
  ≥ 0.30 between claim and cited evidence.
- **Why:** deterministic, fast, no extra model load, and reproducible by
  hand. A reviewer can verify any decision the system made.
- **Cost:** paraphrase-heavy claims can read as "uncertain" even when
  effectively supported. Thresholds were tuned (4-gram → 3-gram, 0.40 →
  0.30) after observing real false-negatives. The operator-override
  button is the explicit escape hatch for borderline cases, and it
  records the override in the draft JSON so the provenance is preserved.

### Reliability: subprocess isolation, not in-process speed

- **Chose:** draft generation and pattern extraction each run in a
  short-lived subprocess.
- **Why:** on Windows, llama-cpp's CUDA context and mmap locks can
  deadlock or OOM-kill the worker when a multi-GB model loads inside the
  live server. Empirically this took the whole server down repeatedly.
- **Cost:** a few seconds of subprocess + model-load overhead per job. In
  return the server cannot be killed by whatever the model is doing.

### Learning: three layers, not one fine-tune

- **Layer 1 — exemplars.** BM25 retrieval over correction source text;
  effective from the very first correction, no training needed.
- **Layer 2 — extracted rules.** Every 20 corrections (or on demand),
  Gemma-4-E4B clusters similar corrections and writes reusable rules to
  `config/learned_rules.yaml`.
- **Layer 3 — prompt consolidation.** Every 50 corrections, stable rules
  fold into the base prompt and now-redundant exemplars are archived.
- **Cost:** more moving parts than a fine-tune. In return: no training
  infrastructure, every layer is human-inspectable, and the loop has an
  effect from correction #1.

### Chunk → page mapping: exact

The chunker used to estimate page numbers as `char_start // 3000`. That
silently produced wrong citations on multi-page PDFs. The ingester now
records the real per-page char spans and the chunker resolves each
chunk's true page. Wrong provenance in a legal tool is the worst kind of
bug; this fix is non-negotiable.

### Bounded scope

No multi-tenant auth, no distributed queue, no fine-tuning, no
non-English support. Effort went into grounding correctness, the
real-improvement loop, and reliability under GPU pressure.

## Known limitations

These are honest known issues, not surprises a reviewer would discover
later.

- **Cross-document verification % is not a quality metric.** Different
  documents have different claim counts and complexity. The learning
  result in `05_EVALUATION.md` uses a same-document A/B specifically to
  avoid this trap.
- **OCR confidence pooling on degraded scans** averages failed pages into
  the document mean. The audit panel may over-count failures cosmetically.
  Functional output is unaffected.
- **Pattern extraction needs a cluster.** Fewer than 2–3 similar
  corrections will produce no rule. By design — it avoids over-fitting to
  a single edit.
- **In-process ingestion model swap.** Draft generation and pattern
  extraction run in isolated subprocesses; the ingestion pipeline (OCR
  followed by structured extraction within the same upload) still runs in
  the web process. Under GPU pressure the OCR → extraction swap inside a
  single upload can fail (the document OCRs cleanly but structuring is
  skipped). Mitigated by a `reclear_gpu()` at the start of every upload,
  so documents are independent. The correct fix — subprocess-isolating
  ingestion the same way — is the natural next step. The shipped demo set
  is unaffected.
