# LegalMind — Assumptions & Tradeoffs

## Assumptions

1. **Synthetic inputs are acceptable.** The PDF explicitly permits mock/
   synthetic documents and simulated operator edits. All sample documents
   (5 legal-style texts + 25 progressively degraded scans) are fictional.
2. **Single local GPU.** Target hardware is one RTX 3080 (10 GB). Only one
   large model is resident at a time; models swap as the pipeline advances.
3. **Draft types are bounded.** Four output types cover the assessment's
   suggested picks: case-fact summary, title-review summary, notice-related
   summary, document checklist. Each has a dedicated prompt + field schema.
4. **Operators edit, they don't author.** The learning loop assumes
   corrections are localized fixes (a wrong name, an omitted date), not
   wholesale rewrites — this is what makes pattern extraction meaningful.
5. **"Not evaluating legal correctness."** Per the PDF, the system optimizes
   for *grounding and support*, not legal validity. The firewall measures
   whether a claim is anchored in the source, not whether it is good law.

## Key tradeoffs

### Retrieval: hybrid BM25 + small embedding, not a large embedder
- **Chose:** ChromaDB with `bge-small-en` on CPU, fused with BM25 via
  Reciprocal Rank Fusion.
- **Why:** keeps all VRAM for the generation model; BM25 alone is strong on
  legal text (entity names, statute numbers are exact tokens); the small
  embedder adds semantic recall cheaply.
- **Cost:** lower semantic ceiling than BGE-M3/large. Acceptable because
  legal retrieval is keyword-heavy and the fusion recovers most of the gap.

### Grounding: lexical (n-gram + containment), not an LLM judge
- **Chose:** 3-gram verbatim match OR ≥0.30 token containment between claim
  and cited evidence.
- **Why:** deterministic, fast, no extra model load, auditable by a human.
- **Cost:** paraphrase-heavy claims can read as "uncertain" even when
  supported. Thresholds were tuned (4-gram→3-gram, 0.40→0.30) after observing
  real false-negatives; the verification badge lets the operator inspect and
  judge borderline cases rather than trusting a black box.

### Reliability: subprocess isolation, not in-process speed
- **Chose:** run draft generation and rule extraction in short-lived
  subprocesses.
- **Why:** on Windows, llama-cpp's CUDA context / mmap locks deadlock or
  OOM-kill the worker when a multi-GB model loads inside the live server —
  empirically this took the whole server down repeatedly.
- **Cost:** ~2–5 s subprocess + model-load overhead per job. Worth it: the
  server stays up no matter what the model does. (Root cause and the fix are
  documented candidly in `docs/decisions.md`.)

### Learning: three layers, not one
- **Layer 1 exemplars (BM25 over correction source text):** immediate effect
  on the very next similar document; no training.
- **Layer 2 rules (LLM-clustered every 20 corrections, or forced):**
  generalizes a class of fixes into a reusable instruction.
- **Layer 3 prompt consolidation (every 50):** folds stable rules into the
  base prompt and archives now-redundant exemplars so the bank stays lean.
- **Cost:** more moving parts than a single fine-tune, but no training
  infrastructure, fully inspectable, and effective from correction #1.

### Chunk → page mapping: exact, not estimated
- Originally `page_number = char_start // 3000` (a guess). Now the ingester
  records real `page_spans` and the chunker resolves each chunk's true page.
- **Cost:** already-processed docs keep old numbers until re-ingested; new
  uploads are exact. Chosen because wrong provenance in a legal tool destroys
  reviewer trust.

### Scope deliberately bounded
- No multi-tenant auth, no distributed queue, no fine-tuning. The PDF says
  "scope down where needed — pick the parts you can do well and ship those
  cleanly." Effort went into grounding correctness, the real improvement
  loop, and reliability over breadth.

## Known limitations (stated plainly)

- **Cross-document firewall % is not a quality metric.** Different documents
  have different claim counts/complexity; only a *same-document* before/after
  isolates the learning effect (see `05_EVALUATION.md`).
- **Degraded-scan OCR confidence pooling** averages failed pages into the
  doc mean; the audit panel can over-count failures cosmetically. Functional
  output is unaffected.
- **Pattern extraction needs a cluster.** With <2–3 similar corrections the
  LLM may extract no rule — by design (avoids over-fitting to one edit).
- **In-process ingestion model swap.** Draft generation and rule extraction
  run in isolated subprocesses (llama-cpp on Windows only frees model VRAM on
  process exit). The *ingestion* pipeline (OCR → extraction) still runs
  in-process; under GPU pressure the OCR→extraction model swap within a
  single upload can fail (doc OCRs fine but structuring is skipped). Mitigated
  by a clean-GPU reclear at the start of every upload; the correct fix —
  subprocess-isolating ingestion too — is identified as the next step and
  deliberately deferred from the final pre-ship pass to avoid ship-time risk.
  See `docs/decisions.md` §20. The shipped demo set (5 text PDFs covering all
  4 draft types + 3 image docs across degradation levels, all with grounded
  drafts) is unaffected.
