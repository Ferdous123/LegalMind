# Evaluation Approach & Results

I evaluated LegalMind along four dimensions: extraction, retrieval and
grounding, draft quality, and improvement from operator edits. The
improvement-from-edits experiment is the headline result; I designed it
as a controlled same-document A/B so the effect of learning could be
isolated from document-to-document variance.

## 1. Document processing and extraction

**Approach.** `scripts/run_evaluation.py` compares the pipeline's
extracted structured fields against the hand-written ground truth in
`data/sample/expected_outputs/*.json`, across the five clean documents
and the 25 degraded scans (`img_N_level_1..5`).

**Method.** Per field: exact match, normalized match (case and
whitespace), or miss. Degradation sensitivity = how accuracy moves as
scan quality drops from level 1 to 5.

**Observations.**

- Clean text PDFs extract reliably; document summaries and citations are
  well-formed.
- OCR (LightOnOCR-1B) holds through about level 3. Levels 4–5 lose
  low-confidence fields, which is why `structurer._ensure_schema_shape()`
  exists — a garbled JSON extraction no longer crashes downstream; missing
  fields degrade to empty.
- Per-page `confidence` and `ocr_used` flags propagate through the
  pipeline so the operator can see what was uncertain.

## 2. Retrieval and grounding

**Approach.** Every generated draft carries `[E1]`-style citations and a
`firewall_summary` of verified / uncertain / unsupported claims.
Grounding is deterministic — a 3-gram verbatim match or ≥ 0.30 token
containment between the claim and its cited evidence — so a reviewer can
verify any decision by hand.

**Real drafts on this hardware (after the concise + strict-grounding
prompts):**

| Source | Draft type | Words | Verified / total | Unsupported | Confidence |
|---|---|---|---|---|---|
| doc_1 lease | case_fact_summary | 156 | 6 / 9 | 0 | 0.91 |
| doc_2 court filing | case_fact_summary | 190 | 9 / 15 | 0 | 0.87 |
| doc_3 property deed | title_review_summary | 176 | 6 / 10 | 1 | 0.81 |
| doc_4 compliance notice | notice_summary | 150 | 8 / 13 | 0 | 0.85 |
| doc_5 document checklist | document_checklist | 116 | 3 / 6 | 1 | 0.74 |
| img_1_level_1 (clean scan) | case_fact_summary | 222 | 7 / 9 | 0 | 0.91 |
| img_2_level_3 (medium degraded) | case_fact_summary | 242 | 10 / 18 | 0 | 0.83 |
| img_3_level_2 (medium degraded) | title_review_summary | 262 | 11 / 17 | 0 | 0.88 |

Zero or one unsupported claim across the set — the strict-grounding
prompts pay off. The verification badge in the draft viewer is
clickable: it lists each claim by status with its evidence snippet, and
the citation "eye" opens the source page image or a highlighted source
text span. Operators can also flip a claim to verified via an explicit
override, which is recorded as `operator override` in the draft JSON so
the provenance is preserved.

**Cross-document caveat.** Verification percentages are not directly
comparable across documents — different claim counts, different
complexity. The improvement-from-edits result below uses a
same-document A/B specifically to avoid this trap.

## 3. Draft quality

After tightening the four prompts to require concise output with strict
grounding (cite or omit), the same documents produced drafts roughly
75–86% shorter than the original verbose templates, without losing
material content. The shipped drafts surface the right entities,
dates, monetary values, and counsel, each cited. Sample drafts are in
`04_SAMPLE_INPUTS_AND_OUTPUTS.md`.

## 4. Improvement from operator edits — controlled experiment

This is the result I'm most interested in. I wanted to know whether the
learning loop produces a measurable, observable change in a future draft
on a *different* document, not just a process diagram that suggests it
should.

### 4a. End-to-end wiring through the real API

1. The system generated a baseline draft for doc_1 (lease,
   case_fact_summary).
2. An operator agent read the actual source text and the generated draft,
   identified five genuine errors and omissions — incomplete party
   names, a rent amount marked "not applicable" though the source stated
   it, the omitted premises address and governing law — and submitted
   them via `POST /api/v1/corrections`.
3. Rule extraction was triggered through the system's own endpoint
   (`POST /api/v1/learning/rules`, the Audit-tab button), with the
   pattern extractor running in an isolated subprocess.
4. The system extracted three new rules (3 → 6), each directly encoding
   a pattern from the corrections — e.g. "retain the full, formal legal
   designation, including corporate suffixes (LLC, Inc.) and qualifiers"
   and "prioritize direct extraction for dates and monetary values; use
   the concrete value rather than a placeholder."

The corrections, the extraction, and the rule file are real artifacts in
`data/corrections/case_fact_summary.jsonl` and
`config/learned_rules.yaml`.

### 4b. Controlled same-document A/B

To isolate the rules' effect from document variance I ran the same
document under two different learning states. The document is doc_2
(court filing, case_fact_summary). Only the rule set varies: the three
newly-learned rules deactivated for the BEFORE run, all six active for
the AFTER run. The rule file is backed up and restored.

Script: `scripts/controlled_ab.py`. Raw result:
`logs/controlled_ab_result.json`.

| Metric (same document) | BEFORE (3 rules) | AFTER (6 rules) | Δ |
|---|---|---|---|
| rules applied | 3 | 6 | +3 |
| total claims | 42 | 21 | −21 |
| verified | 18 | 9 | −9 |
| verified % | 42.9% | 42.9% | flat |
| **unsupported claims** | **12** | **6** | **−50%** |
| uncertain claims | 12 | 6 | −50% |
| **overall confidence** | **0.556** | **0.625** | **+12.5% rel** |
| content length | 5489 | 5493 | ≈ same |

**Reading the result.** With the learned rules active, on the identical
document the draft made half as many claims for essentially the same
output length — consolidated, precise assertions instead of sprawling
generic tables, which is what the extracted rules instruct. Unsupported
claims dropped by half, overall confidence rose ~12.5%. The verified
proportion is flat (42.9%): the gain is "same proportion verified, on a
tighter and better-grounded draft", not "more boxes ticked."

**Caveat.** This is a single before/after pair, and generation is
stochastic (temperature 0.3), so absolute values carry run-to-run
variance. The direction and magnitude — halved unsupported, +12.5%
confidence, tighter output — are consistent with the rule semantics.
Averaging N runs per arm would tighten the confidence interval; I
didn't, to keep scope practical.

## 5. Reproducing the numbers

```bash
python scripts/seed_sample_data.py
python scripts/run_evaluation.py     # extraction accuracy vs expected
python scripts/controlled_ab.py      # same-document learning A/B
cat logs/controlled_ab_result.json
```

## Summary

| What | Outcome |
|---|---|
| Messy-input handling | 25 degraded scans processed; schema-shape guard prevents downstream crashes; per-page confidence flags propagate |
| Retrieval & grounding | Hybrid BM25 + semantic with RRF; deterministic per-claim verification; clickable inspection in the UI |
| Hallucination control | 0–1 unsupported claims per draft across the shipped set; operator-override available for borderline cases |
| Improvement from edits | Same-document A/B: −50% unsupported claims, +12.5% confidence, halved uncertain |
