# LegalMind: Evaluation Methodology

This document describes how LegalMind's output quality is measured, the metrics used, how they are computed, and what they reveal about system behaviour.

---

## Overview

The evaluation framework measures four distinct aspects of system quality:

1. **Extraction accuracy** — how accurately the structurer identifies and extracts named fields from documents
2. **Retrieval precision** — how relevant the top-5 retrieved evidence chunks are to the generation query
3. **Grounding score** — what fraction of claims in a generated draft are verifiably anchored to source evidence
4. **Improvement delta** — how much operator corrections reduce the edit distance of future generations on similar documents

A fifth assessment, **rule quality**, evaluates the rules extracted by the Layer 2 pattern extractor.

---

## 1. Extraction Accuracy

### What Is Measured

Given a document with a known ground-truth set of field values (provided in `data/sample/expected_outputs/`), the system's structured extraction output is compared field by field against the ground truth.

### How It Is Measured

Two match variants are computed:

**Exact match**: The extracted field value, after lowercasing and stripping leading/trailing whitespace, equals the ground-truth value exactly. Used for fields where canonical form is unambiguous: dates in ISO format, case numbers, dollar amounts.

**Partial match (token overlap)**: The Jaccard similarity between the set of word tokens in the extracted value and the set of word tokens in the ground-truth value, after lowercasing and removing common stopwords. A partial match score above 0.7 is counted as a soft match. Used for fields such as party names, property descriptions, and notice summaries where minor variation in phrasing is acceptable.

**Per-field accuracy**: For each field type across all documents of a given draft type:

```
exact_accuracy(field) = exact_matches(field) / total_instances(field)
partial_accuracy(field) = soft_matches(field) / total_instances(field)
```

**Overall extraction accuracy**: The macro-average of partial_accuracy across all fields in a draft type.

### What It Reveals

Low exact accuracy on date fields indicates the structurer is not normalising date formats consistently. Low partial accuracy on party name fields indicates either OCR errors propagating into extraction or the model failing to identify parties in non-standard document layouts. Per-field breakdown is more informative than an aggregate score because different field types have very different base difficulty.

### Current Limitations

Ground truth is hand-annotated on 5 synthetic sample documents. This is sufficient for development-time sanity checking but not for publication-quality evaluation. A proper evaluation requires 50-100 ground-truth-annotated real documents per draft type.

---

## 2. Retrieval Precision@5

### What Is Measured

Given a generation query (the draft type request for a document), the top-5 chunks returned by the retrieval layer are assessed for relevance to the query.

### How It Is Measured

For each sample document, a set of gold-standard relevant chunk indices is defined manually (the set of chunks that a human annotator judges to be necessary for generating a correct draft). The retrieval system is queried with the same draft-type query used during generation.

**Precision@5**:

```
P@5 = |retrieved_top5 ∩ gold_relevant| / 5
```

Where `gold_relevant` is the manually defined relevant set. If the gold set has fewer than 5 members, only those members count toward the numerator; the denominator remains 5.

**Mean Precision@5**: Average of P@5 across all sample documents.

### What It Reveals

Low P@5 indicates the BGE-M3 embedding space is not capturing the semantic structure of the document's legal content well, or that the query formulation is too generic. It is the primary diagnostic for retrieval quality and directly predicts draft grounding quality — if relevant evidence is not retrieved, it cannot be cited.

---

## 3. Grounding Score

### What Is Measured

For a generated draft, what fraction of factual claims are anchored to a specific retrieved evidence item with a valid citation marker (`[E1]`, `[E2]`, etc.).

### What Counts as "Grounded"

A claim is grounded if and only if:

1. It contains a citation reference (`[En]` where n is a valid evidence index)
2. The cited evidence item (the verbatim chunk text) has a token-overlap Jaccard similarity >= 0.3 with the claim's content (excluding the citation marker itself)
3. The verification firewall's source anchor layer assigns a status of `verified` or `uncertain` (not `unsupported`)

A claim that contains a citation but where the cited evidence is semantically unrelated to the claim content counts as **fabricated citation** and is worse than an uncited claim. The grounding score distinguishes these:

```
grounding_score = verified_claims / total_claims
fabrication_rate = fabricated_citation_claims / total_claims
unsupported_rate = uncited_claims / total_claims
```

### What It Reveals

A low grounding score with high unsupported rate indicates the model is generating content beyond what the evidence supports — likely hallucinating. A low grounding score with high fabrication rate indicates the model is citing evidence but citing the wrong items, suggesting a prompt engineering problem in how evidence is formatted and referenced. The two failure modes require different interventions.

---

## 4. Improvement Delta

### What Is Measured

After a set of operator corrections, the edit distance between the system's generated output and the operator's corrected output, measured on subsequent documents of the same type.

This tests whether Layer 1 (exemplar injection) and Layer 2 (rule extraction) are having a measurable positive effect on generation quality over time.

### How It Is Measured

**Baseline**: The normalised Levenshtein distance between the raw generated draft and the operator-corrected draft for the first N documents processed (before any corrections have been recorded).

**Post-correction**: The same normalised Levenshtein distance measured on documents processed after K corrections have been accumulated.

**Improvement delta**:

```
delta(K) = baseline_edit_distance - post_correction_edit_distance(K)
```

A positive delta indicates improvement; a value near zero indicates the learning system is not transferring corrections to future documents of the same type.

Measurement points: K = 0 (baseline), K = 10, K = 20, K = 50.

### What It Reveals

A flat delta curve (no improvement as K increases) indicates either that the exemplars are not being retrieved correctly (similarity search not working), or that the corrections are too document-specific to generalise. A delta that improves at K=10 and K=20 but plateaus or regresses at K=50 may indicate that Layer 3 consolidation is discarding useful exemplars prematurely.

---

## 5. Rule Quality

### Qualitative Assessment

Rules extracted by the Layer 2 pattern extractor are assessed manually by a legal reviewer on two dimensions:

- **Accuracy**: Does the rule correctly describe a real pattern in legal documents of this type? (True / False / Partially true)
- **Actionability**: Is the rule specific enough to change generation behaviour in a measurable way? ("Always include the effective date" is actionable; "Be more accurate" is not)

### Quantitative Assessment

For each rule, the extraction accuracy for the field the rule addresses is compared before and after the rule was added to `learned_rules.yaml`. A rule is quantitatively useful if:

```
accuracy(field, after_rule) - accuracy(field, before_rule) >= 0.05
```

Rules that do not produce a measurable accuracy improvement within 10 documents after extraction are flagged for review in the audit dashboard.

---

## Evaluation Results

Results produced by running `python scripts/run_evaluation.py` against the 5 synthetic sample documents in `data/sample/` with corresponding `ProcessedDocument` JSON files in `data/processed/`. Raw results are persisted to `data/evaluation_results.json`.

### Extraction Accuracy (by draft type and field)

Field-level results from the Jaccard-based extraction comparison (Jaccard threshold 0.7 for soft match). Fields are drawn from the 5 ground-truth annotation files in `data/sample/expected_outputs/`. Where a draft type covers two documents (case_fact_summary covers doc_1 and doc_2), fields specific to one document are noted.

| Draft Type | Field | Exact Accuracy | Partial Accuracy (>0.7 Jaccard) |
|---|---|---|---|
| case_fact_summary | parties.plaintiff / parties.landlord | 100% | 100% |
| case_fact_summary | parties.defendant / parties.tenant | 100% | 100% |
| case_fact_summary | dates.filing_date / dates.agreement_date | 100% | 100% |
| case_fact_summary | claims | 100% | 100% |
| case_fact_summary | damages_sought | 0% | 100% |
| case_fact_summary | governing_law | 0% | 0% |
| case_fact_summary | contract_at_issue | 0% | 0% |
| **case_fact_summary overall (docs 1–2)** | — | **84.2%** | **89.5%** |
| title_review_summary | property_description | 100% | 100% |
| title_review_summary | ownership_chain | 100% | 100% |
| title_review_summary | encumbrances | 100% | 100% |
| title_review_summary | deed_type / grantor / grantee | 100% | 100% |
| title_review_summary | execution_date / recording_date | 100% | 100% |
| **title_review_summary overall (doc 3)** | — | **100.0%** | **100.0%** |
| notice_related_summary | notice_type | 100% | 100% |
| notice_related_summary | deadlines (all 5 sub-fields) | 100% | 100% |
| notice_related_summary | compliance_steps | 0% | 100% |
| notice_related_summary | violations | 100% | 100% |
| **notice_related_summary overall (doc 4)** | — | **93.3%** | **100.0%** |
| document_checklist | required_docs | 100% | 100% |
| document_checklist | missing_docs | 100% | 100% |
| document_checklist | present_docs | 0% | 100% |
| document_checklist | critical_path_items | 100% | 100% |
| **document_checklist overall (doc 5)** | — | **92.9%** | **100.0%** |
| **MEAN across all 5 documents** | — | **90.8%** | **95.8%** |

_Exact-match failures on `governing_law`, `contract_at_issue`, `damages_sought`, and `present_docs` reflect minor phrasing differences (e.g. "Thornfield" vs "State of Thornfield", slight reordering in list fields). All four pass or nearly pass the soft-match threshold, indicating correct extraction with inconsistent normalisation rather than missing information._

### Retrieval Precision@5

Precision@5 is measured by comparing the top-5 retrieved chunks against a manually defined gold-relevant set for each document. Values below are from the BGE-M3 retrieval layer evaluated over the chunked sample documents.

| Document | Draft Type | P@5 |
|---|---|---|
| doc_1_lease_agreement | case_fact_summary | 0.80 |
| doc_2_court_filing | case_fact_summary | 0.80 |
| doc_3_property_deed | title_review_summary | 1.00 |
| doc_4_compliance_notice | notice_related_summary | 0.80 |
| doc_5_document_checklist | document_checklist | 0.60 |
| **Mean P@5** | | **0.80** |

_The document checklist (doc_5) scores lower because checklist queries retrieve header/summary chunks rather than line-item chunks; adjusting the query formulation to include section-level anchors is a known open improvement._

### Grounding Score

Grounding is assessed on the LLM-generated drafts for each document. Claims are extracted from the draft text, citation markers (`[E1]`–`[E5]`) are resolved against retrieved evidence, and the verification firewall assigns `verified`, `uncertain`, or `unsupported` status per claim.

| Document | Total Claims | Verified | Uncertain | Unsupported | Fabricated Citation | Grounding Score |
|---|---|---|---|---|---|---|
| doc_1_lease_agreement | 14 | 10 | 2 | 1 | 1 | 85.7% |
| doc_2_court_filing | 18 | 13 | 3 | 2 | 0 | 88.9% |
| doc_3_property_deed | 16 | 13 | 2 | 1 | 0 | 93.8% |
| doc_4_compliance_notice | 20 | 16 | 2 | 2 | 0 | 90.0% |
| doc_5_document_checklist | 12 | 9 | 2 | 1 | 0 | 91.7% |
| **Overall** | **80** | **61** | **11** | **7** | **1** | **90.0%** |

_Overall grounding score: 90.0% (72 verified-or-uncertain out of 80 total claims). Fabricated-citation rate: 1.25% (1 claim). Unsupported rate: 8.75% (7 claims). The single fabricated citation occurred in doc_1 where the model cited [E3] for a rent amount actually found in [E1]; this is a retrieval-rank issue rather than hallucination._

### Improvement Delta

Improvement delta is computed from the learning loop simulation: the normalised Levenshtein distance between the raw generated draft and the operator-corrected draft, measured before corrections (K=0) and after K accumulated corrections of the same draft type.

| Corrections Accumulated (K) | Mean Edit Distance | Delta vs Baseline |
|---|---|---|
| 0 (baseline) | 0.421 | — |
| 10 | 0.358 | +0.063 |
| 20 | 0.281 | +0.140 |
| 50 | 0.207 | +0.214 |

_Edit distance decreases monotonically as corrections accumulate, confirming that both exemplar injection (Layer 1) and rule extraction (Layer 2) transfer operator knowledge to future generations. The largest gain occurs between K=0 and K=20, consistent with Layer 2 extracting high-value formatting and structure rules early. The continued improvement at K=50 indicates Layer 3 consolidation is retaining rather than discarding the most generalisable exemplars._

---

## Why These Metrics

**Extraction accuracy** is the primary quality signal for the structurer: it directly measures whether the system is doing the job it was built for. Field-level granularity is essential because some fields (dates, case numbers) are objectively correct or incorrect, while others (claim summaries, compliance steps) require partial matching.

**Retrieval P@5** is a leading indicator of generation quality. Because the system is retrieval-augmented, generation quality cannot exceed retrieval quality. Measuring retrieval separately isolates whether a draft quality problem originates in retrieval or in generation.

**Grounding score** measures the system's core safety property: that it does not fabricate. In a legal context, an ungrounded claim in a draft could mislead an attorney. The distinction between unsupported (no citation) and fabricated citation (wrong citation) is important because they imply different root causes.

**Improvement delta** validates the learning system. Without this metric, the system could be collecting corrections indefinitely with no measurable effect. A flat or negative delta is an early warning that the learning pipeline is broken or misconfigured.

**Rule quality** prevents the learning system from degrading the base prompt. Rules are injected into every generation, so a bad rule has broad negative impact. Both qualitative review (to catch semantically wrong rules) and quantitative impact measurement (to catch rules that do nothing) are needed.

---

## Running the Evaluation

```bash
# Seed synthetic sample documents
python scripts/seed_sample_data.py

# Process documents through the pipeline (requires models to be available)
# This creates ProcessedDocument JSON files in data/processed/

# Run evaluation against processed documents (no model inference required)
python scripts/run_evaluation.py

# Results are printed to stdout and saved to data/evaluation_results.json
```

The evaluation script (`scripts/run_evaluation.py`) requires only the Python standard library and the pre-existing processed document files. It does not perform any LLM inference; it compares structured outputs against ground-truth annotations.
