# Sample Inputs & Outputs

All sample inputs are synthetic — fictional parties, cases, addresses,
and numbers. Both the inputs and the drafts the system produced from them
are committed to the repo, so the behaviour is inspectable without
running the models locally.

## Inputs — `data/sample/`

### Clean legal-style text (5 documents, one per draft type)

| File | Draft type | What it exercises |
|---|---|---|
| `doc_1_lease_agreement` (txt + pdf) | case_fact_summary | Parties, dates, monetary terms |
| `doc_2_court_filing` (txt + pdf) | case_fact_summary | Multiple defendants, claims, counsel |
| `doc_3_property_deed` (txt + pdf) | title_review_summary | Chain of title, encumbrances |
| `doc_4_compliance_notice` (txt + pdf) | notice_summary | Deadlines, required actions |
| `doc_5_document_checklist` (txt + pdf) | document_checklist | Present vs missing items |

Both raw `.txt` (`data/sample/`) and PDF-wrapped (`data/sample/pdfs/`)
versions are provided so the text-extraction path is directly testable.

### Degraded scans — the messy-input case (25 images)

`data/sample/images/img_{1..5}_level_{1..5}.png` — five source documents,
each rendered at five increasing degradation levels (level 1 = light
noise; level 5 = heavy blur / skew / low-res). `img_N` shares content
with `doc_N`, so OCR output can be checked against the clean text.
PDF-wrapped copies are in `data/sample/pdfs/img_*_level_*.pdf`.

### Ground truth — `data/sample/expected_outputs/`

One `*_expected.json` per input with the expected structured fields.
`scripts/run_evaluation.py` scores extraction against these.

## Outputs — `data/processed/`

Real drafts produced by the pipeline on this hardware, committed to the
repo (each is the most recent regeneration with the concise + strict
grounding prompts):

| Source document | Draft type | Words | Grounding (verified / total, unsupported) | Confidence |
|---|---|---|---|---|
| `doc_1_lease_agreement.pdf` | case_fact_summary | 156 | 6 / 9, 0 unsupported | 0.91 |
| `doc_2_court_filing.pdf` | case_fact_summary | 190 | 9 / 15, 0 unsupported | 0.87 |
| `doc_3_property_deed.pdf` | title_review_summary | 176 | 6 / 10, 1 unsupported | 0.81 |
| `doc_4_compliance_notice.pdf` | notice_summary | 150 | 8 / 13, 0 unsupported | 0.85 |
| `doc_5_document_checklist.pdf` | document_checklist | 116 | 3 / 6, 1 unsupported | 0.74 |
| `img_1_level_1.png` (clean scan) | case_fact_summary | 222 | 7 / 9, 0 unsupported | 0.91 |
| `img_2_level_3.png` (medium degrad.) | case_fact_summary | 242 | 10 / 18, 0 unsupported | 0.83 |
| `img_3_level_2.png` (medium degrad.) | title_review_summary | 262 | 11 / 17, 0 unsupported | 0.88 |

Each draft JSON contains:

- `content_markdown` — the human-readable draft with inline `[E1]` cites
- `citations` — `[E_]` → evidence map (text, page, char range)
- `firewall_results` and `firewall_summary` — per-claim verification
- `exemplars_used`, `rules_applied`, `confidence_overall`

## Excerpt from a generated draft

`doc_2_court_filing.pdf` → `case_fact_summary`:

```markdown
## CASE FACT SUMMARY

**Summary:** Plaintiff Synthex Industrial Partners Inc. filed a complaint
against Norwood Fabrication Group LLC and Gerald P. Norwood individually,
alleging breach of contract, fraud, and unjust enrichment, seeking
$680,000.00 in compensatory and punitive damages [E1].

**Parties:**
Plaintiff: Synthex Industrial Partners Inc. [E1]
Defendant: Norwood Fabrication Group LLC [E1]
Defendant: Gerald P. Norwood (individually) [E1]

**Key dates:**
Filing Date: March 14, 2024 [E1]

**Claims / causes of action:**
Breach of Contract [E1]
Fraud [E1]
Unjust Enrichment [E1]

**Relief sought:** $680,000.00 compensatory plus punitive damages [E1].

**Gaps:**
- No evidence of any prior filings, motions, or discovery actions.
- No evidence of jurisdictional or venue challenges.
- No evidence of counterclaims or affirmative defenses.

**Citations:**
[E1] "IN THE DISTRICT COURT OF CALDWELL COUNTY ... Case No.: CV-2024-08847 ...
COMPLAINT FOR BREACH OF CONTRACT FRAUD AND UNJUST ENRICHMENT
Filing Date: March 14 2024 Damages: $680,000.00 ..."
(Document: doc_6802495c90ea, Page: 1)
```

Every factual line is cited. Fields not present in the source are
listed under "Gaps" rather than invented. The verification badge in the
UI lists each claim by status with its evidence snippet, and the
citation "eye" opens the source — a page image for PDFs and scans, a
highlighted text span for `.txt` inputs.

## Reproducing the outputs

```bash
python scripts/seed_sample_data.py   # write sample inputs + expected outputs

# start the server (see 01_README.md), then:

curl -X POST http://localhost:7860/api/v1/documents/upload \
  -F file=@data/sample/pdfs/doc_1_lease_agreement.pdf \
  -F draft_type=case_fact_summary

curl -X POST http://localhost:7860/api/v1/documents/upload \
  -F file=@data/sample/images/img_1_level_4.png \
  -F draft_type=case_fact_summary
```

Generated drafts land in `data/processed/draft_<id>_<type>.json` and are
viewable at `http://localhost:7860/documents/<id>/draft`.
