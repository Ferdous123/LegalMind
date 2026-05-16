# LegalMind — Submission Index

**Assessment:** AI Engineer Take-Home — Document Understanding, Grounded Drafting, and Improvement from Edits
**Candidate role:** AI Engineer, Pearson Specter Litt (simulated)

This folder contains every required deliverable. Each item below maps directly
to the assessment PDF's "What to Submit" list and the 100-point rubric.

---

## Required Deliverables → Files

| # | Required item (per PDF) | File |
|---|---|---|
| 1 | Source code | Repository root (`code/`, `webapp/`, `config/`, `scripts/`, `tests/`) |
| 2 | README with setup and run instructions | [`01_README.md`](01_README.md) · also repo-root [`README.md`](../README.md) |
| 3 | Short architecture overview | [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) |
| 4 | Assumptions and tradeoffs write-up | [`03_ASSUMPTIONS_AND_TRADEOFFS.md`](03_ASSUMPTIONS_AND_TRADEOFFS.md) |
| 5 | Sample inputs and outputs | [`04_SAMPLE_INPUTS_AND_OUTPUTS.md`](04_SAMPLE_INPUTS_AND_OUTPUTS.md) + `../data/sample/` |
| 6 | Evaluation approach and results | [`05_EVALUATION.md`](05_EVALUATION.md) |

Optional items (all delivered): API endpoints (FastAPI, `/api/v1/*`), web UI
(Jinja2 + HTMX, 6 panels), tests (`tests/` pytest), Docker
(`Dockerfile` + `docker-compose.yml`).

---

## Rubric Coverage (100 points)

| Rubric area | Pts | Where demonstrated |
|---|---|---|
| **1. Document Processing** | 25 | OCR (LightOnOCR-1B) + text extraction (pdfplumber); sparse-page → OCR routing; per-page confidence; degraded-scan handling. See ARCHITECTURE §1–3, EVALUATION "Extraction". 25 degraded scans (`data/sample/images/img_*_level_1..5.png`) exercise the messy-input path. |
| **2. Retrieval & Grounding** | 25 | Hybrid retrieval (BM25 + ChromaDB semantic, RRF fusion); `[E1]`-style citations; per-claim verification firewall; clickable verification badge in the draft viewer to inspect which evidence supported which claim. ARCHITECTURE §5–7, EVALUATION "Grounding". |
| **3. Draft Quality** | 10 | Four grounded draft types; WYSIWYG draft viewer; sample drafts in SAMPLE_INPUTS_AND_OUTPUTS. |
| **4. Improvement from Edits** | 25 | Three-layer learning loop (exemplars → rules → prompt). **Validated end-to-end** with an *agentic operator* that made source-grounded corrections through the real API → system extracted reusable rules → a later un-edited similar document's draft applied them. Controlled same-document A/B in EVALUATION. |
| **5. Code Quality & System Design** | 10 | Modular `code/` packages, dataclasses, type hints, graceful degradation, subprocess isolation for GPU jobs. ASSUMPTIONS "Design". |
| **6. Documentation & Clarity** | 5 | This index + the five documents + repo README + `docs/decisions.md` (running RCA log). |

---

## Fastest Path for a Reviewer (15 minutes)

1. **Read** `02_ARCHITECTURE.md` (5 min) — the data flow and why each choice was made.
2. **Run** `01_README.md` → Docker quick-start, or `run.bat` on Windows with the local model cache.
3. **Open** `http://localhost:7860` → **Pipeline** panel → drag in
   `data/sample/pdfs/doc_1_lease_agreement.pdf` (clean text) and
   `data/sample/images/img_1_level_4.png` (degraded scan) to see both paths.
4. **Inspect grounding**: open the generated draft, click the **verification
   badge** (`N verified / N uncertain / N unsupported`) to see exactly which
   claim was anchored to which evidence; click the **eye** on a citation to
   see the source page/text span.
5. **See learning**: open the **Audit** panel — correction history + learned
   rules. The end-to-end proof (agentic edits → extracted rules → improved
   later draft) is written up with numbers in `05_EVALUATION.md`.

If you cannot run models locally, `04_SAMPLE_INPUTS_AND_OUTPUTS.md` contains
committed real inputs and their generated outputs so the system's behaviour
is fully inspectable without a GPU.

---

## Honesty Notes (engineering transparency)

- Synthetic sample documents are used (the PDF explicitly permits this) — all
  names/cases are fictional. 25 progressively-degraded scans simulate the
  "messy input" requirement.
- `docs/decisions.md` is a candid running log of every bug found and fixed
  during development, including root-cause analysis. It is included
  deliberately: it shows the engineering process, not a polished facade.
- Where a limitation remains, it is stated plainly in
  `03_ASSUMPTIONS_AND_TRADEOFFS.md` rather than hidden.
