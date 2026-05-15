"""
Generate synthetic scanned legal documents with progressive difficulty.
Produces PNG images and PDF files that simulate real-world messy inputs.

Usage:
    python scripts/generate_sample_documents.py

Output:
    data/sample/images/  — PNG files at various degradation levels
    data/sample/pdfs/    — Single-page PDFs wrapping each image
    data/sample/expected_outputs/ — JSON ground truth per image
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import math
import numpy as np
from pathlib import Path
import json
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
IMAGES_DIR = SAMPLE_DIR / "images"
PDFS_DIR = SAMPLE_DIR / "pdfs"
EXPECTED_DIR = SAMPLE_DIR / "expected_outputs"

# ---------------------------------------------------------------------------
# Page settings
# ---------------------------------------------------------------------------

PAGE_WIDTH = 2550   # 8.5 inches at 300 DPI
PAGE_HEIGHT = 3300  # 11 inches at 300 DPI
MARGIN = 200        # pixel margin

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def load_typed_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a serif font for typed text. Falls back to default."""
    candidates = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    # Fallback: PIL default bitmap font (cannot be resized)
    return ImageFont.load_default()


def load_handwritten_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a script/cursive font for handwritten simulation."""
    candidates = [
        "C:/Windows/Fonts/segoesc.ttf",
        "C:/Windows/Fonts/segoepr.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/comicbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    return load_typed_font(size)


# ---------------------------------------------------------------------------
# Document text content (from seed_sample_data.py)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "id": 1,
        "name": "lease_agreement",
        "title": "Residential Lease Agreement",
        "content": """\
RESIDENTIAL LEASE AGREEMENT

This Residential Lease Agreement ("Agreement") is entered into as of the 1st day
of March, 2024, by and between:

LANDLORD: Hargrove Property Holdings LLC, a limited liability company organised
under the laws of the State of Thornfield, with its principal place of business at
4400 Willowmere Drive, Suite 200, Thornfield, TX 77022 ("Landlord");

AND

TENANT: Marcus Delacroix-Webb, an individual residing at 814 Cedarbrook Lane,
Apartment 3B, Thornfield, TX 77019 ("Tenant").

1. PREMISES
Landlord hereby leases to Tenant the residential premises located at 214 Finchley
Court, Apartment 7, Thornfield, TX 77031 (the "Premises"), consisting of a
two-bedroom, one-bathroom apartment of approximately 875 square feet.

2. TERM
The lease term shall commence on March 1, 2024 and shall terminate on
February 28, 2025 (the "Lease Term"), unless sooner terminated or extended in
accordance with this Agreement.

3. RENT
Tenant agrees to pay Landlord the sum of One Thousand Four Hundred Dollars
($1,400.00) per month ("Monthly Rent") as rent for the Premises, due and payable
on or before the first (1st) day of each calendar month during the Lease Term.
Rent shall be paid by electronic transfer to Hargrove Property Holdings LLC,
Account No. ending in 7742, Routing No. 111000025.

4. SECURITY DEPOSIT
Upon execution of this Agreement, Tenant shall deposit with Landlord the sum of
Two Thousand Eight Hundred Dollars ($2,800.00) ("Security Deposit"), equivalent
to two months' rent, as security for Tenant's faithful performance of all
obligations under this Agreement.

5. PERMITTED USE
The Premises shall be used solely as a private residential dwelling for Tenant
and Tenant's immediate family. No commercial activity of any kind shall be
conducted on the Premises without the prior written consent of Landlord.

6. UTILITIES AND SERVICES
Tenant shall be responsible for payment of the following utilities: electricity,
natural gas, internet, and cable television. Landlord shall be responsible for
water, sewer, and trash collection services.

7. MAINTENANCE AND REPAIRS
Tenant shall maintain the Premises in a clean and sanitary condition and shall
promptly notify Landlord of any damage or defect. Landlord shall be responsible
for structural repairs and maintenance of major systems (HVAC, plumbing,
electrical) provided that such damage is not caused by Tenant's negligence or
misuse.

8. PETS
No pets of any kind are permitted on the Premises without prior written consent
of Landlord. Unauthorised animals shall be grounds for termination of this
Agreement.

9. RENEWAL
Unless either party provides written notice of non-renewal at least thirty (30)
days prior to the expiration of the Lease Term, this Agreement shall
automatically convert to a month-to-month tenancy at the same Monthly Rent.

10. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of
the State of Thornfield.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date
first written above.

HARGROVE PROPERTY HOLDINGS LLC
By: /s/ Evelyn R. Hargrove
    Evelyn R. Hargrove, Managing Member
    Date: February 28, 2024

TENANT:
/s/ Marcus Delacroix-Webb
Marcus Delacroix-Webb
Date: February 28, 2024
""",
        "expected": {
            "document_type": "lease_agreement",
            "parties": {
                "landlord": "Hargrove Property Holdings LLC",
                "tenant": "Marcus Delacroix-Webb",
            },
            "dates": {
                "agreement_date": "2024-03-01",
                "lease_start": "2024-03-01",
                "lease_end": "2025-02-28",
                "execution_date": "2024-02-28",
            },
            "property": "214 Finchley Court, Apartment 7, Thornfield, TX 77031",
            "monthly_rent": "$1,400.00",
            "security_deposit": "$2,800.00",
            "governing_law": "State of Thornfield",
        },
    },
    {
        "id": 2,
        "name": "court_filing",
        "title": "Court Filing - Breach of Contract",
        "content": """\
IN THE DISTRICT COURT OF CALDWELL COUNTY
STATE OF THORNFIELD
CIVIL DIVISION

-----------------------------------------------------------
SYNTHEX INDUSTRIAL PARTNERS, INC.,        Case No.: CV-2024-08847
a Thornfield corporation,

                Plaintiff,

        v.

NORWOOD FABRICATION GROUP LLC,
a Delaware limited liability company,
and GERALD P. NORWOOD, individually,

                Defendants.
-----------------------------------------------------------

COMPLAINT FOR BREACH OF CONTRACT,
FRAUD, AND UNJUST ENRICHMENT

Plaintiff Synthex Industrial Partners, Inc. ("Synthex" or "Plaintiff"), by and
through its undersigned counsel, Kavanaugh & Ashworth LLP, hereby files this
Complaint against Defendants Norwood Fabrication Group LLC ("Norwood") and
Gerald P. Norwood ("Norwood Individual") (collectively "Defendants"), and in
support thereof alleges as follows:

PARTIES

1. Plaintiff Synthex Industrial Partners, Inc. is a corporation incorporated and
   existing under the laws of the State of Thornfield, with its principal place
   of business at 9900 Commerce Parkway, Caldwell City, TX 77045.

2. Defendant Norwood Fabrication Group LLC is a limited liability company
   organised under the laws of the State of Delaware, registered to do business
   in Thornfield, with its registered agent at 302 Industrial Loop, Caldwell
   City, TX 77046.

3. Defendant Gerald P. Norwood is an individual and the sole managing member of
   Norwood Fabrication Group LLC, residing at 512 Ridgecrest Drive, Caldwell
   City, TX 77048.

JURISDICTION AND VENUE

4. This Court has subject matter jurisdiction pursuant to Thornfield Civil
   Practice Code Section 14.001, as the amount in controversy exceeds
   $75,000.00 exclusive of interest and costs.

5. Venue is proper in Caldwell County pursuant to Thornfield Civil Practice
   Code Section 15.002(a)(1), as Defendants' principal place of business is
   located in Caldwell County and the events giving rise to this action occurred
   in Caldwell County.

FACTUAL BACKGROUND

6. On or about January 15, 2023, Synthex and Norwood entered into a Manufacturing
   Services Agreement ("MSA") under which Norwood agreed to fabricate and deliver
   3,500 custom industrial valve assemblies to Synthex at a total contract price
   of $840,000.00, with delivery to occur in four equal quarterly instalments
   beginning April 1, 2023.

7. Synthex paid Norwood an initial deposit of $210,000.00 on January 20, 2023,
   representing 25% of the total contract price, as required by Section 4.1
   of the MSA.

8. Norwood delivered the first instalment of 875 valve assemblies on April 3,
   2023. Upon inspection, Synthex discovered that 312 of the 875 assemblies
   failed to meet the pressure tolerance specifications set forth in Exhibit B
   of the MSA.

9. Synthex provided written notice of the defects to Norwood on April 10, 2023.
   Norwood failed to cure the defects within the 30-day cure period required
   by Section 8.3 of the MSA.

10. Norwood failed to deliver the second, third, and fourth instalments. Despite
    written demands dated July 5, 2023, October 6, 2023, and January 8, 2024,
    Norwood did not deliver the remaining 2,625 valve assemblies.

CAUSES OF ACTION

COUNT I - BREACH OF CONTRACT
11. Paragraphs 1 through 10 are incorporated by reference.
12. Norwood materially breached the MSA by delivering defective goods and
    failing to deliver the remaining contract quantity.
13. Synthex has suffered damages of not less than $680,000.00.

COUNT II - FRAUD
14. Paragraphs 1 through 10 are incorporated by reference.
15. Gerald P. Norwood personally represented that Norwood possessed the
    manufacturing capacity to fulfil the MSA at the time of contracting.
16. This representation was false; Norwood lacked the equipment and workforce
    to manufacture the contracted quantity.
17. Synthex relied on this misrepresentation in entering the MSA and paying the
    initial deposit.

COUNT III - UNJUST ENRICHMENT
18. Defendants have been unjustly enriched by retaining Synthex's $210,000.00
    deposit without performing the corresponding contractual obligations.

PRAYER FOR RELIEF

WHEREFORE, Plaintiff respectfully requests that this Court:
(a) Award compensatory damages of not less than $680,000.00;
(b) Award punitive damages on the fraud claim;
(c) Award pre-judgment and post-judgment interest;
(d) Award attorneys' fees and costs;
(e) Grant such other and further relief as the Court deems just and proper.

Respectfully submitted,
Date: March 14, 2024

KAVANAUGH & ASHWORTH LLP
/s/ Diane L. Kavanaugh
Diane L. Kavanaugh (Bar No. 19872341)
1200 Commerce Tower, Suite 3400
Caldwell City, TX 77002
Tel: (713) 555-0147
Attorneys for Plaintiff Synthex Industrial Partners, Inc.
""",
        "expected": {
            "document_type": "court_filing",
            "case_number": "CV-2024-08847",
            "court": "District Court of Caldwell County, State of Thornfield, Civil Division",
            "parties": {
                "plaintiff": "Synthex Industrial Partners, Inc.",
                "defendants": [
                    "Norwood Fabrication Group LLC",
                    "Gerald P. Norwood",
                ],
            },
            "filing_date": "2024-03-14",
            "claims": [
                "Breach of Contract",
                "Fraud",
                "Unjust Enrichment",
            ],
            "damages_sought": "$680,000.00 compensatory plus punitive damages",
            "contract_at_issue": "Manufacturing Services Agreement dated 2023-01-15",
            "attorneys_for_plaintiff": "Kavanaugh & Ashworth LLP",
        },
    },
    {
        "id": 3,
        "name": "property_deed",
        "title": "General Warranty Deed",
        "content": """\
GENERAL WARRANTY DEED

STATE OF THORNFIELD
COUNTY OF CALDWELL

KNOW ALL PERSONS BY THESE PRESENTS:

That BERTRAM H. LOCKWOOD and DOROTHY M. LOCKWOOD, husband and wife
("Grantor"), for and in consideration of the sum of FOUR HUNDRED TWENTY THOUSAND
DOLLARS ($420,000.00) in hand paid by FIDELITY MERIDIAN TRUST, a Thornfield
statutory trust ("Grantee"), the receipt and sufficiency of which are hereby
acknowledged, do hereby GRANT, BARGAIN, SELL and CONVEY unto Grantee, and
Grantee's successors and assigns forever, all that certain lot or parcel of land
situated in Caldwell County, Thornfield, described as follows:

LEGAL DESCRIPTION:

Lot 14, Block 7, of CEDARBROOK ESTATES, SECOND FILING, a subdivision in Caldwell
County, Thornfield, according to the Plat thereof recorded in Volume 82, Page 341,
of the Plat Records of Caldwell County, Thornfield.

Street Address: 1407 Cedarbrook Lane, Caldwell City, TX 77031

CHAIN OF TITLE SUMMARY (attached as Exhibit A):

1. Original Grant: The State of Thornfield granted the parcel to Elias Monroe
   Whitfield by Patent No. T-1188, dated September 12, 1901, recorded in
   Volume 2, Page 88, Deed Records of Caldwell County.

2. Whitfield to Carmichael: Elias Monroe Whitfield conveyed to Robert J.
   Carmichael by Warranty Deed dated March 3, 1924, recorded in Volume 14,
   Page 212, Deed Records of Caldwell County.

3. Carmichael to Voss Development Corp.: Robert J. Carmichael and Edna L.
   Carmichael conveyed to Voss Development Corporation by Warranty Deed dated
   June 18, 1952, recorded in Volume 31, Page 88, Deed Records of Caldwell County.

4. Voss Development Corp. to Lockwood (Original): Voss Development Corporation
   conveyed Lot 14, Block 7, Cedarbrook Estates Second Filing, to Harold G.
   Lockwood by Warranty Deed dated November 9, 1967, recorded in Volume 48,
   Page 117, Deed Records of Caldwell County.

5. Lockwood (Original) to Lockwood (Current Grantors): Harold G. Lockwood
   devised the property to Bertram H. Lockwood and Dorothy M. Lockwood by
   Last Will and Testament, probated in Caldwell County Probate Court Case
   No. P-1989-0441, Order dated February 2, 1990, recorded in Volume 71,
   Page 209, Deed Records of Caldwell County.

ENCUMBRANCES AND EXCEPTIONS:

This conveyance is made subject to the following:
1. A Deed of Trust lien in favour of Caldwell Savings Bank recorded in Volume
   89, Page 34, Deed of Trust Records of Caldwell County, dated January 15, 2008,
   in the original principal amount of $185,000.00 (to be released at closing).
2. A ten-foot utility easement along the western boundary of the property,
   recorded in Volume 52, Page 17, Easement Records of Caldwell County.
3. Restrictive covenants applicable to Cedarbrook Estates Second Filing, recorded
   in Volume 82, Page 344, Deed Records of Caldwell County.
4. Ad valorem taxes for the current year, not yet due and payable.

TO HAVE AND TO HOLD the above-described premises, together with all and singular
the appurtenances thereunto belonging or in anywise appertaining, to the only
proper use, benefit, and behoof of the said Grantee, Grantee's successors and
assigns forever.

AND Grantor covenants with Grantee that Grantor is lawfully seized of the
premises in fee simple; has good right to convey the same; the premises are free
from all encumbrances except as stated above; and Grantor will warrant and defend
the title to the premises against all lawful claims whatsoever.

IN WITNESS WHEREOF, Grantor has executed this Deed as of September 4, 2024.

/s/ Bertram H. Lockwood
BERTRAM H. LOCKWOOD

/s/ Dorothy M. Lockwood
DOROTHY M. LOCKWOOD

STATE OF THORNFIELD
COUNTY OF CALDWELL

Before me, the undersigned Notary Public, personally appeared Bertram H. Lockwood
and Dorothy M. Lockwood, known to me to be the persons whose names are subscribed
to the foregoing instrument, and acknowledged that they executed the same for the
purposes and consideration therein expressed.

/s/ Patricia O. Gillespie
PATRICIA O. GILLESPIE, Notary Public
Commission No. 98-7741
My Commission Expires: December 31, 2026

Filed for record: September 9, 2024
Instrument No.: 2024-0394817
Volume 114, Page 22, Deed Records of Caldwell County, Thornfield
""",
        "expected": {
            "document_type": "property_deed",
            "deed_type": "General Warranty Deed",
            "grantor": "Bertram H. Lockwood and Dorothy M. Lockwood",
            "grantee": "Fidelity Meridian Trust",
            "property_description": "Lot 14, Block 7, Cedarbrook Estates Second Filing, Caldwell County, Thornfield",
            "street_address": "1407 Cedarbrook Lane, Caldwell City, TX 77031",
            "consideration": "$420,000.00",
            "execution_date": "2024-09-04",
            "recording_date": "2024-09-09",
            "instrument_number": "2024-0394817",
            "ownership_chain": [
                "State of Thornfield (Patent 1901)",
                "Elias Monroe Whitfield",
                "Robert J. Carmichael",
                "Voss Development Corporation",
                "Harold G. Lockwood",
                "Bertram H. Lockwood and Dorothy M. Lockwood",
                "Fidelity Meridian Trust",
            ],
            "encumbrances": [
                "Deed of Trust lien - Caldwell Savings Bank (to be released at closing)",
                "Ten-foot utility easement along western boundary",
                "Restrictive covenants - Cedarbrook Estates Second Filing",
                "Ad valorem taxes for current year",
            ],
        },
    },
    {
        "id": 4,
        "name": "compliance_notice",
        "title": "Notice of Violation and Order to Comply",
        "content": """\
THORNFIELD DEPARTMENT OF ENVIRONMENTAL QUALITY
OFFICE OF REGULATORY COMPLIANCE

NOTICE OF VIOLATION AND ORDER TO COMPLY

Notice No.: TEQ-NOV-2024-1187
Date Issued: August 5, 2024

TO:
Pinnacle Chemical Manufacturing Co.
Attention: Mr. Stewart G. Albrecht, Chief Compliance Officer
3300 Refinery Road
Port Caldwell, TX 77530

RE: NOTICE OF VIOLATION — THORNFIELD CLEAN AIR ACT, CHAPTER 382
    FACILITY PERMIT NO. AIR-2019-4422

FINDINGS OF VIOLATION

Pursuant to an inspection conducted on July 18, 2024, by Compliance Officer
Renata Okonkwo (Inspector ID: TEQ-CO-0091), the Thornfield Department of
Environmental Quality ("Department") has determined that Pinnacle Chemical
Manufacturing Co. ("Facility" or "Respondent") has committed the following
violations at the above-referenced facility:

VIOLATION 1: Excess Volatile Organic Compound (VOC) Emissions
During the inspection period of July 14-18, 2024, continuous emissions monitoring
data recorded VOC emissions from Stack 4-B of 148 parts per million (ppm) on
a rolling 24-hour average basis. This exceeds the permitted limit of 85 ppm
established in Permit Condition 3.4.1 by 74.1%. This constitutes a violation of
Thornfield Clean Air Act Section 382.085(b).

VIOLATION 2: Incomplete Quarterly Emissions Report
Respondent failed to submit the Quarterly Emissions Report for Q2 2024 (April-June
2024), which was due on July 31, 2024. This constitutes a violation of Thornfield
Clean Air Act Section 382.141(a) and Permit Condition 6.1.2.

VIOLATION 3: Deficient Stack Monitoring Equipment
Portable opacity monitor OM-7 was found to be operating with a calibration
certificate expired on March 31, 2024. Data collected by OM-7 since that date is
presumed unreliable. This constitutes a violation of Thornfield Clean Air Act
Section 382.162 and Permit Condition 5.3.1.

REQUIRED CORRECTIVE ACTIONS AND DEADLINES

The Department hereby orders Respondent to:

ACTION 1: IMMEDIATE — Within 48 hours of receipt of this Notice
Reduce VOC emissions from Stack 4-B to at or below the permitted limit of 85 ppm.
Provide written confirmation of corrective action to the Department by email to
compliance@teq.thornfield.gov, Attention: NOV Case Manager.

ACTION 2: WITHIN 14 DAYS — By August 19, 2024
Submit the overdue Q2 2024 Quarterly Emissions Report in full compliance with
Section 382.141(a) using the Department's eDEQ online reporting portal
(portal.teq.thornfield.gov). Late submission penalties may apply pursuant to
Thornfield Administrative Code Section 7.048.

ACTION 3: WITHIN 30 DAYS — By September 4, 2024
(a) Recalibrate or replace portable opacity monitor OM-7 and provide to the
    Department a copy of the new calibration certificate from an accredited
    laboratory.
(b) Submit a Data Quality Assessment Report covering the period April 1 -
    August 5, 2024, identifying any monitoring data affected by the deficient
    instrument and proposing substitute data or conservative estimates in
    accordance with Thornfield Air Monitoring Guidelines, Chapter 6.

ACTION 4: WITHIN 60 DAYS — By October 4, 2024
Submit a Root Cause Analysis and Corrective Action Plan ("RCA/CAP") addressing
all three violations. The RCA/CAP must include:
- Root cause identification for each violation
- Corrective measures implemented or planned
- Preventive measures to ensure future compliance
- A compliance schedule with milestone dates

PENALTY NOTICE

The Department is authorised to assess administrative penalties of up to $10,000
per day per violation under Thornfield Clean Air Act Section 382.352. A penalty
determination will be made following review of Respondent's corrective actions.
Timely and complete compliance will be considered as a mitigating factor.

RESPONSE AND HEARING RIGHTS

Respondent may request a conference with the Department within 20 calendar days
of receipt of this Notice (by August 25, 2024) by submitting a written request to:

Office of Regulatory Compliance
Thornfield Department of Environmental Quality
P.O. Box 13087
Caldwell City, TX 77711-3087

Questions regarding this Notice should be directed to Compliance Officer
Renata Okonkwo at (512) 555-0294 or r.okonkwo@teq.thornfield.gov.

/s/ Commissioner Raymond T. Bledsoe
RAYMOND T. BLEDSOE
Commissioner, Thornfield Department of Environmental Quality
Date: August 5, 2024
""",
        "expected": {
            "document_type": "compliance_notice",
            "notice_number": "TEQ-NOV-2024-1187",
            "notice_type": "Notice of Violation and Order to Comply",
            "issuing_authority": "Thornfield Department of Environmental Quality",
            "respondent": "Pinnacle Chemical Manufacturing Co.",
            "respondent_contact": "Stewart G. Albrecht, Chief Compliance Officer",
            "issue_date": "2024-08-05",
            "permit_number": "AIR-2019-4422",
            "violations": [
                "Excess VOC emissions from Stack 4-B (148 ppm vs 85 ppm limit)",
                "Failure to submit Q2 2024 Quarterly Emissions Report",
                "Expired calibration certificate on opacity monitor OM-7",
            ],
            "deadlines": {
                "action_1_immediate": "Within 48 hours of receipt",
                "action_2": "2024-08-19",
                "action_3": "2024-09-04",
                "action_4": "2024-10-04",
                "conference_request_deadline": "2024-08-25",
            },
            "max_penalty_per_day": "$10,000 per day per violation",
        },
    },
    {
        "id": 5,
        "name": "document_checklist",
        "title": "Document Checklist - Commercial Real Estate Acquisition",
        "content": """\
DOCUMENT CHECKLIST
COMMERCIAL REAL ESTATE ACQUISITION
ACQUISITION OF 88 WESTBROOK PLAZA, CALDWELL CITY, TX 77036

TRANSACTION REFERENCE: ACQ-2024-PSL-0087
Prepared by: Meridian Title and Escrow Services LLC
Date Prepared: October 15, 2024
Buyer: Westbrook Commercial Holdings LLC
Seller: Alcott-Meridian Properties Inc.
Escrow Officer: Howard V. Kincaid, Jr.
Expected Closing Date: November 30, 2024

---

SECTION A — TITLE AND OWNERSHIP DOCUMENTS

[X] 1. Preliminary Title Report                         Received: Oct 10, 2024
[X] 2. Current Vesting Deed (Seller)                   Received: Oct 11, 2024
[ ] 3. Seller's Statement of Title                     PENDING — due Oct 22, 2024
[X] 4. Survey — ALTA/NSPS Land Title Survey            Received: Oct 8, 2024
[ ] 5. Release of Mortgage Lien (First National Bank)  PENDING — lender payoff
        requested Oct 14, 2024; payoff statement expected Oct 18, 2024
[X] 6. HOA/Property Association Estoppel Certificate   Received: Oct 9, 2024

---

SECTION B — ENTITY AND AUTHORITY DOCUMENTS

[X] 7.  Buyer Entity: Articles of Organisation         Received: Oct 1, 2024
[X] 8.  Buyer Entity: Operating Agreement              Received: Oct 1, 2024
[X] 9.  Buyer Entity: Certificate of Good Standing     Received: Oct 2, 2024
[ ] 10. Buyer Entity: Resolution Authorising Purchase  MISSING — to be provided
         by Buyer's counsel no later than Oct 25, 2024
[X] 11. Seller Entity: Certificate of Good Standing    Received: Oct 3, 2024
[X] 12. Seller Entity: Resolution Authorising Sale     Received: Oct 11, 2024
[ ] 13. Seller Entity: IRS Form W-9                    PENDING — requested Oct 12

---

SECTION C — FINANCIAL AND LENDING DOCUMENTS

[X] 14. Purchase and Sale Agreement (fully executed)   Received: Sept 28, 2024
[X] 15. Loan Commitment Letter — Horizon Bank NA       Received: Oct 7, 2024
[ ] 16. Loan Agreement (final form)                    PENDING — expected Oct 20
[ ] 17. Deed of Trust — Horizon Bank NA                PENDING — expected Oct 25
[X] 18. Proof of Earnest Money Deposit ($125,000)      Received: Sept 30, 2024
[ ] 19. Closing Disclosure (HUD-1 equivalent)          PENDING — to be prepared by
         Meridian Title no later than Nov 15, 2024

---

SECTION D — DUE DILIGENCE DOCUMENTS

[X] 20. Phase I Environmental Site Assessment          Received: Oct 5, 2024
         (No Recognised Environmental Conditions found)
[ ] 21. Phase II Environmental Site Assessment        NOT REQUIRED per Phase I
[ ] 22. Property Condition Report                     PENDING — inspector scheduled
         Oct 18, 2024; report expected Oct 25, 2024
[X] 23. Zoning Compliance Letter — City of Caldwell   Received: Oct 10, 2024
[X] 24. Lease Abstracts (all 4 existing tenants)      Received: Oct 12, 2024
[ ] 25. Tenant Estoppel Certificates (4 of 4)         PARTIAL — 2 of 4 received;
         outstanding tenants: Unit 4B (Greystone Consulting) and Unit 6A (open)
         Deadline for remaining: Oct 28, 2024
[X] 26. Rent Roll (current as of Oct 1, 2024)         Received: Oct 11, 2024
[ ] 27. Operating Expense Statement (trailing 12 mo.) PENDING — due from Seller
         by Oct 22, 2024

---

SECTION E — CLOSING AND RECORDING

[ ] 28. Grant Deed (Seller to Buyer) — draft          PENDING — draft to be
         prepared by Meridian Title by Nov 1, 2024
[ ] 29. Escrow Instructions (bilateral)               PENDING — draft circulated;
         awaiting Seller's counsel comments by Oct 18, 2024
[X] 30. Wire Instructions — Meridian Title Escrow     Received: Oct 15, 2024
[ ] 31. Title Insurance Policy — ALTA Owner's Policy  PENDING — to be issued at
         closing by Meridian Title and Escrow Services LLC
[ ] 32. Title Insurance Policy — ALTA Lender's Policy PENDING — to be issued at
         closing per Horizon Bank NA requirements

---

SUMMARY

Total items on checklist:    32
Items received/completed:    16  (50%)
Items pending:               13  (41%)
Items not required:           1  (3%)
Items missing (no response):  2  (6%)

CRITICAL PATH ITEMS (must be resolved to meet Nov 30 closing):
- Item 10: Buyer resolution authorising purchase (due Oct 25)
- Item 16: Final loan agreement (expected Oct 20)
- Item 17: Deed of Trust (expected Oct 25)
- Item 25: Remaining tenant estoppel certificates (due Oct 28)
- Item 28: Grant deed draft (due Nov 1)

Escrow Officer: /s/ Howard V. Kincaid, Jr.
Date: October 15, 2024
""",
        "expected": {
            "document_type": "document_checklist",
            "transaction_reference": "ACQ-2024-PSL-0087",
            "property": "88 Westbrook Plaza, Caldwell City, TX 77036",
            "buyer": "Westbrook Commercial Holdings LLC",
            "seller": "Alcott-Meridian Properties Inc.",
            "escrow_officer": "Howard V. Kincaid, Jr.",
            "expected_closing_date": "2024-11-30",
            "total_items": 32,
            "items_received": 16,
            "items_pending": 13,
            "items_missing": 2,
            "critical_path_items": [
                "Buyer resolution authorising purchase (due 2024-10-25)",
                "Final loan agreement (expected 2024-10-20)",
                "Deed of Trust (expected 2024-10-25)",
                "Remaining tenant estoppel certificates (due 2024-10-28)",
                "Grant deed draft (due 2024-11-01)",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Text rendering helper
# ---------------------------------------------------------------------------

def render_text_to_image(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    page_width: int = PAGE_WIDTH,
    page_height: int = PAGE_HEIGHT,
    margin: int = MARGIN,
) -> Image.Image:
    """Render multi-line text onto a white page image."""
    img = Image.new("L", (page_width, page_height), 255)  # grayscale white
    draw = ImageDraw.Draw(img)

    # Determine line height
    try:
        bbox = font.getbbox("Ag")
        line_height = int((bbox[3] - bbox[1]) * 1.4)
    except AttributeError:
        line_height = 24

    # Word-wrap text to fit within margins
    max_width = page_width - 2 * margin
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            lines.append("")
            continue
        # Simple word wrap
        words = raw_line.split(" ")
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            try:
                w = draw.textlength(test_line, font=font)
            except (TypeError, AttributeError):
                w = len(test_line) * 10
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

    # Draw lines
    y = margin
    for line in lines:
        if y + line_height > page_height - margin:
            break  # don't overflow page
        draw.text((margin, y), line, fill=0, font=font)
        y += line_height

    return img


# ---------------------------------------------------------------------------
# Degradation functions
# ---------------------------------------------------------------------------

def add_noise(img: Image.Image, intensity: float = 0.05) -> Image.Image:
    """Add salt-and-pepper noise to an image."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_specks(img: Image.Image, count: int = 50) -> Image.Image:
    """Add random dark specks (simulating dust/dirt on scanner)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(count):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(1, 4)
        fill = random.randint(0, 80)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    return img


def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image by given angle (degrees), fill with white."""
    return img.rotate(angle, expand=False, fillcolor=255)


def apply_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    """Apply Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_partial_blur(img: Image.Image, num_patches: int = 3, radius: float = 4.0) -> Image.Image:
    """Apply blur to random rectangular patches."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_arr = np.array(blurred)

    for _ in range(num_patches):
        px = random.randint(0, w - 300)
        py = random.randint(0, h - 300)
        pw = random.randint(200, 500)
        ph = random.randint(200, 500)
        arr[py:py+ph, px:px+pw] = blurred_arr[py:py+ph, px:px+pw]

    return Image.fromarray(arr)


def apply_coffee_stain(img: Image.Image) -> Image.Image:
    """Add a semi-transparent brownish ellipse simulating a coffee ring."""
    overlay = Image.new("L", img.size, 255)
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    rx = random.randint(150, 350)
    ry = random.randint(150, 350)
    # Draw a ring (outer minus inner ellipse)
    ring_width = random.randint(15, 40)
    # Outer ellipse (darker)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=200)
    # Inner ellipse (restore white)
    draw.ellipse(
        [cx - rx + ring_width, cy - ry + ring_width,
         cx + rx - ring_width, cy + ry - ring_width],
        fill=255,
    )
    # Composite: darken where the ring is
    img_arr = np.array(img, dtype=np.float32)
    overlay_arr = np.array(overlay, dtype=np.float32)
    # Where overlay is darker than white, blend
    result = np.minimum(img_arr, overlay_arr).astype(np.uint8)
    return Image.fromarray(result)


def apply_vignette(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """Apply vignette effect (darker edges, brighter center)."""
    w, h = img.size
    # Create radial gradient
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    # Normalized distance from center (0 at center, ~1 at corners)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    dist_norm = dist / max_dist
    # Vignette mask: 1.0 at center, decreasing at edges
    mask = 1.0 - strength * (dist_norm ** 2)
    mask = np.clip(mask, 0.3, 1.0)

    arr = np.array(img, dtype=np.float32)
    # Apply: darken pixels at edges
    # For grayscale: lighter = 255, darker = 0
    # We want edges darker, so we blend toward 0
    result = arr * mask
    return Image.fromarray(result.astype(np.uint8))


def apply_fade_regions(img: Image.Image, num_regions: int = 3, fade_amount: float = 0.4) -> Image.Image:
    """Fade text in random rectangular regions (simulate faded ink)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    for _ in range(num_regions):
        px = random.randint(0, w - 400)
        py = random.randint(0, h - 300)
        pw = random.randint(300, 700)
        ph = random.randint(200, 500)
        # Fade toward white
        region = arr[py:py+ph, px:px+pw]
        arr[py:py+ph, px:px+pw] = region + (255 - region) * fade_amount
    return Image.fromarray(arr.astype(np.uint8))


def apply_margin_cutoff(img: Image.Image, side: str = "right", amount: int = 200) -> Image.Image:
    """Simulate text cut off at a margin by painting a strip white (or black for scan edge)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if side == "right":
        draw.rectangle([w - amount, 0, w, h], fill=20)
    elif side == "left":
        draw.rectangle([0, 0, amount, h], fill=20)
    elif side == "bottom":
        draw.rectangle([0, h - amount, w, h], fill=20)
    return img


def apply_word_deletion(img: Image.Image, num_smudges: int = 8) -> Image.Image:
    """Simulate ink smudges that obscure words (dark blobs)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(num_smudges):
        x = random.randint(MARGIN, w - MARGIN)
        y = random.randint(MARGIN, h - MARGIN)
        sw = random.randint(60, 200)
        sh = random.randint(20, 50)
        draw.ellipse([x, y, x + sw, y + sh], fill=random.randint(10, 60))
    return img


def apply_torn_page(img: Image.Image) -> Image.Image:
    """Simulate a torn corner by drawing a black polygon."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Tear from a random corner
    corner = random.choice(["top_right", "bottom_left", "bottom_right"])
    if corner == "top_right":
        points = [
            (w - random.randint(200, 500), 0),
            (w, 0),
            (w, random.randint(200, 600)),
        ]
    elif corner == "bottom_left":
        points = [
            (0, h - random.randint(200, 500)),
            (0, h),
            (random.randint(200, 500), h),
        ]
    else:  # bottom_right
        points = [
            (w - random.randint(200, 500), h),
            (w, h),
            (w, h - random.randint(200, 600)),
        ]
    # Add some irregularity
    jittered = []
    for px, py in points:
        jittered.append((px + random.randint(-30, 30), py + random.randint(-30, 30)))
    draw.polygon(jittered, fill=15)
    return img


def apply_watermark(img: Image.Image, text: str = "COPY") -> Image.Image:
    """Overlay a diagonal watermark text."""
    w, h = img.size
    # Create watermark layer
    wm = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(wm)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 200)
    except (OSError, IOError):
        font = load_typed_font(200)

    # Draw text rotated at 45 degrees in center
    # Create a temporary image for the text
    txt_img = Image.new("L", (1500, 400), 255)
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((50, 50), text, fill=230, font=font)
    txt_img = txt_img.rotate(35, expand=True, fillcolor=255)

    # Paste centered
    tw, th = txt_img.size
    offset_x = (w - tw) // 2
    offset_y = (h - th) // 2
    wm.paste(txt_img, (offset_x, offset_y))

    # Composite: min of both (watermark is lighter gray, so it shows through)
    img_arr = np.array(img, dtype=np.float32)
    wm_arr = np.array(wm, dtype=np.float32)
    result = np.minimum(img_arr, wm_arr).astype(np.uint8)
    return Image.fromarray(result)


def apply_ghost_image(img: Image.Image, ghost_img: Image.Image, opacity: float = 0.15) -> Image.Image:
    """Overlay a ghost image (simulating two pages overlapping in scanner)."""
    w, h = img.size
    # Resize ghost to match
    ghost_resized = ghost_img.resize((w, h), Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32)
    ghost_arr = np.array(ghost_resized, dtype=np.float32)
    # Blend: where ghost is dark, slightly darken the main image
    blended = img_arr * (1.0 - opacity) + ghost_arr * opacity
    return Image.fromarray(blended.astype(np.uint8))


def apply_stamp(img: Image.Image) -> Image.Image:
    """Add a rectangular stamp-like marking (simulating an official stamp)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Position: lower-right area
    sx = random.randint(w // 2, w - 600)
    sy = random.randint(h // 2, h - 400)
    sw, sh = 400, 200
    # Draw rectangle border
    for i in range(4):
        draw.rectangle([sx + i, sy + i, sx + sw - i, sy + sh - i], outline=100)
    # Add stamp text
    try:
        stamp_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 40)
    except (OSError, IOError):
        stamp_font = load_typed_font(40)
    draw.text((sx + 30, sy + 40), "RECEIVED", fill=100, font=stamp_font)
    draw.text((sx + 30, sy + 100), "OCT 2024", fill=120, font=stamp_font)
    # Rotate the stamp slightly by re-drawing rotated
    return img


# ---------------------------------------------------------------------------
# Degradation levels
# ---------------------------------------------------------------------------

def degrade_level_1(img: Image.Image) -> Image.Image:
    """Level 1 (Easy): Clean typed text, slight skew, minor noise."""
    img = add_specks(img, count=30)
    img = add_noise(img, intensity=0.02)
    angle = random.uniform(1.0, 2.0) * random.choice([-1, 1])
    img = apply_rotation(img, angle)
    return img


def degrade_level_2(img: Image.Image) -> Image.Image:
    """Level 2 (Medium): Photocopy artifacts, faded regions, coffee stain."""
    img = apply_fade_regions(img, num_regions=2, fade_amount=0.3)
    img = add_noise(img, intensity=0.04)
    img = add_specks(img, count=100)
    img = apply_blur(img, radius=0.8)
    img = apply_coffee_stain(img)
    angle = random.uniform(3.0, 5.0) * random.choice([-1, 1])
    img = apply_rotation(img, angle)
    return img


def degrade_level_3(img: Image.Image) -> Image.Image:
    """Level 3 (Hard): Poor scan quality, vignette, margin cutoff."""
    img = add_noise(img, intensity=0.08)
    img = add_specks(img, count=200)
    img = apply_vignette(img, strength=0.6)
    img = apply_fade_regions(img, num_regions=4, fade_amount=0.5)
    img = apply_margin_cutoff(img, side="right", amount=180)
    angle = random.uniform(5.0, 8.0) * random.choice([-1, 1])
    img = apply_rotation(img, angle)
    img = apply_blur(img, radius=1.2)
    return img


def degrade_level_4(img: Image.Image, use_handwritten_font: bool = True) -> Image.Image:
    """Level 4 (Very Hard): Handwritten + degraded. If use_handwritten_font,
    the image should already be rendered with handwritten font."""
    img = add_noise(img, intensity=0.06)
    img = apply_word_deletion(img, num_smudges=10)
    img = apply_partial_blur(img, num_patches=4, radius=3.5)
    img = apply_torn_page(img)
    img = add_specks(img, count=150)
    angle = random.uniform(3.0, 6.0) * random.choice([-1, 1])
    img = apply_rotation(img, angle)
    return img


def degrade_level_5(
    img: Image.Image,
    ghost_source: Image.Image | None = None,
) -> Image.Image:
    """Level 5 (Extreme): Multi-issue — combination of all degradations."""
    img = add_noise(img, intensity=0.07)
    img = add_specks(img, count=250)
    img = apply_vignette(img, strength=0.4)
    img = apply_fade_regions(img, num_regions=3, fade_amount=0.45)
    img = apply_coffee_stain(img)
    img = apply_watermark(img, text="COPY")
    img = apply_stamp(img)
    img = apply_word_deletion(img, num_smudges=6)
    if ghost_source is not None:
        img = apply_ghost_image(img, ghost_source, opacity=0.12)
    img = apply_partial_blur(img, num_patches=2, radius=2.5)
    img = apply_margin_cutoff(img, side="bottom", amount=120)
    angle = random.uniform(4.0, 7.0) * random.choice([-1, 1])
    img = apply_rotation(img, angle)
    return img


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("LegalMind — Synthetic Scanned Document Generator")
    print("=" * 70)
    print()

    # Ensure output directories exist
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    # Load fonts
    typed_font = load_typed_font(36)
    handwritten_font = load_handwritten_font(40)

    print(f"Typed font: {typed_font}")
    print(f"Handwritten font: {handwritten_font}")
    print()

    generated_images: list[str] = []
    generated_pdfs: list[str] = []
    generated_expected: list[str] = []

    # Pre-render all base images (typed) for ghost overlay use
    base_images: list[Image.Image] = []
    for doc in DOCUMENTS:
        base_img = render_text_to_image(doc["content"], typed_font)
        base_images.append(base_img)

    for doc_idx, doc in enumerate(DOCUMENTS):
        doc_id = doc["id"]
        doc_name = doc["name"]
        doc_content = doc["content"]
        doc_expected = doc["expected"]

        print(f"Processing document {doc_id}: {doc['title']}")

        for level in range(1, 6):
            # Reset random state per document-level pair for reproducibility
            seed_val = 42 + doc_id * 100 + level
            random.seed(seed_val)
            np.random.seed(seed_val)

            # Render text: level 4+ uses handwritten font for part/all
            if level == 4:
                base_img = render_text_to_image(doc_content, handwritten_font)
            elif level == 5:
                # Mixed: render typed, but we'll also use handwritten for
                # the ghost overlay effect
                base_img = render_text_to_image(doc_content, typed_font)
            else:
                base_img = render_text_to_image(doc_content, typed_font)

            # Apply degradations
            if level == 1:
                degraded = degrade_level_1(base_img.copy())
            elif level == 2:
                degraded = degrade_level_2(base_img.copy())
            elif level == 3:
                degraded = degrade_level_3(base_img.copy())
            elif level == 4:
                degraded = degrade_level_4(base_img.copy(), use_handwritten_font=True)
            elif level == 5:
                # Use a different document as ghost source
                ghost_idx = (doc_idx + 1) % len(DOCUMENTS)
                ghost_source = base_images[ghost_idx]
                degraded = degrade_level_5(base_img.copy(), ghost_source=ghost_source)
            else:
                degraded = base_img

            # Convert to RGB for PDF compatibility
            degraded_rgb = degraded.convert("RGB")

            # Save PNG
            img_filename = f"img_{doc_id}_level_{level}.png"
            img_path = IMAGES_DIR / img_filename
            degraded_rgb.save(str(img_path), "PNG")
            generated_images.append(img_filename)

            # Save PDF (single-page PDF from image)
            pdf_filename = f"img_{doc_id}_level_{level}.pdf"
            pdf_path = PDFS_DIR / pdf_filename
            degraded_rgb.save(str(pdf_path), "PDF", resolution=300.0)
            generated_pdfs.append(pdf_filename)

            # Save expected output JSON
            expected_filename = f"img_{doc_id}_level_{level}_expected.json"
            expected_path = EXPECTED_DIR / expected_filename
            expected_data = {
                "source_document": f"doc_{doc_id}_{doc_name}",
                "image_file": img_filename,
                "pdf_file": pdf_filename,
                "difficulty_level": level,
                "difficulty_label": {
                    1: "easy",
                    2: "medium",
                    3: "hard",
                    4: "very_hard",
                    5: "extreme",
                }[level],
                "degradations_applied": {
                    1: ["minor_noise", "specks", "slight_rotation"],
                    2: ["faded_regions", "noise", "specks", "blur", "coffee_stain", "moderate_rotation"],
                    3: ["heavy_noise", "specks", "vignette", "faded_regions", "margin_cutoff", "significant_rotation", "blur"],
                    4: ["handwritten_font", "noise", "ink_smudges", "partial_blur", "torn_page", "specks", "rotation"],
                    5: ["noise", "specks", "vignette", "faded_regions", "coffee_stain", "watermark", "stamp", "ink_smudges", "ghost_image", "partial_blur", "margin_cutoff", "rotation"],
                }[level],
                "expected_extracted_fields": doc_expected,
            }
            expected_path.write_text(
                json.dumps(expected_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            generated_expected.append(expected_filename)

            print(f"  Level {level} ({expected_data['difficulty_label']:>9s}): "
                  f"{img_filename} + {pdf_filename}")

        print()

    # Print summary
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Images generated:          {len(generated_images)}")
    print(f"PDFs generated:            {len(generated_pdfs)}")
    print(f"Expected output JSONs:     {len(generated_expected)}")
    print()
    print(f"Image output directory:    {IMAGES_DIR}")
    print(f"PDF output directory:      {PDFS_DIR}")
    print(f"Expected outputs directory:{EXPECTED_DIR}")
    print()
    print("Files:")
    print("-" * 40)
    for name in generated_images:
        print(f"  images/{name}")
    print()
    for name in generated_pdfs:
        print(f"  pdfs/{name}")
    print()
    for name in generated_expected:
        print(f"  expected_outputs/{name}")


if __name__ == "__main__":
    main()
