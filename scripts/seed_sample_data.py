"""
Seed sample data for LegalMind demo and evaluation.
Creates synthetic legal documents (fully fictional) and their expected outputs.

All names, addresses, case numbers, and other identifying details are entirely
fictional and do not refer to any real person, company, property, or legal matter.

Run:
    python scripts/seed_sample_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
EXPECTED_DIR = SAMPLE_DIR / "expected_outputs"


# ---------------------------------------------------------------------------
# Document definitions
# ---------------------------------------------------------------------------

DOCUMENTS: list[dict] = []

# ------ Document 1: Residential Lease Agreement ----------------------------

DOCUMENTS.append(
    {
        "filename": "doc_1_lease_agreement.txt",
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
            "document": "doc_1_lease_agreement.txt",
            "draft_type": "case_fact_summary",
            "expected_fields": {
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
    }
)

# ------ Document 2: Court Filing -------------------------------------------

DOCUMENTS.append(
    {
        "filename": "doc_2_court_filing.txt",
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
            "document": "doc_2_court_filing.txt",
            "draft_type": "case_fact_summary",
            "expected_fields": {
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
    }
)

# ------ Document 3: Property Deed with Chain of Title ----------------------

DOCUMENTS.append(
    {
        "filename": "doc_3_property_deed.txt",
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
            "document": "doc_3_property_deed.txt",
            "draft_type": "title_review_summary",
            "expected_fields": {
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
                "title_gaps": [],
            },
        },
    }
)

# ------ Document 4: Compliance Notice with Deadlines ----------------------

DOCUMENTS.append(
    {
        "filename": "doc_4_compliance_notice.txt",
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
During the inspection period of July 14–18, 2024, continuous emissions monitoring
data recorded VOC emissions from Stack 4-B of 148 parts per million (ppm) on
a rolling 24-hour average basis. This exceeds the permitted limit of 85 ppm
established in Permit Condition 3.4.1 by 74.1%. This constitutes a violation of
Thornfield Clean Air Act Section 382.085(b).

VIOLATION 2: Incomplete Quarterly Emissions Report
Respondent failed to submit the Quarterly Emissions Report for Q2 2024 (April–June
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
(b) Submit a Data Quality Assessment Report covering the period April 1 –
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
            "document": "doc_4_compliance_notice.txt",
            "draft_type": "notice_related_summary",
            "expected_fields": {
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
                "compliance_steps": [
                    "Reduce VOC emissions from Stack 4-B to at or below 85 ppm",
                    "Submit overdue Q2 2024 Quarterly Emissions Report via eDEQ portal",
                    "Recalibrate or replace opacity monitor OM-7",
                    "Submit Data Quality Assessment Report for April 1 - August 5 2024",
                    "Submit Root Cause Analysis and Corrective Action Plan",
                ],
                "max_penalty_per_day": "$10,000 per day per violation",
            },
        },
    }
)

# ------ Document 5: Document Checklist for a Transaction -------------------

DOCUMENTS.append(
    {
        "filename": "doc_5_document_checklist.txt",
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
            "document": "doc_5_document_checklist.txt",
            "draft_type": "document_checklist",
            "expected_fields": {
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
                "required_docs": [
                    "Preliminary Title Report",
                    "Current Vesting Deed (Seller)",
                    "Seller's Statement of Title",
                    "Survey - ALTA/NSPS Land Title Survey",
                    "Release of Mortgage Lien (First National Bank)",
                    "HOA/Property Association Estoppel Certificate",
                    "Buyer Entity: Articles of Organisation",
                    "Buyer Entity: Operating Agreement",
                    "Buyer Entity: Certificate of Good Standing",
                    "Buyer Entity: Resolution Authorising Purchase",
                    "Seller Entity: Certificate of Good Standing",
                    "Seller Entity: Resolution Authorising Sale",
                    "Seller Entity: IRS Form W-9",
                    "Purchase and Sale Agreement",
                    "Loan Commitment Letter",
                    "Loan Agreement (final form)",
                    "Deed of Trust - Horizon Bank NA",
                    "Proof of Earnest Money Deposit",
                    "Closing Disclosure",
                    "Phase I Environmental Site Assessment",
                    "Property Condition Report",
                    "Zoning Compliance Letter",
                    "Lease Abstracts",
                    "Tenant Estoppel Certificates (4 of 4)",
                    "Rent Roll",
                    "Operating Expense Statement",
                    "Grant Deed",
                    "Escrow Instructions",
                    "Wire Instructions",
                    "Title Insurance Policy - ALTA Owner's Policy",
                    "Title Insurance Policy - ALTA Lender's Policy",
                ],
                "present_docs": [
                    "Preliminary Title Report",
                    "Current Vesting Deed (Seller)",
                    "Survey - ALTA/NSPS Land Title Survey",
                    "HOA/Property Association Estoppel Certificate",
                    "Buyer Entity: Articles of Organisation",
                    "Buyer Entity: Operating Agreement",
                    "Buyer Entity: Certificate of Good Standing",
                    "Seller Entity: Certificate of Good Standing",
                    "Seller Entity: Resolution Authorising Sale",
                    "Purchase and Sale Agreement",
                    "Loan Commitment Letter",
                    "Proof of Earnest Money Deposit",
                    "Phase I Environmental Site Assessment",
                    "Zoning Compliance Letter",
                    "Lease Abstracts",
                    "Rent Roll",
                    "Wire Instructions",
                ],
                "missing_docs": [
                    "Buyer Entity: Resolution Authorising Purchase",
                    "Seller Entity: IRS Form W-9",
                ],
                "critical_path_items": [
                    "Buyer resolution authorising purchase (due 2024-10-25)",
                    "Final loan agreement (expected 2024-10-20)",
                    "Deed of Trust (expected 2024-10-25)",
                    "Remaining tenant estoppel certificates (due 2024-10-28)",
                    "Grant deed draft (due 2024-11-01)",
                ],
            },
        },
    }
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import sys
    sys.path.insert(0, str(REPO_ROOT))

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    written_docs: list[str] = []
    written_expected: list[str] = []

    for doc in DOCUMENTS:
        doc_path = SAMPLE_DIR / doc["filename"]
        doc_path.write_text(doc["content"], encoding="utf-8")
        written_docs.append(doc["filename"])

        stem = Path(doc["filename"]).stem
        expected_filename = f"{stem}_expected.json"
        expected_path = EXPECTED_DIR / expected_filename
        expected_path.write_text(
            json.dumps(doc["expected"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written_expected.append(expected_filename)

    print("Sample data seeded successfully.")
    print(f"  {len(written_docs)} documents written to {SAMPLE_DIR}")
    print(f"  {len(written_expected)} expected outputs written to {EXPECTED_DIR}")

    # Prebuilt processed JSONs in data/sample/prebuilt/ are NOT copied to
    # data/processed/ by default — operators want the library to reflect only
    # documents they have actually uploaded. Pass --copy-prebuilt to opt in
    # (e.g. for a quick demo against canned sample drafts).
    import sys as _sys
    if "--copy-prebuilt" in _sys.argv:
        try:
            from config.paths import ensure_dirs, PROCESSED_DIR
            ensure_dirs()

            import shutil
            prebuilt_dir = SAMPLE_DIR / "prebuilt"
            if prebuilt_dir.exists():
                copied = 0
                for f in prebuilt_dir.glob("*.json"):
                    dest = PROCESSED_DIR / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
                        copied += 1
                print(f"  {copied} prebuilt documents/drafts copied to processed/ (--copy-prebuilt)")
            else:
                print("  [WARN] No prebuilt data directory found")

            from code.retrieval.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            indexed = 0
            for f in PROCESSED_DIR.glob("*.json"):
                if f.stem.startswith("draft_"):
                    continue
                try:
                    doc_data = json.loads(f.read_text(encoding="utf-8"))
                    chunks = doc_data.get("chunks", [])
                    if chunks:
                        indexer.index_document(doc_data["id"], chunks)
                        indexed += 1
                except Exception:
                    pass
            if indexed:
                print(f"  {indexed} documents indexed in ChromaDB")
        except Exception as e:
            print(f"  [WARN] Prebuilt data setup: {e}")
    else:
        print("  Prebuilt processed docs left untouched (pass --copy-prebuilt to seed them)")

    # Seed sample corrections for the learning system demo
    try:
        from code.learning.correction_store import Correction, CorrectionStore
        store = CorrectionStore()

        if store.get_count() == 0:
            sample_corrections = [
                Correction(
                    document_id="sample_doc_1",
                    draft_type="case_fact_summary",
                    field_path="parties.defendant",
                    source_ocr_chunk="NORWOOD FABRICATION GROUP LLC and GERALD P. NORWOOD individually Defendants",
                    generated_text="Defendant: Norwood Fabrication Group LLC",
                    edited_text="Defendants: Norwood Fabrication Group LLC (Delaware LLC); Gerald P. Norwood (individually)",
                    correction_type="omission",
                ),
                Correction(
                    document_id="sample_doc_1",
                    draft_type="case_fact_summary",
                    field_path="key_dates.filing",
                    source_ocr_chunk="Date: March 14, 2024",
                    generated_text="Filing date: unclear",
                    edited_text="Filing date: March 14, 2024",
                    correction_type="error",
                ),
                Correction(
                    document_id="sample_doc_2",
                    draft_type="case_fact_summary",
                    field_path="parties.plaintiff",
                    source_ocr_chunk="Plaintiff Synthex Industrial Partners Inc by and through its counsel",
                    generated_text="Plaintiff: Synthex",
                    edited_text="Plaintiff: Synthex Industrial Partners, Inc.",
                    correction_type="omission",
                ),
                Correction(
                    document_id="sample_doc_3",
                    draft_type="title_review_summary",
                    field_path="chain_of_title.grantee",
                    source_ocr_chunk="FIDELITY MERIDIAN TRUST a Thornfield statutory trust Grantee",
                    generated_text="Grantee: Fidelity Trust",
                    edited_text="Grantee: Fidelity Meridian Trust (Thornfield statutory trust)",
                    correction_type="error",
                ),
                Correction(
                    document_id="sample_doc_4",
                    draft_type="notice_summary",
                    field_path="deadlines.action_1",
                    source_ocr_chunk="ACTION 1 IMMEDIATE Within 48 hours of receipt of this Notice",
                    generated_text="Deadline: Not specified",
                    edited_text="Deadline: Within 48 hours of receipt",
                    correction_type="omission",
                ),
            ]
            for c in sample_corrections:
                store.save_correction(c)
            print(f"  {len(sample_corrections)} sample corrections seeded")
        else:
            print(f"  Corrections already exist ({store.get_count()}), skipping seed")
    except Exception as e:
        print(f"  [WARN] Correction seeding skipped: {e}")

    print("\nDone. Start the server with: run.bat")


if __name__ == "__main__":
    main()
