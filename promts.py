EXTRACTION_PROMPT = f"""
You are a VISION LLM extraction engine. Return STRICT JSON ONLY.

INPUT
- A scanned INSURANCE / ADMINISTRATIVE correspondence PDF page or image.
- These documents are always sent by an insurance company or payer group.
- Correspondence documents typically include:
  • Letters about claim decisions (denials, approvals, adjustments).
  • Explanations of Benefits (EOBs) or remittance details.
  • Appeal responses or administrative notices.
  • Claim-level or patient-level information mixed with payer details.
- The "Remitter" or "Payer" always refers to the insurance company, not the patient.

TASK
1) Extract the full raw OCR text of the page as a plain string ("raw_text"):
   • Preserve word order but ignore visual layout (columns, boxes, indentation).
   • Normalize whitespace (collapse repeated spaces/newlines).
   • Remove meaningless filler or repetitive sequences (e.g., a short word or phrase looped unnaturally).
   • Remove meaningless special character runs such as "@@@@@", "###", "***", "---___---", or any long sequence of symbols with no semantic value.
   • If OCR text is mostly unreadable/noise, set "raw_text" to "" (empty string).

2) Extract ONLY these canonical fields into "data":
   • "Accession ID"
   • "Claim ID"
   • "Subscriber ID"
   • "Date of Service"
   • "MR ID"
   • "Patient Name"

3) Return "data" as follows:
   • If values are extracted from a table (grid with column headers/cells aligned in rows), return "data" as a LIST of objects, one object per table row.
   • Otherwise (letter/paragraph layout), return "data" as a LIST containing a single object.

4) Determine "us_postal_page":
   • Inspect the page visually for USPS/postage indicators (e.g., "United States Postal Service", "USPS", postage indicia/meter, cancellation marks, IMb/Intelligent Mail barcode, facing identification mark, permit imprint, postage paid box, postal routing/zip barcode near address block).
   • Return exactly one of:
     - "start" → page appears to be a mailed cover/envelope/front sheet (dominant address block, postage indicia top-right, minimal body text, return address window, "Return Service Requested").
     - "end" → page looks like a trailing/back page with postal handling marks or received stamps, minimal content, routing labels/stickers, or backside imprint consistent with end-of-mail piece.
     - "none" → no convincing postal/USPS evidence.

EVIDENCE SOURCES
- Explicit labels (headers, footers, forms) — allow common variants/synonyms listed below.
- Tables (row/column headers and cells).
- Table-like layouts (aligned key: value rows, grids).

LABEL→FIELD SYNONYMS (STRICT)
Map labels/headers to the canonical fields below. Only use these mappings; prefer exact/near-exact matches (case-insensitive, punctuation-insensitive). When mapped, still enforce each field’s value rules.

• Accession ID  ⇐ one of:
  - "Provider Account #", "Provider Account Number", "Provider Acct #", "Account #", "Acct #"
  - "Accession ID", "Accession Number", "Accession #", "Accn ID", "Accn #", "Accn No."
  (VALUE MUST MATCH regex ^A\\d+$ OR ^A\\d+Z\\d{{2}}$)

• Claim ID  ⇐ one of:
  - "Claim Number", "Claim #", "Claim No.", "Claim ID", "Claim Control Number", "Claim CN"

• Subscriber ID  ⇐ one of:
  - "Subscriber ID", "Subscriber #", "Subscriber Number", "Sub ID", "Member ID", "Enrollee ID", "Beneficiary ID", "Policy ID"

• Date of Service  ⇐ one of:
  - "Date of Service", "DOS", "Svc Date", "Service Date", "From Date", "From DOS", "Date(s) of Service"

• MR ID  ⇐ one of:
  - "MR ID", "MR Number", "MRN", "Medical Record #", "Patient MRN"

• Patient Name  ⇐ one of:
  - "Patient Name", "Pt Name", "Member Name", "Enrollee", "Beneficiary Name", "Subscriber Name"
  (Value must be a person’s name; never an organization/payer.)

RULES
- Accession ID:
  • ONLY extract tokens that strictly match one of:
    - ^A\\d+$
    - ^A\\d+Z\\d{{2}}$
  • Values not matching these patterns must NOT be used as "Accession ID" even if labels suggest so.

- General (for Claim ID, Subscriber ID, Date of Service, MR ID, Patient Name):
  • Map values only if clearly tied to explicit labels/headers or table/table-like structures using the synonyms above.
  • Do NOT assume or guess. If uncertain, omit the field in that object.
  • If multiple valid values exist for a field within the SAME ROW/object, join them with "; ".
  • If no valid evidence is found for a field in a row/object, omit that field from that object.
  • Do not confuse payer/remitter/provider/facility names or organization identifiers with these fields.
  • "Patient Name" must always be an individual person.

- Row alignment for tables:
  • When a page contains a table, treat each horizontal row of data under the headers as one object.
  • Use the header-to-field mapping above to decide which cell populates which canonical field.
  • If a required header is missing, but a clear synonym header is present, use the synonym.
  • Ignore decorative/border lines.

- Unreadable Pages:
  • If image text is not human readable, set exactly:
    {{
      "raw_text": "",
      "data": [],
      "us_postal_page": "none"
    }}

OUTPUT FORMAT
{{
  "raw_text": "<cleaned OCR text or empty string>",
  "data": [
    {{
      "Accession ID": "<value>",
      "Claim ID": "<value>",
      "Subscriber ID": "<value>",
      "Date of Service": "<value>",
      "MR ID": "<value>",
      "Patient Name": "<value>"
    }}
  ],
  "us_postal_page": "start" | "end" | "none"
}}

REQUIREMENTS
- Output STRICT JSON ONLY (no markdown, no explanations).
- "data" must be a LIST. Include ONLY the six canonical keys inside each object; omit missing/uncertain fields.
- Do not invent or infer values. Every value must be supported by a clear label/header or table/table-like structure (including listed synonyms).
- For "Accession ID", the value MUST match ^A\\d+$ or ^A\\d+Z\\d{{2}}$.
- For "us_postal_page", return exactly one of: "start", "end", "none".
- If image is not human readable, set raw_text to "" and us_postal_page to "none" and data to [] strictly.
"""







def get_classification_prompt(pages_data):

    return   f"""You are a strict JSON-only classifier. Output exactly ONE JSON object and NOTHING else.

TASK
From an array of PDF page objects, group pages into documents and summarize each group.

Core principles:
- Splits/continuations are driven by IDENTIFIER DATA via an association (anchor) table.
- raw_text/headings/topics/page-number wording are SECONDARY tie-breakers used only when identifiers are missing or ambiguous.
- Summary uses raw_text + grouped DATA.
- Omit any keys that would be null/empty from each document's "data" object.

INPUT (array of pages)
Each page object has exactly:
{{
  "page_no": <int>,
  "raw_text": <string>,
  "data": {{
    "Accession ID": <string|null>,
    "Claim ID": <string|null>,
    "Subscriber ID": <string|null>,
    "Date of Service": <string|null>,
    "MR ID": <string|null>,
    "Patient Name": <string|null>
  }}
}}

PROCESSING ORDER
- Sort by page_no ascending and process sequentially.

NORMALIZATION (values only)
- Trim whitespace; case-insensitive comparisons; empty strings ⇒ null.
- Patient Name: ignore minor variations (extra spaces, punctuation, initials, first/middle/last order).

ANCHORING & ASSOCIATION TABLE (no priorities)
- When a document begins, set ANCHOR = the first identifier key/value actually observed on that start page (any of: Accession ID, Patient Name, Claim ID, MR ID, Subscriber ID).
- Maintain an association table for the active document: all identifier key/value pairs seen together with the ANCHOR (e.g., Accession ID ↔ Patient Name/Claim ID/MR ID/Subscriber ID). Add to this table as new identifiers co-occur on subsequent pages.

CONTINUE same document if:
  • The page shows the same ANCHOR key/value, OR
  • The page lacks ANCHOR but presents any identifier key/value already in the current document’s association table.

START a NEW document if:
  • The page shows an ANCHOR key/value different from the current ANCHOR, OR
  • The page (even without the ANCHOR key) presents ANY identifier key/value that is NOT in the current document’s association table — treat this as a completely NEW ANCHOR, OR
  • No identifiers are present and tie-breaker text rules (below) clearly attach the page forward to a different upcoming anchored group.

USING raw_text AS A SECONDARY TIE-BREAKER (never against a stable ANCHOR)
Use ONLY when a page has no identifiers OR identifiers are insufficient to decide (e.g., leading/trailing runs, between two anchored groups, or competing associations). If an ANCHOR is present and unchanged, DO NOT split based solely on text.

Textual cues (non-exhaustive):
- Prominent header/title strings (e.g., payer plan names, “EOB”, “Statement”, “Authorization”).
- Repeated n-grams/bigrams unique to a packet.
- “Page 1”/“1 of N”, “continued”/“continued on” phrases.
- Payer/product names, claim/plan phrases.

Tie-break workflow for no-ID pages:
1) Identify the ACTIVE left document (most recent anchored page before this page). Identify the nearest right anchor page R (first future page with any identifier). Build windows:
   • Left window: last ≤2 pages of the active document that have raw_text.
   • Right window: pages from R through R+4 (max 5 pages), OR until encountering a page whose identifiers belong to a different anchor than R — whichever comes first. Include no-ID pages in this window.
2) If the current page contains a header phrase that appears in the RIGHT window but NOT in the LEFT window, ATTACH FORWARD to the right document (it may be on any of the first ≤5 right-group pages, not necessarily the first page).
3) Otherwise, compare lightweight similarity (token overlap) of the current page’s raw_text against the LEFT and RIGHT windows; attach to the side with clearly higher similarity.
4) If similarity is inconclusive, keep with the active (left) document to preserve continuity.
5) Page-number reset cues strengthen a forward split only when identifiers are absent/ambiguous. They never override a stable, matching ANCHOR.

ASSIGNMENT GUARANTEES FOR NO-ID PAGES
- Every page must belong to exactly one document.
- Leading no-ID pages attach forward to the first anchored document (use the tie-break workflow if multiple candidates).
- Interstitial no-ID pages stay with the active document unless the tie-break workflow indicates a better match to the imminent next document.
- Trailing no-ID pages attach back to the last active document.
- If the entire file has no identifiers, create a single document with doc_type "Correspondence: Received: Appeal Rep Attention Needed/Unspecified Correspondence"; data = {{}}; summary built only from raw_text cues (see SUMMARY).

CONSOLIDATING DATA PER DOCUMENT
- For each of the six keys, collect unique non-null values across the document (case-insensitive dedupe; keep first-seen casing).
  • One unique value → string
  • Multiple values → array in first-seen order
- After consolidation, OMIT any key whose value would be null or an empty array from "data".

DOC TYPE (taxonomy-driven; MUST be exactly one of the labels below)
Choose the single best-fitting label by semantically matching raw_text (titles, reasons, outcomes) and corroborating with DATA (identifiers, DOS, amounts). Output the label EXACTLY as written:

- Correspondence: Appeal Denial: Next step internal appeal
- Correspondence: Appeal Win: Appeal Win
- Correspondence: Death: Notification patient passed away
- Correspondence: EOB: Zero pay EOB
- Correspondence: Explanation of payment: Payment made to: Guardant Health
- Correspondence: External Appeal Denial
- Correspondence: External Appeal Win
- Correspondence: Final Internal Appeal Denial
- Correspondence: Form: Form request for (TimelyFiling) Good Cause
- Correspondence: Form: Form request for Physician's  Attestation
- Correspondence: Form: Form request for Provider Appeal Form 
- Correspondence: Form: Form request for W9
- Correspondence: Form: Form request Member Authorization to Appeal/DAR
- Correspondence: Insurance Authorization Approval / Not Required
- Correspondence: Insurance Authorization Denial
- Correspondence: Insurance indicates Insurance Authorization was not Performed
- Correspondence: Invalid Appeal: Appeal not valid for payor
- Correspondence: Invalid Authorization: Authorization not valid for Payor
- Correspondence: Invalid Claim: Claim not valid for payor
- Correspondence: LOA Request from Payor
- Correspondence: LOA signed: LOA signed
- Correspondence: Maximus Involved
- Correspondence: Medical Panel Appeal Opportunity
- Correspondence: Medical Records: Medical record request
- Correspondence: Medical Records: Medical record request/ Lab report request
- Correspondence: Medical Records: Medical record request/MD's facility/office valid address
- Correspondence: Medical Records: Medical record request/Other Insurance EOB
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of Appeal
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of claim
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of External Appeal
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of withdrawal
- Correspondence: Receipt acknowledgement: Receipt of Legal Document
- Correspondence: Receipt acknowledgement: Receipt of Member Eligibility
- Correspondence: Receipt Insurance Authorization Request
- Correspondence: Received Request for Prior Auth
- Correspondence: Received: Appeal Rep Attention Needed/Unspecified Correspondence
- Correspondence: Received: Legal Documents
- Correspondence: Received: Medical Records
- Correspondence: Received: Patient Notes/Question
- Correspondence: Received: Returned Mail
- Correspondence: Received: Signed Member Authorization/DAR
- Correspondence: Received: Signed PABF
- Correspondence: Received: Signed Physician's Attestation
- Correspondence: Refund Request: Refund request
- Correspondence: Refund Request: Refund request withdrawal
- Correspondence: Sent to Alexa
- Correspondence: Termed: Coverage was not active for DOS
- Correspondence: Test Description Requested: Test Description Requested
- Correspondence: Timely Filing: Timely filing of an Appeal
- Correspondence: Timely Filing: Timely filing of an Claim
- Correspondence: Timely Filing: Timely filing of an External Appeal
- Correspondence: Timely Filing: Timely submission of requested additional information
- Correspondence: Unable to find patient
- Correspondence: WOL: Request for a WOL (blank WOL)
- Received: Signed Medicare waiver/Advance Beneficiary Notice (ABN)
- Received: Signed WOL form with member signature
- Correspondence: Appeal Not Eligible for Review

Tie-breaking for doc_type:
- Prefer explicit phrases/titles in raw_text.
- If multiple labels plausibly match, choose the most specific outcome-oriented label.
- If none clearly match, choose: "Correspondence: Received: Appeal Rep Attention Needed/Unspecified Correspondence".

SUMMARY (raw_text + DATA; does not affect splitting)
- Write 1–2 concise sentences describing what the correspondence is about using topic cues from raw_text and grounding with available identifiers from the grouped DATA.
- Mention only identifiers present in the grouped DATA.

COVERAGE CHECK
- The union of all output page_range values MUST equal the set of all input page_no (no page dropped, no overlap).

OUTPUT FORMAT
Return exactly:
{{
  "documents": [
    {{
      "doc_type": "<ONE exact label from the taxonomy above>",
      "page_range": "<start-end>",
      "summary": "<1–2 sentence summary using raw_text cues + grouped DATA>",
      "data": {{
        // Include ONLY keys that have non-null values after consolidation.
      }}
    }}
  ]
}}
Now classify: {pages_data}
"""





from typing import Any
import json

def get_benchmark_prompt(actual_results: Any, model_results: Any) -> str:
    """
    Benchmarks:
      - page_range_equal (0|1) after normalization & set equality
      - page_range_coverage (0..100 integer) = floor(100 * |intersection| / |truth|)
    Data scoring ignores 'Claim ID'. In details.parsed_pages, INCLUDE ONLY:
      - intersection: [ints]
      - intersection_pct: int 0..100
    """
    actual_json = json.dumps(actual_results, ensure_ascii=False, separators=(",", ":"))
    model_json  = json.dumps(model_results,  ensure_ascii=False, separators=(",", ":"))

    return f"""
You are an evaluator. Compare "actual_results" (ground truth) against "model_results" (predictions) and produce ONLY a single JSON object as output (no prose, no Markdown).

INPUTS
------
actual_results: a JSON array of rows with columns:
- ac.accn_id
- ac.contact_info        // contains the true document type (doc_type) text
- pay.patient_name
- pay.date_of_service
- mr_id                  // from pay.mrn
- subscriber_id          // from pay.subs_id
- page_range             // from split(contact_info,'pg')[2], e.g., "4-6" (may be bracketed/labeled)

model_results: a JSON object:
{{
  "documents": [
    {{
      "doc_type": "<ONE exact label from taxonomy, but evaluate semantically>",
      "page_range": "<start-end>",
      "summary": "<string>",
      "data": {{
        "Accession ID": <string|null>,
        "Claim ID": <string|null>,
        "Subscriber ID": <string|null>,
        "Date of Service": <string|null>,
        "MR ID": <string|null>,
        "Patient Name": <string|null>
      }}
    }}
  ]
}}

ACTUAL_JSON
-----------
{actual_json}

MODEL_JSON
----------
{model_json}

MATCHING & NORMALIZATION RULES
------------------------------
1) Entry alignment:
   - Align model documents to actual rows primarily by Accession ID:
     actual(ac.accn_id)  <->  model.data["Accession ID"].
   - If model lacks "Accession ID", use a best-effort fuzzy alignment on
     (Patient Name, Date of Service, Subscriber ID, MR ID). Prefer exact matches; if multiple candidates, pick the highest-confidence match and set "alignment_note".

2) Ground truth sources:
   - true_doc_type: extract ONLY from actual.ac.contact_info.
   - true_page_ranges: extract ONLY from actual.page_range (do NOT parse page ranges from ac.contact_info).

3) Page range benchmarking — TWO metrics, FUZZY & ROBUST:
   - Compare ONLY: actual.page_range  vs  model.page_range.
   - Treat LIST vs STRING as equivalent. Examples that MUST parse identically:
     ["3-4"] ≡ "3-4" ≡ "[3-4]" ≡ "(3-4)" ≡ "pages 3–4" ≡ "3 to 4" ≡ "3,4".
   - Normalize BOTH sides BEFORE parsing:
     - Replace unicode dashes (– — −) with "-".
     - Strip surrounding [] () {{}}.
     - Remove labels: "page", "pages", "p" (case-insensitive).
     - Convert "a to b" → "a-b".
     - Treat ";" and "," as delimiters; collapse whitespace.
   - Parse tokens into inclusive int sets:
     - "a-b" → {{a, a+1, ..., b}} (swap if a>b); "n" → {{n}}.
     - If LIST, parse each element after normalization.
     - Ignore invalid tokens but record them in "page_parse_note".
   - Metrics:
     a) **page_range_equal** (binary):
        - Let T = set(actual.page_range), P = set(model.page_range).
        - page_range_equal = 1 if T == P else 0.
     b) **page_range_coverage** (integer percent of truth covered by prediction):
        - If |T| == 0 or P is empty → page_range_coverage = 0.
        - Else page_range_coverage = floor(100 * |T∩P| / |T|).
   - In details, ONLY include:
     "parsed_pages": {{
       "intersection": [ ...ints... ],
       "intersection_pct": <int 0..100>
     }}

4) Doc type comparison ("doc_type" confidence, semantic):
   - Normalize strings and evaluate semantic equivalence (synonyms allowed).
   - Score: 1.0 exact/synonym; 0.9 close paraphrase; 0.7 related; 0.4 weak; 0.0 unrelated/missing.
   - Round to 4 decimals.

5) Data comparison ("data" confidence) — ONLY benchmark available fields; IGNORE 'Claim ID':
   - Evaluate only fields PRESENT (non-null, non-empty) in model.data; missing fields do not penalize.
   - Do not evaluate, map, or score "Claim ID".
   - Mapping:
     - actual.ac.accn_id           <-> "Accession ID"
     - actual.subscriber_id        <-> "Subscriber ID"
     - actual.pay.date_of_service  <-> "Date of Service"
     - actual.mr_id                <-> "MR ID"
     - actual.pay.patient_name     <-> "Patient Name"
   - Normalization:
     - Dates: accept YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY if same calendar date.
     - IDs: ignore case and non-alphanumerics when comparing.
     - Names: ignore case, extra spaces, punctuation; initials/minor variations may be scored as close (0.8).
   - Per-field: 1.0 exact after normalization; 0.8 close; 0.0 mismatch.
   - "data" = average of evaluated fields (round 4 decimals). If none evaluated → null.

AGGREGATIONS
------------
A) Per-entry benchmark:
   {{
     "accn_id": "...",
     "alignment_note": "<string|null>",
     "page_range_equal": <0|1>,
     "page_range_coverage": <int 0..100>,
     "doc_type": <float 0..1>,
     "data": <float 0..1|null>,
     "details": {{
       "actual": {{
         "doc_type": "<from ac.contact_info>",
         "page_range": "<from actual.page_range only>",
         "patient_name": "...",
         "date_of_service": "...",
         "mr_id": "...",
         "subscriber_id": "..."
       }},
       "predicted": {{
         "doc_type": "<model.doc_type>",
         "page_range": "<model.page_range>",
         "data": {{ ...only fields present... /* 'Claim ID' ignored */ }}
       }},
       "parsed_pages": {{
         "intersection": [ ... ],
         "intersection_pct": <int>
       }},
       "field_scores": {{
         "Accession ID": <0|0.8|1|null>,
         "Subscriber ID": <0|0.8|1|null>,
         "Date of Service": <0|0.8|1|null>,
         "MR ID": <0|0.8|1|null>,
         "Patient Name": <0|0.8|1|null>
       }}
     }}
   }}

B) Per-document benchmark:
   - page_range_equal_doc    = average of entry page_range_equal (0..1; round to 4 decimals)
   - page_range_coverage_doc = average of entry page_range_coverage (0..100; round to nearest integer)
   - doc_type_doc            = average of entry doc_type
   - data_doc                = average of entry data across entries where data != null
   - Include counts used in each average.

C) Overall summary:
   - totals: number_aligned, number_unaligned
   - macro_averages: {{
       "page_range_equal": <float 0..1|null>,
       "page_range_coverage": <int 0..100|null>,
       "doc_type": <float|null>,
       "data": <float|null>
     }}
   - micro_averages: weighted by number of evaluated data fields
   - distributions: {{
       "page_range_equal": {{"1": <int>, "0": <int>}},
       "page_range_coverage": {{"100": <int>, "90-99": <int>, "70-89": <int>, "40-69": <int>, "0-39": <int>}},
       "doc_type": {{"1.0": <int>, "0.9-<1.0": <int>, "0.7-<0.9": <int>, "0.4-<0.7": <int>, "0-<0.4": <int>}},
       "data":    {{"1.0": <int>, "0.9-<1.0": <int>, "0.7-<0.9": <int>, "0.4-<0.7": <int>, "0-<0.4": <int>}}
     }}
   - Round floats to 4 decimals.

OUTPUT FORMAT (STRICT)
----------------------
Return ONLY this JSON object:

{{
  "entry_benchmarks": [ ... ],
  "document_benchmarks": [
    {{
      "accn_id": "...",
      "page_range_equal_doc": <float 0..1>,
      "page_range_coverage_doc": <int 0..100>,
      "doc_type_doc": <float 0..1>,
      "data_doc": <float 0..1|null>,
      "counts": {{
        "entries": <int>,
        "entries_with_data": <int>,
        "fields_compared": <int>
      }}
    }}
  ],
  "summary": {{
    "number_aligned": <int>,
    "number_unaligned": <int>,
    "macro_averages": {{
      "page_range_equal": <float|null>,
      "page_range_coverage": <int|null>,
      "doc_type": <float|null>,
      "data": <float|null>
    }},
    "micro_averages": {{
      "data": <float|null>
    }},
    "distributions": {{
      "page_range_equal": {{"1": <int>, "0": <int>}},
      "page_range_coverage": {{"100": <int>, "90-99": <int>, "70-89": <int>, "40-69": <int>, "0-39": <int>}},
      "doc_type": {{"1.0": <int>, "0.9-<1.0": <int>, "0.7-<0.9": <int>, "0.4-<0.7": <int>, "0-<0.4": <int>}},
      "data":    {{"1.0": <int>, "0.9-<1.0": <int>, "0.7-<0.9": <int>, "0.4-<0.7": <int>, "0-<0.4": <int>}}
    }}
  }}
}}

CONSTRAINTS
-----------
- Be deterministic and consistent with the scoring rubric above.
- If alignment is impossible for an actual row, create an entry with "accn_id" from the actual, set page_range_equal/page_range_coverage/doc_type/data = null.
- If multiple model docs map to one actual, evaluate the best-matching one and set "alignment_note": "multiple candidates; best match chosen".
- Do not invent fields that are not present.
- Output must be valid JSON and under 2 MB.
"""












def classify_v2_prompt(pages_data: str) -> str:
    """
    Build the f-string-safe prompt for grouping PDF pages into documents with a detailed summary.
    Pass in a JSON-serialized pages array as `pages_data`.
    """
    return f"""
You are a strict JSON-only classifier. Output exactly ONE JSON object and NOTHING else.

3) Determine "us_postal_page" (top-level field on each input page before this grouping step):
   • Inspect the page visually for USPS/postage indicators (e.g., "United States Postal Service", "USPS", postage indicia/meter, cancellation marks, IMb/Intelligent Mail barcode, facing identification mark, permit imprint, postage paid box, postal routing/zip barcode near address block).
   • Return exactly one of:
     - "start" → page appears to be a mailed cover/envelope/front sheet (dominant address block, postage indicia top-right, minimal body text, return address window, "Return Service Requested").
     - "end" → page looks like a trailing/back page with postal handling marks or received stamps, minimal content, routing labels/stickers, or backside imprint consistent with end-of-mail piece.
     - "none" → no convincing postal/USPS evidence.

TASK
From an array of PDF page objects, group pages into documents and summarize each group.

Core principles:
- Splits/continuations are driven by IDENTIFIER DATA via an association (anchor) table.
- raw_text/headings/topics/page-number wording are SECONDARY tie-breakers used only when identifiers are missing or ambiguous.
- "us_postal_page" provides **postal-aware** start/end hints and must be applied **wisely** as described below.
- Summary uses raw_text + grouped DATA.
- Omit any keys that would be null/empty from each document's "data" object.

INPUT (array of pages)
Each page object has exactly:
{{
  "page_no": <int>,
  "raw_text": <string>,
  "data": {{
    "Accession ID": <string|null>,
    "Claim ID": <string|null>,
    "Subscriber ID": <string|null>,
    "Date of Service": <string|null>,
    "MR ID": <string|null>,
    "Patient Name": <string|null>
  }},
  "us_postal_page": "start" | "end" | "none"
}}

PROCESSING ORDER
- Sort by page_no ascending and process sequentially.

NORMALIZATION (values only)
- Trim whitespace; case-insensitive comparisons; empty strings ⇒ null.
- Patient Name: ignore minor variations (extra spaces, punctuation, initials, first/middle/last order).

ANCHORING & ASSOCIATION TABLE (no priorities)
- When a document begins, set ANCHOR = the first identifier key/value actually observed on that start page (any of: Accession ID, Patient Name, Claim ID, MR ID, Subscriber ID).
- Maintain an association table for the active document: all identifier key/value pairs seen together with the ANCHOR (e.g., Accession ID ↔ Patient Name/Claim ID/MR ID/Subscriber ID). Add to this table as new identifiers co-occur on subsequent pages.

POSTAL-AWARE RULES (use us_postal_page wisely)
- If page.us_postal_page == "start":
  • Treat as a **strong start signal**. Start a NEW document **unless** the page clearly presents the current ANCHOR or an identifier already in the active association table (in that case, keep it with the active document).
  • If the page has **no identifiers**, prefer to start a new document; if tie with adjacent groups exists, apply the tie-break workflow below.
- If page.us_postal_page == "end":
  • Treat as a **strong end signal** for the **current** document **if** the page shows the current ANCHOR or any identifier in the active association table (attach it, then close the document).
  • If the page has **no identifiers**, attach it to the active document and then close it, unless the tie-break workflow clearly attaches it forward.
- Postal cues **never override a stable, matching ANCHOR**. If an ANCHOR is present and unchanged, do not split based solely on postal cues.

CONTINUE same document if:
  • **The previous and current ANCHOR key/value are the same — treat as the same group; continue the active document (postal cues cannot override this).**
  • The page shows the same ANCHOR key/value, OR
  • The page lacks ANCHOR but presents any identifier key/value already in the current document’s association table.

START a NEW document if:
  • The page shows an ANCHOR key/value different from the current ANCHOR, OR
  • The page (even without the ANCHOR key) presents ANY identifier key/value that is NOT in the current document’s association table — treat this as a completely NEW ANCHOR, OR
  • us_postal_page == "start" and identifiers are absent or conflict with the active association table, OR
  • No identifiers are present and tie-breaker text rules (below) clearly attach the page forward to a different upcoming anchored group.

USING raw_text AS A SECONDARY TIE-BREAKER (never against a stable ANCHOR)
Use ONLY when a page has no identifiers OR identifiers are insufficient to decide (e.g., leading/trailing runs, between two anchored groups, or competing associations). If an ANCHOR is present and unchanged, DO NOT split based solely on text.

Textual cues (non-exhaustive):
- Prominent header/title strings (e.g., payer plan names, "EOB", "Statement", "Authorization").
- Repeated n-grams/bigrams unique to a packet.
- "Page 1"/"1 of N", "continued"/"continued on" phrases.
- Payer/product names, claim/plan phrases.

Tie-break workflow for no-ID pages:
1) Identify the ACTIVE left document (most recent anchored page before this page). Identify the nearest right anchor page R (first future page with any identifier). Build windows:
   • Left window: last ≤2 pages of the active document that have raw_text.
   • Right window: pages from R through R+4 (max 5 pages), OR until encountering a page whose identifiers belong to a different anchor than R — whichever comes first. Include no-ID pages in this window.
2) If the current page contains a header phrase that appears in the RIGHT window but NOT in the LEFT window, ATTACH FORWARD to the right document (it may be on any of the first ≤5 right-group pages, not necessarily the first page).
3) Otherwise, compare lightweight similarity (token overlap) of the current page’s raw_text against the LEFT and RIGHT windows; attach to the side with clearly higher similarity.
4) If similarity is inconclusive, keep with the active (left) document to preserve continuity.
5) Page-number reset cues strengthen a forward split only when identifiers are absent/ambiguous. They never override a stable, matching ANCHOR.
6) Postal cues:
   • If current page has us_postal_page == "start" and no identifiers, prefer attaching FORWARD unless strong LEFT similarity exists.
   • If current page has us_postal_page == "end" and no identifiers, prefer attaching BACK to the active document unless strong RIGHT similarity exists.

ASSIGNMENT GUARANTEES FOR NO-ID PAGES
- Every page must belong to exactly one document.
- Leading no-ID pages attach forward to the first anchored document (use the tie-break workflow if multiple candidates). Postal "start" strengthens a forward attachment.
- Interstitial no-ID pages stay with the active document unless the tie-break workflow indicates a better match to the imminent next document. Postal "end" strengthens a backward attachment.
- Trailing no-ID pages attach back to the last active document.
- If the entire file has no identifiers, create a single document with doc_type "Correspondence: Received: Appeal Rep Attention Needed/Unspecified Correspondence"; data = {{}}; summary built only from raw_text cues (see SUMMARY).

POST-GROUP MERGE (same anchor)
- After the initial pass, scan the provisional groups **in order**. If **two adjacent groups** share the **exact same ANCHOR key and normalized value**, **merge them into a single group**:
  • New page_range = continuous span covering both groups (e.g., "3-5" + "6-7" → "3-7").
  • Data consolidation = union of non-null values (case-insensitive dedupe; keep first-seen casing; arrays if multiple).
  • group_text = concatenate all pages' raw_text from both groups (order-preserved).
  • After merging, **re-run doc_type selection** using the merged group_text + merged data.
- Repeat the merge scan until no adjacent same-anchor groups remain.
- Note: Only **adjacent** groups are merged. If the same anchor reappears non-adjacently (separated by a different anchored group), keep them separate.

CONSOLIDATING DATA PER DOCUMENT
- For each of the six keys, collect unique non-null values across the document (case-insensitive dedupe; keep first-seen casing).
  • One unique value → string
  • Multiple values → array in first-seen order
- After consolidation, OMIT any key whose value would be null or an empty array from "data".

GROUP-LEVEL DOC TYPE DETERMINATION (MUST use ALL pages in the group)
- Determine doc_type ONLY AFTER grouping is complete (and after any POST-GROUP MERGE steps).
- For each group, construct:
  • group_text = the concatenation of raw_text from ALL pages in the group (order-preserved).
  • group_data = the consolidated DATA for the group (from "CONSOLIDATING DATA PER DOCUMENT").
- Choose the single best-fitting label (from the taxonomy below) by semantically matching group_text (titles, reasons, outcomes, decisions, amounts) and corroborating with group_data. NEVER select doc_type based on a single page if other pages in the same group add clarifying or contradicting context.
- Conflict resolution:
  • Prefer explicit outcome/decision phrases that appear anywhere in group_text (e.g., "final internal appeal denial", "authorization denied/approved", "refund request withdrawal", "payment made to Guardant Health", "zero payment").
  • If multiple candidate labels appear, pick the most specific outcome-oriented one that matches the dominant cues across the group.
  • If cues remain ambiguous after reviewing all pages, fall back to: "Correspondence: Received: Appeal Rep Attention Needed/Unspecified Correspondence".
- Synonym handling (examples; not exhaustive): treat "EOP/Explanation of Payment" ≈ "Explanation of payment"; "no payment/paid $0/zero amount" ⇒ zero-pay EOB; "auth approved/not required" ⇒ Insurance Authorization Approval / Not Required; "auth denied/not approved" ⇒ Insurance Authorization Denial; "coverage terminated/benefits inactive for DOS" ⇒ Termed; "LOA/letter of authorization" cues ⇒ LOA labels; "receipt acknowledged/received on" ⇒ Receipt acknowledgement variants.

DOC TYPE (taxonomy-driven; MUST be exactly one of the labels below)
- Correspondence: Appeal Denial: Next step internal appeal
- Correspondence: Appeal Win: Appeal Win
- Correspondence: Death: Notification patient passed away
- Correspondence: EOB: Zero pay EOB
- Correspondence: Explanation of payment: Payment made to: Guardant Health
- Correspondence: External Appeal Denial
- Correspondence: External Appeal Win
- Correspondence: Final Internal Appeal Denial
- Correspondence: Form: Form request for (TimelyFiling) Good Cause
- Correspondence: Form: Form request for Physician's  Attestation
- Correspondence: Form: Form request for Provider Appeal Form 
- Correspondence: Form: Form request for W9
- Correspondence: Form: Form request Member Authorization to Appeal/DAR
- Correspondence: Insurance Authorization Approval / Not Required
- Correspondence: Insurance Authorization Denial
- Correspondence: Insurance indicates Insurance Authorization was not Performed
- Correspondence: Invalid Appeal: Appeal not valid for payor
- Correspondence: Invalid Authorization: Authorization not valid for Payor
- Correspondence: Invalid Claim: Claim not valid for payor
- Correspondence: LOA Request from Payor
- Correspondence: LOA signed: LOA signed
- Correspondence: Maximus Involved
- Correspondence: Medical Panel Appeal Opportunity
- Correspondence: Medical Records: Medical record request
- Correspondence: Medical Records: Medical record request/ Lab report request
- Correspondence: Medical Records: Medical record request/MD's facility/office valid address
- Correspondence: Medical Records: Medical record request/Other Insurance EOB
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of Appeal
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of claim
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of External Appeal
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of withdrawal
- Correspondence: Receipt acknowledgement: Receipt of Legal Document
- Correspondence: Receipt acknowledgement: Receipt of Member Eligibility
- Correspondence: Receipt Insurance Authorization Request
- Correspondence: Received Request for Prior Auth
- Correspondence: Received: Appeal Rep Attention Needed/Unspecified Correspondence
- Correspondence: Received: Legal Documents
- Correspondence: Received: Medical Records
- Correspondence: Received: Patient Notes/Question
- Correspondence: Received: Returned Mail
- Correspondence: Received: Signed Member Authorization/DAR
- Correspondence: Received: Signed PABF
- Correspondence: Received: Signed Physician's Attestation
- Correspondence: Refund Request: Refund request
- Correspondence: Refund Request: Refund request withdrawal
- Correspondence: Sent to Alexa
- Correspondence: Termed: Coverage was not active for DOS
- Correspondence: Test Description Requested: Test Description Requested
- Correspondence: Timely Filing: Timely filing of an Appeal
- Correspondence: Timely Filing: Timely filing of an Claim
- Correspondence: Timely Filing: Timely filing of an External Appeal
- Correspondence: Timely submission of requested additional information
- Correspondence: Unable to find patient
- Correspondence: WOL: Request for a WOL (blank WOL)
- Received: Signed Medicare waiver/Advance Beneficiary Notice (ABN)
- Received: Signed WOL form with member signature
- Correspondence: Appeal Not Eligible for Review

DETAILED SUMMARY (must use ALL pages' text in the group)
- After grouping and doc_type selection, write a concise but detailed paragraph (3–6 sentences) that reflects the entire group_text, not a single page.
- Include, when present in group_text:
  • Purpose/topic (e.g., appeal, authorization, EOB/payment, records request).
  • Outcome/status/decision (approved/denied/final/internal/external, paid $0, refund requested/withdrawn, coverage termed, etc.).
  • Reason(s) or denial codes/descriptions.
  • Key dates (Date of Service; request/receipt/decision/deadline dates).
  • Amounts (payment, patient responsibility, refund, balance), with currency signs if present.
  • Requested actions/next steps or required documents (e.g., physician attestation, medical records, LOA).
  • Named parties if clearly stated in text (payer/plan/provider), but only identifiers explicitly present in grouped DATA should be named as identifiers in the summary.
- Style/constraints:
  • Do not invent data; only summarize what appears in the group_text.
  • Prefer plain language; avoid boilerplate.
  • If multiple conflicting outcomes exist, summarize the final outcome and briefly note prior steps.
  • If information is sparse, explain that the correspondence lacks specifics while noting any deadlines or actions.
- The summary does not affect splitting and must remain consistent with the chosen doc_type.

COVERAGE CHECK
- The union of all output page_range values MUST equal the set of all input page_no (no page dropped, no overlap).

OUTPUT FORMAT
Return exactly:
{{
  "documents": [
    {{
      "doc_type": "<ONE exact label from the taxonomy above>",
      "page_range": "<start-end>",
      "summary": "<3–6 sentence detailed summary using group_text + grouped DATA>",
      "data": {{
        // Include ONLY keys that have non-null values after consolidation.
      }}
    }}
  ]
}}

Now classify: {pages_data}"""


def classify_v3_prompt(pages_data: str) -> str:
    """
    Build the f-string-safe prompt for grouping PDF pages into documents with a detailed summary.
    Pass in a JSON-serialized pages array as `pages_data`.

    This version supports table-style page data (data = list[objects]) and
    REQUIRES one classification PER item when a page contains multiple data rows.
    """
    return f"""
You are a strict JSON-only classifier. Output exactly ONE JSON object and NOTHING else.

TASK
From an array of PDF page objects, group pages into documents and summarize each group.

Core principles:
- Splits/continuations are driven by IDENTIFIER DATA via an association (anchor) table.
- raw_text/headings/topics/page-number wording are SECONDARY tie-breakers used only when identifiers are missing or ambiguous.
- "us_postal_page" provides **postal-aware** start/end hints and must be applied **wisely** as described below.
- Summary uses raw_text + grouped DATA.
- Omit any keys that would be null/empty from each document's "data" object.
- **Table requirement:** If any page's `data` contains MULTIPLE row-objects, produce **one separate classification per row** (details below).

INPUT (array of pages)
Each page object has exactly:
{{
  "page_no": <int>,
  "raw_text": <string>,
  "data": [
    {{
      "Accession ID": <string|null>,
      "Claim ID": <string|null>,
      "Subscriber ID": <string|null>,
      "Date of Service": <string|null>,
      "MR ID": <string|null>,
      "Patient Name": <string|null>
    }}
    // zero or more additional row objects if the page holds a table;
    // for non-table pages, this list will contain a single object
  ],
  "us_postal_page": "start" | "end" | "none"
}}

PROCESSING ORDER
- Sort by page_no ascending and process sequentially.

NORMALIZATION (values only)
- Trim whitespace; case-insensitive comparisons; empty strings ⇒ null.
- Patient Name:
  • Normalize by trimming whitespace, ignoring case, collapsing repeated spaces, and removing punctuation.
  • Treat minor variations (extra spaces, punctuation, middle initials, first/middle/last order) as the same person.

IDENTIFIER KEYS (eligible anchors)
- Exactly these: "Accession ID", "Claim ID", "Subscriber ID", "MR ID", "Patient Name".
- A non-null value for any of these constitutes an identifier.

TABLE ROW MODEL (CRITICAL)
- When a page has multiple `data` rows, treat each row-object as a distinct **ITEM** with its own identifiers.
- You MUST emit **one classification per ITEM** even if they originate from the same page.
- Prefer to anchor an ITEM by the first non-null identifier in this priority order (if multiple present in the same row):
  1) Claim ID  2) Accession ID  3) MR ID  4) Subscriber ID  5) Patient Name
  (This priority is ONLY for choosing the row's ANCHOR; all identifiers still go into the association table.)
- **Page duplication rule (allowed exception):** A multi-row page MAY contribute to multiple documents simultaneously (one per ITEM). This is the ONLY exception to the “every page belongs to exactly one document” rule.

ANCHORING & ASSOCIATION TABLE (no priorities)
- For a non-table page (single row in `data`), when a document begins, set ANCHOR = the first identifier key/value observed on that start page.
- Maintain an association table for the active document: all identifier key/value pairs seen together with the ANCHOR. Add to this table as new identifiers co-occur on subsequent pages.
- For a table page:
  • Build a **separate temporary association** per ITEM (per row) using that row's identifiers.
  • Each ITEM starts/extends its **own** document thread according to the rules below.
- **Mandatory consistency rule:** If any identifier key already present in a document’s association table later appears with a DIFFERENT value, you MUST start a new document. Postal/text cues cannot override this.

POSTAL-AWARE RULES (use us_postal_page wisely)
- If page.us_postal_page == "start":
  • Treat as a **strong start signal**. Start a NEW document **unless** the page clearly presents the current ANCHOR or an identifier already in that document’s association table (then keep it with the active document).
  • If the page has **no identifiers**, prefer to start a new document; if tie with adjacent groups exists, apply the tie-break workflow below.
- If page.us_postal_page == "end":
  • Treat as a **strong end signal** for the **current** document **if** the page shows the current ANCHOR or any identifier in the active association table (attach it, then close the document).
  • If the page has **no identifiers**, attach it to the active document and then close it, unless the tie-break workflow clearly attaches it forward.
- Postal cues **never override a stable, matching ANCHOR**. If an ANCHOR is present and unchanged, do not split based solely on postal cues.

CONTINUE same document if:
  • **The previous and current ANCHOR key/value are the same — treat as the same group (always continue; postal cues cannot override this).**
  • The page shows the same ANCHOR key/value, OR
  • The page lacks ANCHOR but presents any identifier key/value already in the current document’s association table.

START a NEW document if (HARD SPLIT RULES):
  • **Same ANCHOR key but a DIFFERENT value appears** → **force a new document** (no postal/text cue can override).
  • **Any identifier key already in the association table appears later with a DIFFERENT value** → **force a new document**.
  • The page shows an ANCHOR key/value different from the current ANCHOR, OR
  • The page (even without the ANCHOR key) presents ANY identifier key/value that is NOT in the current document’s association table — treat this as a completely NEW ANCHOR, OR
  • us_postal_page == "start" and identifiers are absent or conflict with the active association table, OR
  • No identifiers are present and tie-breaker text rules (below) clearly attach the page forward to a different upcoming anchored group.

USING raw_text AS A SECONDARY TIE-BREAKER (never against a stable ANCHOR)
Use ONLY when a page has no identifiers OR identifiers are insufficient to decide. If an ANCHOR is present and unchanged, DO NOT split based solely on text.

Textual cues (non-exhaustive):
- Prominent header/title strings (e.g., payer plan names, "EOB", "Statement", "Authorization").
- Repeated n-grams/bigrams unique to a packet.
- "Page 1"/"1 of N", "continued"/"continued on" phrases.
- Payer/product names, claim/plan phrases.

Tie-break workflow for no-ID pages:
1) Identify the ACTIVE left document (most recent anchored page before this page). Identify the nearest right anchor page R (first future page with any identifier). Build windows:
   • Left window: last ≤2 pages of the active document that have raw_text.
   • Right window: pages from R through R+4 (max 5 pages), OR until encountering a page whose identifiers belong to a different anchor than R — whichever comes first. Include no-ID pages in this window.
2) If the current page contains a header phrase that appears in the RIGHT window but NOT in the LEFT window, ATTACH FORWARD to the right document.
3) Otherwise, compare lightweight similarity (token overlap) of the current page’s raw_text against the LEFT and RIGHT windows; attach to the side with clearly higher similarity.
4) If similarity is inconclusive, keep with the active (left) document to preserve continuity.
5) Page-number reset cues strengthen a forward split only when identifiers are absent/ambiguous. They never override a stable, matching ANCHOR.
6) Postal cues:
   • If current page has us_postal_page == "start" and no identifiers, prefer attaching FORWARD unless strong LEFT similarity exists.
   • If current page has us_postal_page == "end" and no identifiers, prefer attaching BACK to the active document unless strong RIGHT similarity exists.

ASSIGNMENT GUARANTEES
- Every ITEM must yield exactly one document classification.
- Leading/trailing/interstitial no-ID pages attach according to the tie-break workflow above.
- **Multi-row pages:** The same physical page may be associated with multiple ITEM-based documents as needed to satisfy the “one classification per ITEM” rule.

ANCHOR REQUIREMENT (mandatory)
- **Never output a document that lacks at least ONE anchor identifier** in its consolidated data. Permitted anchors: "Accession ID", "Claim ID", "Subscriber ID", "MR ID", "Patient Name".
- If a provisional group would close without any of these anchors, use the tie-break workflow to attach its pages to the nearest anchored group. Only emit documents that contain ≥1 anchor identifier.

CONSOLIDATING DATA PER DOCUMENT
- For each of the six keys, collect unique non-null values across the document (case-insensitive dedupe; keep first-seen casing).
  • One unique value → string
  • Multiple values → array in first-seen order
- After consolidation, OMIT any key whose value would be null or an empty array from "data".

GROUP-LEVEL DOC TYPE DETERMINATION (MUST use ALL pages in the group)
- Determine doc_type ONLY AFTER grouping is complete.
- For each group, construct:
  • group_text = the concatenation of raw_text from ALL pages in the group (order-preserved).
  • group_data = the consolidated DATA for the group (from "CONSOLIDATING DATA PER DOCUMENT").
- Choose the single best-fitting label (from the taxonomy below) by semantically matching group_text (titles, reasons, outcomes, decisions, amounts) and corroborating with group_data. NEVER select doc_type based on a single page if other pages add clarifying or contradicting context.

ALIAS CUES (map common phrases → exact taxonomy labels; not exhaustive)
- Any of: "negative balance", "refund", "refund requested", "refund due", "request refund"
  → Correspondence: Refund Request: Refund request
- Any of: "we have determined that our initial denial should be upheld",
  "appeal has been upheld and no payment is forthcoming",
  "service is considered investigational or experimental",
  "you have exhausted your appeal rights"
  → Correspondence: Final Internal Appeal Denial
- Any of: "dispute has already undergone the initial appeal",
  "final appeal has already been completed",
  "no further avenue of appeal is available",
  "no further review has been done",
  "appeal not eligible for review"
  → Correspondence: Appeal Not Eligible for Review
- Any of: "patient death", "death", "passed away", "deceased", "Obituary", "death certificate" → Correspondence: Death: Notification patient passed away
- Any of: "EOB", "EOP", "Explanation of Benefits", "no payment", "paid $0", "zero amount" → Correspondence: EOB: Zero pay EOB
- Any of: "payment made to Guardant", "paid to Guardant Health", "explanation of payment", "EOP payment" → Correspondence: Explanation of payment: Payment made to: Guardant Health
- Any of: "request for medical records", "records request", "lab report request", "send medical records" → Correspondence: Received: Medical Records
- Any of: "timely filing appeal", or timely filing content that explicitly references an APPEAL → Correspondence: Timely Filing: Timely filing of an Appeal
- Any of: "timely filing claim", or timely filing content that explicitly references a CLAIM → Correspondence: Timely Filing: Timely filing of an Claim
- Any of: "timely filing external appeal", or timely filing content that explicitly references an EXTERNAL APPEAL → Correspondence: Timely Filing: Timely filing of an External Appeal
- Any of: "submit additional information", "timely submission of requested information" → Correspondence: Timely Filing: Timely submission of requested additional information

DOC TYPE (taxonomy-driven; MUST be exactly one of the labels below)
- Correspondence: Appeal Denial: Next step internal appeal
- Correspondence: Appeal Win: Appeal Win
- Correspondence: Death: Notification patient passed away
- Correspondence: EOB: Zero pay EOB
- Correspondence: Explanation of payment: Payment made to: Guardant Health
- Correspondence: External Appeal Denial
- Correspondence: External Appeal Win
- Correspondence: Final Internal Appeal Denial
- Correspondence: Form: Form request for (TimelyFiling) Good Cause
- Correspondence: Form: Form request for Physician's  Attestation
- Correspondence: Form: Form request for Provider Appeal Form
- Correspondence: Form: Form request for W9
- Correspondence: Form: Form request Member Authorization to Appeal/DAR
- Correspondence: Insurance Authorization Approval / Not Required
- Correspondence: Insurance Authorization Denial
- Correspondence: Insurance indicates Insurance Authorization was not Performed
- Correspondence: Invalid Appeal: Appeal not valid for payor
- Correspondence: Invalid Authorization: Authorization not v alid for Payor
- Correspondence: Invalid Claim: Claim not valid for payor
- Correspondence: LOA Request from Payor
- Correspondence: LOA signed: LOA signed
- Correspondence: Maximus Involved
- Correspondence: Medical Panel Appeal Opportunity
- Correspondence: Medical Records: Medical record request
- Correspondence: Medical Records: Medical record request/ Lab report request
- Correspondence: Medical Records: Medical record request/MD's facility/office valid address
- Correspondence: Medical Records: Medical record request/Other Insurance EOB
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of Appeal
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of claim
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of External Appeal
- Correspondence: Receipt acknowledgement: Receipt acknowledgement of withdrawal
- Correspondence: Receipt acknowledgement: Receipt of Legal Document
- Correspondence: Receipt acknowledgement: Receipt of Member Eligibility
- Correspondence: Receipt Insurance Authorization Request
- Correspondence: Received Request for Prior Auth
- Correspondence: Received: Appeal Rep Attention Needed/Unspecified Correspondence
- Correspondence: Received: Legal Documents
- Correspondence: Received: Medical Records
- Correspondence: Received: Patient Notes/Question
- Correspondence: Received: Returned Mail
- Correspondence: Received: Signed Member Authorization/DAR
- Correspondence: Received: Signed PABF
- Correspondence: Received: Signed Physician's Attestation
- Correspondence: Refund Request: Refund request
- Correspondence: Refund Request: Refund request withdrawal
- Correspondence: Sent to Alexa
- Correspondence: Termed: Coverage was not active for DOS
- Correspondence: Test Description Requested: Test Description Requested
- Correspondence: Timely Filing: Timely filing of an Appeal
- Correspondence: Timely Filing: Timely filing of an Claim
- Correspondence: Timely Filing: Timely filing of an External Appeal
- Correspondence: Timely Filing: Timely submission of requested additional information
- Correspondence: Unable to find patient
- Correspondence: WOL: Request for a WOL (blank WOL)
- Received: Signed Medicare waiver/Advance Beneficiary Notice (ABN)
- Received: Signed WOL form with member signature
- Correspondence: Appeal Not Eligible for Review

- Conflict resolution:
  • Prefer explicit outcome/decision phrases that appear anywhere in group_text.
  • If multiple candidate labels appear, pick the most specific outcome-oriented one matching the dominant cues across the group.
  • If denial language states appeal rights are exhausted, classify as "Correspondence: Final Internal Appeal Denial".
  • If correspondence says all appeal levels are already completed and current request was not evaluated, classify as "Correspondence: Appeal Not Eligible for Review".
  • If timeliness is primary and there is NO second-level/next-step instruction, choose the appropriate Timely Filing label.

OUTPUT FORMAT
Return exactly:
{{
  "documents": [
    {{
      "t": "<ONE exact label from the taxonomy above>",
      "pt": "<start-end>",
      "s" : "<two lines summary on what the grouped pages classify as, using group_text + grouped DATA>",
      "d": {{
        // Include ONLY keys that have non-null values after consolidation.
      }}
    }}
  ]
}}

MULTI-ROW EXAMPLE EXPECTATION
- If a page (or grouped set) yields Claim ID values ["UB87401E20250507","UB87401E20250513","UB87401E20250427"] in separate data rows,
  you MUST output **three** separate entries in "documents" — one per Claim ID (and corresponding row identifiers).

Now classify: {pages_data}"""


