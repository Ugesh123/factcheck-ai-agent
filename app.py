"""Fact-Check Agent: Streamlit entry point.

Upload a PDF -> extract checkable claims -> search the live web for each
claim -> label it Verified, Inaccurate or False, with the correct fact and
sources. Layout uses containers only (no st.dataframe / st.table).
"""

import csv
import hashlib
import io

import streamlit as st

from factcheck.claims import (
    MAX_CLAIMS,
    build_batches,
    extract_from_batch,
    extract_missed,
    finalise_claims,
    find_candidates,
    uncovered,
)
from factcheck.config import ConfigError, llm_api_key, llm_models, tavily_api_key
from factcheck.llm import LLMError, make_client
from factcheck.pdf_reader import MAX_FILE_MB, MAX_PAGES, PDFReadError, read_pdf
from factcheck.search import SearchError, gather_evidence
from factcheck.verify import (
    NOT_CHECKED,
    VERIFY_BATCH_SIZE,
    build_payload,
    finalise_results,
    make_batches,
    today_string,
    verify_batch,
)

st.set_page_config(page_title="Fact-Check Agent", layout="wide")

VERDICT_STYLE = {
    "Verified": ":green[**Verified**]",
    "Inaccurate": ":orange[**Inaccurate**]",
    "False": ":red[**False**]",
    NOT_CHECKED: ":gray[**Not checked**]",
}

VERDICT_ORDER = ["Inaccurate", "False", "Verified", NOT_CHECKED]


MISSED_PASSES = 2  # extra requests per batch for sentences the first reply skipped


@st.cache_resource(ttl=24 * 3600)
def complete_extractions() -> dict:
    """Extraction results per batch, stored only once they passed the
    completeness check. An incomplete reply is never cached, so the next run
    asks again instead of repeating a bad result for 24 hours."""
    return {}


def extract_batch(client, models: tuple[str, ...], batch, candidates, log: list[dict]):
    """Extract one batch, then re-ask about any number-bearing sentence it skipped.

    Returns (raw claims, error message or None).
    """
    key = (hashlib.sha256(batch.text.encode()).hexdigest(), models)
    store = complete_extractions()
    if key in store:
        log.append({"batch": batch.label, "pass": "saved result", "model": "-", "claims": len(store[key])})
        return list(store[key]), None

    found: list[dict] = []
    error = None
    passes = [("first pass", lambda m: extract_from_batch(client, list(models), batch.text, m))]
    for n in range(1, MISSED_PASSES + 1):
        passes.append((f"retry {n}", None))

    for name, call in passes:
        if call is None:
            missing = uncovered(candidates, found)
            if not missing:
                break
            call = lambda m, missing=missing: extract_missed(client, list(models), missing, m)
        meta: dict = {}
        try:
            got = call(meta)
        except LLMError as err:
            if err.fatal:
                raise
            error = str(err)
            log.append({"batch": batch.label, "pass": name, "model": meta.get("model", "?"),
                        "claims": 0, "note": error})
            continue
        found.extend(got)
        log.append({"batch": batch.label, "pass": name, "model": meta.get("model", "?"),
                    "claims": len(got), "note": meta.get("mode", "")})

    if found and not uncovered(candidates, found):
        store[key] = list(found)
        error = None  # a retry recovered what the failed request missed
    return found, error


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_verify(payload: str, models: tuple[str, ...], today: str) -> list[dict]:
    """One LLM call per batch of claims, cached on the exact claims + evidence."""
    client = make_client(llm_api_key())
    return verify_batch(client, list(models), payload, today)


def md(text: str) -> str:
    """Escape text for st.markdown ($ would otherwise start a maths formula)."""
    return str(text).replace("\\", "\\\\").replace("$", "\\$").replace("[", "\\[").replace("]", "\\]")


def results_csv(results: list[dict]) -> str:
    fields = ["id", "page", "claim", "quote", "verdict", "reason_label", "correct_fact",
              "explanation", "confidence", "sources"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        writer.writerow({**r, "sources": " | ".join(s["url"] for s in r["sources"])})
    return buffer.getvalue()


def show_result(r: dict):
    with st.container(border=True):
        st.markdown(f"{VERDICT_STYLE[r['verdict']]} · {r['reason_label']}")
        meta = f"{r['id']} · page {r['page'] or '?'}"
        if r["confidence"]:
            meta += f" · {r['confidence']} confidence"
        st.caption(meta)

        st.markdown(f"**Claim:** {md(r['claim'])}")
        st.caption(f"In the document: “{md(r['quote'])}”")

        if r["correct_fact"]:
            st.markdown(f"**Correct fact:** :green[{md(r['correct_fact'])}]")
        if r["explanation"]:
            st.markdown(md(r["explanation"]))
        if r["verdict"] == "False":
            st.caption("No reliable evidence was found for this claim. That is not proof it is untrue.")

        if r["sources"]:
            links = []
            for s in r["sources"]:
                date = f" · {s['published_date'][:10]}" if s.get("published_date") else ""
                links.append(f"- [{md(s['title'])}]({s['url']}) · {s['domain']}{date}")
            st.markdown("**Sources**\n" + "\n".join(links))


def claims_to_csv(claims: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(claims[0].keys()))
    writer.writeheader()
    writer.writerows(claims)
    return buffer.getvalue()


# ---------------------------------------------------------------- header

st.title("Fact-Check Agent")
st.caption(
    "Upload a PDF. The app extracts factual claims (stats, dates, financial and "
    "technical figures), checks each one against the live web, and labels it "
    "**Verified**, **Inaccurate** or **False**, with the correct fact and sources."
)

uploaded = st.file_uploader(
    f"Upload a PDF (max {MAX_FILE_MB} MB, {MAX_PAGES} pages)", type=["pdf"]
)

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_id = hashlib.sha256(file_bytes).hexdigest()

# A new file clears the previous file's results.
if st.session_state.get("file_id") != file_id:
    st.session_state["file_id"] = file_id
    for key in ("claims", "claim_warnings", "claim_missing", "claim_log", "claim_total",
                "results", "verify_warnings"):
        st.session_state.pop(key, None)

try:
    doc = read_pdf(file_bytes)
except PDFReadError as err:
    st.error(str(err))
    st.stop()

# ---------------------------------------------------------------- document summary

col1, col2, col3 = st.columns(3)
col1.metric("Pages", doc.page_count)
col2.metric("Pages with text", sum(1 for p in doc.pages if p.text.strip()))
col3.metric("Characters", f"{doc.char_count:,}")

if doc.image_only_pages:
    st.warning(
        "These pages have no extractable text (they may be scanned images) and "
        f"will be skipped: {', '.join(map(str, doc.image_only_pages))}"
    )

with st.expander("Preview extracted text"):
    for page in doc.pages:
        if page.text.strip():
            st.markdown(f"**Page {page.number}**")
            st.text(page.text[:1500] + ("…" if len(page.text) > 1500 else ""))

# ---------------------------------------------------------------- claim extraction

st.divider()
st.subheader("1. Extract claims")

if st.button("Extract claims", type="primary"):
    try:
        llm_api_key()
        models = tuple(llm_models())
    except ConfigError as err:
        st.error(str(err))
        st.stop()

    batches = build_batches(doc)
    candidates = find_candidates(doc)
    client = make_client(llm_api_key())

    raw_claims: list[dict] = []
    warnings: list[str] = []
    log: list[dict] = []
    progress = st.progress(0.0, text="Reading the document…")

    for i, batch in enumerate(batches, start=1):
        progress.progress((i - 1) / len(batches), text=f"Extracting claims from {batch.label}…")
        batch_candidates = [c for c in candidates if c.page in batch.pages]
        try:
            found, error = extract_batch(client, models, batch, batch_candidates, log)
        except LLMError as err:  # fatal: bad key, daily limit, unknown model
            progress.empty()
            st.error(str(err))
            st.stop()
        raw_claims.extend(found)
        if error:
            warnings.append(f"Part of {batch.label} may be missing: {error}")

    progress.empty()
    claims = finalise_claims(raw_claims, doc)
    total = len(finalise_claims(raw_claims, doc, limit=10**6))
    missing = uncovered(candidates, raw_claims)
    st.session_state["claims"] = [c.to_dict() for c in claims]
    st.session_state["claim_warnings"] = warnings
    st.session_state["claim_missing"] = [{"page": m.page, "text": m.text} for m in missing]
    st.session_state["claim_log"] = log
    st.session_state["claim_total"] = total
    st.session_state.pop("results", None)  # new claims need a new check

claims = st.session_state.get("claims")

if claims is None:
    st.caption(f"Finds up to {MAX_CLAIMS} checkable claims in the document.")
    st.stop()

claim_warnings = st.session_state.get("claim_warnings", [])
claim_missing = st.session_state.get("claim_missing", [])

for warning in claim_warnings:
    st.warning(warning)

if claim_missing:
    lines = "\n".join(f"- Page {m['page']}: {md(m['text'])}" for m in claim_missing)
    st.warning(
        f"**Extraction may be incomplete.** {len(claim_missing)} sentence(s) in the PDF contain "
        f"a number or date but no claim was extracted from them, even after "
        f"{MISSED_PASSES} retries. Some may not be checkable facts (a document date, a "
        f"section number); check the rest by hand or click Extract claims again.\n\n{lines}"
    )

if st.session_state.get("claim_log"):
    with st.expander("Extraction details (which free model answered each request)"):
        for row in st.session_state["claim_log"]:
            note = f" · {row['note']}" if row.get("note") else ""
            st.write(f"{row['batch']} · {row['pass']} · `{row['model']}` · {row['claims']} claims{note}")

if not claims:
    st.info(
        "No checkable factual claims were found. The document may contain only "
        "opinions or general statements, with no specific numbers, dates or figures."
    )
    st.stop()

if claim_missing or claim_warnings:
    st.info(f"Showing the {len(claims)} claims that were extracted.")
else:
    st.success(f"Found {len(claims)} checkable claims.")

if st.session_state.get("claim_total", 0) > len(claims):
    st.info(
        f"The document has {st.session_state['claim_total']} distinct claims; the "
        f"{len(claims)} most important are shown (limit {MAX_CLAIMS})."
    )

unmatched = [c["id"] for c in claims if not c["quote_found"]]
if unmatched:
    st.caption(
        f"The quote for {', '.join(unmatched)} couldn't be matched exactly to the "
        "PDF text (often due to line breaks or special characters). Check these by hand."
    )

st.subheader("Extracted Claims")

for c in claims:
    with st.container(border=True):
        st.markdown(f"**{c['id']} — {md(c['claim'])}**")
        st.write(f"Page: {c['page'] or '?'}")
        st.write(f"Type: {c['claim_type']}")
        st.write(f"Value: {c['value']}")
        st.write(f"Time reference: {c['time_reference']}")
        st.write(f"Search query: {c['search_query']}")

with st.expander("Exact quotes from the document"):
    for c in claims:
        st.markdown(f"**{c['id']}** · page {c['page'] or '?'}")
        st.markdown(f"> {c['quote']}")

st.download_button(
    "Download claims (CSV)",
    data=claims_to_csv(claims),
    file_name="claims.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------- verification

st.divider()
st.subheader("2. Verify against the live web")

if st.button("Verify claims", type="primary"):
    try:
        search_key = tavily_api_key()
        llm_api_key()
        models = tuple(llm_models())
    except ConfigError as err:
        st.error(str(err))
        st.stop()

    progress = st.progress(0.0, text="Searching the web…")

    def on_search_progress(done, total):
        progress.progress(0.6 * done / total, text=f"Searching the web… ({done}/{total} claims)")

    try:
        evidence, search_errors = gather_evidence(claims, search_key, on_search_progress)
    except SearchError as err:
        progress.empty()
        st.error(str(err))
        st.stop()

    not_checked = dict(search_errors)
    to_verify = [c for c in claims if c["id"] not in not_checked and evidence.get(c["id"])]
    by_id = {c["id"]: c for c in to_verify}
    verify_batches = make_batches([c["id"] for c in to_verify])
    raw_results: list[dict] = []
    warnings: list[str] = []
    today = today_string()

    for i, batch_ids in enumerate(verify_batches, start=1):
        progress.progress(0.6 + 0.4 * (i - 1) / len(verify_batches),
                          text=f"Checking claims against the evidence… (batch {i} of {len(verify_batches)})")
        payload = build_payload([by_id[cid] for cid in batch_ids], evidence)
        try:
            raw_results.extend(cached_verify(payload, models, today))
        except LLMError as err:
            remaining = [cid for b in verify_batches[i - 1:] for cid in b] if err.fatal else batch_ids
            for cid in remaining:
                not_checked[cid] = str(err)
            warnings.append(str(err))
            if err.fatal:
                break

    results = finalise_results(claims, evidence, raw_results, not_checked)

    # Retry once, uncached, any claim whose verdict came back without a cited source.
    retry_ids = [r["id"] for r in results if r.get("uncited")]
    if retry_ids:
        progress.progress(0.95, text=f"Re-checking {len(retry_ids)} claim(s) that came back without a source…")
        try:
            retry_raw = verify_batch(make_client(llm_api_key()), list(models),
                                     build_payload([by_id[cid] for cid in retry_ids], evidence), today)
            retry_set = set(retry_ids)
            raw_results = [r for r in raw_results
                           if not (isinstance(r, dict) and str(r.get("claim_id", "")).strip() in retry_set)] + retry_raw
            results = finalise_results(claims, evidence, raw_results, not_checked)
        except LLMError as err:
            warnings.append(f"Retry of uncited claims failed: {err}")

    progress.empty()
    st.session_state["results"] = results
    st.session_state["verify_warnings"] = warnings

results = st.session_state.get("results")

if results is None:
    st.caption(
        f"Searches the web for each claim and checks up to {VERIFY_BATCH_SIZE} claims "
        "per model request."
    )
    st.stop()

for warning in st.session_state.get("verify_warnings", []):
    st.warning(warning)

counts = {v: sum(1 for r in results if r["verdict"] == v) for v in VERDICT_ORDER}

m1, m2, m3 = st.columns(3)
m1.metric("Verified", counts["Verified"])
m2.metric("Inaccurate", counts["Inaccurate"])
m3.metric("False", counts["False"])

st.caption(
    "**Verified** = matches live web data. "
    "**Inaccurate** = contradicted or outdated; the correct fact is shown. "
    "**False** = no reliable evidence found. This means nothing online supports the claim, "
    "not that it was proven wrong."
)

if counts[NOT_CHECKED]:
    st.warning(
        f"{counts[NOT_CHECKED]} claim(s) could not be checked: a search or model request failed, "
        "or the model gave a verdict without citing a source (already retried once). "
        "They have no verdict; open each one under 'Not checked' for details."
    )

filters = ["All", "Inaccurate", "False", "Verified"] + ([NOT_CHECKED] if counts[NOT_CHECKED] else [])
choice = st.radio("Show", filters, horizontal=True)
shown = [r for r in results if choice == "All" or r["verdict"] == choice]
shown.sort(key=lambda r: (VERDICT_ORDER.index(r["verdict"]), r["id"]))

if not shown:
    st.caption("No claims in this category.")
for r in shown:
    show_result(r)

st.download_button(
    "Download report (CSV)",
    data=results_csv(results),
    file_name="fact_check_report.csv",
    mime="text/csv",
)