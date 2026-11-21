"""Fact-Check Agent — Streamlit entry point.

Step 1 skeleton: upload a PDF, extract its text, and show a page summary.
Claim extraction and live verification are added in later steps.
"""

import streamlit as st

from factcheck.pdf_reader import MAX_FILE_MB, MAX_PAGES, PDFReadError, read_pdf

st.set_page_config(page_title="Fact-Check Agent", layout="wide")

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

try:
    doc = read_pdf(uploaded.getvalue())
except PDFReadError as err:
    st.error(str(err))
    st.stop()

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
