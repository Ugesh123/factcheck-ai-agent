# Fact-Check Agent

A web app that reads a PDF, pulls out its factual claims (statistics, dates, financial and technical figures), checks each claim against the live web, and labels it with one of three verdicts, along with the correct fact and its sources.

**Live app:** https://factcheck-ai-agent-egowdvhxhafxbfdytp7koa.streamlit.app/

## Verdicts

| Verdict | Meaning | What the app shows |
|---|---|---|
| **Verified** | Matches live web data | Supporting sources |
| **Inaccurate** | Contradicted by sources, or outdated | The correct fact, with its date, and sources |
| **False** | No reliable evidence found | A note that nothing online supports the claim |

"False" follows the assessment's definition: *no evidence found*. It means no reliable source supports the claim, not that the claim was proven wrong. Claims that sources actively contradict are labelled **Inaccurate** and come with the correct figure.

If a search or model request fails, or the model gives a verdict without citing a source (even after one automatic retry), the claim is shown as **Not checked** with the reason. It is never silently labelled False.

## How it works

```
PDF → text by page (PyMuPDF)
    → claim extraction (LLM, structured JSON)
    → live web search per claim (Tavily: the claim itself + a "latest figure" query)
    → verification in batches of 8 claims (LLM judges only from the retrieved sources)
    → report: verdict, correct fact, explanation, clickable sources, CSV export
```

Design choices:
- **No vector database.** The evidence comes from the live web, not from stored documents, so search replaces retrieval.
- **Evidence-only judging.** The model is given today's date and must use only the retrieved sources, not its own memory, which may be out of date.
- **No invented citations.** The model cites sources by ID; the code maps IDs back to URLs it actually retrieved and drops anything else.
- **Rules in code.** The model picks a reason (`matches`, `outdated`, `contradicted`, `not_supported`, `insufficient_evidence`); the code turns it into the verdict. A Verified or Inaccurate verdict that cites no retrieved source is rejected and retried once; if it still has no source, the claim is shown as Not checked, never as False.
- **Outdated statistics.** A claim tied to a past date ("in 2023, X was Y") counts as Verified if it was true then. A figure presented as current but superseded is Inaccurate, with the latest figure.

## Tech stack

Python · Streamlit · PyMuPDF · Tavily (live search) · OpenRouter free models via the OpenAI SDK

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then add your keys
streamlit run app.py
```

`.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "sk-or-..."
TAVILY_API_KEY = "tvly-..."
LLM_MODEL = "openrouter/free"   # free models only; paid model IDs are rejected
```

On Streamlit Community Cloud, paste the same lines into **App settings → Secrets**. Keys are never stored in the repository.

## Testing

`tests/trap_test_ai_search_brief.pdf` mixes true, outdated and invented statements:

| Claim in the PDF | Expected verdict |
|---|---|
| ChatGPT now has 100 million weekly active users | Inaccurate (outdated) |
| In February 2026, OpenAI reported 900 million weekly ChatGPT users | Verified |
| Gemini app has 250 million monthly users | Inaccurate (outdated) |
| Google was founded in 2004 | Inaccurate (1998) |
| Tesla was founded in 2003 | Verified |
| Chandrayaan-3 landed near the Moon's south pole on 23 August 2023 | Verified |
| Mount Everest stands 7,849 metres above sea level | Inaccurate (8,849 m) |
| 2025 "Bengaluru Institute of Digital Futures" study: 92% of Indian SMEs use AI agents daily | False |
| "Global Marketing Truth Index 2026" ranks India first of 140 countries | False |

## Limitations

- Free-tier limits: OpenRouter free models allow about 50 requests a day, and Tavily's free plan gives 1,000 searches a month. One PDF uses about 3 model requests and about 3 searches per claim. Results are cached, so re-running the same PDF costs nothing.
- Up to 25 claims per PDF, 40 pages, 15 MB. Scanned (image-only) pages are skipped; there is no OCR.
- A claim can be labelled False when it is true but poorly covered online. The app says so in each False result.
- Free models vary in quality; the app falls back from strict JSON schemas to plain JSON mode when a free model doesn't support them.