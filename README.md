# Fact-Check Agent

A web app that reads a PDF, extracts its factual claims (statistics, dates, financial and technical figures), checks each claim against the live web, and labels it:

| Label | Meaning |
|---|---|
| **Verified** | Matches current, reliable sources |
| **Inaccurate** | Contradicted or outdated; the correct fact is shown |
| **False** | No supporting evidence found |

**Live app:** _link added after deployment_

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your keys
streamlit run app.py
```

_Architecture, verdict rules, evaluation results and limitations are added as the build progresses._
