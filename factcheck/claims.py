"""Claim extraction.

The document is split into batches of whole pages. Each batch is sent to the
LLM with a strict JSON schema. The results are then cleaned up in code:
quotes are checked against the real page text, duplicates are removed, and
the list is ranked and capped at MAX_CLAIMS.

Free models vary a lot from one request to the next, so the app doesn't
trust a single reply. find_candidates() lists every full sentence in the PDF
that contains a number; uncovered() shows which of them no claim came from.
Those sentences are sent back once more (extract_missed), and anything still
uncovered is reported to the user instead of being silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .llm import structured_completion

if TYPE_CHECKING:
    from openai import OpenAI

    from .pdf_reader import Document

MAX_CLAIMS = 25
PER_BATCH_LIMIT = 15
BATCH_CHAR_LIMIT = 12000

CLAIM_TYPES = ["statistic", "date", "financial", "technical"]
IMPORTANCE_RANK = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------- schema

_FIELDS = {
    "claim": {
        "type": "string",
        "description": "The claim restated as one self-contained sentence, with the subject named (no 'it' or 'the company').",
    },
    "quote": {
        "type": "string",
        "description": "The exact sentence or phrase from the document that contains the claim, copied character for character.",
    },
    "page": {"type": "integer", "description": "Page number from the [PAGE n] marker the quote appears under."},
    "claim_type": {"type": "string", "enum": CLAIM_TYPES},
    "entity": {"type": "string", "description": "Who or what the claim is about, e.g. 'ChatGPT', 'India', 'Tesla'."},
    "metric": {"type": "string", "description": "What is measured, e.g. 'weekly active users', 'founding year', 'annual revenue'."},
    "value": {"type": "string", "description": "The figure or fact as stated in the document, with units."},
    "time_reference": {
        "type": "string",
        "description": "The date or period the claim refers to, as stated. Use 'not stated' if the document gives none.",
    },
    "search_query": {
        "type": "string",
        "description": "A short web search query (under 12 words) that would find an authoritative source for this fact. Name the entity and metric. Do not include the claimed value.",
    },
    "importance": {
        "type": "string",
        "enum": ["high", "medium", "low"],
        "description": "high = a headline number or fact the document relies on; low = a minor detail.",
    },
}

CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_FIELDS),
                "properties": _FIELDS,
            },
        }
    },
}

SYSTEM_PROMPT = f"""You extract checkable factual claims from documents for a fact-checking tool.

Extract a claim only if it can be confirmed or disproved with a web search. That means it contains at least one of:
- a statistic or number (users, market size, percentages, growth rates, rankings, counts)
- a date or year (founding, launch, release, event)
- a financial figure (revenue, valuation, funding, price, profit)
- a technical fact (specifications, capabilities, versions, measurements)

Do not extract:
- opinions, predictions, promises or marketing language ("industry-leading", "will transform")
- vague statements with no checkable detail ("many companies use AI")
- the document's own instructions, headings or contact details

Rules:
- Copy the quote exactly from the document. Do not fix typos or rephrase it.
- Use the page number from the [PAGE n] marker above the quote.
- If one sentence contains two separate facts, make two claims.
- If the same fact appears twice, extract it once.
- Extract at most {PER_BATCH_LIMIT} claims, choosing the most important ones.
- Claims inside tables and bullet lists count too.
- If there are no checkable claims, return an empty list."""


# ---------------------------------------------------------------- data

@dataclass
class Claim:
    id: str
    claim: str
    quote: str
    page: int
    claim_type: str
    entity: str
    metric: str
    value: str
    time_reference: str
    search_query: str
    importance: str
    quote_found: bool  # False if the quote couldn't be matched to the PDF text

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Batch:
    pages: list[int]
    text: str

    @property
    def label(self) -> str:
        if self.pages[0] == self.pages[-1]:
            return f"page {self.pages[0]}"
        return f"pages {self.pages[0]}-{self.pages[-1]}"


# ---------------------------------------------------------------- batching

def build_batches(doc: Document, char_limit: int = BATCH_CHAR_LIMIT) -> list[Batch]:
    """Group pages into batches under char_limit; split any oversized page."""
    batches: list[Batch] = []
    pages: list[int] = []
    parts: list[str] = []
    size = 0

    def flush():
        nonlocal pages, parts, size
        if parts:
            batches.append(Batch(pages=pages, text="\n\n".join(parts)))
        pages, parts, size = [], [], 0

    for page in doc.pages:
        text = page.text.strip()
        if not text:
            continue
        chunks = [text[i : i + char_limit] for i in range(0, len(text), char_limit)]
        for chunk in chunks:
            block = f"[PAGE {page.number}]\n{chunk}"
            if size + len(block) > char_limit:
                flush()
            pages.append(page.number)
            parts.append(block)
            size += len(block)
    flush()

    for batch in batches:
        batch.pages = sorted(set(batch.pages))
    return batches


# ---------------------------------------------------------------- LLM call

def _claim_list(result: dict) -> list[dict]:
    claims = result.get("claims", [])
    return [c for c in claims if isinstance(c, dict)] if isinstance(claims, list) else []


def extract_from_batch(client: OpenAI, models: list[str], batch_text: str,
                       meta: dict | None = None) -> list[dict]:
    """Send one batch to the LLM and return the raw claim dicts."""
    result = structured_completion(
        client=client,
        models=models,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Extract the checkable factual claims from this document text.\n\n{batch_text}",
        schema_name="claim_list",
        schema=CLAIM_SCHEMA,
        temperature=0,
        meta=meta,
    )
    return _claim_list(result)


def extract_missed(client: OpenAI, models: list[str], sentences: list[Candidate],
                   meta: dict | None = None) -> list[dict]:
    """Second pass: ask only about the sentences the first reply skipped."""
    listing = "\n".join(f"[PAGE {c.page}] {c.text}" for c in sentences)
    result = structured_completion(
        client=client,
        models=models,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "An earlier pass skipped the sentences below. Each one contains a number or a date. "
            "Go through every sentence and extract each checkable factual claim it contains. "
            "Leave out a sentence only if it has nothing that a web search could confirm or "
            "disprove (for example the document's own date or a section number).\n\n" + listing
        ),
        schema_name="claim_list",
        schema=CLAIM_SCHEMA,
        temperature=0,
        meta=meta,
    )
    return _claim_list(result)


# ---------------------------------------------------------------- completeness check

@dataclass
class Candidate:
    """A full sentence in the PDF that contains a number, so it probably holds a claim."""
    page: int
    text: str


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_LIST_NUMBER = re.compile(r"^\(?\d{1,2}[.)]\s+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def _blocks(page_text: str) -> list[str]:
    """Join wrapped lines back into paragraphs, keeping headings separate.

    A short line with no closing punctuation, followed by a line that starts
    with a capital or a digit, is treated as a heading rather than the start
    of the next sentence.
    """
    lines = [l.strip() for l in page_text.replace("-\n", "").splitlines()]
    blocks, current = [], []
    for i, line in enumerate(lines):
        if not line:
            if current:
                blocks.append(" ".join(current)); current = []
            continue
        current.append(line)
        nxt = next((l for l in lines[i + 1:] if l), "")
        heading = len(line) < 60 and line[-1] not in ".!?,;:" and (nxt[:1].isupper() or nxt[:1].isdigit())
        if heading:
            blocks.append(" ".join(current)); current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def find_candidates(doc: Document) -> list[Candidate]:
    """Every full sentence (ending in . ! or ?) of 20+ characters with a digit in it.

    Headings and table cells are left out: they rarely end in a full stop, and
    this check is a safety net, not the extractor itself.
    """
    found: list[Candidate] = []
    for page in doc.pages:
        for block in _blocks(page.text):
            flat = re.sub(r"\s+", " ", block).strip()
            for sentence in _SENTENCE_END.split(flat):
                sentence = _LIST_NUMBER.sub("", sentence.strip())
                if len(sentence) >= 20 and sentence[-1] in ".!?" and _numbers(sentence):
                    found.append(Candidate(page=page.number, text=sentence))
    return found


def _covers(claim: dict, sentence_norm: str, sentence_numbers: set[str]) -> bool:
    quote = _normalise(str(claim.get("quote", "")))
    if quote and (quote in sentence_norm or sentence_norm in quote
                  or (len(quote) >= 40 and quote[:40] in sentence_norm)):
        return True
    # Quote paraphrased: accept if a figure and the entity both appear in the sentence.
    entity = _normalise(str(claim.get("entity", "")))
    figures = _numbers(f"{claim.get('value', '')} {claim.get('quote', '')}")
    return bool(entity) and entity in sentence_norm and bool(figures & sentence_numbers)


def uncovered(candidates: list[Candidate], claims: list[dict]) -> list[Candidate]:
    """Candidate sentences that none of the claims were taken from."""
    missing = []
    for cand in candidates:
        norm, nums = _normalise(cand.text), _numbers(cand.text)
        if not any(_covers(c, norm, nums) for c in claims):
            missing.append(cand)
    return missing


# ---------------------------------------------------------------- clean-up

_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "•": " ",
}


def _normalise(text: str) -> str:
    text = text.replace("-\n", "")  # words hyphenated across a line break
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip().lower()


def _find_quote_page(quote: str, claimed_page: int, page_texts: dict[int, str]) -> int | None:
    """Return the page the quote really appears on, or None if it isn't found.

    Checks the claimed page first, then every other page. If the full quote
    doesn't match (models sometimes trim punctuation), the first 60
    characters are tried instead.
    """
    q = _normalise(quote)
    if not q:
        return None
    candidates = [q] if len(q) <= 60 else [q, q[:60]]
    order = [claimed_page] + [p for p in page_texts if p != claimed_page]
    for needle in candidates:
        for page in order:
            if needle in page_texts.get(page, ""):
                return page
    return None


def finalise_claims(raw_claims: list[dict], doc: Document, limit: int = MAX_CLAIMS) -> list[Claim]:
    """Validate, de-duplicate, rank and cap the claims from all batches."""
    page_texts = {p.number: _normalise(p.text) for p in doc.pages}
    seen: set[str] = set()
    cleaned: list[Claim] = []

    for raw in raw_claims:
        if not isinstance(raw, dict):
            continue
        claim_text = str(raw.get("claim", "")).strip()
        quote = str(raw.get("quote", "")).strip()
        if not claim_text or not quote:
            continue

        try:
            page = int(raw.get("page", 0))
        except (TypeError, ValueError):
            page = 0

        found_page = _find_quote_page(quote, page, page_texts)
        if found_page is not None:
            page = found_page
        elif page not in page_texts:
            page = 0  # unknown

        claim_type = raw.get("claim_type", "statistic")
        if claim_type not in CLAIM_TYPES:
            claim_type = "statistic"
        importance = raw.get("importance", "medium")
        if importance not in IMPORTANCE_RANK:
            importance = "medium"

        entity = str(raw.get("entity", "")).strip()
        metric = str(raw.get("metric", "")).strip()
        value = str(raw.get("value", "")).strip()

        keys = {_normalise(claim_text)}
        if entity and metric and value:
            keys.add(_normalise(f"{entity}|{metric}|{value}"))
        if keys & seen:
            continue
        seen |= keys

        cleaned.append(
            Claim(
                id="",
                claim=claim_text,
                quote=quote,
                page=page,
                claim_type=claim_type,
                entity=entity,
                metric=metric,
                value=value,
                time_reference=str(raw.get("time_reference", "")).strip() or "not stated",
                search_query=str(raw.get("search_query", "")).strip() or claim_text,
                importance=importance,
                quote_found=found_page is not None,
            )
        )

    # Quotes that match the PDF first, then by importance, then reading order.
    cleaned.sort(key=lambda c: (not c.quote_found, IMPORTANCE_RANK[c.importance], c.page))
    selected = cleaned[:limit]
    selected.sort(key=lambda c: (c.page if c.page else 10_000))

    for i, claim in enumerate(selected, start=1):
        claim.id = f"C{i:02d}"
    return selected
