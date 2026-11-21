"""Claim verification against retrieved web evidence.

Claims are checked in batches (VERIFY_BATCH_SIZE per LLM request) to stay
within OpenRouter's free-model limits. The model picks a *reason* for each
claim; the verdict is derived from the reason in code, so the three
assessment labels are always applied consistently:

    reason                  verdict
    matches                 Verified
    outdated                Inaccurate   (correct fact required)
    contradicted            Inaccurate   (correct fact required)
    not_supported           False        no evidence: sources cover the topic, none support the claim
    insufficient_evidence   False        no evidence: too little relevant information online

"False" follows the assessment's definition (no evidence found). It means
no reliable source supports the claim, not that the claim was disproven;
claims that sources actively contradict are labelled Inaccurate.

The model cites sources by ID only. IDs are mapped back to URLs in code,
so it can't cite a page that wasn't actually retrieved. A Verified or
Inaccurate verdict that cites no retrieved source is not accepted: the claim
is marked "uncited" so the app can retry it, and stays Not checked if the
retry also fails. It is never turned into False.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from .llm import structured_completion

if TYPE_CHECKING:
    from openai import OpenAI

    from .search import Source

VERIFY_BATCH_SIZE = 8

REASON_TO_VERDICT = {
    "matches": "Verified",
    "outdated": "Inaccurate",
    "contradicted": "Inaccurate",
    "not_supported": "False",
    "insufficient_evidence": "False",
}

REASON_LABELS = {
    "matches": "Matches the sources",
    "outdated": "Outdated: newer data exists",
    "contradicted": "Contradicted by the sources",
    "not_supported": "No reliable evidence found: sources on this topic don't support it",
    "insufficient_evidence": "No reliable evidence found: little relevant information online",
}

NOT_CHECKED = "Not checked"
CONFIDENCE_LEVELS = ["high", "medium", "low"]

_RESULT_FIELDS = {
    "claim_id": {"type": "string"},
    "reason": {"type": "string", "enum": list(REASON_TO_VERDICT)},
    "correct_fact": {
        "type": "string",
        "description": "For outdated or contradicted claims: the correct fact from the sources, with units and its date. Otherwise an empty string.",
    },
    "explanation": {"type": "string", "description": "One or two plain sentences on what the sources show."},
    "source_ids": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
}

VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_RESULT_FIELDS),
                "properties": _RESULT_FIELDS,
            },
        }
    },
}

SYSTEM_PROMPT = """You are a careful fact-checker. Today's date is {today}.

For each claim, judge it using ONLY the sources listed under that claim. Do not use your own knowledge to accept or reject a claim; it may be out of date.

Pick one reason per claim:
- matches: a reliable source states the same fact. Allow rounding and approximate wording ("over 1 billion" matches 1.2 billion). A claim tied to a past date ("in 2023, X was Y") matches if it was true at that date, even if newer figures exist.
- outdated: the figure was true at some point, but the sources show a newer one, and the claim presents the old figure as current (no date given, or words like "now", "currently", "today").
- contradicted: the sources clearly state a different fact (a different number, date, name or detail).
- not_supported: the sources discuss this topic, but none of them state or support this specific fact. This means no evidence was found; it does not prove the claim is untrue.
- insufficient_evidence: the sources are off-topic or too thin to judge either way.

correct_fact: for outdated or contradicted, write the correct fact as the sources give it, with units and its date or period, e.g. "About 1.2 billion weekly users (September 2026)". For any other reason, use an empty string.
explanation: one or two plain sentences on what the sources show. For not_supported and insufficient_evidence, say what was searched for and that no reliable source confirms it; do not say the claim is disproven.
source_ids: the IDs of the sources you relied on, exactly as written (e.g. "C03-S2"). For matches, outdated and contradicted, cite at least one. Never make up an ID.
confidence: high if the sources are authoritative and agree; medium if they are secondary or slightly unclear; low if they conflict or are weak.

When sources disagree, prefer official sources (the company itself, government, regulators, original research) over news, news over blogs and forums, and newer over older.

Return exactly one result for every claim ID."""


# ---------------------------------------------------------------- batching

def make_batches(claim_ids: list[str], size: int = VERIFY_BATCH_SIZE) -> list[list[str]]:
    return [claim_ids[i : i + size] for i in range(0, len(claim_ids), size)]


def build_payload(claims: list[dict], evidence: dict[str, list[Source]]) -> str:
    """The user message for one batch: each claim followed by its sources."""
    blocks = []
    for c in claims:
        lines = [
            f"### Claim {c['id']}",
            f"Claim: {c['claim']}",
            f"Exact wording in the document: \"{c['quote']}\"",
            f"Stated value: {c['value']} | Time reference: {c['time_reference']}",
            "Sources:",
        ]
        for n, s in enumerate(evidence.get(c["id"], []), start=1):
            published = f" | published {s.published_date}" if s.published_date else ""
            lines.append(f"[{c['id']}-S{n}] {s.title} ({s.domain}{published})\n{s.content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def verify_batch(client: OpenAI, models: list[str], payload: str, today: str) -> list[dict]:
    """One LLM request for one batch. Returns the raw result dicts."""
    result = structured_completion(
        client=client,
        models=models,
        system_prompt=SYSTEM_PROMPT.format(today=today),
        user_prompt=f"Fact-check each claim below against its sources.\n\n{payload}",
        schema_name="verification_results",
        schema=VERIFY_SCHEMA,
    )
    results = result.get("results", [])
    return results if isinstance(results, list) else []


# ---------------------------------------------------------------- clean-up

def _not_checked(claim: dict, note: str, uncited: bool = False) -> dict:
    return {
        **claim,
        "verdict": NOT_CHECKED,
        "reason": "",
        "reason_label": "Could not be checked",
        "correct_fact": "",
        "explanation": note,
        "sources": [],
        "confidence": "",
        "uncited": uncited,
    }


def no_evidence_result(claim: dict) -> dict:
    """For claims where the search returned nothing: no LLM request needed."""
    return {
        **claim,
        "verdict": "False",
        "reason": "insufficient_evidence",
        "reason_label": REASON_LABELS["insufficient_evidence"],
        "correct_fact": "",
        "explanation": "The live web search returned no relevant results, so no reliable evidence supports this claim. This does not prove it is untrue.",
        "sources": [],
        "confidence": "low",
    }


def finalise_results(
    claims: list[dict],
    evidence: dict[str, list[Source]],
    raw_results: list[dict],
    not_checked: dict[str, str],
) -> list[dict]:
    """Combine claims with the model's results, enforcing the verdict rules."""
    by_id = {}
    for raw in raw_results:
        if isinstance(raw, dict) and raw.get("claim_id"):
            by_id.setdefault(str(raw["claim_id"]).strip(), raw)

    final = []
    for claim in claims:
        cid = claim["id"]
        if cid in not_checked:
            final.append(_not_checked(claim, not_checked[cid]))
            continue
        if not evidence.get(cid):
            final.append(no_evidence_result(claim))
            continue
        raw = by_id.get(cid)
        if raw is None:
            final.append(_not_checked(claim, "The model didn't return a result for this claim. Run verification again."))
            continue

        reason = raw.get("reason")
        if reason not in REASON_TO_VERDICT:
            reason = "insufficient_evidence"
        confidence = raw.get("confidence") if raw.get("confidence") in CONFIDENCE_LEVELS else "low"
        explanation = str(raw.get("explanation", "")).strip()
        correct_fact = str(raw.get("correct_fact", "")).strip()

        # Keep only source IDs that belong to this claim's retrieved evidence.
        claim_sources = evidence[cid]
        cited, seen = [], set()
        for sid in raw.get("source_ids") or []:
            prefix = f"{cid}-S"
            sid = str(sid).strip().strip("[]")
            if not sid.startswith(prefix) or not sid[len(prefix):].isdigit():
                continue
            n = int(sid[len(prefix):])
            if 1 <= n <= len(claim_sources) and n not in seen:
                seen.add(n)
                s = claim_sources[n - 1]
                cited.append({"url": s.url, "title": s.title, "domain": s.domain, "published_date": s.published_date})

        # A verdict that rests on evidence must cite some. Without a citation
        # the verdict is rejected (not turned into False) so it can be retried.
        if reason in ("matches", "outdated", "contradicted") and not cited:
            final.append(_not_checked(
                claim,
                "The model gave a verdict without citing any retrieved source, so it was not accepted.",
                uncited=True,
            ))
            continue

        if reason in ("matches", "not_supported", "insufficient_evidence"):
            correct_fact = ""
        elif not correct_fact:
            confidence = "low"
            explanation = (explanation + " (The sources didn't give a clear corrected figure.)").strip()

        final.append(
            {
                **claim,
                "verdict": REASON_TO_VERDICT[reason],
                "reason": reason,
                "reason_label": REASON_LABELS[reason],
                "correct_fact": correct_fact,
                "explanation": explanation,
                "sources": cited,
                "confidence": confidence,
            }
        )
    return final


def today_string() -> str:
    return date.today().strftime("%d %B %Y")