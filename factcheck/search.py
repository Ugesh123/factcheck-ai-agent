"""Live web search with Tavily (free plan: 1,000 credits a month).

Each claim gets two searches:
  1. the claim's own search query (advanced depth, 2 credits) to find the fact
  2. a "latest" query (basic depth, 1 credit) to catch outdated statistics
Results are merged, de-duplicated by URL and trimmed so a batch of claims
fits comfortably in one LLM request. Searches run in parallel, and results
are cached in memory for 24 hours so repeat runs use no credits.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from urllib.parse import urlparse

from tavily import TavilyClient

MAX_RESULTS_PER_QUERY = 5
MAX_SOURCES_PER_CLAIM = 6
SNIPPET_CHARS = 700
SEARCH_WORKERS = 4
SEARCH_TIMEOUT_SECONDS = 45
CACHE_TTL_SECONDS = 24 * 3600


class SearchError(Exception):
    """A user-facing search failure. fatal=True means stop all searching."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class Source:
    url: str
    title: str
    content: str
    published_date: str  # "" when Tavily doesn't give one

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.removeprefix("www.")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- cache

_cache: dict[tuple[str, str], tuple[float, list[Source]]] = {}
_cache_lock = threading.Lock()


def _cache_get(key):
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]
    return None


def _cache_put(key, value):
    with _cache_lock:
        _cache[key] = (time.time(), value)


# ---------------------------------------------------------------- errors

def _to_search_error(exc: Exception) -> SearchError:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "invalidapikey" in name or "missingapikey" in name or "unauthorized" in text or "invalid api key" in text:
        return SearchError("Tavily rejected the API key. Check TAVILY_API_KEY in your secrets.", fatal=True)
    if "usagelimit" in name or "usage limit" in text or "exceeds your plan" in text:
        return SearchError("The Tavily free plan's monthly search credits have run out.", fatal=True)
    if "forbidden" in name:
        return SearchError("Tavily refused the request (forbidden).", fatal=True)
    return SearchError(f"Web search failed: {exc}")


# ---------------------------------------------------------------- search

def _search(client: TavilyClient, query: str, depth: str) -> list[Source]:
    key = (query.strip().lower(), depth)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        response = client.search(
            query=query,
            search_depth=depth,
            max_results=MAX_RESULTS_PER_QUERY,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        raise _to_search_error(exc) from exc

    sources = []
    for item in response.get("results", []):
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        if not url.startswith("http") or not content:
            continue
        sources.append(
            Source(
                url=url,
                title=(item.get("title") or "").strip() or url,
                content=content[:SNIPPET_CHARS],
                published_date=(item.get("published_date") or "").strip(),
            )
        )
    _cache_put(key, sources)
    return sources


def queries_for(claim: dict, today: date) -> list[tuple[str, str]]:
    """The (query, depth) pairs to run for one claim."""
    queries = [(claim["search_query"], "advanced")]
    entity, metric = claim.get("entity", ""), claim.get("metric", "")
    if entity and metric:
        latest = f"{entity} {metric} latest {today.year}"
        if latest.lower() != claim["search_query"].lower():
            queries.append((latest, "basic"))
    return queries


def _evidence_for_claim(client: TavilyClient, claim: dict, today: date) -> list[Source]:
    results_per_query = [_search(client, q, depth) for q, depth in queries_for(claim, today)]

    # Interleave results from both queries so neither crowds out the other.
    merged: list[Source] = []
    seen: set[str] = set()
    for rank in range(MAX_RESULTS_PER_QUERY):
        for results in results_per_query:
            if rank < len(results):
                source = results[rank]
                key = source.url.rstrip("/").lower()
                if key not in seen:
                    seen.add(key)
                    merged.append(source)
    return merged[:MAX_SOURCES_PER_CLAIM]


def gather_evidence(claims: list[dict], api_key: str, on_progress=None):
    """Search for every claim in parallel.

    Returns (evidence, errors): evidence maps claim id -> list[Source];
    errors maps claim id -> message for claims whose search failed.
    Raises SearchError if a fatal error (bad key, no credits) occurs.
    on_progress(done, total) is called from the calling thread.
    """
    client = TavilyClient(api_key=api_key)
    today = date.today()
    evidence: dict[str, list[Source]] = {}
    errors: dict[str, str] = {}

    # Generous overall limit: two queries per claim, SEARCH_WORKERS at a time.
    rounds = -(-len(claims) // SEARCH_WORKERS)
    overall_timeout = SEARCH_TIMEOUT_SECONDS * 2 * max(1, rounds)

    pool = ThreadPoolExecutor(max_workers=SEARCH_WORKERS)
    futures = {pool.submit(_evidence_for_claim, client, c, today): c["id"] for c in claims}
    try:
        for done, future in enumerate(as_completed(futures, timeout=overall_timeout), start=1):
            claim_id = futures[future]
            try:
                evidence[claim_id] = future.result()
            except SearchError as err:
                if err.fatal:
                    raise
                errors[claim_id] = str(err)
            except Exception as exc:  # unexpected failure in one claim
                errors[claim_id] = f"Web search failed: {exc}"
            if on_progress:
                on_progress(done, len(claims))
    except TimeoutError:
        for claim_id in futures.values():
            if claim_id not in evidence and claim_id not in errors:
                errors[claim_id] = "Web search timed out."
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return evidence, errors