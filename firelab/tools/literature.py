"""Literature search through the OpenAlex API (free, no key)."""

from __future__ import annotations

import requests

from firelab import config
from firelab.tools._common import ToolError, get_json, tool

OPENALEX = "https://api.openalex.org"


def _abstract(inverted: dict | None, max_chars: int = 1200) -> str:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
    if not inverted:
        return ""
    positions = sorted((pos, word) for word, poss in inverted.items() for pos in poss)
    text = " ".join(word for _, word in positions)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


@tool
def openalex_search(query: str, max_results: int = 10, from_year: int | None = None) -> dict:
    """Search works; returns source ids ready to cite in EvidenceCards."""
    if not query or not query.strip():
        raise ToolError("query must not be empty")
    params = {
        "search": query,
        "per-page": max(1, min(int(max_results), 25)),
        "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index",
    }
    if from_year:
        params["filter"] = f"from_publication_date:{int(from_year)}-01-01"
    if config.OPENALEX_MAILTO:
        params["mailto"] = config.OPENALEX_MAILTO
    data = get_json(f"{OPENALEX}/works", params)
    results = []
    for w in data.get("results", []):
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        results.append({
            "source_id": "openalex:" + w["id"].rsplit("/", 1)[-1],
            "doi": f"doi:{doi}" if doi else None,
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count"),
            "abstract": _abstract(w.get("abstract_inverted_index")),
        })
    return {"query": query, "n_results": len(results), "results": results}


def resolve_source(source_id: str) -> bool | None:
    """True if OpenAlex knows the work, False if it does not, None if OpenAlex is unreachable."""
    if source_id.startswith("openalex:"):
        url = f"{OPENALEX}/works/{source_id.split(':', 1)[1]}"
    elif source_id.startswith("doi:"):
        url = f"{OPENALEX}/works/https://doi.org/{source_id.split(':', 1)[1]}"
    else:
        return False
    try:
        get_json(url, {"select": "id"})
        return True
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return False
        return None
    except requests.RequestException:
        return None
