"""Find a model's official pricing page via DuckDuckGo HTML (no API key)."""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable

DDG = "https://duckduckgo.com/html/"
_RESULT = re.compile(r'result__a"[^>]*href="([^"]+)"')


def _default_fetch(query: str) -> str:
    import httpx

    return httpx.get(
        DDG,
        params={"q": query},
        timeout=20,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    ).text


def parse_results(html: str) -> list[str]:
    urls: list[str] = []
    for href in _RESULT.findall(html):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if "uddg" in params:  # DDG wraps the real URL in ?uddg=
            urls.append(params["uddg"][0])
        elif href.startswith("http"):
            urls.append(href)
    return urls


def search_web(
    query: str, *, limit: int = 5, fetch: Callable[[str], str] | None = None
) -> list[str]:
    return parse_results((fetch or _default_fetch)(query))[:limit]
