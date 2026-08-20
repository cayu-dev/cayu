"""Browse, extract, and verify canonical web evidence with failure isolation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cayu import AgentSpec, CayuApp, ToolContext, WebBridge


@dataclass(frozen=True, slots=True)
class VerifiedPage:
    rank: int
    source_url: str
    final_url: str
    title: str | None
    content: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class PageFailure:
    rank: int
    source_url: str
    error: str


@dataclass(frozen=True, slots=True)
class ResearchEvidence:
    query: str
    pages: tuple[VerifiedPage, ...]
    failures: tuple[PageFailure, ...]


def register_researcher(
    app: CayuApp,
    bridge: WebBridge,
    *,
    model: str,
    environment_name: str | None = None,
) -> AgentSpec:
    """Register the same prompt/tool contract for any explicit bridge profile."""

    return bridge.register_agent(
        app,
        AgentSpec(name="web_researcher", model=model),
        environment_name=environment_name,
    )


async def browse_extract_verify(
    bridge: WebBridge,
    ctx: ToolContext,
    query: str,
    *,
    max_pages: int = 5,
) -> ResearchEvidence:
    """Search once, fetch each canonical source, and isolate page failures.

    Tool results remain untrusted evidence. Verification here binds every fetch
    to the canonical URL selected by ``web_search``; it does not claim that page
    assertions are true merely because the page returned them.
    """

    tools = {tool.spec.name: tool for tool in bridge.tools}
    search = tools.get("web_search")
    fetch = tools.get("web_fetch")
    if search is None or fetch is None:
        raise ValueError("browse_extract_verify requires web_search and web_fetch.")
    if type(max_pages) is not int or not 1 <= max_pages <= 20:
        raise ValueError("max_pages must be between 1 and 20.")

    searched = await search.run(ctx, {"query": query, "num_results": max_pages})
    if searched.is_error:
        error = _bounded_error(searched.structured)
        return ResearchEvidence(
            query=query,
            pages=(),
            failures=(PageFailure(rank=0, source_url="", error=error),),
        )

    pages: list[VerifiedPage] = []
    failures: list[PageFailure] = []
    if not isinstance(searched.structured, Mapping):
        return ResearchEvidence(
            query=query,
            pages=(),
            failures=(PageFailure(rank=0, source_url="", error="malformed_search_result"),),
        )
    raw_results = searched.structured.get("results")
    if isinstance(raw_results, str | bytes) or not isinstance(raw_results, Sequence):
        return ResearchEvidence(
            query=query,
            pages=(),
            failures=(PageFailure(rank=0, source_url="", error="malformed_search_result"),),
        )
    for raw_source in raw_results[:max_pages]:  # untrusted provider evidence
        if not isinstance(raw_source, Mapping):
            failures.append(PageFailure(rank=0, source_url="", error="malformed_search_result"))
            continue
        rank = raw_source.get("rank")
        source_url = raw_source.get("url")
        if type(rank) is not int or type(source_url) is not str:
            failures.append(PageFailure(rank=0, source_url="", error="malformed_search_result"))
            continue
        fetched = await fetch.run(ctx, {"url": source_url})
        if fetched.is_error:
            failures.append(
                PageFailure(
                    rank=rank,
                    source_url=source_url,
                    error=_bounded_error(fetched.structured),
                )
            )
            continue
        if not isinstance(fetched.structured, Mapping):
            failures.append(
                PageFailure(rank=rank, source_url=source_url, error="malformed_fetch_result")
            )
            continue
        if fetched.structured.get("requested_url") != source_url:
            failures.append(
                PageFailure(
                    rank=rank,
                    source_url=source_url,
                    error="source_binding_mismatch",
                )
            )
            continue
        final_url = fetched.structured.get("final_url")
        content = fetched.structured.get("content")
        title = fetched.structured.get("title")
        truncated = fetched.structured.get("truncated")
        if (
            type(final_url) is not str
            or type(content) is not str
            or (title is not None and type(title) is not str)
            or type(truncated) is not bool
        ):
            failures.append(
                PageFailure(rank=rank, source_url=source_url, error="malformed_fetch_result")
            )
            continue
        pages.append(
            VerifiedPage(
                rank=rank,
                source_url=source_url,
                final_url=final_url,
                title=title,
                content=content,
                truncated=truncated,
            )
        )
    return ResearchEvidence(query=query, pages=tuple(pages), failures=tuple(failures))


def _bounded_error(structured: Mapping[str, Any] | None) -> str:
    if not isinstance(structured, Mapping):
        return "web_operation_failed"
    error = structured.get("error")
    if type(error) is not str or not error or len(error) > 64:
        return "web_operation_failed"
    return error
