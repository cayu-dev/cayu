"""Cayu tools called by the repository's catalog-verifier agent.

Browser tools execute ``agent-browser`` through the ToolContext runner when one is configured;
otherwise, including in the default scheduled workflow, Chromium runs on the host. ``search_web``
is a plain HTTPS request and always stays on the host.
"""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from cayu.artifacts.attachments import file_attachment
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.runners.base import ExecCommand
from maintenance.model_catalog import browser, search
from maintenance.model_catalog.security import (
    allowed_hosts,
    provider_from_context,
    validate_official_url,
)

_MAX_PAGE_CHARS = 22_000
_THIN_TEXT_CHARS = 200  # below this the page didn't load -> fall back to flattened text
_PAGE_COMMAND_TIMEOUT_SECONDS = 120
_SCREENSHOT_COMMAND_TIMEOUT_SECONDS = 180
_PRICING_MODES = ("standard", "batch", "flex", "priority")
_PRICING_MODE_OPTIONS = ("all", *_PRICING_MODES)

# Keep only a11y-snapshot lines that carry pricing-table signal — table structure (rows/cells/
# headers), tab/radio state (which pricing mode is selected), and pricing prose/numbers. This drops
# the nav-menu cruft (list/listitem/link entries) that otherwise dominates the snapshot and pushes
# the table past the size budget.
_KEEP_LINE = re.compile(
    r"\$|"
    r"\b(table|row|rowgroup|cell|columnheader|rowheader|gridcell|grid|radio|checkbox|tab|tablist|heading)\b|"
    r"context|token|price|cache|batch|tier|million|window|standard|flex|priority|input|output",
    re.I,
)


def _compress_snapshot(snap: str) -> str:
    """Strip nav cruft from an a11y snapshot, keeping table/pricing structure so it fits the budget."""
    kept = [ln for ln in snap.splitlines() if _KEEP_LINE.search(ln)]
    return "\n".join(kept)


@dataclass(frozen=True)
class _PricingControl:
    ref: str
    selected: bool


def _control_state(line: str) -> _PricingControl | None:
    checked = re.search(r"\bchecked=(true|false)\b", line)
    ref = re.search(r"\bref=([a-z0-9_-]+)\b", line)
    if checked is None or ref is None:
        return None
    return _PricingControl(ref=ref.group(1), selected=checked.group(1) == "true")


def _pricing_controls(snapshot: str, mode: str) -> list[_PricingControl]:
    label = mode.title()
    lines = snapshot.splitlines()
    controls: list[_PricingControl] = []
    for line in lines:
        if f'radio "{label}"' in line:
            control = _control_state(line)
            if control is not None:
                controls.append(control)
    if controls:
        return controls
    for index, line in enumerate(lines):
        if '"Batch API price"' not in line:
            continue
        for candidate in lines[index : index + 4]:
            if "switch" not in candidate:
                continue
            control = _control_state(candidate)
            if control is None:
                continue
            if mode == "batch":
                controls.append(control)
            if mode == "standard":
                controls.append(_PricingControl(ref=control.ref, selected=not control.selected))
    return list({control.ref: control for control in controls}.values())


def _has_interactive_pricing_modes(snapshot: str) -> bool:
    return any(f'radio "{mode.title()}"' in snapshot for mode in _PRICING_MODES) or (
        '"Batch API price"' in snapshot and "switch" in snapshot
    )


def _has_explicit_static_mode(snapshot: str, mode: str) -> bool:
    mode_pattern = re.escape(mode)
    return any(
        re.search(
            rf"\b(heading|columnheader|rowheader|table)\b.*\b{mode_pattern}\b"
            rf"|\b{mode_pattern}\b.*\b(pricing|prices)\b",
            line,
            re.IGNORECASE,
        )
        is not None
        for line in snapshot.splitlines()
    )


async def _exec(
    ctx: ToolContext,
    command: str,
    *,
    timeout: int = _PAGE_COMMAND_TIMEOUT_SECONDS,
) -> str:
    runner = getattr(ctx, "runner", None)
    if runner is not None:
        result = await runner.exec(ExecCommand.bash(command), timeout_s=timeout)
        return result.stdout
    return await asyncio.to_thread(browser.run_bash_host, command, timeout=timeout)


async def _prepare_pricing_page(
    ctx: ToolContext,
    url: str,
    mode: str,
) -> tuple[str, bool, str, str]:
    provider_name = provider_from_context(ctx)
    hosts = allowed_hosts(provider_name)
    requested_url = validate_official_url(url, provider_name=provider_name)
    await _exec(
        ctx,
        browser.open_page_command(requested_url, session=ctx.session_id, allowed_hosts=hosts),
    )
    effective_url = validate_official_url(
        (
            await _exec(
                ctx,
                browser.current_url_command(session=ctx.session_id, allowed_hosts=hosts),
            )
        ).strip(),
        provider_name=provider_name,
    )
    snapshot = await _exec(
        ctx,
        browser.snapshot_command(session=ctx.session_id, allowed_hosts=hosts),
    )
    controls = _pricing_controls(snapshot, mode)
    if not controls:
        if mode == "standard":
            if _has_interactive_pricing_modes(snapshot):
                return (
                    snapshot,
                    False,
                    "page pricing controls do not expose Standard state",
                    effective_url,
                )
            return snapshot, True, "default page state", effective_url
        if not _has_interactive_pricing_modes(snapshot) and _has_explicit_static_mode(
            snapshot, mode
        ):
            return snapshot, True, f"static {mode} pricing section", effective_url
        return (
            snapshot,
            False,
            f"page exposes no selectable or explicit {mode} pricing",
            effective_url,
        )
    if all(control.selected for control in controls):
        return snapshot, True, f"{mode} pricing controls already selected", effective_url

    # Some official pages contain several independently switched pricing tables. Select every
    # table that exposes the requested mode so a row lower on the page cannot remain in Standard
    # while an unrelated table above it proves Batch. Re-snapshot after each click because browser
    # references can change when the page re-renders.
    for _ in range(20):
        control = next((item for item in controls if not item.selected), None)
        if control is None:
            return snapshot, True, f"{mode} pricing controls selected", effective_url
        await _exec(
            ctx,
            browser.click_command(control.ref, session=ctx.session_id, allowed_hosts=hosts),
        )
        effective_url = validate_official_url(
            (
                await _exec(
                    ctx,
                    browser.current_url_command(session=ctx.session_id, allowed_hosts=hosts),
                )
            ).strip(),
            provider_name=provider_name,
        )
        snapshot = await _exec(
            ctx,
            browser.snapshot_command(session=ctx.session_id, allowed_hosts=hosts),
        )
        controls = _pricing_controls(snapshot, mode)
        if not controls:
            break
    if controls and all(control.selected for control in controls):
        return snapshot, True, f"{mode} pricing controls selected", effective_url
    return (
        snapshot,
        False,
        f"could not confirm that all {mode} pricing controls became selected",
        effective_url,
    )


def _requested_pricing_mode(args: dict[str, Any], *, default: str = "all") -> str:
    mode = args.get("pricing_mode", default)
    if mode not in _PRICING_MODE_OPTIONS:
        raise ValueError(f"pricing_mode must be one of {', '.join(_PRICING_MODE_OPTIONS)}")
    return mode


async def _read_bytes(ctx: ToolContext, path: str) -> bytes:
    """Pull a file's bytes off the sandbox (base64 over exec stdout) or the host (direct read)."""
    runner = getattr(ctx, "runner", None)
    if runner is not None:
        b64 = (await _exec(ctx, browser.read_file_b64_command(path), timeout=60)).strip()
        return base64.b64decode(b64) if b64 else b""
    return await asyncio.to_thread(Path(path).read_bytes)


class SearchWebTool(Tool):
    spec = ToolSpec(
        name="search_web",
        description="Search the web; returns the top result URLs for a query.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        provider_name = provider_from_context(ctx)
        candidates = await asyncio.to_thread(search.search_web, args["query"])
        urls: list[str] = []
        for candidate in candidates:
            try:
                urls.append(validate_official_url(candidate, provider_name=provider_name))
            except ValueError:
                continue
        return ToolResult(content="\n".join(urls), structured={"urls": urls})


class ReadPageTool(Tool):
    spec = ToolSpec(
        name="read_page",
        description=(
            "Open a URL in a real browser and return its ACCESSIBILITY TREE: "
            "table rows/cells with their columns intact. Keep pricing_mode=all (the default) to "
            "capture Standard and, when offered, Batch in separately labeled sections of one "
            "trusted result. Explicit single modes are available for retries. The tool selects "
            "and verifies every relevant tab/switch before returning values. Never derive "
            "Batch/Flex/Priority values from a Standard section. If it comes back empty "
            "(JS-blocked), use screenshot with an explicit pricing_mode."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "pricing_mode": {
                    "type": "string",
                    "enum": list(_PRICING_MODE_OPTIONS),
                    "default": "all",
                },
            },
            "required": ["url"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        url = args["url"]
        mode = _requested_pricing_mode(args)
        if mode != "all":
            return await _read_page_mode(ctx, url, mode)

        standard = await _read_page_mode(ctx, url, "standard")
        if standard.is_error:
            return standard
        results = [standard]
        batch = await _read_page_mode(ctx, url, "batch")
        if not batch.is_error:
            results.append(batch)
        modes = [
            item.structured["pricing_mode"]
            for item in results
            if item.structured is not None and item.structured.get("pricing_mode_verified") is True
        ]
        effective_url = (
            standard.structured.get("effective_url") if standard.structured is not None else None
        )
        return ToolResult(
            content="\n\n".join(item.content for item in results),
            structured={
                "source": "multi_mode",
                "pricing_mode": "all",
                "pricing_mode_verified": True,
                "pricing_modes_verified": modes,
                "selection_note": "captured " + ", ".join(modes),
                "effective_url": effective_url,
            },
        )


async def _read_page_mode(ctx: ToolContext, url: str, mode: str) -> ToolResult:
    """Read one pricing mode. ``ReadPageTool`` composes this for its default all-mode read."""

    if mode not in _PRICING_MODES:
        raise ValueError(f"single pricing mode must be one of {', '.join(_PRICING_MODES)}")
    snap, mode_verified, selection_note, effective_url = await _prepare_pricing_page(ctx, url, mode)
    result_details = {
        "pricing_mode": mode,
        "pricing_mode_verified": mode_verified,
        "selection_note": selection_note,
        "effective_url": effective_url,
    }
    if not mode_verified:
        return ToolResult(
            content=f"{url}: {selection_note}. Do not report {mode} prices.",
            structured={"source": "selection_error", **result_details},
            is_error=True,
        )
    mode_header = f"[verified pricing_mode={mode}; {selection_note}]\n"
    if len(snap.strip()) >= _THIN_TEXT_CHARS:
        compressed = _compress_snapshot(snap).strip()
        if compressed:
            return ToolResult(
                content=mode_header + compressed[:_MAX_PAGE_CHARS],
                structured={"source": "a11y", **result_details},
            )
    # snapshot empty (page failed to render structure) -> flattened text on the same open page
    text = await _exec(
        ctx,
        browser.text_body_command(
            session=ctx.session_id,
            allowed_hosts=allowed_hosts(provider_from_context(ctx)),
        ),
    )
    if text.strip():
        return ToolResult(
            content=mode_header + text[:_MAX_PAGE_CHARS],
            structured={"source": "text", **result_details},
        )
    return ToolResult(
        content=f"{url} returned no readable structure or text — try the screenshot tool.",
        structured={"source": "empty", **result_details},
        is_error=True,
    )


class ScreenshotTool(Tool):
    spec = ToolSpec(
        name="screenshot",
        description=(
            "Open a URL in a real browser, select and verify the requested pricing_mode, and "
            "capture a FULL-PAGE screenshot. Use this when read_page returns empty or unreadable "
            "text. Never report mode-specific prices from a screenshot of another mode."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "pricing_mode": {
                    "type": "string",
                    "enum": list(_PRICING_MODES),
                    "default": "standard",
                },
            },
            "required": ["url"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        store = getattr(ctx, "artifact_store", None)
        if store is None:
            return ToolResult(
                content="screenshot unavailable: no artifact store configured", is_error=True
            )
        url = args["url"]
        mode = _requested_pricing_mode(args, default="standard")
        _, mode_verified, selection_note, effective_url = await _prepare_pricing_page(
            ctx, url, mode
        )
        if not mode_verified:
            return ToolResult(
                content=f"{url}: {selection_note}. Do not report {mode} prices.",
                structured={
                    "url": url,
                    "effective_url": effective_url,
                    "pricing_mode": mode,
                    "pricing_mode_verified": False,
                    "selection_note": selection_note,
                },
                is_error=True,
            )
        path = f"/tmp/shot-{uuid4().hex}.png"
        await _exec(
            ctx,
            browser.screenshot_current_page_command(
                path,
                session=ctx.session_id,
                allowed_hosts=allowed_hosts(provider_from_context(ctx)),
            ),
            timeout=_SCREENSHOT_COMMAND_TIMEOUT_SECONDS,
        )
        data = await _read_bytes(ctx, path)
        if not data:
            return ToolResult(
                content="screenshot produced no bytes (page may have failed to load)", is_error=True
            )
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            # base64-over-exec can truncate a large image -> a corrupt PNG the model API rejects.
            return ToolResult(
                content="screenshot bytes were not a valid PNG; try read_page instead",
                is_error=True,
            )
        meta = await store.put_bytes(
            data, filename="page.png", content_type="image/png", session_id=ctx.session_id
        )
        return ToolResult(
            content=(
                f"Captured a full-page screenshot of {url} with verified pricing_mode={mode} "
                f"({meta.size_bytes} bytes). Read only the {mode} prices from the attached image."
            ),
            artifacts=[
                file_attachment(
                    artifact_id=meta.id,
                    kind="image",
                    filename="page.png",
                    content_type="image/png",
                    size_bytes=meta.size_bytes,
                )
            ],
            structured={
                "url": url,
                "effective_url": effective_url,
                "size_bytes": meta.size_bytes,
                "pricing_mode": mode,
                "pricing_mode_verified": True,
                "selection_note": selection_note,
            },
        )
