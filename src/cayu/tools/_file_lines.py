"""Find where a line range sits in a workspace file, for line-range reads."""

from __future__ import annotations

from cayu.workspaces.base import Workspace, WorkspaceLineSpan, WorkspaceReadOffsetError

# Without a line locator, lines are found by paging through ``read_bytes``.
# Lines past the range are counted only in files this small, so a read near
# the start of a large file doesn't page through all of it.
FALLBACK_LINE_COUNT_LIMIT_BYTES = 4 * 1024 * 1024
_FALLBACK_PAGE_BYTES = 1024 * 1024


async def locate_workspace_lines(
    workspace: Workspace,
    path: str,
    *,
    start_line: int,
    end_line: int | None,
) -> WorkspaceLineSpan:
    """Use the workspace's line locator, or page through the file in bounded reads."""

    locator = workspace.line_locator()
    if locator is not None:
        span = await locator.locate_lines(path, start_line=start_line, end_line=end_line)
        if type(span) is not WorkspaceLineSpan:
            raise TypeError(
                f"{type(workspace).__name__} line locator must return a WorkspaceLineSpan."
            )
        return span
    return await _locate_lines_by_paging(workspace, path, start_line=start_line, end_line=end_line)


async def _locate_lines_by_paging(
    workspace: Workspace,
    path: str,
    *,
    start_line: int,
    end_line: int | None,
) -> WorkspaceLineSpan:
    scan = _LineScan(start_line, end_line)
    page_bytes = workspace.bounded_read_limit(_FALLBACK_PAGE_BYTES)
    offset = 0
    while True:
        try:
            result = await workspace.read_bytes(path, offset=offset, max_bytes=page_bytes)
        except WorkspaceReadOffsetError:
            # The file shrank under the scan; what was seen is all there is.
            return scan.span(total_bytes=offset, counted_all=True)
        scan.feed(result.content)
        offset += len(result.content)
        if not result.content or offset >= result.total_bytes:
            return scan.span(total_bytes=offset, counted_all=True)
        if scan.located and result.total_bytes > FALLBACK_LINE_COUNT_LIMIT_BYTES:
            return scan.span(total_bytes=result.total_bytes, counted_all=False)


class _LineScan:
    """Track newline positions across consecutive pages of one file.

    Line N starts after the (N-1)th newline and ends after the Nth; text after
    the last newline is one more line. This matches the guest guard's scan.
    """

    def __init__(self, start_line: int, end_line: int | None) -> None:
        self._start_line = start_line
        self._end_line = end_line
        self._newlines = 0
        self._offset = 0
        self._start_offset = 0 if start_line == 1 else None
        self._end_offset: int | None = None
        self._ends_with_newline = True

    @property
    def located(self) -> bool:
        return self._start_offset is not None and (
            self._end_line is None or self._end_offset is not None
        )

    def feed(self, chunk: bytes) -> None:
        pos = 0
        while not self.located:
            wanted = self._start_line - 1 if self._start_offset is None else self._end_line
            assert wanted is not None
            needed = wanted - self._newlines
            if chunk.count(b"\n", pos) < needed:
                break
            for _ in range(needed):
                pos = chunk.index(b"\n", pos) + 1
            self._newlines = wanted
            if self._start_offset is None:
                self._start_offset = self._offset + pos
            else:
                self._end_offset = self._offset + pos
        self._newlines += chunk.count(b"\n", pos)
        self._offset += len(chunk)
        if chunk:
            self._ends_with_newline = chunk.endswith(b"\n")

    def span(self, *, total_bytes: int, counted_all: bool) -> WorkspaceLineSpan:
        if counted_all:
            total_lines = self._newlines if self._ends_with_newline else self._newlines + 1
        else:
            total_lines = None
        start_offset = self._start_offset if self._start_offset is not None else total_bytes
        end_offset = self._end_offset if self._end_offset is not None else total_bytes
        return WorkspaceLineSpan(
            start_offset=start_offset,
            end_offset=end_offset,
            total_bytes=total_bytes,
            total_lines=total_lines,
        )
