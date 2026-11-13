"""Compare two prototype builds of the Developer dashboard pixel by pixel.

Build the prototype (``CAYU_PROTOTYPE=1 npx vite build`` in ``dashboard/``, which
writes ``dist/prototype``) once from the base commit and once from the change, then
run::

    uv run python scripts/dashboard_pixel_diff.py BASE_DIST HEAD_DIST [--out DIR]
        [--dashboard-config JSON]

``--dashboard-config`` is the runtime config the server injects into the dashboard
(``window.__CAYU_DASHBOARD_CONFIG__``). With a ``humanReviewPurpose`` in it, approvals
and questions show protected review, and the report adds the Inbox before and after
**Refresh review**, which the prototype's fake API can't answer.

The prototype answers from a fake API on its seed's pinned clock, so the same page
renders the same pixels on both builds unless the change altered it. Each page is
shot in light and dark; differing pages get a diff image in ``--out``.

Chromium occasionally antialiases a few edges one level differently between two
loads of the same build, so a page that differs is shot again, up to ``ATTEMPTS``
times, and the report says how many shots it took. The report also compares each
page's rendered DOM, which has no such noise.

This is a one-off check for refactors that must not change the dashboard, not a
standing test: later changes alter pages on purpose.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import Page, sync_playwright

# Tall enough that every card shows: the dashboard scrolls inside its own pane.
VIEWPORT = {"width": 1280, "height": 2400}
ATTEMPTS = 3
THEME_KEY = "cayu-dashboard-theme"


@dataclass(frozen=True)
class Shot:
    name: str
    seed: str
    path: str
    # Click every button with this name before the shot.
    click: str | None = None


# The month-end seed has an approval, a question and a recovery waiting; the new-app
# seed has nothing waiting. The session is found from the Inbox's first card.
SHOTS = (
    Shot("inbox-waiting", "month-end", "/inbox"),
    Shot("inbox-approvals", "month-end", "/inbox?kind=approval"),
    Shot("inbox-empty", "new-app", "/inbox"),
    Shot("knowledge", "month-end", "/knowledge"),
    Shot("home-waiting", "month-end", "/"),
)
# With a review purpose, approvals and questions show protected review instead.
REVIEW_SHOTS = (
    Shot("review-inbox", "month-end", "/inbox"),
    Shot("review-refreshed", "month-end", "/inbox", click="Refresh review"),
)


_MERGE_CONFIG = """(() => {
  const extra = EXTRA_JSON
  let value = { ...extra }
  Object.defineProperty(window, "__CAYU_DASHBOARD_CONFIG__", {
    configurable: true,
    get: () => value,
    set: (next) => {
      value = { ...next, ...extra }
    },
  })
})()"""


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextlib.contextmanager
def _serve(directory: Path) -> Iterator[str]:
    handler = functools.partial(_Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _settle(page: Page) -> None:
    """Wait until the fake API's delayed answers have rendered: the text stops changing."""
    page.evaluate("window.__settle = undefined")
    page.wait_for_function(
        """() => {
          const text = document.body.innerText
          const now = performance.now()
          if (window.__settle?.text !== text) window.__settle = { text, since: now }
          return now - window.__settle.since > 600
        }""",
        polling=100,
        timeout=15_000,
    )


def _open(page: Page, base: str, seed: str, path: str) -> None:
    page.goto(f"{base}/prototype.html#/")
    _settle(page)
    page.select_option("#prototype-app", seed)
    _settle(page)
    page.evaluate("path => { location.hash = '#' + path }", path)
    _settle(page)
    page.evaluate("document.fonts.ready.then(() => undefined)")
    page.mouse.move(0, 0)


def _session_with_card(page: Page, base: str) -> str:
    _open(page, base, "month-end", "/inbox?kind=approval")
    href = page.locator("[data-inbox-item] a[href*='#/sessions/']").first.get_attribute("href")
    if not href:
        raise SystemExit("The month-end seed's Inbox links to no session.")
    return href[href.index("#") + 1 :]


def _shoot(page: Page, base: str, shot: Shot, out: Path) -> tuple[Path, str]:
    """Shoot one page; return the image and the app's rendered DOM."""
    _open(page, base, shot.seed, shot.path)
    if shot.click:
        buttons = page.get_by_role("button", name=shot.click, exact=True)
        if buttons.count() == 0:
            raise SystemExit(f"{shot.name}: no {shot.click!r} button to click.")
        for index in range(buttons.count()):
            buttons.nth(index).click()
        _settle(page)
        page.mouse.move(0, 0)
    path = out / f"{shot.name}.png"
    page.screenshot(path=str(path), full_page=True, animations="disabled", caret="hide")
    return path, page.evaluate("document.getElementById('root').outerHTML")


def _compare(base: Path, head: Path, diff: Path) -> tuple[int, int]:
    """Return (differing pixels, total pixels), writing a diff image when they differ."""
    diff.unlink(missing_ok=True)
    with Image.open(base) as a, Image.open(head) as b:
        width, height = max(a.width, b.width), max(a.height, b.height)
        total = width * height
        if a.size != b.size:
            return total, total
        delta = ImageChops.difference(a.convert("RGBA"), b.convert("RGBA"))
        changed = sum(1 for pixel in delta.get_flattened_data() if any(pixel))
        if changed:
            delta.save(diff)
        return changed, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="The base build's dist/prototype.")
    parser.add_argument("head", type=Path, help="The change's dist/prototype.")
    parser.add_argument("--out", type=Path, default=Path("dist/pixel-diff"))
    parser.add_argument(
        "--dashboard-config",
        type=json.loads,
        default=None,
        help='The injected runtime config, such as \'{"humanReviewPurpose": "review"}\'.',
    )
    args = parser.parse_args(argv)
    for directory in (args.base, args.head):
        if not (directory / "prototype.html").is_file():
            parser.error(f"{directory} has no prototype.html; build the prototype first.")

    results: list[tuple[str, int, int, int, bool]] = []
    with (
        _serve(args.base) as base_url,
        _serve(args.head) as head_url,
        sync_playwright() as playwright,
    ):
        browser = playwright.chromium.launch()
        try:
            for theme in ("light", "dark"):
                context = browser.new_context(
                    viewport=VIEWPORT,
                    device_scale_factor=1,
                    timezone_id="America/Los_Angeles",
                    locale="en-US",
                    color_scheme=theme,
                )
                context.add_init_script(f"localStorage.setItem({THEME_KEY!r}, {theme!r})")
                if args.dashboard_config is not None:
                    # The prototype assigns its own config at startup; merge ours into it.
                    context.add_init_script(
                        _MERGE_CONFIG.replace("EXTRA_JSON", json.dumps(args.dashboard_config))
                    )
                page = context.new_page()
                shots = (
                    *SHOTS,
                    Shot("session-card", "month-end", _session_with_card(page, base_url)),
                    *(
                        REVIEW_SHOTS
                        if (args.dashboard_config or {}).get("humanReviewPurpose")
                        else ()
                    ),
                )
                for shot in shots:
                    name = f"{shot.name}-{theme}"
                    attempts = 0
                    while True:
                        attempts += 1
                        shot_of = {}
                        for side, url in (("base", base_url), ("head", head_url)):
                            folder = args.out / side / theme
                            folder.mkdir(parents=True, exist_ok=True)
                            shot_of[side] = _shoot(page, url, shot, folder)
                        changed, total = _compare(
                            shot_of["base"][0], shot_of["head"][0], args.out / f"{name}-diff.png"
                        )
                        if not changed or attempts == ATTEMPTS:
                            break
                    same_dom = shot_of["base"][1] == shot_of["head"][1]
                    results.append((name, changed, total, attempts, same_dom))
                context.close()
        finally:
            browser.close()

    width = max(len(name) for name, *_ in results)
    for name, changed, total, attempts, same_dom in results:
        dom = "same DOM" if same_dom else "DOM DIFFERS"
        shots = "" if attempts == 1 else f", {attempts} shots"
        print(f"{name:<{width}}  {changed:>9,} of {total:>11,} pixels differ ({dom}{shots})")
    changed = sum(result[1] for result in results)
    total = sum(result[2] for result in results)
    print(f"{'total':<{width}}  {changed:>9,} of {total:>11,} pixels differ")
    return 1 if changed or not all(result[4] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
