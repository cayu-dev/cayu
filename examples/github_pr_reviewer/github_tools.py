from __future__ import annotations

import httpx

from cayu import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.vaults import SecretRef

GITHUB_API = "https://api.github.com"
_GITHUB_PAGE_SIZE = 100
_MAX_DIFF_FILES = 200
_MAX_SUMMARY_FILES = 80
_MAX_TOTAL_PATCH_PREVIEW_CHARS = 24_000
_MAX_PATCH_PREVIEW_CHARS_PER_FILE = 1_200


def _pr_diff_result(
    *,
    pr_number: int,
    title: str,
    body: str | None,
    head_ref: str,
    head_sha: str,
    base_ref: str,
    files: list[dict],
    total_files: int | None = None,
    files_truncated: bool = False,
    patches_truncated: bool = False,
    patch_preview_chars: int = 0,
) -> ToolResult:
    total_label = total_files if total_files is not None else len(files)
    summary = [
        f"PR #{pr_number} {title!r} ({head_ref} -> {base_ref})",
        f"Showing {len(files)} of {total_label} changed files.",
        f"Patch previews use {patch_preview_chars} chars.",
    ]
    if files_truncated or patches_truncated:
        summary.append(
            "Output truncated; use list_files/read_file in the checked-out workspace for exact content."
        )
    summary.extend(
        [
            "",
            *[
                f"- {c['path']} (+{c['additions']}/-{c['deletions']})"
                for c in files[:_MAX_SUMMARY_FILES]
            ],
        ]
    )
    if len(files) > _MAX_SUMMARY_FILES:
        summary.append(f"- ... {len(files) - _MAX_SUMMARY_FILES} more files omitted from summary")
    return ToolResult(
        content="\n".join(summary),
        structured={
            "title": title,
            "body": body,
            "head_ref": head_ref,
            "head_sha": head_sha,
            "base_ref": base_ref,
            "total_files": total_files,
            "files_returned": len(files),
            "files_truncated": files_truncated,
            "patches_truncated": patches_truncated,
            "patch_preview_chars": patch_preview_chars,
            "files": files,
        },
    )


class GetPRDiffTool(Tool):
    """Fetch PR metadata plus bounded changed-file metadata and patch previews."""

    spec = ToolSpec(
        name="get_pr_diff",
        description=(
            "Fetch a GitHub pull request's title, body, base/head refs, and the list "
            "of changed files with bounded patch previews and truncation flags. Use "
            "read_file on the checked-out workspace for exact file contents. Uses "
            "the PR identity from the session's environment metadata when the caller "
            "omits it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repo owner/org."},
                "repo": {"type": "string", "description": "Repo name."},
                "pr_number": {"type": "integer", "description": "Pull request number."},
            },
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        owner = args.get("owner") or ctx.metadata.get("repo_owner")
        repo = args.get("repo") or ctx.metadata.get("repo_name")
        pr_number = args.get("pr_number") or ctx.metadata.get("pr_number")
        if not (owner and repo and pr_number):
            return ToolResult(
                content="Missing owner/repo/pr_number (not in tool args or session metadata).",
                is_error=True,
            )

        headers = {"Accept": "application/vnd.github+json"}
        token = await _resolve_github_token(ctx, tool="get_pr_diff")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        base = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            pr_resp = await client.get(base, headers=headers)
            if pr_resp.status_code != 200:
                return ToolResult(
                    content=f"GitHub API error {pr_resp.status_code} fetching PR: "
                    f"{pr_resp.text[:500]}",
                    is_error=True,
                )
            pr = pr_resp.json()
            current_head_sha = pr["head"]["sha"]
            expected_head_sha = _expected_head_sha_for_request(
                ctx,
                owner=owner,
                repo=repo,
                pr_number=pr_number,
            )
            should_validate_head = type(expected_head_sha) is str and bool(expected_head_sha)
            stale_head_result = _stale_head_result(
                expected_head_sha=expected_head_sha,
                current_head_sha=current_head_sha,
            )
            if stale_head_result is not None:
                return stale_head_result
            try:
                files, files_truncated = await _fetch_pr_files(client, base=base, headers=headers)
            except RuntimeError as exc:
                return ToolResult(content=str(exc), is_error=True)
            if should_validate_head:
                pr_resp = await client.get(base, headers=headers)
                if pr_resp.status_code != 200:
                    return ToolResult(
                        content=f"GitHub API error {pr_resp.status_code} rechecking PR head: "
                        f"{pr_resp.text[:500]}",
                        is_error=True,
                    )
                current_head_sha = pr_resp.json()["head"]["sha"]
                stale_head_result = _stale_head_result(
                    expected_head_sha=expected_head_sha,
                    current_head_sha=current_head_sha,
                )
                if stale_head_result is not None:
                    return stale_head_result

        total_files = pr.get("changed_files")
        if type(total_files) is not int:
            total_files = None
        if total_files is not None:
            files_truncated = total_files > len(files)
        changed, patches_truncated, patch_preview_chars = _changed_files_with_patch_budget(files)
        return _pr_diff_result(
            pr_number=pr_number,
            title=pr["title"],
            body=pr.get("body"),
            head_ref=pr["head"]["ref"],
            head_sha=current_head_sha,
            base_ref=pr["base"]["ref"],
            files=changed,
            total_files=total_files,
            files_truncated=files_truncated,
            patches_truncated=patches_truncated,
            patch_preview_chars=patch_preview_chars,
        )


def _expected_head_sha_for_request(
    ctx: ToolContext, *, owner: object, repo: object, pr_number: object
) -> object:
    if (
        owner == ctx.metadata.get("repo_owner")
        and repo == ctx.metadata.get("repo_name")
        and pr_number == ctx.metadata.get("pr_number")
    ):
        return ctx.metadata.get("head_sha")
    return None


def _stale_head_result(*, expected_head_sha: object, current_head_sha: str) -> ToolResult | None:
    if (
        type(expected_head_sha) is str
        and expected_head_sha
        and current_head_sha != expected_head_sha
    ):
        return ToolResult(
            content=(
                "PR head changed before review: expected "
                f"{expected_head_sha}, but GitHub now reports {current_head_sha}. "
                "Enqueue a new review for the current head."
            ),
            structured={
                "expected_head_sha": expected_head_sha,
                "current_head_sha": current_head_sha,
            },
            is_error=True,
        )
    return None


async def _fetch_pr_files(
    client: httpx.AsyncClient, *, base: str, headers: dict[str, str]
) -> tuple[list[dict], bool]:
    files: list[dict] = []
    page = 1
    truncated = False
    while len(files) < _MAX_DIFF_FILES:
        files_resp = await client.get(
            f"{base}/files",
            headers=headers,
            params={"per_page": _GITHUB_PAGE_SIZE, "page": page},
        )
        if files_resp.status_code != 200:
            raise RuntimeError(
                f"GitHub API error {files_resp.status_code} fetching files: {files_resp.text[:500]}"
            )
        batch = files_resp.json()
        if type(batch) is not list:
            raise RuntimeError("GitHub API returned a non-list response for PR files.")
        remaining = _MAX_DIFF_FILES - len(files)
        if len(batch) > remaining:
            files.extend(batch[:remaining])
            truncated = True
            break
        files.extend(batch)
        if len(batch) < _GITHUB_PAGE_SIZE:
            break
        page += 1
    if len(files) >= _MAX_DIFF_FILES:
        truncated = True
    return files, truncated


def _changed_files_with_patch_budget(files: list[dict]) -> tuple[list[dict], bool, int]:
    changed: list[dict] = []
    remaining_patch_chars = _MAX_TOTAL_PATCH_PREVIEW_CHARS
    patches_truncated = False
    patch_preview_chars = 0
    for file in files:
        patch = file.get("patch")
        patch_text = patch if type(patch) is str else ""
        preview_limit = min(_MAX_PATCH_PREVIEW_CHARS_PER_FILE, max(remaining_patch_chars, 0))
        patch_preview = patch_text[:preview_limit]
        patch_preview_chars += len(patch_preview)
        remaining_patch_chars -= len(patch_preview)
        patch_truncated = len(patch_preview) < len(patch_text)
        if patch_truncated:
            patches_truncated = True
        changed.append(
            {
                "path": file["filename"],
                "status": file["status"],
                "additions": file["additions"],
                "deletions": file["deletions"],
                "patch_available": bool(patch_text),
                "patch_preview": patch_preview,
                "patch_preview_truncated": patch_truncated,
            }
        )
    return changed, patches_truncated, patch_preview_chars


class DemoPRDiffTool(Tool):
    """Deterministic stand-in for ``get_pr_diff`` used by the no-key demo."""

    spec = GetPRDiffTool.spec

    def __init__(self, *, pr_number: int, head_sha: str) -> None:
        self._pr_number = pr_number
        self._head_sha = head_sha

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        files = [
            {
                "filename": "review_target.py",
                "status": "added",
                "additions": 3,
                "deletions": 0,
                "patch": ("@@ -0,0 +1,3 @@\n+def answer() -> int:\n+    return 42\n+\n"),
            }
        ]
        changed, patches_truncated, patch_preview_chars = _changed_files_with_patch_budget(files)
        return _pr_diff_result(
            pr_number=self._pr_number,
            title="Add deterministic review target",
            body="Fixture PR used by the no-key demo.",
            head_ref="fixture-pr",
            head_sha=self._head_sha,
            base_ref="main",
            files=changed,
            total_files=len(files),
            patches_truncated=patches_truncated,
            patch_preview_chars=patch_preview_chars,
        )


class PostPRCommentTool(Tool):
    """Post one review comment on a GitHub pull request (issue-comment style)."""

    spec = ToolSpec(
        name="post_pr_comment",
        description="Post one review comment on the pull request being reviewed.",
        parallel_safe=False,
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pr_number": {"type": "integer"},
                "body": {"type": "string", "description": "Markdown comment body."},
            },
            "required": ["body"],
            "additionalProperties": False,
        },
        effect=ToolEffect.EXTERNAL,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        owner = args.get("owner") or ctx.metadata.get("repo_owner")
        repo = args.get("repo") or ctx.metadata.get("repo_name")
        pr_number = args.get("pr_number") or ctx.metadata.get("pr_number")
        marker = _review_comment_marker(ctx.session_id)
        body = _body_with_review_marker(args["body"], marker)
        if not (owner and repo and pr_number):
            return ToolResult(
                content="Missing owner/repo/pr_number (not in tool args or session metadata).",
                is_error=True,
            )
        if ctx.proxy is None:
            return ToolResult(
                content=(
                    "No credential proxy configured for this environment -- cannot post to "
                    "GitHub. Wire Environment(vault=..., proxy=...) with a github_token secret."
                ),
                is_error=True,
            )

        destination = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{pr_number}/comments"
        credential = SecretRef(name="github_token")
        authorization = await ctx.proxy.authorize_request(
            destination=destination,
            credential=credential,
            action="post_pr_comment",
            metadata={"owner": owner, "repo": repo, "pr_number": pr_number},
        )
        if not authorization.allowed:
            return ToolResult(
                content=f"Blocked by credential proxy: {authorization.reason}", is_error=True
            )

        resolved = await ctx.proxy.resolve(
            credential, scope={"session_id": ctx.session_id, "tool": "post_pr_comment"}
        )
        token = resolved.value.get_secret_value()
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            existing = None
            page = 1
            while True:
                listing = await client.get(
                    destination, headers=headers, params={"per_page": 100, "page": page}
                )
                if listing.status_code >= 300:
                    return ToolResult(
                        content=(
                            f"GitHub API error {listing.status_code} listing comments: "
                            f"{listing.text[:500]}"
                        ),
                        is_error=True,
                    )
                batch = listing.json()
                existing = _find_existing_review_comment(batch, marker)
                if existing is not None or len(batch) < 100:
                    break
                page += 1
            if existing is not None:
                edit_url = existing.get("url")
                if not edit_url:
                    return ToolResult(
                        content="GitHub comment is missing its edit URL.", is_error=True
                    )
                edit_authorization = await ctx.proxy.authorize_request(
                    destination=edit_url,
                    credential=credential,
                    action="update_pr_comment",
                    metadata={"owner": owner, "repo": repo, "pr_number": pr_number},
                )
                if not edit_authorization.allowed:
                    return ToolResult(
                        content=f"Blocked by credential proxy: {edit_authorization.reason}",
                        is_error=True,
                    )
                resp = await client.patch(edit_url, headers=headers, json={"body": body})
                operation = "updated"
            else:
                resp = await client.post(destination, headers=headers, json={"body": body})
                operation = "posted"
        if resp.status_code >= 300:
            return ToolResult(
                content=f"GitHub API error {resp.status_code}: {resp.text[:500]}", is_error=True
            )
        posted = resp.json()
        return ToolResult(
            content=f"{operation.title()} comment {posted.get('html_url', posted.get('id'))}",
            structured={
                "id": posted.get("id"),
                "html_url": posted.get("html_url"),
                "operation": operation,
            },
        )


def _review_comment_marker(session_id: str) -> str:
    return f"<!-- cayu-pr-reviewer:{session_id} -->"


def _body_with_review_marker(body: str, marker: str) -> str:
    if marker in body:
        return body
    return f"{body.rstrip()}\n\n{marker}"


def _find_existing_review_comment(comments: list[dict], marker: str) -> dict | None:
    for comment in comments:
        if isinstance(comment, dict) and marker in str(comment.get("body", "")):
            return comment
    return None


async def _resolve_github_token(ctx: ToolContext, *, tool: str) -> str | None:
    """Resolve the github_token secret via the proxy, or None for anonymous reads."""
    if ctx.proxy is None:
        return None
    try:
        resolved = await ctx.proxy.resolve(
            SecretRef(name="github_token"), scope={"session_id": ctx.session_id, "tool": tool}
        )
        return resolved.value.get_secret_value()
    except Exception:
        return None
