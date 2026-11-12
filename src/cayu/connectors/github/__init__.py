"""GitHub from code, through an application's ``github`` connection slot.

``GitHubClient(client=None, slot="github")`` calls GitHub's REST and GraphQL APIs: typed errors
with GitHub's own words, rate limits waited out, ``list()`` and ``paginate()`` over next-page
links, GraphQL answers whose errors the caller may accept (``forbidden_under``,
``errors_under``), and a read-only mode for dry runs that refuses every write before it leaves
and lists it in ``refused``. On Cayu Cloud the broker attaches the credential; off it,
``connections_from_environment(permissions=...)`` reads ``GITHUB_TOKEN`` or a GitHub App's id
and private key (``token_connections``, ``app_connections``). ``read_pull`` and ``list_pulls``
answer a pull request's facts as a ``PullRequestFacts`` record: mergeability, CI checks with
their workflow run attempts and one ``CheckState``, the reviewer App's check apart, reviews and
comments. ``comment_once`` and ``file_issue_once`` write once however often a task restarts:
the body ends with a hidden ``Marker``, looked up first among the caller's own login's writes
(``trusted_markers`` reads them back). ``tools(client=None, slot="github")`` answers GitHub to a
model as ``github_read_pull``, ``github_list_pulls``, ``github_read_issue`` and
``github_read_job_log``; with ``include_writes=True`` also ``github_comment``,
``github_rerun_failed_jobs``, ``github_update_branch`` and ``github_merge_pull``, which
``approval_policy(inner)`` sends for approval and ``review_fields()`` shows by repository,
number and head commit. See ``docs/connector-toolkits.md``.
"""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu._api import wildcard_names as _wildcard_names
from cayu.connectors.github._exports import EXPORTS as _EXPORTS
from cayu.connectors.github._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _wildcard_names(_PUBLIC_NAMES, _EXPORTS)


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
