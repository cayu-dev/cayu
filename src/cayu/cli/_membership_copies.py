"""Notice a project still carrying a copy of the membership code `cayu.memberships` replaced.

Organizations projects generated before `cayu.memberships` existed carry the access policy
(`policies/access.py`) and the membership SQL (`database/memberships.py`) as their own copies.
A copy that is byte-for-byte one Cayu generated can move to the library in one commit; one
the project edited is its own and is left alone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic

# The SHA-256 of each file as Cayu releases generated it, and which releases did. Every
# release from 0.16.1 to 0.20.0 generated the same two files.
GENERATED_COPIES: dict[str, dict[str, str]] = {
    "policies/access.py": {
        "40c51983cd0ba925de7769d3c3a60f4362cbffd4839449ab013cf6489ffcbcb0": "0.16.1 to 0.20.0",
    },
    "database/memberships.py": {
        "a4e39fd2440c9c1f6cdfc053797c1a12a8d6252db0cd669bdeb9b98cd7f863e9": "0.16.1 to 0.20.0",
    },
}
# Larger than any generated copy; a bigger file is the project's own.
_MAX_BYTES = 64 * 1024


def membership_copy_findings(root: Path) -> tuple[ProjectDiagnostic, ...]:
    """A note for each unedited generated copy of the code `cayu.memberships` replaced."""

    findings: list[ProjectDiagnostic] = []
    for path, known in GENERATED_COPIES.items():
        file = root / path
        try:
            if not file.is_file() or file.stat().st_size > _MAX_BYTES:
                continue
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
        except OSError:
            continue
        releases = known.get(digest)
        if releases is None:
            continue
        findings.append(
            ProjectDiagnostic(
                code="MEMBERSHIPS_GENERATED_COPY",
                severity=DiagnosticSeverity.INFO,
                subject="organizations",
                path=path,
                message=(
                    f"{path} is the copy Cayu {releases} generated, unedited. "
                    "`cayu.memberships` now provides it, with the same authority, so queued "
                    "work stays admitted for the person who started it when the project moves "
                    "to it."
                ),
                hint=(
                    "Replace it with the imports `cayu new` generates now, as `cayu guide "
                    "b2b#move-to-cayumemberships` describes; no database migration is needed."
                ),
                tags=("configuration",),
                parameters={"generated_by": releases},
                documentation_anchor="cayu guide b2b#move-to-cayumemberships",
                verification_command="cayu check",
            )
        )
    return tuple(findings)
