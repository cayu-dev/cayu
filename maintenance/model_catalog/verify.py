"""Result shared by the browser verifier and catalog refresh runner."""

from __future__ import annotations

from dataclasses import dataclass

from cayu import ModelInfo


@dataclass(frozen=True)
class VerifyOutcome:
    verified: bool
    model: ModelInfo | None = None  # the corrected/confirmed record (Phase B)
    note: str = ""
    usage: dict[str, int] | None = None  # tokens consumed by this verification (Phase B)
    evidence: str = ""  # verbatim pricing row the agent grounded its numbers on (Phase B)


@dataclass(frozen=True)
class RecommendationOutcome:
    verified: bool
    provider_name: str
    models: tuple[str, ...] = ()
    source_url: str = ""
    evidence: str = ""
    note: str = ""
    usage: dict[str, int] | None = None
