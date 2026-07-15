"""Result shared by the browser verifier and catalog refresh runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from cayu import ModelInfo, ModelPrice, Provenance


@dataclass(frozen=True)
class VerifyOutcome:
    verified: bool
    model: ModelInfo | None = None  # the corrected/confirmed record (Phase B)
    price: ModelPrice | None = None
    note: str = ""
    usage: dict[str, int] | None = None  # tokens consumed by this verification (Phase B)
    evidence: str = ""  # verbatim pricing row the agent grounded its numbers on (Phase B)
    pricing_provenance: Provenance | None = None
    pricing_effective_from: date | None = None
    model_evidence: str = ""


@dataclass(frozen=True)
class RecommendationOutcome:
    verified: bool
    provider_name: str
    models: tuple[str, ...] = ()
    source_url: str = ""
    evidence: str = ""
    note: str = ""
    usage: dict[str, int] | None = None
