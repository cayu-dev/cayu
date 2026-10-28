"""Application configuration and entrances for protected human review."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.approvals.review import (
    HumanReviewContext,
    HumanReviewPolicy,
    HumanReviewReference,
    HumanReviewView,
)
from cayu.runtime.application_lifecycle import _tracked_entrance

if TYPE_CHECKING:
    from cayu.applications import CayuApp


def _human_review_dashboard_purpose(policy: HumanReviewPolicy | None) -> str | None:
    purpose = getattr(policy, "dashboard_purpose", None)
    if purpose is None:
        return None
    if type(purpose) is not str or not 1 <= len(purpose) <= 128 or purpose != purpose.strip():
        raise ValueError(
            "human_review_policy.dashboard_purpose must be 1 to 128 characters "
            "without surrounding spaces."
        )
    return purpose


class HumanReviewEntrances:
    _human_review_purpose: str | None = None

    def _set_human_review_purpose(self, policy: HumanReviewPolicy | None) -> None:
        self._human_review_purpose = _human_review_dashboard_purpose(policy)

    @property
    def human_review_purpose(self) -> str | None:
        """The purpose the configured human review policy asks the dashboard to use.

        None without a policy, or when the policy names none (``dashboard_purpose``).
        """

        return self._human_review_purpose

    @_tracked_entrance
    async def require_human_review_resolution_authority(
        self: CayuApp,
        session_id: str,
        reference: HumanReviewReference | None,
    ) -> None:
        """Check decision permission independently of inspection and receipt replay."""
        session_id = await self._resolve_public_session_id(session_id)
        await self._recovery_coordinator.require_human_review_resolution_authority(
            session_id, reference
        )

    @_tracked_entrance
    async def inspect_human_review(
        self: CayuApp,
        session_id: str,
        *,
        context: HumanReviewContext,
    ) -> HumanReviewView:
        """Inspect protected review content without executing or claiming work.

        Direct SDK callers are trusted to supply verified recipient provenance.
        The configured policy must authorize the exact session and tenant.
        """
        context = HumanReviewContext.model_validate(context.model_dump())
        session_id = await self._resolve_public_session_id(session_id)
        return await self._recovery_coordinator.inspect_human_review(session_id, context)
