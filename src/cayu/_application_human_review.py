"""Application configuration and entrances for protected human review."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.approvals.review import (
    HumanReviewContext,
    HumanReviewLongFieldView,
    HumanReviewPolicy,
    HumanReviewReference,
    HumanReviewView,
)
from cayu.runtime.application_lifecycle import _tracked_entrance

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.runtime.public_authority import PublicAuthorityAliasCodec


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
    _human_review_configured: bool = False

    def _configure_human_review(
        self,
        policy: HumanReviewPolicy | None,
        alias_codec: PublicAuthorityAliasCodec | None,
    ) -> None:
        if policy is not None:
            # A policy without its own key (ToolArgumentReview) derives one here.
            policy._bind_application_keyring(None if alias_codec is None else alias_codec.keyring)
        self._human_review_purpose = _human_review_dashboard_purpose(policy)
        self._human_review_configured = policy is not None

    @property
    def human_review_purpose(self) -> str | None:
        """The purpose the configured human review policy asks the dashboard to use.

        None without a policy, or when the policy names none (``dashboard_purpose``).
        """

        return self._human_review_purpose

    @property
    def human_review_configured(self) -> bool:
        """Whether the app has a human review policy that can show approval arguments."""

        return self._human_review_configured

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

    @_tracked_entrance
    async def read_human_review_long_field(
        self: CayuApp,
        session_id: str,
        *,
        reference: HumanReviewReference,
        key: str,
    ) -> HumanReviewLongFieldView:
        """Read one long field of the current review, whole.

        ``reference`` is the reference of the view that lists ``key`` in its
        ``long_fields``. The recipient it names needs both inspect and decide
        permission, the view must still be current and permitted, and the
        complete value is checked for known secrets. Raises
        ``HumanReviewDenied`` or ``HumanReviewConflict`` (inspect again).
        """
        reference = HumanReviewReference.model_validate(reference.model_dump())
        if type(key) is not str:
            raise TypeError("key must be a string.")
        session_id = await self._resolve_public_session_id(session_id)
        return await self._recovery_coordinator.read_human_review_long_field(
            session_id, reference, key
        )
