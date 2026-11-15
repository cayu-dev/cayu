"""How a new conversation is shared: the labels Cayu enforces, from verified membership.

Never take these labels from a request. Build them with :func:`conversation_labels` from the
signed-in person and a membership read from the application's own records; Cayu refuses any
the person may not create.
"""

from __future__ import annotations

from typing import Literal, get_args

from cayu.memberships.directory import Membership

Sharing = Literal["personal", "organization"]
# The labels that decide sharing; nobody may change them on an existing conversation.
SHARING_LABELS = ("organization", "owner", "sharing")


def conversation_labels(
    *, subject: str, membership: Membership, sharing: Sharing
) -> dict[str, str]:
    """The labels a conversation ``subject`` starts and shares as ``sharing`` takes."""

    if sharing not in get_args(Sharing):
        raise ValueError("A conversation is shared as 'personal' or 'organization'.")
    return {
        "organization": membership.organization,
        "owner": subject,
        "sharing": sharing,
    }
