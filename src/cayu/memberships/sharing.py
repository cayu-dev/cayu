"""How a new conversation is shared: the labels Cayu enforces, from verified membership.

Never take these labels from a request. Build them with :func:`conversation_labels` from the
signed-in person and a membership read from the application's own records; Cayu refuses any
the person may not create.
"""

from __future__ import annotations

from typing import Literal

from cayu.memberships.directory import Membership

Sharing = Literal["personal", "department", "organization"]
# The labels that decide sharing; nobody may change them on an existing conversation.
SHARING_LABELS = ("organization", "department", "owner", "sharing")


def conversation_labels(
    *, subject: str, membership: Membership, sharing: Sharing
) -> dict[str, str]:
    """The labels a conversation ``subject`` starts and shares as ``sharing`` takes."""

    labels = {
        "organization": membership.organization,
        "owner": subject,
        "sharing": sharing,
    }
    if sharing == "department":
        if membership.department is None:
            raise ValueError("Only a member of a department can share with it.")
        labels["department"] = membership.department
    return labels
