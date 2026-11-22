"""How many sessions are sources of a knowledge entry, as its metadata records it.

An entry starts with one source: the session (or application) that wrote it. A session
that finds the same fact and corroborates the entry instead of proposing a copy is
recorded in the entry's metadata summary under ``cayu:corroboration``, whose
``source_count`` counts the writer plus every corroborating session. Readers that rank
by corroboration (proposal lookups and the review queue) read the count here, so an
entry without that summary, or with one Cayu didn't write, counts as one source.
"""

from __future__ import annotations

from cayu.knowledge.records import KnowledgeEntry

#: The entry metadata key that holds the corroboration summary.
KNOWLEDGE_CORROBORATION_METADATA_KEY = "cayu:corroboration"


def knowledge_source_count(entry: KnowledgeEntry) -> int:
    """The entry's recorded number of sources; 1 when it records none."""

    summary = entry.metadata.get(KNOWLEDGE_CORROBORATION_METADATA_KEY)
    if not isinstance(summary, dict):
        return 1
    count = summary.get("source_count")
    if type(count) is not int or count < 1:
        return 1
    return count
