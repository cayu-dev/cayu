"""Schema versioning / compatibility model (ADR 0001, Decision 7)."""

from __future__ import annotations

import pytest

from cayu.storage import migrations as m


def test_baseline_constants_are_coherent():
    assert m.REVISIONS[0].revision == m.BASELINE_REVISION
    assert m.REVISIONS[-1].revision == m.LATEST_REVISION
    assert m.REVISIONS[-1].compatible_from == m.MIN_SUPPORTED_REVISION
    assert m.MIGRATIONS_TABLE == "cayu_schema_migrations"


def test_pending_returns_revisions_after_current():
    assert m.pending(m.LATEST_REVISION) == ()
    assert m.pending(0) == m.REVISIONS  # a fresh DB has every revision pending


def test_validate_rejects_uninitialized():
    with pytest.raises(m.SchemaUninitialized):
        m.validate(m.SchemaState(revision=m.UNINITIALIZED, compatible_from=0))


def test_validate_accepts_matching_revision():
    # The common case: DB at the binary's latest revision.
    m.validate(m.SchemaState(revision=m.LATEST_REVISION, compatible_from=m.MIN_SUPPORTED_REVISION))


def test_validate_rejects_too_old_db():
    # DB revision below what this binary supports → needs migrate.
    with pytest.raises(m.SchemaTooOld):
        m.validate(
            m.SchemaState(revision=1, compatible_from=1),
            app_latest=3,
            app_min_supported=2,
        )


def test_validate_rejects_incompatibly_new_db():
    # DB migrated past a BREAKING revision the binary doesn't understand:
    # compatible_from (5) > app_latest (4) → upgrade the app.
    with pytest.raises(m.SchemaTooNew):
        m.validate(
            m.SchemaState(revision=5, compatible_from=5),
            app_latest=4,
            app_min_supported=1,
        )


def test_validate_tolerates_additively_newer_db():
    # DB is newer than the binary, but only by ADDITIVE revisions: the floor
    # (compatible_from=3) is still <= the binary's latest (3), so an older binary
    # keeps running. This is what makes rolling deploys / rollback safe (Q1/Q3).
    m.validate(
        m.SchemaState(revision=5, compatible_from=3),
        app_latest=3,
        app_min_supported=1,
    )


def test_additive_revision_inherits_floor_breaking_raises_it():
    # Documents the intended authoring rule for REVISIONS entries.
    base = m.Revision(revision=2, kind=m.RevisionKind.ADDITIVE, compatible_from=1)
    breaking = m.Revision(revision=3, kind=m.RevisionKind.BREAKING, compatible_from=3)
    assert base.compatible_from == 1  # additive keeps the prior floor
    assert breaking.compatible_from == breaking.revision  # breaking floors at itself


def test_revision_fourteen_remains_compatible_with_older_binaries() -> None:
    m.validate(
        m.SchemaState(revision=14, compatible_from=10),
        app_latest=13,
        app_min_supported=10,
    )


def test_revision_nineteen_rejects_pre_queue_session_workers() -> None:
    state = m.SchemaState(revision=19, compatible_from=19)

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 19"):
        m.validate(
            state,
            app_latest=18,
            app_min_supported=18,
        )

    m.validate(
        state,
        app_latest=19,
        app_min_supported=19,
    )
