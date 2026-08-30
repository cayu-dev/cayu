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


def test_revision_thirty_one_rejects_pre_input_contract_readers() -> None:
    state = m.SchemaState(revision=31, compatible_from=31)

    # Pre-31 readers do not know that input_contract is a private runtime marker
    # and would expose it as ordinary event payload.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 31"):
        m.validate(state, app_latest=30, app_min_supported=30)
    m.validate(state, app_latest=31, app_min_supported=31)


def test_revision_thirty_two_adds_opt_in_eval_storage() -> None:
    state = m.SchemaState(revision=32, compatible_from=31)

    # Revision-31 stores do not use the new eval tables, so they may continue
    # operating on the expanded database.
    m.validate(state, app_latest=31, app_min_supported=31)
    m.validate(state, app_latest=32, app_min_supported=32)
    with pytest.raises(m.SchemaTooOld, match="requires >= 32"):
        m.validate(
            m.SchemaState(revision=31, compatible_from=31),
            app_latest=32,
            app_min_supported=32,
        )


def test_revision_thirty_three_adds_target_scoped_eval_indexes() -> None:
    state = m.SchemaState(revision=33, compatible_from=31)

    # Existing writers maintain every indexed column, while EvalStore requires
    # the target-leading query plans introduced at revision 33.
    m.validate(state, app_latest=31, app_min_supported=31)
    m.validate(state, app_latest=33, app_min_supported=33)
    with pytest.raises(m.SchemaTooOld, match="requires >= 33"):
        m.validate(
            m.SchemaState(revision=32, compatible_from=31),
            app_latest=33,
            app_min_supported=33,
        )


def test_revision_forty_seven_adds_origin_aware_eval_results_and_baselines() -> None:
    state = m.SchemaState(revision=47, compatible_from=47)

    # A revision-46 EvalStore can publish a fresh result without maintaining the
    # new origin-aware index, so it cannot share the migrated database.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 47"):
        m.validate(state, app_latest=46, app_min_supported=46)
    m.validate(state, app_latest=47, app_min_supported=47)
    with pytest.raises(m.SchemaTooOld, match="requires >= 47"):
        m.validate(
            m.SchemaState(revision=46, compatible_from=46),
            app_latest=47,
            app_min_supported=47,
        )


def test_revision_forty_eight_adds_explicit_captured_only_eval_cases() -> None:
    state = m.SchemaState(revision=48, compatible_from=48)

    # A revision-47 EvalStore requires at least one input message and therefore
    # cannot safely interpret the zero-message marker for a captured-only case.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 48"):
        m.validate(state, app_latest=47, app_min_supported=47)
    m.validate(state, app_latest=48, app_min_supported=48)
    with pytest.raises(m.SchemaTooOld, match="requires >= 48"):
        m.validate(
            m.SchemaState(revision=47, compatible_from=47),
            app_latest=48,
            app_min_supported=48,
        )


def test_revision_forty_nine_adds_durable_verified_work_authority() -> None:
    state = m.SchemaState(revision=49, compatible_from=49)

    # A revision-48 task worker can complete contracted work through an
    # ordinary terminal entrance, so it cannot share the migrated database.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 49"):
        m.validate(state, app_latest=48, app_min_supported=48)
    m.validate(state, app_latest=49, app_min_supported=49)
    with pytest.raises(m.SchemaTooOld, match="requires >= 49"):
        m.validate(
            m.SchemaState(revision=48, compatible_from=48),
            app_latest=49,
            app_min_supported=49,
        )


def test_revision_fifty_persists_eval_run_invocation_authority() -> None:
    state = m.SchemaState(revision=50, compatible_from=50)

    # A revision-49 worker would recover queued work without the authenticated
    # HTTP origin and runtime contractions persisted at admission.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 50"):
        m.validate(state, app_latest=49, app_min_supported=49)
    m.validate(state, app_latest=50, app_min_supported=50)
    with pytest.raises(m.SchemaTooOld, match="requires >= 50"):
        m.validate(
            m.SchemaState(revision=49, compatible_from=49),
            app_latest=50,
            app_min_supported=50,
        )


def test_revision_fifty_one_adds_empty_memory_evidence_tables_additively() -> None:
    state = m.SchemaState(revision=51, compatible_from=50)

    # Revision 51 only adds unused tables. Revision-50 binaries neither need to
    # read nor maintain them, and no historical runtime rows are synthesized.
    m.validate(state, app_latest=50, app_min_supported=50)
    m.validate(state, app_latest=51, app_min_supported=50)
    m.validate(
        m.SchemaState(revision=50, compatible_from=50),
        app_latest=51,
        app_min_supported=50,
    )


def test_revision_fifty_three_adds_compatible_portable_eval_scenarios() -> None:
    state = m.SchemaState(revision=53, compatible_from=52)

    # Revision-52 workers remain safe because they do not use the independent
    # scenario catalog, while scenario-aware stores require revision 53.
    m.validate(state, app_latest=52, app_min_supported=52)
    m.validate(state, app_latest=53, app_min_supported=53)
    with pytest.raises(m.SchemaTooOld, match="requires >= 53"):
        m.validate(
            m.SchemaState(revision=52, compatible_from=52),
            app_latest=53,
            app_min_supported=53,
        )


def test_revision_fifty_four_rejects_readers_that_expose_private_input_attestations() -> None:
    state = m.SchemaState(revision=54, compatible_from=54)

    # Pre-54 readers do not hide input_contract on resume and queue events, so
    # they cannot share a database once new session writers persist those facts.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 54"):
        m.validate(state, app_latest=53, app_min_supported=52)
    m.validate(state, app_latest=54, app_min_supported=54)
    with pytest.raises(m.SchemaTooOld, match="requires >= 54"):
        m.validate(
            m.SchemaState(revision=53, compatible_from=52),
            app_latest=54,
            app_min_supported=54,
        )


def test_revision_fifty_five_breaks_for_rejected_reconciliation_idempotency() -> None:
    state = m.SchemaState(revision=55, compatible_from=55)

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 55"):
        m.validate(state, app_latest=54, app_min_supported=54)
    m.validate(state, app_latest=55, app_min_supported=55)
    with pytest.raises(m.SchemaTooOld, match="requires >= 55"):
        m.validate(
            m.SchemaState(revision=54, compatible_from=54),
            app_latest=55,
            app_min_supported=55,
        )


def test_revision_fifty_six_adds_compatible_scenario_progress() -> None:
    state = m.SchemaState(revision=56, compatible_from=55)

    # Revision-55 writers ignore the nullable progress document, while
    # scenario-aware eval stores require the column before admitting work.
    m.validate(state, app_latest=55, app_min_supported=55)
    m.validate(state, app_latest=56, app_min_supported=56)
    with pytest.raises(m.SchemaTooOld, match="requires >= 56"):
        m.validate(
            m.SchemaState(revision=55, compatible_from=55),
            app_latest=56,
            app_min_supported=56,
        )


def test_revision_fifty_seven_breaks_for_typed_queued_messages() -> None:
    state = m.SchemaState(revision=57, compatible_from=57)

    # Pre-57 session workers would ignore message_json and deliver only the
    # compatibility text projection, so they cannot share the migrated store.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 57"):
        m.validate(state, app_latest=56, app_min_supported=55)
    m.validate(state, app_latest=57, app_min_supported=57)
    with pytest.raises(m.SchemaTooOld, match="requires >= 57"):
        m.validate(
            m.SchemaState(revision=56, compatible_from=55),
            app_latest=57,
            app_min_supported=57,
        )


def test_revision_fifty_eight_breaks_for_completion_verifier_profiles() -> None:
    state = m.SchemaState(revision=58, compatible_from=58)

    # Pre-58 task workers cannot preserve immutable verifier-profile authority
    # across claims, decisions, replacement, or restart.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 58"):
        m.validate(state, app_latest=57, app_min_supported=57)
    m.validate(state, app_latest=58, app_min_supported=58)
    with pytest.raises(m.SchemaTooOld, match="requires >= 58"):
        m.validate(
            m.SchemaState(revision=57, compatible_from=57),
            app_latest=58,
            app_min_supported=58,
        )


def test_revision_fifty_nine_breaks_for_result_resolver_authority() -> None:
    state = m.SchemaState(revision=59, compatible_from=59)

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 59"):
        m.validate(state, app_latest=58, app_min_supported=58)
    m.validate(state, app_latest=59, app_min_supported=59)
    with pytest.raises(m.SchemaTooOld, match="requires >= 59"):
        m.validate(
            m.SchemaState(revision=58, compatible_from=58),
            app_latest=59,
            app_min_supported=59,
        )


def test_revision_sixty_breaks_for_revision_bound_knowledge_relations() -> None:
    state = m.SchemaState(revision=60, compatible_from=60)

    # Pre-60 knowledge workers cannot preserve relation records, receipts, or
    # their atomic outbox events.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 60"):
        m.validate(state, app_latest=59, app_min_supported=59)
    m.validate(state, app_latest=60, app_min_supported=60)
    with pytest.raises(m.SchemaTooOld, match="requires >= 60"):
        m.validate(
            m.SchemaState(revision=59, compatible_from=59),
            app_latest=60,
            app_min_supported=60,
        )


def test_revision_sixty_one_breaks_for_work_attempt_admission() -> None:
    state = m.SchemaState(revision=61, compatible_from=61)

    # Pre-61 task workers do not maintain admission intents or process-aware
    # execution claims, so they cannot share the migrated task store.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 61"):
        m.validate(state, app_latest=60, app_min_supported=60)
    m.validate(state, app_latest=61, app_min_supported=61)
    with pytest.raises(m.SchemaTooOld, match="requires >= 61"):
        m.validate(
            m.SchemaState(revision=60, compatible_from=60),
            app_latest=61,
            app_min_supported=61,
        )


def test_revision_sixty_two_breaks_for_deferred_interaction_payloads() -> None:
    state = m.SchemaState(revision=62, compatible_from=62)

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 62"):
        m.validate(state, app_latest=61, app_min_supported=61)
    m.validate(state, app_latest=62, app_min_supported=62)
    with pytest.raises(m.SchemaTooOld, match="requires >= 62"):
        m.validate(
            m.SchemaState(revision=61, compatible_from=61),
            app_latest=62,
            app_min_supported=62,
        )


def test_revision_sixty_three_breaks_for_reviewed_knowledge_maintenance() -> None:
    state = m.SchemaState(revision=63, compatible_from=63)

    # Pre-63 knowledge workers cannot preserve atomic multi-entry decisions or
    # distinguish their immutable review receipts from partial lifecycle writes.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 63"):
        m.validate(state, app_latest=62, app_min_supported=62)
    m.validate(state, app_latest=63, app_min_supported=63)
    with pytest.raises(m.SchemaTooOld, match="requires >= 63"):
        m.validate(
            m.SchemaState(revision=62, compatible_from=62),
            app_latest=63,
            app_min_supported=63,
        )


def test_revision_sixty_four_adds_authored_eval_suites_without_new_compatibility_break() -> None:
    revision = m.revision(64)

    assert revision.kind is m.RevisionKind.ADDITIVE
    assert revision.compatible_from == 63
    m.validate(
        m.SchemaState(revision=64, compatible_from=63),
        app_latest=64,
        app_min_supported=63,
    )


def test_revision_sixty_five_breaks_for_bounded_knowledge_entry_reads() -> None:
    revision = m.revision(65)
    state = m.SchemaState(revision=65, compatible_from=65)

    assert revision.kind is m.RevisionKind.BREAKING
    assert revision.compatible_from == 65
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 65"):
        m.validate(state, app_latest=64, app_min_supported=63)
    m.validate(state, app_latest=65, app_min_supported=65)
    with pytest.raises(m.SchemaTooOld, match="requires >= 65"):
        m.validate(
            m.SchemaState(revision=64, compatible_from=63),
            app_latest=65,
            app_min_supported=65,
        )


def test_revision_sixty_six_breaks_for_local_execution_attempt_fencing() -> None:
    revision = m.revision(66)
    state = m.SchemaState(revision=66, compatible_from=66)

    assert revision.kind is m.RevisionKind.BREAKING
    assert revision.compatible_from == 66
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 66"):
        m.validate(state, app_latest=65, app_min_supported=65)
    m.validate(state, app_latest=66, app_min_supported=66)
    with pytest.raises(m.SchemaTooOld, match="requires >= 66"):
        m.validate(
            m.SchemaState(revision=65, compatible_from=65),
            app_latest=66,
            app_min_supported=66,
        )


def test_revision_sixty_seven_breaks_for_pending_maintenance_proposals() -> None:
    revision = m.revision(67)
    state = m.SchemaState(revision=67, compatible_from=67)

    assert revision.kind is m.RevisionKind.BREAKING
    assert revision.compatible_from == 67
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 67"):
        m.validate(state, app_latest=66, app_min_supported=66)
    m.validate(state, app_latest=67, app_min_supported=67)
    with pytest.raises(m.SchemaTooOld, match="requires >= 67"):
        m.validate(
            m.SchemaState(revision=66, compatible_from=66),
            app_latest=67,
            app_min_supported=67,
        )


def test_revision_sixty_eight_adds_judge_calibration_reports() -> None:
    revision = m.revision(68)
    state = m.SchemaState(revision=68, compatible_from=67)

    assert revision.kind is m.RevisionKind.ADDITIVE
    assert revision.compatible_from == 67
    # Older writers do not use the new table and remain safe during rollout.
    m.validate(state, app_latest=67, app_min_supported=67)
    m.validate(state, app_latest=68, app_min_supported=68)
    with pytest.raises(m.SchemaTooOld, match="requires >= 68"):
        m.validate(
            m.SchemaState(revision=67, compatible_from=67),
            app_latest=68,
            app_min_supported=68,
        )


def test_revision_sixty_nine_adds_agent_work_context_without_raising_floor() -> None:
    revision = m.revision(69)
    state = m.SchemaState(revision=69, compatible_from=67)

    assert revision.kind is m.RevisionKind.ADDITIVE
    assert revision.compatible_from == 67
    m.validate(state, app_latest=68, app_min_supported=67)
    m.validate(state, app_latest=69, app_min_supported=67)


def test_revision_seventy_adds_interrupted_task_handoff_receipts() -> None:
    revision = m.revision(70)
    state = m.SchemaState(revision=70, compatible_from=67)

    assert revision.kind is m.RevisionKind.ADDITIVE
    assert revision.compatible_from == 67
    m.validate(state, app_latest=69, app_min_supported=67)
    m.validate(state, app_latest=70, app_min_supported=67)


def test_revision_seventy_one_requires_atomic_recall_delivery_writers() -> None:
    revision = m.revision(71)
    state = m.SchemaState(revision=71, compatible_from=71)

    assert revision.kind is m.RevisionKind.BREAKING
    assert revision.compatible_from == 71
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 71"):
        m.validate(state, app_latest=70, app_min_supported=67)
    m.validate(state, app_latest=71, app_min_supported=71)
    with pytest.raises(m.SchemaTooOld, match="requires >= 71"):
        m.validate(
            m.SchemaState(revision=70, compatible_from=67),
            app_latest=71,
            app_min_supported=71,
        )


def test_revision_seventy_two_fences_current_durable_eval_execution() -> None:
    revision = m.revision(72)
    state = m.SchemaState(revision=72, compatible_from=72)

    assert revision.kind is m.RevisionKind.BREAKING
    assert revision.compatible_from == 72
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 72"):
        m.validate(state, app_latest=71, app_min_supported=71)
    m.validate(state, app_latest=72, app_min_supported=72)
    with pytest.raises(m.SchemaTooOld, match="requires >= 72"):
        m.validate(
            m.SchemaState(revision=71, compatible_from=71),
            app_latest=72,
            app_min_supported=72,
        )


def test_revision_seventy_three_requires_input_bound_recall_results() -> None:
    revision = m.revision(73)
    state = m.SchemaState(revision=73, compatible_from=73)

    assert revision.kind is m.RevisionKind.BREAKING
    assert revision.compatible_from == 73
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 73"):
        m.validate(state, app_latest=72, app_min_supported=72)
    m.validate(state, app_latest=73, app_min_supported=73)
    with pytest.raises(m.SchemaTooOld, match="requires >= 73"):
        m.validate(
            m.SchemaState(revision=72, compatible_from=72),
            app_latest=73,
            app_min_supported=73,
        )


def test_revision_seventy_four_fences_recoverable_eval_trials() -> None:
    revision = m.revision(74)
    state = m.SchemaState(revision=74, compatible_from=74)

    assert revision.kind is m.RevisionKind.BREAKING
    assert revision.compatible_from == 74
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 74"):
        m.validate(state, app_latest=73, app_min_supported=73)
    m.validate(state, app_latest=74, app_min_supported=74)
    with pytest.raises(m.SchemaTooOld, match="requires >= 74"):
        m.validate(
            m.SchemaState(revision=73, compatible_from=73),
            app_latest=74,
            app_min_supported=74,
        )


def test_revision_thirty_four_adds_delayed_task_availability() -> None:
    revision = m.revision(34)
    state = m.SchemaState(
        revision=revision.revision,
        compatible_from=revision.compatible_from,
    )

    # An older TaskStore would ignore available_at and could claim future work
    # early, so no pre-34 binary may start against the migrated database.
    with pytest.raises(m.SchemaTooNew, match="understands revision >= 34"):
        m.validate(state, app_latest=33, app_min_supported=31)
    m.validate(state, app_latest=34, app_min_supported=34)
    with pytest.raises(m.SchemaTooOld, match="requires >= 34"):
        m.validate(
            m.SchemaState(revision=33, compatible_from=31),
            app_latest=34,
            app_min_supported=34,
        )


def test_revision_thirty_eight_adds_idempotent_task_terminalization_receipts() -> None:
    revision = m.revision(38)
    state = m.SchemaState(
        revision=revision.revision,
        compatible_from=revision.compatible_from,
    )

    # Older writers may keep using the legacy terminal methods against the new
    # receipt table; they simply cannot claim acknowledgement-loss replay safety.
    m.validate(state, app_latest=37, app_min_supported=37)
    m.validate(state, app_latest=38, app_min_supported=38)
    with pytest.raises(m.SchemaTooOld, match="requires >= 38"):
        m.validate(
            m.SchemaState(revision=37, compatible_from=37),
            app_latest=38,
            app_min_supported=38,
        )


def test_revision_forty_indexes_queued_dispatch_terminal_handoffs() -> None:
    revision = m.revision(40)
    state = m.SchemaState(
        revision=revision.revision,
        compatible_from=revision.compatible_from,
    )

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 40"):
        m.validate(state, app_latest=39, app_min_supported=39)
    m.validate(state, app_latest=40, app_min_supported=40)
    with pytest.raises(m.SchemaTooOld, match="requires >= 40"):
        m.validate(
            m.SchemaState(revision=39, compatible_from=39),
            app_latest=40,
            app_min_supported=40,
        )


def test_revision_thirty_six_requires_session_invocation_provenance() -> None:
    revision = m.revision(36)
    state = m.SchemaState(
        revision=revision.revision,
        compatible_from=revision.compatible_from,
    )

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 36"):
        m.validate(state, app_latest=35, app_min_supported=35)
    m.validate(state, app_latest=36, app_min_supported=36)
    with pytest.raises(m.SchemaTooOld, match="requires >= 36"):
        m.validate(
            m.SchemaState(revision=35, compatible_from=35),
            app_latest=36,
            app_min_supported=36,
        )


def test_revision_thirty_nine_requires_task_invocation_provenance() -> None:
    revision = m.revision(39)
    state = m.SchemaState(
        revision=revision.revision,
        compatible_from=revision.compatible_from,
    )

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 39"):
        m.validate(state, app_latest=38, app_min_supported=38)
    m.validate(state, app_latest=39, app_min_supported=39)
    with pytest.raises(m.SchemaTooOld, match="requires >= 39"):
        m.validate(
            m.SchemaState(revision=38, compatible_from=37),
            app_latest=39,
            app_min_supported=39,
        )


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


def test_revision_twenty_side_effect_handoff_is_rolling_deploy_compatible() -> None:
    state = m.SchemaState(revision=20, compatible_from=19)

    m.validate(
        state,
        app_latest=19,
        app_min_supported=19,
    )
    m.validate(
        state,
        app_latest=20,
        app_min_supported=20,
    )


def test_revision_twenty_one_rejects_pre_billing_identity_readers() -> None:
    state = m.SchemaState(revision=21, compatible_from=21)

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 21"):
        m.validate(
            state,
            app_latest=20,
            app_min_supported=19,
        )

    with pytest.raises(m.SchemaTooOld, match="requires >= 21"):
        m.validate(
            m.SchemaState(revision=20, compatible_from=19),
            app_latest=21,
            app_min_supported=21,
        )

    m.validate(
        state,
        app_latest=21,
        app_min_supported=21,
    )


def test_revision_twenty_two_rejects_workers_without_manifest_baselines() -> None:
    state = m.SchemaState(revision=22, compatible_from=22)

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 22"):
        m.validate(
            state,
            app_latest=21,
            app_min_supported=21,
        )

    with pytest.raises(m.SchemaTooOld, match="requires >= 22"):
        m.validate(
            m.SchemaState(revision=21, compatible_from=21),
            app_latest=22,
            app_min_supported=22,
        )

    m.validate(
        state,
        app_latest=22,
        app_min_supported=22,
    )


def test_revision_twenty_three_rejects_pre_execution_identity_workers() -> None:
    state = m.SchemaState(revision=23, compatible_from=23)

    with pytest.raises(m.SchemaTooNew, match="understands revision >= 23"):
        m.validate(
            state,
            app_latest=22,
            app_min_supported=22,
        )

    with pytest.raises(m.SchemaTooOld, match="requires >= 23"):
        m.validate(
            m.SchemaState(revision=22, compatible_from=22),
            app_latest=23,
            app_min_supported=23,
        )

    m.validate(
        state,
        app_latest=23,
        app_min_supported=23,
    )
