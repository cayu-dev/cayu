from contextvars import Context

from cayu.runtime import _interruption_coordinator as interruption_coordinator


def test_interruption_cascade_suppression_restores_same_context() -> None:
    context = Context()

    def exercise() -> tuple[bool, bool, bool]:
        before = interruption_coordinator.interruption_cascade_suppressed()
        with interruption_coordinator.suppress_interruption_cascade():
            during = interruption_coordinator.interruption_cascade_suppressed()
        after = interruption_coordinator.interruption_cascade_suppressed()
        return before, during, after

    assert context.run(exercise) == (False, True, False)


def test_cross_context_suppression_close_does_not_overwrite_closer() -> None:
    creator = Context()
    closer = Context()
    scope = interruption_coordinator.suppress_interruption_cascade()

    creator.run(scope.__enter__)
    closer.run(interruption_coordinator._SUPPRESS_BACKGROUND_INTERRUPTION_CASCADE.set, True)
    closer.run(scope.__exit__, None, None, None)

    assert closer.run(interruption_coordinator.interruption_cascade_suppressed) is True
