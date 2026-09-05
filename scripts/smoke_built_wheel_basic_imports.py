"""Fresh-process smoke check; run with the installed wheel on the import path."""

from __future__ import annotations

import sys
import time


def main() -> None:
    started = time.perf_counter()
    from cayu.core.events import Event

    elapsed = time.perf_counter() - started
    for module_name in (
        "cayu.evals.browser_acceptance",
        "cayu.evals.browser_acceptance_fixture",
        "cayu.evals.browser_acceptance_manifests",
        "cayu.evals.causal_memory_campaign",
    ):
        assert module_name not in sys.modules, module_name
    print(
        f"core import: {elapsed:.3f}s; Cayu modules: {sum(n.startswith('cayu') for n in sys.modules)}"
    )

    from cayu import CayuApp, WorkflowEvalTarget
    from cayu.evals import (
        BrowserAcceptanceFixtureV1,
        deterministic_browser_acceptance_manifest,
        run_causal_memory_reference_campaign,
    )

    assert Event.__name__ == "Event"
    assert CayuApp.__name__ == "CayuApp"
    assert WorkflowEvalTarget.__name__ == "WorkflowEvalTarget"
    assert BrowserAcceptanceFixtureV1.__name__ == "BrowserAcceptanceFixtureV1"
    assert callable(deterministic_browser_acceptance_manifest)
    assert callable(run_causal_memory_reference_campaign)
    print("supported lazy public imports passed")


if __name__ == "__main__":
    main()
