"""Process-local ownership backstops for factory-created runners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from cayu._validation import canonical_durable_json_bytes
from cayu.runners._exit_cleanup import release_held_runners_at_exit
from cayu.runners.base import Runner
from cayu.sessions.store import SessionStore

if TYPE_CHECKING:
    from cayu.runtime._shared_environment_allocations import SharedEnvironmentAllocations


def session_store_survives_restart(store: SessionStore) -> bool:
    # Imported here: the application registry imports runtime modules.
    from cayu._application_agent_registry import _session_store_survives_restart as survives
    from cayu.runtime._checkpoint_store import public_session_store

    return survives(public_session_store(store))


class EnvironmentExitOwnership:
    """Keep preserved factory runners visible until their allocation is settled.

    A preserved factory result can fail before setup promotion. Its runner
    still needs exit cleanup when its reconnect state is process-local.
    Group by exact allocation so reconnect promotion or proven disposal
    can retire old runners even when reconnect creates a new instance.
    """

    def __init__(
        self,
        session_store: SessionStore,
        setups: Mapping[str, Any],
        shared_allocations: SharedEnvironmentAllocations,
    ) -> None:
        self._exit_setups = setups
        self._exit_shared_allocations = shared_allocations
        self._preserved_factory_runners: dict[tuple[str, str, str, bytes], dict[int, Runner]] = {}
        self._exit_hook: Callable[[], list[Runner]] | None = None
        self._process_local_store = False
        if not session_store_survives_restart(session_store):
            self.declare_process_local_session_store()

    def _remember_preserved_factory_runner(
        self,
        session_id: str,
        session_instance_id: str,
        environment_name: str,
        reconnect_metadata: dict[str, Any],
        runner: object,
    ) -> None:
        if not isinstance(runner, Runner):
            return
        key = (
            session_id,
            session_instance_id,
            environment_name,
            canonical_durable_json_bytes(reconnect_metadata, "reconnect_metadata"),
        )
        self._preserved_factory_runners.setdefault(key, {})[id(runner)] = runner

    def _retire_preserved_factory_runners(
        self,
        session_id: str,
        session_instance_id: str,
        environment_name: str,
        reconnect_metadata: dict[str, Any],
    ) -> None:
        """Forget the backstop only for the allocation disposed or transferred."""

        self._preserved_factory_runners.pop(
            (
                session_id,
                session_instance_id,
                environment_name,
                canonical_durable_json_bytes(reconnect_metadata, "reconnect_metadata"),
            ),
            None,
        )

    def declare_process_local_session_store(self) -> None:
        """Treat allocation records as ending with this process.

        For a session store that does not outlive the process, no later process
        can resume its sessions or reconnect to what they allocated. Two things
        follow, and this enables both (idempotently):

        - Factory requests, for creation and reconnect alike, carry
          ``process_local_owner``, so a factory can tag what it creates and a
          later process can reap it once this one is gone
          (``reap_orphaned_docker_coding_containers`` for Docker coding).
        - At normal interpreter exit, each factory runner still held (a session
          whose release failed, or that never released because the app was not
          closed) has its own close action applied synchronously; for Docker
          coding that is ``docker rm -f`` of the exact container. Environments
          the application registered directly are never touched.
        """

        self._process_local_store = True
        if self._exit_hook is not None:
            return
        # The hook holds the owner tables, not this lifecycle: an app dropped
        # without aclose() still has its held allocations released at exit.
        setups = self._exit_setups
        preserved_runners = self._preserved_factory_runners
        shared_runners = self._exit_shared_allocations.held_runners_source()

        def held_runners() -> list[Runner]:
            runners: list[Runner] = []
            for setup in list(setups.values()):
                registered = setup.registered_environment
                if not registered.factory_backed:
                    continue
                for environment in (
                    registered.environment,
                    getattr(registered.unclaimed_factory_result, "environment", None),
                    getattr(registered.retained_factory_result, "environment", None),
                ):
                    runner = getattr(environment, "runner", None)
                    if isinstance(runner, Runner):
                        runners.append(runner)
            for allocation_runners in preserved_runners.values():
                runners.extend(allocation_runners.values())
            runners.extend(shared_runners())
            return runners

        self._exit_hook = held_runners
        release_held_runners_at_exit(self, held_runners)
