"""PostgreSQL implementation of the durable verified-work task lifecycle."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar

from psycopg import AsyncConnection as PsycopgAsyncConnection
from psycopg import AsyncCursor
from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.runtime.completion_verifier_profiles import (
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierProfileRecord,
    completion_verifier_profile_preparation_request_sha256,
    completion_verifier_profile_record_from_document,
    completion_verifier_profile_record_from_preparation,
    copy_completion_verifier_profile_preparation_request,
    copy_completion_verifier_profile_record,
    require_completion_verifier_profile_transition,
)
from cayu.runtime.tasks import (
    CompletionDecisionApplicationReceipt,
    Task,
    TaskClaimLost,
    TaskStatus,
    TaskTopologyInconsistent,
    _ensure_owned_active_task_lease,
    copy_task,
)
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionVerdict,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    TaskCompletionDecisionRequired,
    WorkAttempt,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    WorkContractRef,
    completion_decision_application_request_sha256,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
    completion_proposal_request_sha256,
    completion_verification_claim_authority_sha256,
    completion_verification_claim_request_sha256,
    copy_completion_decision_application_request,
    copy_completion_decision_create,
    copy_completion_proposal_create,
    copy_completion_verification_claim_request,
    copy_work_attempt_create,
    copy_work_contract,
    copy_work_contract_ref,
    validate_completion_decision_contract,
    validate_work_completion_idempotency_key,
    work_attempt_request_sha256,
)
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _verified_work_support as verified_work_support

_T = TypeVar("_T")
_POSTGRES_MUTATION_CANCELLATION_GRACE_SECONDS = 1.0
_POSTGRES_MUTATION_UNWIND_GRACE_SECONDS = 1.0
_RETAINED_POSTGRES_MUTATION_TASKS: set[asyncio.Task[Any]] = set()
_POSTGRES_CONNECTION_BEHAVIOR_KWARGS = frozenset(("context", "cursor_factory", "row_factory"))


def _require_quiescent_postgres_mutation_pool(
    pool: Any,
    *,
    allowed_configure: Any = None,
) -> None:
    """Authenticate the driver behavior used by bounded mutation settlement."""

    if type(pool) is not AsyncConnectionPool:
        raise TypeError(
            "Postgres verified-work mutations require the built-in "
            "AsyncConnectionPool implementation."
        )
    if pool.connection_class is not PsycopgAsyncConnection:
        raise TypeError(
            "Postgres verified-work mutations require the built-in psycopg "
            "AsyncConnection implementation."
        )
    if pool._configure is not allowed_configure:
        raise TypeError(
            "Postgres verified-work mutations do not accept an unverified pool configure callback."
        )
    if pool._check is not None:
        raise TypeError("Postgres verified-work mutations do not accept a pool check callback.")
    if pool._reset is not None:
        raise TypeError("Postgres verified-work mutations do not accept a pool reset callback.")
    if pool._reconnect_failed is not None:
        raise TypeError("Postgres verified-work mutations do not accept a pool reconnect callback.")
    if type(pool.conninfo) is not str:
        raise TypeError("Postgres verified-work mutations require a static pool connection string.")
    kwargs = pool.kwargs
    if kwargs is not None:
        if type(kwargs) is not dict:
            raise TypeError(
                "Postgres verified-work mutations require static pool connection options."
            )
        if _POSTGRES_CONNECTION_BEHAVIOR_KWARGS.intersection(kwargs):
            raise TypeError(
                "Postgres verified-work mutations do not accept connection behavior "
                "factories in pool options."
            )
        if kwargs.get("autocommit", False) is not False:
            raise TypeError("Postgres verified-work mutations require transactional connections.")


def _require_quiescent_postgres_mutation_connection(connection: Any) -> None:
    """Verify that checkout preserved the pool's authenticated connection type."""

    if type(connection) is not PsycopgAsyncConnection:
        raise TypeError(
            "Postgres verified-work mutation checkout returned an unsupported "
            "connection implementation."
        )
    if connection.autocommit is not False:
        raise TypeError(
            "Postgres verified-work mutation checkout returned an autocommit connection."
        )
    if connection.cursor_factory is not AsyncCursor or connection.row_factory is not tuple_row:
        raise TypeError(
            "Postgres verified-work mutation checkout returned unsupported cursor behavior."
        )


class _PostgresMutationOwnerCancellation:
    """Private marker for cancellation injected into the mutation owner."""


_POSTGRES_MUTATION_OWNER_CANCELLATION = _PostgresMutationOwnerCancellation()


class _PostgresMutationConnectionOwner:
    """Keep exact mutation authority reachable and synchronously revocable."""

    def __init__(self, pool: Any, *, allowed_configure: Any = None) -> None:
        _require_quiescent_postgres_mutation_pool(
            pool,
            allowed_configure=allowed_configure,
        )
        self.connection: Any | None = None
        self._revoked = False

    def acquire(self, connection: Any) -> None:
        if self.connection is not None:
            raise RuntimeError("Postgres mutation already owns a connection.")
        if self._revoked:
            _abort_postgres_connection(connection)
            raise RuntimeError("Cancelled Postgres mutation cannot acquire new authority.")
        _require_quiescent_postgres_mutation_connection(connection)
        if not callable(getattr(getattr(connection, "pgconn", None), "finish", None)):
            raise TypeError("Postgres mutation connections must expose psycopg pgconn.finish().")
        self.connection = connection

    def release(self, connection: Any) -> None:
        if self.connection is connection:
            self.connection = None

    def revoke(self) -> tuple[BaseException, ...]:
        """Fence future acquisition and synchronously abort current DB authority."""

        self._revoked = True
        connection = self.connection
        if connection is None or connection.closed:
            return ()
        try:
            _abort_postgres_connection(connection)
        except BaseException as failure:
            failure.__context__ = None
            return (failure,)
        return ()

    @property
    def revocation_is_proven(self) -> bool:
        connection = self.connection
        return connection is None or connection.closed


def _abort_postgres_connection(connection: Any) -> None:
    """Synchronously revoke one checked-out psycopg connection's authority."""

    if connection.closed:
        return
    finish = getattr(getattr(connection, "pgconn", None), "finish", None)
    if not callable(finish):
        raise TypeError("Postgres mutation connection cannot be synchronously aborted.")
    finish()
    if not connection.closed:
        raise RuntimeError("Postgres mutation connection remained open after abort.")


def _without_injected_owner_cancellation(
    failure: BaseException,
) -> BaseException | None:
    """Remove only the cancellation propagated from the public caller."""

    if isinstance(failure, asyncio.CancelledError):
        if len(failure.args) == 1 and failure.args[0] is _POSTGRES_MUTATION_OWNER_CANCELLATION:
            return None
        return failure
    if isinstance(failure, BaseExceptionGroup):
        remaining = tuple(
            retained
            for child in failure.exceptions
            if (retained := _without_injected_owner_cancellation(child)) is not None
        )
        return None if not remaining else failure.derive(remaining)
    return failure


def _ordered_postgres_mutation_failure(
    failures: list[BaseException],
) -> BaseException | None:
    if not failures:
        return None
    if len(failures) == 1:
        return failures[0]
    return BaseExceptionGroup(
        "Postgres mutation cancellation encountered settlement failures",
        failures,
    )


def _retain_postgres_mutation_task(task: asyncio.Task[Any]) -> None:
    """Keep a closed-connection owner alive until its Python cleanup settles."""

    _RETAINED_POSTGRES_MUTATION_TASKS.add(task)

    def consume(completed: asyncio.Task[Any]) -> None:
        _RETAINED_POSTGRES_MUTATION_TASKS.discard(completed)
        with suppress(BaseException):
            completed.exception()

    task.add_done_callback(consume)


async def _wait_for_postgres_mutation_task(
    task: asyncio.Task[Any],
    *,
    timeout_seconds: float,
) -> bool:
    """Wait to one fixed deadline without losing repeated caller cancellation."""

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        try:
            done, _ = await asyncio.wait((task,), timeout=remaining)
        except asyncio.CancelledError:
            # The first cancellation remains authoritative in the caller. A
            # repeated delivery must not extend the settlement deadline.
            continue
        if task in done:
            return True
    return True


def _json_document(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


class PostgresVerifiedWorkMixin:
    """Transactional verified-work behavior mixed into ``PostgresTaskStore``."""

    verified_work_mutations_are_cancellation_quiescent: ClassVar[bool] = True

    if TYPE_CHECKING:
        _pool: Any
        _clock_is_injected: bool
        _clock: Callable[[], datetime]
        _postgres_mutation_allowed_configure: Any

        async def _ensure_ready(self) -> None: ...

    async def _await_owned_store_mutation(
        self,
        operation: Coroutine[Any, Any, _T],
        *,
        connection_owner: _PostgresMutationConnectionOwner,
    ) -> _T:
        """Bound cancellation while retaining exact database mutation ownership."""

        owner = asyncio.create_task(operation, name="cayu-postgres-store-mutation")
        cancellation: asyncio.CancelledError | None = None
        settlement_failures: list[BaseException] = []
        while not owner.done():
            try:
                await asyncio.wait((owner,))
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                    owner.cancel(_POSTGRES_MUTATION_OWNER_CANCELLATION)
                    settled = await _wait_for_postgres_mutation_task(
                        owner,
                        timeout_seconds=_POSTGRES_MUTATION_CANCELLATION_GRACE_SECONDS,
                    )
                    if settled:
                        break

                    settlement_failures.extend(connection_owner.revoke())
                    settled = await _wait_for_postgres_mutation_task(
                        owner,
                        timeout_seconds=_POSTGRES_MUTATION_UNWIND_GRACE_SECONDS,
                    )
                    if not settled:
                        if not connection_owner.revocation_is_proven:
                            # A broken extension boundary left live database
                            # authority. Fail closed and keep ownership until
                            # the operation itself proves settlement.
                            while not owner.done():
                                try:
                                    await asyncio.wait((owner,))
                                except asyncio.CancelledError:
                                    continue
                        else:
                            _retain_postgres_mutation_task(owner)
                            settlement_failures.append(
                                TimeoutError(
                                    "Postgres mutation did not finish Python cleanup "
                                    "after its database authority was revoked."
                                )
                            )
                    break

        if owner.done():
            try:
                owner.result()
            except BaseException as failure:
                if cancellation is None:
                    raise
                retained = _without_injected_owner_cancellation(failure)
                if retained is not None:
                    settlement_failures.insert(0, retained)
        if cancellation is not None:
            cause = _ordered_postgres_mutation_failure(settlement_failures)
            if cause is not None:
                raise cancellation from cause
            raise cancellation
        assert owner.done()
        return owner.result()

    @asynccontextmanager
    async def _owned_store_connection(
        self,
        connection_owner: _PostgresMutationConnectionOwner,
    ) -> AsyncIterator[Any]:
        """Keep the checked-out connection reachable through pool return."""

        connection: Any | None = None
        try:
            async with self._pool.connection() as checked_out_connection:
                connection = checked_out_connection
                connection_owner.acquire(connection)
                yield connection
        finally:
            if connection is not None:
                connection_owner.release(connection)

    async def _run_verified_work_mutation(
        self,
        operation: Callable[[Any, Any], Coroutine[Any, Any, _T]],
    ) -> _T:
        """Defer caller cancellation until the owned DB transaction settles."""

        connection_owner = _PostgresMutationConnectionOwner(
            self._pool,
            allowed_configure=getattr(
                self,
                "_postgres_mutation_allowed_configure",
                None,
            ),
        )

        async def transact() -> _T:
            async with self._owned_store_connection(connection_owner) as conn:
                try:
                    async with conn.cursor() as cur:
                        result = await operation(conn, cur)
                    await conn.commit()
                    return result
                except BaseException as primary:
                    try:
                        await conn.rollback()
                    except BaseException as rollback_failure:
                        rollback_failure.__context__ = None
                        failures = [primary, rollback_failure]
                        try:
                            # AsyncConnection.close() may return a connection to
                            # a close_returns pool. Revoke the libpq authority
                            # directly so the enclosing pool context owns the
                            # only return and must discard this connection.
                            _abort_postgres_connection(conn)
                        except BaseException as abort_failure:
                            abort_failure.__context__ = None
                            failures.append(abort_failure)
                        raise BaseExceptionGroup(
                            "Postgres verified-work transaction failed and rollback "
                            "could not be proven",
                            failures,
                        ) from None
                    raise

        return await self._await_owned_store_mutation(
            transact(),
            connection_owner=connection_owner,
        )

    async def _database_now(self, cur: Any) -> datetime:
        # Lease authority must be evaluated after any preceding row-lock wait.
        # ``transaction_timestamp()`` is frozen before that wait and can make an
        # already-expired worker or verifier appear live under contention.
        await cur.execute("SELECT clock_timestamp()")
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Postgres did not return a current database timestamp.")
        return pg_support.to_utc(row[0])

    async def _lock_verified_work_identity(
        self,
        cur: Any,
        namespace: str,
        identity: str,
    ) -> None:
        """Serialize one bounded verified-work identity for this transaction."""

        await cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 449))",
            (f"cayu:verified-work:{namespace}:{identity}",),
        )

    async def _lock_verified_work_task(self, cur: Any, task_id: str) -> None:
        """Give every mutation for one task the same first authority lock."""

        await self._lock_verified_work_identity(cur, "task", task_id)

    async def _verified_now(self, cur: Any) -> datetime:
        if self._clock_is_injected:
            return self._clock()
        return await self._database_now(cur)

    async def _load_work_contract_row(
        self,
        cur: Any,
        reference: WorkContractRef,
        *,
        for_update: bool = False,
    ) -> WorkContract | None:
        await cur.execute(
            "SELECT contract_id, version, fingerprint, contract_json "
            "FROM cayu_work_contracts WHERE contract_id = %s AND version = %s"
            + (" FOR UPDATE" if for_update else ""),
            (reference.contract_id, reference.version),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        contract = WorkContract.model_validate(_json_document(row[3]))
        if (
            contract.contract_id != row[0]
            or contract.version != row[1]
            or contract.fingerprint != row[2]
        ):
            raise WorkContractConflict(
                "Stored work-contract indexes conflict with canonical content."
            )
        return verified_work_support.require_contract_reference(contract, reference)

    async def _load_task_locked(self, cur: Any, task_id: str) -> Task:
        await cur.execute(
            f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks WHERE id = %s FOR UPDATE",
            (task_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        return pg_support.task_from_row(row)

    async def _load_attempt_row(
        self,
        cur: Any,
        attempt_id: str,
        *,
        for_update: bool = False,
    ) -> WorkAttempt | None:
        await cur.execute(
            "SELECT attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json "
            "FROM cayu_work_attempts WHERE attempt_id = %s" + (" FOR UPDATE" if for_update else ""),
            (attempt_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        attempt = WorkAttempt.model_validate(_json_document(row[5]))
        if (
            attempt.attempt_id != row[0]
            or attempt.task_id != row[1]
            or attempt.ordinal != row[2]
            or attempt.request_sha256 != row[3]
            or attempt.started_at != pg_support.to_utc(row[4])
        ):
            raise WorkCompletionConflict(
                "Stored work-attempt indexes conflict with canonical content."
            )
        return attempt

    async def _latest_attempt_id(self, cur: Any, task_id: str) -> str | None:
        await cur.execute(
            "SELECT attempt_id FROM cayu_work_attempts "
            "WHERE task_id = %s ORDER BY ordinal DESC LIMIT 1",
            (task_id,),
        )
        row = await cur.fetchone()
        return None if row is None else row[0]

    async def _load_proposal_row(
        self,
        cur: Any,
        proposal_id: str,
        *,
        for_update: bool = False,
    ) -> CompletionProposal | None:
        await cur.execute(
            "SELECT proposal_id, attempt_id, task_id, request_sha256, proposed_at, "
            "proposal_json FROM cayu_completion_proposals WHERE proposal_id = %s"
            + (" FOR UPDATE" if for_update else ""),
            (proposal_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        proposal = CompletionProposal.model_validate(_json_document(row[5]))
        if (
            proposal.proposal_id != row[0]
            or proposal.attempt_id != row[1]
            or proposal.task_id != row[2]
            or proposal.request_sha256 != row[3]
            or proposal.proposed_at != pg_support.to_utc(row[4])
        ):
            raise WorkCompletionConflict(
                "Stored completion-proposal indexes conflict with canonical content."
            )
        return proposal

    async def _load_verifier_profile_row(
        self,
        cur: Any,
        proposal_id: str,
        *,
        for_update: bool = False,
    ) -> CompletionVerifierProfileRecord | None:
        await cur.execute(
            "SELECT proposal_id, task_id, attempt_id, profile_fingerprint, "
            "request_sha256, prepared_at, profile_json "
            "FROM cayu_completion_verifier_profiles WHERE proposal_id = %s"
            + (" FOR UPDATE" if for_update else ""),
            (proposal_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        profile = completion_verifier_profile_record_from_document(_json_document(row[6]))
        if (
            profile.proposal_id != row[0]
            or profile.task_id != row[1]
            or profile.attempt_id != row[2]
            or profile.profile.fingerprint != row[3]
            or profile.request_sha256 != row[4]
            or profile.prepared_at != pg_support.to_utc(row[5])
        ):
            raise WorkCompletionConflict(
                "Stored completion-verifier profile indexes conflict with canonical content."
            )
        return profile

    async def _load_verifier_profile_adoption_row(
        self,
        cur: Any,
        *,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionVerifierProfileRecord | None:
        await cur.execute(
            "SELECT proposal_id FROM cayu_completion_verifier_profiles "
            "WHERE task_id = %s "
            "AND profile_json #>> '{adoption,idempotency_key}' = %s "
            "ORDER BY proposal_id LIMIT 2",
            (task_id, idempotency_key),
        )
        rows = await cur.fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise WorkCompletionConflict(
                "Stored completion-verifier adoption idempotency authority is ambiguous."
            )
        profile = await self._load_verifier_profile_row(cur, rows[0][0], for_update=True)
        if (
            profile is None
            or profile.task_id != task_id
            or profile.adoption is None
            or profile.adoption.idempotency_key != idempotency_key
        ):
            raise WorkCompletionConflict(
                "Stored completion-verifier adoption idempotency authority is invalid."
            )
        return profile

    async def _load_prior_verifier_profile_row(
        self,
        cur: Any,
        proposal: CompletionProposal,
        *,
        for_update: bool = False,
    ) -> CompletionVerifierProfileRecord | None:
        await cur.execute(
            "SELECT prior_proposal.proposal_id "
            "FROM cayu_work_attempts AS current_attempt "
            "JOIN cayu_work_attempts AS prior_attempt "
            "ON prior_attempt.task_id = current_attempt.task_id "
            "AND prior_attempt.ordinal = current_attempt.ordinal - 1 "
            "LEFT JOIN cayu_completion_proposals AS prior_proposal "
            "ON prior_proposal.attempt_id = prior_attempt.attempt_id "
            "WHERE current_attempt.attempt_id = %s",
            (proposal.attempt_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        if row[0] is None:
            raise WorkCompletionConflict("Prior work attempt has no completion proposal authority.")
        profile = await self._load_verifier_profile_row(
            cur,
            row[0],
            for_update=for_update,
        )
        if profile is None:
            raise WorkCompletionConflict("Prior work attempt has no verifier-profile authority.")
        return profile

    @staticmethod
    def _claim_from_row(row: Any) -> CompletionVerificationClaim:
        claim = CompletionVerificationClaim.model_validate(_json_document(row[6]))
        if (
            claim.claim_id != row[0]
            or claim.proposal_id != row[1]
            or claim.attempt_number != row[2]
            or claim.verifier_profile_fingerprint != row[3]
            or claim.request_sha256 != row[4]
            or claim.lease_expires_at != pg_support.to_utc(row[5])
        ):
            raise WorkCompletionConflict(
                "Stored verification-claim indexes conflict with canonical content."
            )
        return claim

    async def _load_current_claim(
        self,
        cur: Any,
        proposal_id: str,
        *,
        for_update: bool = False,
    ) -> CompletionVerificationClaim | None:
        await cur.execute(
            "SELECT claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, request_sha256, "
            "lease_expires_at, claim_json FROM cayu_completion_verification_claims "
            "WHERE proposal_id = %s AND is_current" + (" FOR UPDATE" if for_update else ""),
            (proposal_id,),
        )
        row = await cur.fetchone()
        return None if row is None else self._claim_from_row(row)

    async def _load_claim_by_id(
        self,
        cur: Any,
        claim_id: str,
        *,
        for_update: bool = False,
    ) -> CompletionVerificationClaim | None:
        await cur.execute(
            "SELECT claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, request_sha256, "
            "lease_expires_at, claim_json FROM cayu_completion_verification_claims "
            "WHERE claim_id = %s" + (" FOR UPDATE" if for_update else ""),
            (claim_id,),
        )
        row = await cur.fetchone()
        return None if row is None else self._claim_from_row(row)

    @staticmethod
    def _decision_from_row(row: Any) -> CompletionDecision:
        decision = CompletionDecision.model_validate(_json_document(row[10]))
        if (
            decision.decision_id != row[0]
            or decision.proposal_id != row[1]
            or decision.task_id != row[2]
            or decision.attempt_id != row[3]
            or decision.claim_id != row[4]
            or decision.verifier_profile_fingerprint != row[5]
            or decision.verdict.value != row[6]
            or decision.gap_fingerprint != row[7]
            or decision.request_sha256 != row[8]
            or decision.decided_at != pg_support.to_utc(row[9])
        ):
            raise WorkCompletionConflict(
                "Stored completion-decision indexes conflict with canonical content."
            )
        return decision

    async def _load_decision_row(
        self,
        cur: Any,
        decision_id: str,
        *,
        for_update: bool = False,
    ) -> CompletionDecision | None:
        await cur.execute(
            "SELECT decision_id, proposal_id, task_id, attempt_id, claim_id, "
            "verifier_profile_fingerprint, verdict, "
            "gap_fingerprint, request_sha256, decided_at, decision_json "
            "FROM cayu_completion_decisions WHERE decision_id = %s"
            + (" FOR UPDATE" if for_update else ""),
            (decision_id,),
        )
        row = await cur.fetchone()
        return None if row is None else self._decision_from_row(row)

    async def _load_decision_for_proposal(
        self,
        cur: Any,
        proposal_id: str,
        *,
        for_update: bool = False,
    ) -> CompletionDecision | None:
        await cur.execute(
            "SELECT decision_id, proposal_id, task_id, attempt_id, claim_id, "
            "verifier_profile_fingerprint, verdict, "
            "gap_fingerprint, request_sha256, decided_at, decision_json "
            "FROM cayu_completion_decisions WHERE proposal_id = %s"
            + (" FOR UPDATE" if for_update else ""),
            (proposal_id,),
        )
        row = await cur.fetchone()
        return None if row is None else self._decision_from_row(row)

    async def _load_application_receipt(
        self,
        cur: Any,
        task_id: str,
        idempotency_key: str,
        *,
        for_update: bool = False,
    ) -> CompletionDecisionApplicationReceipt | None:
        await cur.execute(
            "SELECT task_id, idempotency_key, decision_id, request_sha256, applied_at, "
            "receipt_json FROM cayu_completion_decision_application_receipts "
            "WHERE task_id = %s AND idempotency_key = %s" + (" FOR UPDATE" if for_update else ""),
            (task_id, idempotency_key),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        receipt = CompletionDecisionApplicationReceipt.model_validate(_json_document(row[5]))
        if (
            receipt.task_id != row[0]
            or receipt.idempotency_key != row[1]
            or receipt.decision_id != row[2]
            or receipt.request_sha256 != row[3]
            or receipt.applied_at != pg_support.to_utc(row[4])
        ):
            raise WorkCompletionConflict(
                "Stored decision-application receipt indexes conflict with canonical content."
            )
        return receipt

    async def _ensure_session_authority(
        self,
        cur: Any,
        session_id: str,
        authority_kind: Literal["ordinary", "contracted"],
    ) -> None:
        now = await self._database_now(cur)
        await cur.execute(
            "INSERT INTO cayu_task_session_execution_authority "
            "(session_id, authority_kind, committed_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (session_id) DO NOTHING",
            (session_id, authority_kind, now),
        )
        await cur.execute(
            "SELECT authority_kind FROM cayu_task_session_execution_authority "
            "WHERE session_id = %s FOR UPDATE",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise TaskTopologyInconsistent("Session execution authority was not persisted.")
        if row[0] != authority_kind:
            if authority_kind == "ordinary":
                raise TaskCompletionDecisionRequired(
                    "Contracted tasks require the verifier-aware execution entrance."
                )
            raise WorkCompletionConflict(
                "Work-contract attachment conflicts with prior ordinary session execution."
            )

    async def _require_task_contract(
        self,
        cur: Any,
        task: Task,
        reference: WorkContractRef,
    ) -> WorkContract:
        contract = await self._load_work_contract_row(cur, reference, for_update=True)
        return verified_work_support.require_task_contract(task, reference, contract)

    async def _update_task_snapshot(self, cur: Any, task: Task) -> None:
        if task.work_contract is not None:
            task = copy_task(task)
        await cur.execute(
            """
            UPDATE cayu_tasks
            SET status = %s, session_id = %s, session_instance_id = %s,
                worker_id = %s, lease_expires_at = %s,
                status_reason = %s, status_payload = %s, result = %s, error = %s,
                updated_at = %s, started_at = %s, completed_at = %s, retry_series = %s,
                work_contract = %s
            WHERE id = %s
            """,
            (
                str(task.status),
                task.session_id,
                task.session_instance_id,
                task.worker_id,
                pg_support.to_utc_optional(task.lease_expires_at),
                task.status_reason,
                None if task.status_payload is None else json.dumps(task.status_payload),
                None if task.result is None else json.dumps(task.result),
                None if task.error is None else json.dumps(task.error),
                pg_support.to_utc(task.updated_at),
                pg_support.to_utc_optional(task.started_at),
                pg_support.to_utc_optional(task.completed_at),
                None
                if task.retry_series is None
                else json.dumps(task.retry_series.model_dump(mode="json")),
                None
                if task.work_contract is None
                else json.dumps(task.work_contract.model_dump(mode="json", warnings=False)),
                task.id,
            ),
        )
        if cur.rowcount != 1:
            raise KeyError(f"Task not found: {task.id}")

    async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
        contract = copy_work_contract(contract)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> WorkContract:
            del conn
            await self._lock_verified_work_identity(
                cur,
                "contract",
                contract.contract_id,
            )
            await cur.execute(
                "SELECT contract_id, version, fingerprint, contract_json "
                "FROM cayu_work_contracts WHERE contract_id = %s AND version = %s "
                "FOR UPDATE",
                (contract.contract_id, contract.version),
            )
            row = await cur.fetchone()
            if row is not None:
                existing = WorkContract.model_validate(_json_document(row[3]))
                if (
                    existing.contract_id != row[0]
                    or existing.version != row[1]
                    or existing.fingerprint != row[2]
                    or existing != contract
                ):
                    raise WorkContractConflict(
                        "Work-contract identity is already bound to different content."
                    )
                return copy_work_contract(existing)
            if contract.supersedes is not None:
                predecessor = await self._load_work_contract_row(
                    cur,
                    contract.supersedes,
                    for_update=True,
                )
                verified_work_support.require_contract_reference(
                    predecessor,
                    contract.supersedes,
                )
            await cur.execute(
                "INSERT INTO cayu_work_contracts "
                "(contract_id, version, fingerprint, contract_json) VALUES (%s, %s, %s, %s)",
                (
                    contract.contract_id,
                    contract.version,
                    contract.fingerprint,
                    json.dumps(contract.model_dump(mode="json", warnings=False)),
                ),
            )
            return copy_work_contract(contract)

        return await self._run_verified_work_mutation(operation)

    async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
        copied = copy_work_contract_ref(reference)
        if copied is None:
            raise TypeError("reference must be a WorkContractRef.")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            contract = await self._load_work_contract_row(cur, copied)
            return None if contract is None else copy_work_contract(contract)

    async def load_active_work_contract_task_for_session(
        self,
        session_id: str,
    ) -> Task | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT authority_kind FROM cayu_task_session_execution_authority "
                "WHERE session_id = %s",
                (session_id,),
            )
            authority = await cur.fetchone()
            if authority is None or authority[0] == "ordinary":
                return None
            await cur.execute(
                f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks "
                "WHERE session_id = %s AND work_contract IS NOT NULL "
                'ORDER BY created_at, id COLLATE "C" LIMIT 1',
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise TaskTopologyInconsistent(
                    "Contracted session authority has no matching durable task."
                )
            return pg_support.task_from_row(row)

    async def admit_ordinary_session_execution(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> None:
            del conn
            await self._ensure_session_authority(cur, session_id, "ordinary")

        await self._run_verified_work_mutation(operation)

    async def hold_claimed_work_contract_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        contract: WorkContractRef,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        copied_contract = copy_work_contract_ref(contract)
        if copied_contract is None:
            raise TypeError("contract must be a WorkContractRef.")
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> Task:
            del conn
            await self._lock_verified_work_task(cur, task_id)
            task = await self._load_task_locked(cur, task_id)
            now = await self._database_now(cur)
            _ensure_owned_active_task_lease(task, worker_id, now=now)
            if task.status is not TaskStatus.CLAIMED or task.session_id is not None:
                raise TaskClaimLost("Only the current worker may park its unattached claimed task.")
            await self._require_task_contract(cur, task, copied_contract)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.NEEDS_ATTENTION,
                    "status_reason": "verified_work_contract_runner_required",
                    "status_payload": {
                        "contract_id": copied_contract.contract_id,
                        "contract_version": copied_contract.version,
                    },
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            await self._update_task_snapshot(cur, updated)
            return updated.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def begin_work_attempt(self, request: WorkAttemptCreate) -> WorkAttempt:
        request = copy_work_attempt_create(request)
        request_sha256 = work_attempt_request_sha256(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> WorkAttempt:
            del conn
            await self._lock_verified_work_identity(cur, "attempt", request.attempt_id)
            existing = await self._load_attempt_row(cur, request.attempt_id, for_update=True)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Work-attempt identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            await self._lock_verified_work_task(cur, request.task_id)
            task = await self._load_task_locked(cur, request.task_id)
            contract = await self._require_task_contract(cur, task, request.contract)
            if task.status is not TaskStatus.RUNNING:
                raise ValueError("Work attempts require a running contracted task.")
            if task.session_id != request.session_id:
                raise WorkCompletionConflict("Work attempt is bound to a different task session.")
            lease_now = await self._database_now(cur)
            verified_work_support.require_attempt_worker(
                task,
                request.worker_id,
                now=lease_now,
            )
            prior_id = await self._latest_attempt_id(cur, task.id)
            prior = (
                None
                if prior_id is None
                else await self._load_attempt_row(cur, prior_id, for_update=True)
            )
            ordinal = 1 if prior is None else prior.ordinal + 1
            if ordinal > contract.continuation_policy.max_attempts:
                raise WorkCompletionConflict(
                    "Work-contract attempt limit forbids another work attempt."
                )
            if prior is not None:
                await cur.execute(
                    "SELECT decision.decision_id, receipt.decision_id "
                    "FROM cayu_completion_proposals AS proposal "
                    "LEFT JOIN cayu_completion_decisions AS decision "
                    "ON decision.proposal_id = proposal.proposal_id "
                    "LEFT JOIN cayu_completion_decision_application_receipts AS receipt "
                    "ON receipt.decision_id = decision.decision_id "
                    "WHERE proposal.attempt_id = %s",
                    (prior.attempt_id,),
                )
                row = await cur.fetchone()
                if row is None or row[0] is None:
                    raise WorkCompletionConflict(
                        "A prior work attempt has not reached a durable decision."
                    )
                if row[1] is None:
                    raise WorkCompletionConflict(
                        "A prior verifier decision has not reached durable task application."
                    )
            attempt = WorkAttempt(
                attempt_id=request.attempt_id,
                task_id=request.task_id,
                session_id=request.session_id,
                contract=request.contract,
                execution_profile_fingerprint=request.execution_profile_fingerprint,
                worker_id=request.worker_id,
                ordinal=ordinal,
                request_sha256=request_sha256,
                started_at=await self._verified_now(cur),
            )
            await cur.execute(
                "INSERT INTO cayu_work_attempts "
                "(attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    attempt.attempt_id,
                    attempt.task_id,
                    attempt.ordinal,
                    attempt.request_sha256,
                    attempt.started_at,
                    json.dumps(attempt.model_dump(mode="json", warnings=False)),
                ),
            )
            return attempt.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def load_work_attempt(self, attempt_id: str) -> WorkAttempt | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            attempt = await self._load_attempt_row(cur, attempt_id)
            return None if attempt is None else attempt.model_copy(deep=True)

    async def submit_completion_proposal(
        self,
        request: CompletionProposalCreate,
    ) -> CompletionProposal:
        request = copy_completion_proposal_create(request)
        request_sha256 = completion_proposal_request_sha256(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> CompletionProposal:
            del conn
            await self._lock_verified_work_identity(cur, "proposal", request.proposal_id)
            existing = await self._load_proposal_row(cur, request.proposal_id, for_update=True)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Completion-proposal identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            attempt_snapshot = await self._load_attempt_row(cur, request.attempt_id)
            if attempt_snapshot is None:
                raise KeyError(f"Work attempt not found: {request.attempt_id}")
            await self._lock_verified_work_task(cur, attempt_snapshot.task_id)
            attempt = await self._load_attempt_row(cur, request.attempt_id, for_update=True)
            if attempt is None:
                raise KeyError(f"Work attempt not found: {request.attempt_id}")
            await cur.execute(
                "SELECT proposal_id FROM cayu_completion_proposals "
                "WHERE attempt_id = %s FOR UPDATE",
                (request.attempt_id,),
            )
            if await cur.fetchone() is not None:
                raise WorkCompletionConflict(
                    "Work attempt already has a different completion proposal."
                )
            task = await self._load_task_locked(cur, attempt.task_id)
            contract = await self._load_work_contract_row(
                cur,
                attempt.contract,
                for_update=True,
            )
            verified_work_support.require_attempt_current(
                task,
                attempt,
                latest_attempt_id=await self._latest_attempt_id(cur, task.id),
                contract=contract,
                now=await self._database_now(cur),
            )
            proposal = CompletionProposal(
                proposal_id=request.proposal_id,
                attempt_id=request.attempt_id,
                result=request.result,
                evidence_references=request.evidence_references,
                task_id=attempt.task_id,
                contract=attempt.contract,
                request_sha256=request_sha256,
                proposed_at=await self._verified_now(cur),
            )
            await cur.execute(
                "INSERT INTO cayu_completion_proposals "
                "(proposal_id, attempt_id, task_id, request_sha256, proposed_at, "
                "proposal_json) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    proposal.proposal_id,
                    proposal.attempt_id,
                    proposal.task_id,
                    proposal.request_sha256,
                    proposal.proposed_at,
                    json.dumps(proposal.model_dump(mode="json", warnings=False)),
                ),
            )
            return proposal.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def load_completion_proposal(self, proposal_id: str) -> CompletionProposal | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            proposal = await self._load_proposal_row(cur, proposal_id)
            return None if proposal is None else proposal.model_copy(deep=True)

    async def prepare_completion_verifier_profile(
        self,
        request: CompletionVerifierProfilePreparationRequest,
    ) -> CompletionVerifierProfileRecord:
        request = copy_completion_verifier_profile_preparation_request(request)
        request_sha256 = completion_verifier_profile_preparation_request_sha256(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> CompletionVerifierProfileRecord:
            del conn
            await self._lock_verified_work_identity(cur, "verifier-profile", request.proposal_id)
            proposal_snapshot = await self._load_proposal_row(cur, request.proposal_id)
            if proposal_snapshot is None:
                raise KeyError(f"Completion proposal not found: {request.proposal_id}")
            await self._lock_verified_work_task(cur, proposal_snapshot.task_id)
            existing = await self._load_verifier_profile_row(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Completion-verifier profile is already bound to another request."
                    )
                return copy_completion_verifier_profile_record(existing)
            proposal = await self._load_proposal_row(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if proposal is None:
                raise KeyError(f"Completion proposal not found: {request.proposal_id}")
            attempt = await self._load_attempt_row(cur, proposal.attempt_id, for_update=True)
            if attempt is None:
                raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
            contract = verified_work_support.require_contract_reference(
                await self._load_work_contract_row(cur, proposal.contract, for_update=True),
                proposal.contract,
            )
            if (
                request.task_id != proposal.task_id
                or request.attempt_id != attempt.attempt_id
                or request.attempt_request_sha256 != attempt.request_sha256
                or request.source_execution_profile_fingerprint
                != attempt.execution_profile_fingerprint
                or request.proposal_request_sha256 != proposal.request_sha256
                or request.contract != contract.reference()
                or request.profile.verifier != contract.verifier
            ):
                raise WorkCompletionConflict(
                    "Completion-verifier profile conflicts with its durable proposal authority."
                )
            prior = await self._load_prior_verifier_profile_row(
                cur,
                proposal,
                for_update=True,
            )
            require_completion_verifier_profile_transition(request, prior)
            adoption = request.adoption
            if (
                adoption is not None
                and await self._load_verifier_profile_adoption_row(
                    cur,
                    task_id=request.task_id,
                    idempotency_key=adoption.idempotency_key,
                )
                is not None
            ):
                raise WorkCompletionConflict(
                    "Completion-verifier profile adoption idempotency key is already "
                    "bound to another proposal."
                )
            record = completion_verifier_profile_record_from_preparation(
                request,
                request_sha256=request_sha256,
                prepared_at=await self._verified_now(cur),
            )
            await cur.execute(
                "INSERT INTO cayu_completion_verifier_profiles "
                "(proposal_id, task_id, attempt_id, profile_fingerprint, request_sha256, "
                "prepared_at, profile_json) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    record.proposal_id,
                    record.task_id,
                    record.attempt_id,
                    record.profile.fingerprint,
                    record.request_sha256,
                    record.prepared_at,
                    json.dumps(record.model_dump(mode="json", warnings=False)),
                ),
            )
            return copy_completion_verifier_profile_record(record)

        return await self._run_verified_work_mutation(operation)

    async def load_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            profile = await self._load_verifier_profile_row(cur, proposal_id)
            return None if profile is None else copy_completion_verifier_profile_record(profile)

    async def load_prior_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            proposal = await self._load_proposal_row(cur, proposal_id)
            if proposal is None:
                raise KeyError(f"Completion proposal not found: {proposal_id}")
            profile = await self._load_prior_verifier_profile_row(cur, proposal)
            return None if profile is None else copy_completion_verifier_profile_record(profile)

    async def claim_completion_verification(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> CompletionVerificationClaim:
            del conn
            await self._lock_verified_work_identity(cur, "claim", request.claim_id)
            claim_by_id = await self._load_claim_by_id(
                cur,
                request.claim_id,
                for_update=True,
            )
            if claim_by_id is not None and (
                claim_by_id.proposal_id != request.proposal_id
                or claim_by_id.request_sha256 != request_sha256
            ):
                raise WorkCompletionConflict(
                    "Verification-claim identity is already bound to another request."
                )
            proposal_snapshot = await self._load_proposal_row(cur, request.proposal_id)
            if proposal_snapshot is None:
                raise KeyError(f"Completion proposal not found: {request.proposal_id}")
            await self._lock_verified_work_task(cur, proposal_snapshot.task_id)
            proposal = await self._load_proposal_row(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if proposal is None:
                raise KeyError(f"Completion proposal not found: {request.proposal_id}")
            contract = await self._load_work_contract_row(
                cur,
                proposal.contract,
                for_update=True,
            )
            contract = verified_work_support.require_contract_reference(
                contract,
                proposal.contract,
            )
            if request.verifier != contract.verifier:
                raise WorkCompletionConflict(
                    "Verification claim uses a verifier other than the frozen contract verifier."
                )
            profile = await self._load_verifier_profile_row(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if (
                profile is None
                or profile.profile.fingerprint != request.verifier_profile_fingerprint
            ):
                raise WorkCompletionConflict(
                    "Verification claim requires the exact prepared verifier profile."
                )
            current = await self._load_current_claim(
                cur,
                request.proposal_id,
                for_update=True,
            )
            decision = await self._load_decision_for_proposal(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if (
                current is not None
                and current.claim_id == request.claim_id
                and current.request_sha256 == request_sha256
            ):
                replay_now = await self._verified_now(cur)
                if current.lease_expires_at > replay_now or decision is not None:
                    return current.model_copy(deep=True)
                raise CompletionVerificationClaimLost(
                    "Verification claim expired and cannot regain authority by replay."
                )
            if decision is not None:
                raise WorkCompletionConflict("Completion proposal already has a durable decision.")
            if claim_by_id is not None:
                raise CompletionVerificationClaimLost(
                    "Verification claim expired and cannot regain authority by replay."
                )
            attempt = await self._load_attempt_row(cur, proposal.attempt_id, for_update=True)
            if attempt is None:
                raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
            task = await self._load_task_locked(cur, proposal.task_id)
            verified_work_support.require_proposal_chain(
                proposal,
                attempt,
                task,
                latest_attempt_id=await self._latest_attempt_id(cur, task.id),
                contract=contract,
            )
            now = await self._verified_now(cur)
            if current is not None and current.lease_expires_at > now:
                raise CompletionVerificationClaimLost(
                    "Completion proposal is owned by another live verifier claim."
                )
            attempt_number = 1 if current is None else current.attempt_number + 1
            claim = CompletionVerificationClaim(
                claim_id=request.claim_id,
                proposal_id=request.proposal_id,
                worker_id=request.worker_id,
                execution_owner_id=request.execution_owner_id,
                execution_timeout_seconds=request.execution_timeout_seconds,
                verifier=request.verifier,
                verifier_profile_fingerprint=request.verifier_profile_fingerprint,
                attempt_number=attempt_number,
                request_sha256=request_sha256,
                claimed_at=now,
                lease_expires_at=now + timedelta(seconds=request.lease_seconds),
            )
            await cur.execute(
                "UPDATE cayu_completion_verification_claims SET is_current = FALSE "
                "WHERE proposal_id = %s AND is_current",
                (request.proposal_id,),
            )
            await cur.execute(
                "INSERT INTO cayu_completion_verification_claims "
                "(claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, "
                "request_sha256, lease_expires_at, is_current, claim_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s)",
                (
                    claim.claim_id,
                    claim.proposal_id,
                    claim.attempt_number,
                    claim.verifier_profile_fingerprint,
                    claim.request_sha256,
                    claim.lease_expires_at,
                    json.dumps(claim.model_dump(mode="json", warnings=False)),
                ),
            )
            return claim.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def load_completion_verification_claim(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            claim = await self._load_current_claim(cur, proposal_id)
            return None if claim is None else claim.model_copy(deep=True)

    async def renew_completion_verification_claim(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> CompletionVerificationClaim:
            del conn
            await self._lock_verified_work_identity(cur, "claim", request.claim_id)
            proposal_snapshot = await self._load_proposal_row(cur, request.proposal_id)
            if proposal_snapshot is None:
                raise KeyError(f"Completion proposal not found: {request.proposal_id}")
            await self._lock_verified_work_task(cur, proposal_snapshot.task_id)
            proposal = await self._load_proposal_row(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if proposal is None:
                raise KeyError(f"Completion proposal not found: {request.proposal_id}")
            current = await self._load_current_claim(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if (
                current is None
                or current.claim_id != request.claim_id
                or current.worker_id != request.worker_id
                or current.execution_owner_id != request.execution_owner_id
                or current.execution_timeout_seconds != request.execution_timeout_seconds
                or current.verifier != request.verifier
                or current.verifier_profile_fingerprint != request.verifier_profile_fingerprint
                or current.request_sha256 != request_sha256
            ):
                raise CompletionVerificationClaimLost(
                    "Verification claim cannot be renewed without exact current live authority."
                )
            decision = await self._load_decision_for_proposal(
                cur,
                proposal.proposal_id,
                for_update=True,
            )
            if decision is not None:
                raise CompletionVerificationClaimLost(
                    "Verification claim cannot be renewed without exact current live authority."
                )
            attempt = await self._load_attempt_row(cur, proposal.attempt_id, for_update=True)
            if attempt is None:
                raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
            task = await self._load_task_locked(cur, proposal.task_id)
            contract = await self._load_work_contract_row(
                cur,
                proposal.contract,
                for_update=True,
            )
            verified_work_support.require_proposal_chain(
                proposal,
                attempt,
                task,
                latest_attempt_id=await self._latest_attempt_id(cur, task.id),
                contract=contract,
            )
            now = await self._verified_now(cur)
            if current.lease_expires_at <= now:
                raise CompletionVerificationClaimLost(
                    "Verification claim cannot be renewed without exact current live authority."
                )
            renewed = current.model_copy(
                update={
                    "lease_expires_at": max(
                        current.lease_expires_at,
                        now + timedelta(seconds=request.lease_seconds),
                    )
                }
            )
            await cur.execute(
                "UPDATE cayu_completion_verification_claims "
                "SET lease_expires_at = %s, claim_json = %s "
                "WHERE claim_id = %s AND is_current",
                (
                    renewed.lease_expires_at,
                    json.dumps(renewed.model_dump(mode="json", warnings=False)),
                    renewed.claim_id,
                ),
            )
            if cur.rowcount != 1:
                raise CompletionVerificationClaimLost(
                    "Verification claim cannot be renewed without exact current live authority."
                )
            return renewed.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def record_completion_decision(
        self,
        request: CompletionDecisionCreate,
    ) -> CompletionDecision:
        request = copy_completion_decision_create(request)
        request_sha256 = completion_decision_request_sha256(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> CompletionDecision:
            del conn
            await self._lock_verified_work_identity(cur, "decision", request.decision_id)
            existing = await self._load_decision_row(cur, request.decision_id, for_update=True)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Completion-decision identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            proposal_snapshot = await self._load_proposal_row(cur, request.proposal_id)
            if proposal_snapshot is None:
                raise KeyError(f"Completion proposal not found: {request.proposal_id}")
            await self._lock_verified_work_task(cur, proposal_snapshot.task_id)
            proposal = await self._load_proposal_row(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if proposal is None:
                raise KeyError(f"Completion proposal not found: {request.proposal_id}")
            prior = await self._load_decision_for_proposal(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if prior is not None:
                raise WorkCompletionConflict(
                    "Completion proposal already has a different durable decision."
                )
            claim = await self._load_current_claim(
                cur,
                proposal.proposal_id,
                for_update=True,
            )
            profile = await self._load_verifier_profile_row(
                cur,
                request.proposal_id,
                for_update=True,
            )
            if (
                claim is None
                or claim.claim_id != request.claim_id
                or claim.worker_id != request.worker_id
                or claim.verifier != request.verifier
                or claim.verifier_profile_fingerprint != request.verifier_profile_fingerprint
                or profile is None
                or profile.profile.fingerprint != request.verifier_profile_fingerprint
            ):
                raise CompletionVerificationClaimLost(
                    "Completion decision requires the current live verifier claim."
                )
            attempt = await self._load_attempt_row(cur, proposal.attempt_id, for_update=True)
            if attempt is None:
                raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
            task = await self._load_task_locked(cur, proposal.task_id)
            contract = await self._load_work_contract_row(
                cur,
                proposal.contract,
                for_update=True,
            )
            contract = verified_work_support.require_proposal_chain(
                proposal,
                attempt,
                task,
                latest_attempt_id=await self._latest_attempt_id(cur, task.id),
                contract=contract,
            )
            validate_completion_decision_contract(contract, request)
            now = await self._verified_now(cur)
            if claim.lease_expires_at <= now:
                raise CompletionVerificationClaimLost(
                    "Completion decision requires the current live verifier claim."
                )
            decision = CompletionDecision(
                decision_id=request.decision_id,
                proposal_id=request.proposal_id,
                claim_id=request.claim_id,
                worker_id=request.worker_id,
                verifier=request.verifier,
                verifier_profile_fingerprint=request.verifier_profile_fingerprint,
                decision_version=request.decision_version,
                verdict=request.verdict,
                criterion_outcomes=request.criterion_outcomes,
                constraint_outcomes=request.constraint_outcomes,
                gaps=request.gaps,
                evidence_references=request.evidence_references,
                task_id=proposal.task_id,
                attempt_id=proposal.attempt_id,
                contract=proposal.contract,
                claim_authority_sha256=completion_verification_claim_authority_sha256(claim),
                request_sha256=request_sha256,
                gap_fingerprint=completion_gap_fingerprint(request),
                decided_at=now,
            )
            await cur.execute(
                "INSERT INTO cayu_completion_decisions "
                "(decision_id, proposal_id, task_id, attempt_id, claim_id, "
                "verifier_profile_fingerprint, verdict, "
                "gap_fingerprint, request_sha256, decided_at, decision_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    decision.decision_id,
                    decision.proposal_id,
                    decision.task_id,
                    decision.attempt_id,
                    decision.claim_id,
                    decision.verifier_profile_fingerprint,
                    decision.verdict.value,
                    decision.gap_fingerprint,
                    decision.request_sha256,
                    decision.decided_at,
                    json.dumps(decision.model_dump(mode="json", warnings=False)),
                ),
            )
            return decision.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def load_completion_decision(
        self,
        decision_id: str,
    ) -> CompletionDecision | None:
        decision_id = require_clean_nonblank(decision_id, "decision_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            decision = await self._load_decision_row(cur, decision_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def load_completion_decision_for_proposal(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            decision = await self._load_decision_for_proposal(cur, proposal_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def apply_completion_decision(
        self,
        request: CompletionDecisionApplicationRequest,
    ) -> Task:
        try:
            copied_request = copy_completion_decision_application_request(request)
        except BaseException:
            del request
            raise
        request = copied_request
        del copied_request
        request_sha256 = completion_decision_application_request_sha256(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> Task:
            del conn
            await self._lock_verified_work_identity(cur, "decision", request.decision_id)
            await cur.execute(
                "SELECT task_id, idempotency_key FROM "
                "cayu_completion_decision_application_receipts "
                "WHERE decision_id = %s FOR UPDATE",
                (request.decision_id,),
            )
            prior_receipt_key = await cur.fetchone()
            if prior_receipt_key is not None:
                prior_task_id, prior_idempotency_key = prior_receipt_key
                if (
                    prior_task_id != request.task_id
                    or prior_idempotency_key != request.idempotency_key
                ):
                    raise WorkCompletionConflict(
                        "Completion decision was already applied under another identity."
                    )
                receipt = await self._load_application_receipt(
                    cur,
                    prior_task_id,
                    prior_idempotency_key,
                    for_update=True,
                )
                if receipt is None:
                    raise WorkCompletionConflict(
                        "Completion decision application has inconsistent durable authority."
                    )
                if receipt.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Decision-application identity is already bound to another request."
                    )
                return receipt.task.model_copy(deep=True)
            await self._lock_verified_work_task(cur, request.task_id)
            task = await self._load_task_locked(cur, request.task_id)
            receipt = await self._load_application_receipt(
                cur,
                request.task_id,
                request.idempotency_key,
                for_update=True,
            )
            if receipt is not None:
                if receipt.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Decision-application identity is already bound to another request."
                    )
                return receipt.task.model_copy(deep=True)
            decision = await self._load_decision_row(
                cur,
                request.decision_id,
                for_update=True,
            )
            if decision is None:
                raise KeyError(f"Completion decision not found: {request.decision_id}")
            if decision.task_id != task.id:
                raise WorkCompletionConflict("Completion decision belongs to another task.")
            contract = await self._require_task_contract(cur, task, decision.contract)
            attempt = await self._load_attempt_row(cur, decision.attempt_id, for_update=True)
            if attempt is None:
                raise WorkCompletionConflict("Completion decision has no work attempt.")
            verified_work_support.require_decision_attempt_current(
                task,
                attempt,
                latest_attempt_id=await self._latest_attempt_id(cur, task.id),
                contract=contract,
            )
            proposal = await self._load_proposal_row(
                cur,
                decision.proposal_id,
                for_update=True,
            )
            if proposal is None:
                raise WorkCompletionConflict("Completion decision has no completion proposal.")
            profile = await self._load_verifier_profile_row(
                cur,
                proposal.proposal_id,
                for_update=True,
            )
            if (
                profile is None
                or profile.profile.fingerprint != decision.verifier_profile_fingerprint
            ):
                raise WorkCompletionConflict(
                    "Completion decision has no exact verifier-profile authority."
                )
            await cur.execute(
                "SELECT COUNT(*) FROM cayu_completion_decisions "
                "WHERE task_id = %s AND verdict = %s AND gap_fingerprint = %s",
                (task.id, CompletionVerdict.REJECTED.value, decision.gap_fingerprint),
            )
            count_row = await cur.fetchone()
            matching_gap_count = 0 if count_row is None else int(count_row[0])
            updated, receipt = verified_work_support.plan_decision_application(
                request,
                request_sha256=request_sha256,
                task=task,
                decision=decision,
                proposal=proposal,
                attempt=attempt,
                contract=contract,
                matching_gap_count=matching_gap_count,
                now=await self._database_now(cur),
            )
            if updated != task:
                await self._update_task_snapshot(cur, updated)
            await cur.execute(
                "INSERT INTO cayu_completion_decision_application_receipts "
                "(task_id, idempotency_key, decision_id, request_sha256, applied_at, "
                "receipt_json) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    receipt.task_id,
                    receipt.idempotency_key,
                    receipt.decision_id,
                    receipt.request_sha256,
                    receipt.applied_at,
                    json.dumps(receipt.model_dump(mode="json", warnings=False)),
                ),
            )
            return updated.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def load_completion_decision_application_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        idempotency_key = validate_work_completion_idempotency_key(idempotency_key)
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            receipt = await self._load_application_receipt(
                cur,
                task_id,
                idempotency_key,
            )
            return None if receipt is None else receipt.model_copy(deep=True)
