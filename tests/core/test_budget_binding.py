from decimal import Decimal

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets import (
    BudgetBinding,
    BudgetBindingRegistrationConflict,
    BudgetLimit,
    BudgetReservation,
)
from cayu.budgets._batch import BudgetBatchMember
from cayu.budgets.base import (
    BudgetBindingAllowanceExhausted,
    BudgetReservationRecord,
    InMemoryBudgetLedger,
    _effective_budget_limits,
)
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions.outcomes import run_to_completion
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresBudgetLedger, PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.inference import AuxiliaryInferencePolicy, InferenceLimits


def _limit() -> BudgetLimit:
    return BudgetLimit(
        scope="causal",
        key="root-causal",
        max_estimated_cost=Decimal("1"),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="provider",
                    model="model",
                    match="exact",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            ),
        ),
        reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=1),
    )


def _binding(**updates: object) -> BudgetBinding:
    values: dict[str, object] = {
        "binding_id": "binding-1",
        "application_scope": "app",
        "initiator": "principal-1",
        "sponsor": "sponsor-1",
        "purpose": "model-step",
        "root_budget_id": "root-causal",
        "limits": (_limit(),),
        "ledger_owner": "ledger-1",
        "receiver_id": "receiver-1",
        "receiver_generation": 1,
        "allowance": 16,
        "retention_policy": "conservative",
        "settlement_policy": "exact-or-conservative",
        "provider_name": "provider",
        "model": "model",
    }
    values.update(updates)
    return BudgetBinding(**values)


def test_binding_digest_changes_for_each_decision_field() -> None:
    original = _binding()
    assert original.exact_match(original.model_copy(deep=True))
    for field, value in {
        "sponsor": "sponsor-2",
        "purpose": "tool-call",
        "root_budget_id": "root-2",
        "receiver_generation": 2,
        "settlement_policy": "released-only",
    }.items():
        changed = original.model_copy(update={field: value}, deep=True)
        assert changed.authority_digest != original.authority_digest
        assert not original.exact_match(changed)


def test_binding_reconstructs_from_durable_json_without_changing_authority() -> None:
    original = _binding()
    reconstructed = BudgetBinding.model_validate(original.model_dump(mode="json"))
    assert reconstructed.exact_match(original)
    assert reconstructed.authority_digest == original.authority_digest


def test_binding_rejects_duplicate_ancestors() -> None:
    with pytest.raises(ValueError):
        _binding(ancestor_budget_ids=("ancestor", "ancestor"))


def test_binding_requires_a_limit() -> None:
    with pytest.raises(ValueError):
        _binding(limits=())


def test_binding_rejects_oversized_allowance() -> None:
    from cayu._validation import MAX_DURABLE_JSON_INTEGER

    with pytest.raises(ValueError):
        _binding(allowance=MAX_DURABLE_JSON_INTEGER + 1)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_binding_allowance_is_not_consumed_by_rejected_reservation(
    backend, tmp_path, request
) -> None:
    import asyncio
    from functools import cache

    postgres_dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario() -> None:
        binding = _binding(binding_id="allowance-retry", allowance=1)
        if backend == "memory":
            ledger = InMemoryBudgetLedger()
        elif backend == "sqlite":
            ledger = SQLiteBudgetLedger(tmp_path / "allowance-retry.sqlite")
        else:
            ledger = PostgresBudgetLedger(
                postgres_dsn,
                schema_mode=SchemaMode.CREATE,
            )
        await ledger.register_budget_binding(
            binding_id=binding.binding_id,
            authority_digest=binding.authority_digest,
            allowance=binding.allowance,
        )
        limit = _effective_budget_limits((_limit(),), identity_namespace="allowance-test")[0]

        @cache
        def member(*, reservation_id: str, attempt: str, amount: str) -> BudgetBatchMember:
            return BudgetBatchMember(
                limit=limit,
                record=BudgetReservationRecord(
                    reservation_id=reservation_id,
                    budget_limit_id=limit.budget_limit_id,
                    model_step_id="mstep_" + ("1" * 32),
                    model_attempt_id=attempt,
                    scope="causal",
                    key=limit.key,
                    currency="USD",
                    session_id="session",
                    agent_name="assistant",
                    provider_name="provider",
                    model="model",
                    reserved_amount=Decimal(amount),
                ),
            )

        rejected = await ledger.reserve_batch(
            members=(
                member(
                    reservation_id="bres_" + ("1" * 32),
                    attempt="matt_" + ("1" * 32),
                    amount="2",
                ),
            ),
            binding_id=binding.binding_id,
            binding_authority_digest=binding.authority_digest,
            binding_allowance=binding.allowance,
            binding_consumption_id="matt_" + ("1" * 32),
        )
        assert rejected.failure is not None
        accepted = await ledger.reserve_batch(
            members=(
                member(
                    reservation_id="bres_" + ("2" * 32),
                    attempt="matt_" + ("2" * 32),
                    amount="0.5",
                ),
            ),
            binding_id=binding.binding_id,
            binding_authority_digest=binding.authority_digest,
            binding_allowance=binding.allowance,
            binding_consumption_id="matt_" + ("2" * 32),
        )
        assert accepted.failure is None
        assert len(accepted.records) == 1
        if backend != "memory":
            await ledger.close()
            ledger = (
                SQLiteBudgetLedger(tmp_path / "allowance-retry.sqlite")
                if backend == "sqlite"
                else PostgresBudgetLedger(postgres_dsn)
            )
        # Reopening must preserve both exact replay and exhausted allowance;
        # do not register again or seed any state on the replacement ledger.
        replay = await ledger.reserve_batch(
            members=(
                member(
                    reservation_id="bres_" + ("2" * 32),
                    attempt="matt_" + ("2" * 32),
                    amount="0.5",
                ),
            ),
            binding_id=binding.binding_id,
            binding_authority_digest=binding.authority_digest,
            binding_allowance=binding.allowance,
            binding_consumption_id="matt_" + ("2" * 32),
        )
        assert replay.failure is None
        assert replay.records == accepted.records
        with pytest.raises(BudgetBindingAllowanceExhausted):
            await ledger.reserve_batch(
                members=(
                    member(
                        reservation_id="bres_" + ("3" * 32),
                        attempt="matt_" + ("3" * 32),
                        amount="0.1",
                    ),
                ),
                binding_id=binding.binding_id,
                binding_authority_digest=binding.authority_digest,
                binding_allowance=binding.allowance,
                binding_consumption_id="matt_" + ("3" * 32),
            )
        # Monetary headroom remains: the new operation was refused by the
        # retained allowance, and must not leave a reservation behind.
        assert await ledger.load_reservation("bres_" + ("3" * 32)) is None
        if backend == "sqlite" or backend == "postgres":
            await ledger.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("missing", ["root", "ancestor"])
def test_binding_requires_reserved_root_and_ancestor_ceilings(missing) -> None:
    root = _limit().model_dump(mode="python")
    ancestor_limit = _limit().model_dump(mode="python")
    ancestor_limit["key"] = "ancestor"
    updates = {"ancestor_budget_ids": ("ancestor",)}
    if missing == "root":
        root["reservation"] = None
    else:
        ancestor_limit["reservation"] = None
    limits = [root, ancestor_limit]
    with pytest.raises(ValueError, match="reservation"):
        _binding(limits=limits, **updates)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_binding_registration_is_idempotent_and_conflicting(backend, tmp_path) -> None:
    import asyncio

    async def scenario() -> None:
        if backend == "memory":
            ledger = InMemoryBudgetLedger()
            replacement = ledger
        else:
            ledger = SQLiteBudgetLedger(tmp_path / "binding-registration.sqlite")
            replacement = SQLiteBudgetLedger(tmp_path / "binding-registration.sqlite")
        await ledger.register_budget_binding(
            binding_id="binding-1", authority_digest="a" * 64, allowance=1
        )
        await ledger.register_budget_binding(
            binding_id="binding-1", authority_digest="a" * 64, allowance=1
        )
        first_registrar = ledger if backend == "sqlite" else replacement
        results = await asyncio.gather(
            first_registrar.register_budget_binding(
                binding_id="binding-2", authority_digest="c" * 64, allowance=1
            ),
            replacement.register_budget_binding(
                binding_id="binding-2", authority_digest="c" * 64, allowance=1
            ),
        )
        assert results == [None, None]
        with pytest.raises(BudgetBindingRegistrationConflict):
            await replacement.register_budget_binding(
                binding_id="binding-1",
                authority_digest="b" * 64,
                allowance=1,
            )
        if backend == "sqlite":
            await ledger.close()
            await replacement.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_bound_public_auxiliary_settlement(backend, tmp_path, request):
    import asyncio

    bounds = InferenceLimits(max_input_tokens=1000, max_output_tokens=10, timeout_seconds=30)

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("summary",)),
        )

        async def run(self, ctx, args):
            response = await ctx.inference.invoke(
                ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                purpose="summary",
                limits=bounds,
            )
            assert response.text == "summary"
            return ToolResult(content=response.text)

    async def scenario():
        limit = _limit().model_copy(
            update={
                "reservation": BudgetReservation(max_input_tokens=1000, max_output_tokens=10),
            }
        )
        binding = _binding(binding_id="public-auxiliary", limits=(limit,))
        requests = []

        class Receiver:
            async def resolve_budget_binding(self, *, request):
                requests.append(request)
                return binding

        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", id="call", arguments={}),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
            ],
            name="provider",
        )
        session_store = None
        ledger = None
        if backend == "sqlite":
            session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
            ledger = SQLiteBudgetLedger(tmp_path / "budget.sqlite")
        elif backend == "postgres":
            postgres_dsn = request.getfixturevalue("postgres_dsn")
            session_store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
            ledger = PostgresBudgetLedger(postgres_dsn, schema_mode=SchemaMode.CREATE)
        app = CayuApp(
            session_store=session_store,
            budget_ledger=ledger,
            budget_binding_receiver=Receiver(),
            enable_common_root_budget_binding=True,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="bound-aux",
                messages=[Message.text("user", "go")],
            )
        ):
            pass
        events = await app.session_store.load_events("bound-aux")
        settled = [e for e in events if e.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED]
        assert len(settled) == 1, [
            (e.type, e.payload)
            for e in events
            if "fail" in e.type or "interrupt" in e.type or "tool.call" in e.type
        ]
        assert settled[0].payload["auxiliary_outcome"] == "completed"
        assert len(provider.requests) == 3
        assert {r["kind"] for r in requests} == {"model", "auxiliary"}
        reservations = [e for e in events if e.type is EventType.BUDGET_RESERVED]
        assert len(reservations) == 3
        records = [
            await app.budget_ledger.load_reservation(e.payload["reservation_id"])
            for e in reservations
        ]
        assert len({r.budget_limit_id for r in records}) == 1
        assert all(r.status == "reconciled" for r in records)
        assert all(
            r.settlement_event_payload["budget_binding_authority_sha256"]
            == binding.authority_digest
            for r in records
        )
        assert sum((r.actual_amount for r in records), Decimal(0)) == Decimal("0.000007")
        if session_store is not None:
            await session_store.close()
        if ledger is not None:
            await ledger.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_bound_public_auxiliary_concurrency_rejects_shared_ceiling(backend, tmp_path, request):
    import asyncio

    bounds = InferenceLimits(max_input_tokens=1000, max_output_tokens=10, timeout_seconds=30)

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("summary",)),
        )

        async def run(self, ctx, args):
            response = await ctx.inference.invoke(
                ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                purpose="summary",
                limits=bounds,
            )
            return ToolResult(content=response.text)

    async def scenario():
        # PostgreSQL specialist groups share one database across tests; keep
        # this ceiling identity isolated while preserving sharing between the
        # two concurrent sessions in this scenario.
        ceiling = _limit().model_copy(update={"key": f"root-causal-{tmp_path.name}"})
        binding = _binding(
            binding_id=f"binding-{tmp_path.name}",
            root_budget_id=ceiling.key,
            limits=(ceiling.model_copy(update={"max_estimated_cost": Decimal("0.000005")}),),
        )

        class Receiver:
            async def resolve_budget_binding(self, *, request):
                return binding

        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", id="call", arguments={}),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
                ],
            ],
            name="provider",
        )
        session_store = None
        ledger = None
        if backend == "sqlite":
            session_store = SQLiteSessionStore(tmp_path / "sessions-concurrent.sqlite")
            ledger = SQLiteBudgetLedger(tmp_path / "budget-concurrent.sqlite")
        elif backend == "postgres":
            postgres_dsn = request.getfixturevalue("postgres_dsn")
            session_store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
            ledger = PostgresBudgetLedger(postgres_dsn, schema_mode=SchemaMode.CREATE)
        app = CayuApp(
            session_store=session_store,
            budget_ledger=ledger,
            budget_binding_receiver=Receiver(),
            enable_common_root_budget_binding=True,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])

        async def run(session_id):
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "go")],
                    )
                )
            ]

        first, second = await asyncio.gather(
            run(f"bound-aux-concurrent-{tmp_path.name}-1"),
            run(f"bound-aux-concurrent-{tmp_path.name}-2"),
        )
        all_events = [
            *await app.session_store.load_events(f"bound-aux-concurrent-{tmp_path.name}-1"),
            *await app.session_store.load_events(f"bound-aux-concurrent-{tmp_path.name}-2"),
        ]
        assert first and second
        assert any(event.type is EventType.BUDGET_RESERVATION_FAILED for event in all_events)
        assert len(provider.requests) == 2
        if session_store is not None:
            await session_store.close()
        if ledger is not None:
            await ledger.close()

    asyncio.run(scenario())


def test_bound_public_auxiliary_cancellation_after_dispatch_settles() -> None:
    import asyncio

    bounds = InferenceLimits(max_input_tokens=1000, max_output_tokens=10, timeout_seconds=30)
    dispatched = asyncio.Event()
    release = asyncio.Event()

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("summary",)),
        )

        async def run(self, ctx, args):
            response = await ctx.inference.invoke(
                ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                purpose="summary",
                limits=bounds,
            )
            return ToolResult(content=response.text)

    class Provider(ScriptedModelProvider):
        name = "provider"

        def __init__(self) -> None:
            super().__init__(events=[[ModelStreamEvent.completed({})]], name="provider")
            self.calls = 0

        async def stream(self, request):
            del request
            self.calls += 1
            if self.calls == 1:
                yield ModelStreamEvent.tool_call(name="summarize", id="call", arguments={})
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            dispatched.set()
            await release.wait()
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}})

    class Receiver:
        async def resolve_budget_binding(self, *, request):
            del request
            return _binding()

    async def scenario() -> None:
        app = CayuApp(
            budget_binding_receiver=Receiver(),
            enable_common_root_budget_binding=True,
            enable_logging=False,
        )
        provider = Provider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
        task = asyncio.create_task(
            run_to_completion(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="bound-aux-cancel",
                    messages=[Message.text("user", "go")],
                ),
            )
        )
        await dispatched.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelling() == 1
        assert task.cancelled()
        active = await app.session_store.load_active_model_completion_stage("bound-aux-cancel")
        assert active is not None
        assert active.stage.state in {"completed", "in_flight"}
        assert active.stage.reservation_ids
        reservations = [
            await app.budget_ledger.load_reservation(reservation_id)
            for reservation_id in active.stage.reservation_ids
        ]
        assert all(record is not None for record in reservations)
        assert all(record.dispatch_id is not None for record in reservations)
        assert all(record.status in {"active", "reconciled"} for record in reservations)
        binding_digest = _binding().authority_digest
        assert all(
            record.settlement_event_payload["budget_binding_authority_sha256"] == binding_digest
            for record in reservations
        )
        release.set()
        from cayu.runtime import IncompleteSessionRecoveryRequest

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="bound-aux-cancel")
        )
        for _ in range(100):
            events = await app.session_store.load_events("bound-aux-cancel")
            if any(e.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED for e in events):
                break
            await asyncio.sleep(0.01)
        settled = [e for e in events if e.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED]
        assert settled
        assert settled[-1].payload["auxiliary_outcome"] in {"cancelled", "failed"}
        assert provider.calls == 2
        assert (
            await app.session_store.load_active_model_completion_stage("bound-aux-cancel") is None
        )
        recovered = [
            await app.budget_ledger.load_reservation(record.reservation_id)
            for record in reservations
        ]
        assert all(record.status == "reconciled" for record in recovered)
        assert all(record.actual_amount == record.reserved_amount for record in recovered)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_bound_public_auxiliary_acknowledgement_loss_recovers_without_redispatch(
    backend, tmp_path, request, monkeypatch
) -> None:
    import asyncio

    bounds = InferenceLimits(max_input_tokens=1000, max_output_tokens=10, timeout_seconds=30)

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("summary",)),
        )

        async def run(self, ctx, args):
            response = await ctx.inference.invoke(
                ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                purpose="summary",
                limits=bounds,
            )
            return ToolResult(content=response.text)

    async def scenario() -> None:
        binding = _binding(
            limits=(
                _limit().model_copy(
                    update={
                        "reservation": BudgetReservation(
                            max_input_tokens=1000, max_output_tokens=10
                        ),
                    }
                ),
            )
        )

        class Receiver:
            async def resolve_budget_binding(self, *, request):
                del request
                return binding

        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", id="call", arguments={}),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
            ],
            name="provider",
        )
        session_store = None
        ledger = None
        if backend == "sqlite":
            session_store = SQLiteSessionStore(tmp_path / "sessions-ack.sqlite")
            ledger = SQLiteBudgetLedger(tmp_path / "budget-ack.sqlite")
        elif backend == "postgres":
            postgres_dsn = request.getfixturevalue("postgres_dsn")
            session_store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
            ledger = PostgresBudgetLedger(postgres_dsn, schema_mode=SchemaMode.CREATE)
        app = CayuApp(
            session_store=session_store,
            budget_ledger=ledger,
            budget_binding_receiver=Receiver(),
            enable_common_root_budget_binding=True,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
        complete = app.session_store.complete_model_completion_stage
        lost = False

        async def lose_ack(session_id, *, stage_id, publication):
            nonlocal lost
            result = await complete(session_id, stage_id=stage_id, publication=publication)
            if not lost and publication.events[0].type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED:
                lost = True
                raise RuntimeError("auxiliary settlement acknowledgement lost")
            return result

        monkeypatch.setattr(app.session_store, "complete_model_completion_stage", lose_ack)
        await run_to_completion(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="bound-aux-ack",
                messages=[Message.text("user", "go")],
            ),
        )
        assert lost
        from cayu.runtime import IncompleteSessionRecoveryRequest

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="bound-aux-ack")
        )
        stored = await app.session_store.load_events("bound-aux-ack")
        settled = [e for e in stored if e.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED]
        assert len(settled) == 1
        assert settled[0].payload["auxiliary_outcome"] == "completed"
        assert len(provider.requests) == 2
        if session_store is not None:
            await session_store.close()
        if ledger is not None:
            await ledger.close()

    asyncio.run(scenario())


def test_common_root_binding_is_explicitly_opt_in() -> None:
    with pytest.raises(ValueError, match="trusted budget binding receiver"):
        CayuApp(enable_common_root_budget_binding=True)


def test_application_defensively_resolves_trusted_binding() -> None:
    class Receiver:
        async def resolve_budget_binding(self, *, request: object) -> BudgetBinding:
            assert request == {"operation": "test"}
            return _binding()

    async def scenario() -> None:
        app = CayuApp(
            budget_binding_receiver=Receiver(),
            enable_common_root_budget_binding=True,
            enable_logging=False,
        )
        resolved = await app.resolve_budget_binding(request={"operation": "test"})
        assert resolved.exact_match(_binding())
        assert resolved is not app.budget_binding_receiver

    import asyncio

    asyncio.run(scenario())


def test_application_accepts_explicit_register_boundary() -> None:
    class Receiver:
        async def register(self, *, request: object) -> BudgetBinding:
            assert request == {"operation": "register"}
            return _binding()

    async def scenario() -> None:
        app = CayuApp(
            budget_binding_receiver=Receiver(),
            enable_common_root_budget_binding=True,
            enable_logging=False,
        )
        resolved = await app.resolve_budget_binding(request={"operation": "register"})
        assert resolved.exact_match(_binding())

    import asyncio

    asyncio.run(scenario())


def test_public_run_uses_atomic_bound_model_admission() -> None:
    class Provider(ModelProvider):
        name = "provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            del request
            self.calls += 1
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1}}
            )

    class Receiver:
        async def resolve_budget_binding(self, *, request: object) -> BudgetBinding:
            assert request["kind"] == "model"
            return _binding()

    async def scenario() -> int:
        provider = Provider()
        app = CayuApp(
            budget_binding_receiver=Receiver(),
            enable_common_root_budget_binding=True,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"))
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="bound-public-run",
                    messages=[Message.text("user", "hello")],
                )
            )
        ]
        assert events
        return provider.calls

    import asyncio

    assert asyncio.run(scenario()) == 1


def test_public_bound_runs_share_one_atomic_ceiling() -> None:
    class Provider(ModelProvider):
        name = "provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            del request
            self.calls += 1
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1}}
            )

    class Receiver:
        async def resolve_budget_binding(self, *, request: object) -> BudgetBinding:
            del request
            return _binding(
                limits=(_limit().model_copy(update={"max_estimated_cost": Decimal("0.0000025")}),)
            )

    async def scenario() -> int:
        provider = Provider()
        app = CayuApp(
            budget_binding_receiver=Receiver(),
            enable_common_root_budget_binding=True,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"))

        async def run(session_id: str) -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            ):
                pass

        await asyncio.gather(run("bound-concurrent-1"), run("bound-concurrent-2"))
        return provider.calls

    import asyncio

    assert asyncio.run(scenario()) == 1
