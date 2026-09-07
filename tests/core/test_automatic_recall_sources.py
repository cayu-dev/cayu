from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from itertools import repeat

import pytest

from cayu import (
    AgentSpec,
    AutomaticRecallContextPolicy,
    AutomaticRecallSourceConfig,
    AutomaticRecallSourceContext,
    AutomaticRecallSourceDescriptor,
    AutomaticRecallSourceRegistration,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    MemoryDeltaPolicy,
    Message,
    ModelStreamEvent,
    RecallEngineConfig,
    RequestFootprintConfig,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.memory_evidence import ContextExposureState, RecallEvidenceQuery
from cayu.recall import KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL, RecallSituation
from cayu.runtime._memory_evidence import MemoryEvidenceKey, memory_evidence_key_scope
from cayu.runtime.context import ContextBuildError


def _descriptor(**updates):
    return AutomaticRecallSourceDescriptor(
        **{
            "name": "enterprise",
            "channel_names": ("enterprise_lexical",),
            "configuration_version": "enterprise-v1",
            "candidate_limit": 2,
            **updates,
        }
    )


def _source(**updates):
    from test_recall import _StaticRecallSource

    return _StaticRecallSource(
        **{
            "name": "enterprise",
            "channel": "enterprise_lexical",
            "record_id": "enterprise-1",
            **updates,
        }
    )


def _registration(factory=None, **updates):
    if factory is None:

        async def factory(context):
            return _source()

    return AutomaticRecallSourceRegistration(descriptor=_descriptor(**updates), factory=factory)


def _policy(*registrations, **kwargs):
    from test_automatic_recall_context import _admission, _fusion

    policy_type = kwargs.pop("policy_type", AutomaticRecallContextPolicy)
    return policy_type(
        admission_policy=_admission(),
        fusion_config=_fusion(
            KNOWLEDGE_LEXICAL_CHANNEL,
            KNOWLEDGE_SEMANTIC_CHANNEL,
            *(channel for item in registrations for channel in item.descriptor.channel_names),
        ),
        sources=AutomaticRecallSourceConfig(
            include_transcript=False, knowledge_namespace="project:cayu"
        ),
        custom_sources=registrations,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _evidence_key():
    with memory_evidence_key_scope(MemoryEvidenceKey(key_id="test-memory-key", key=b"m" * 32)):
        yield


@pytest.mark.parametrize(
    "updates",
    [
        {"name": " "},
        {"name": "é" * 129},
        {"configuration_version": ""},
        {"configuration_version": "v" * 257},
        {"channel_names": ()},
        {"channel_names": "channel"},
        {"channel_names": ("duplicate", "duplicate")},
        {"channel_names": ("é" * 129,)},
        {"channel_names": tuple(str(i) for i in range(101))},
        {"continuation_channels": ("undeclared",)},
        {"required": 1},
        {"candidate_limit": True},
        {"candidate_limit": 0},
        {"candidate_limit": 101},
    ],
)
def test_descriptor_rejects_invalid_or_unbounded_declarations(updates):
    with pytest.raises(ValueError):
        _descriptor(**updates)


def test_registration_detaches_descriptor_and_revalidates_unchecked_copies():
    channels = ["enterprise_lexical"]
    descriptor = _descriptor(channel_names=channels)
    channels.append("later")
    registration = AutomaticRecallSourceRegistration(
        descriptor=descriptor, factory=_registration().factory
    )
    assert descriptor.channel_names == ("enterprise_lexical",)
    assert registration.descriptor is not descriptor
    with pytest.raises(FrozenInstanceError):
        registration.factory = None
    with pytest.raises(ValueError):
        AutomaticRecallSourceRegistration(
            descriptor=descriptor.model_copy(update={"candidate_limit": True}),
            factory=registration.factory,
        )
    with pytest.raises(TypeError):
        AutomaticRecallSourceRegistration(descriptor=descriptor, factory=None)


@pytest.mark.parametrize(
    "registrations,match",
    [
        ((_registration(), _registration()), "names must be unique"),
        ((_registration(name="other"), _registration()), "globally unique"),
        ((_registration(name="knowledge"),), "reserved built-in"),
        ((_registration(channel_names=(KNOWLEDGE_LEXICAL_CHANNEL,)),), "reserved built-in"),
        ((_registration(candidate_limit=21),), "fusion ceiling"),
    ],
)
def test_policy_validates_global_contract_without_constructing_sources(registrations, match):
    with pytest.raises(ValueError, match=match):
        _policy(*registrations)


def test_policy_rejects_unbounded_registration_input_and_unqualified_deltas():
    from test_automatic_recall_context import _admission, _fusion

    with pytest.raises(ValueError, match="32 custom sources"):
        AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL),
            custom_sources=repeat(_registration()),
        )
    with pytest.raises(ValueError, match="do not support memory deltas"):
        _policy(_registration(), delta_policy=MemoryDeltaPolicy())


@pytest.mark.parametrize(
    "updates", [{"name": "knowledge"}, {"channel_names": (KNOWLEDGE_LEXICAL_CHANNEL,)}]
)
def test_disabled_builtin_names_and_channels_remain_reserved(updates):
    from test_automatic_recall_context import _admission, _fusion

    from cayu.recall import TRANSCRIPT_LEXICAL_CHANNEL

    registration = _registration(**updates)
    with pytest.raises(ValueError, match="reserved built-in"):
        AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                TRANSCRIPT_LEXICAL_CHANNEL, *registration.descriptor.channel_names
            ),
            sources=AutomaticRecallSourceConfig(include_knowledge=False, knowledge_required=False),
            custom_sources=(registration,),
        )


def test_configuration_is_order_independent_versioned_and_preserves_builtin_identity():
    a = _registration()
    b = _registration(name="other", channel_names=("other_lexical",))
    first = _policy(a, b)
    assert first.configuration_fingerprint() == _policy(b, a).configuration_fingerprint()
    assert (
        first.configuration_fingerprint()
        != _policy(
            _registration(configuration_version="enterprise-v2"), b
        ).configuration_fingerprint()
    )
    assert "factory" not in str(first.configuration_material())
    assert "custom_sources" not in _policy().configuration_material()


def test_native_checkpoint_freezes_custom_source_and_rebuilds_for_changed_version():
    from test_automatic_recall_context import _fixture, _manifest, _request

    async def run():
        sessions, knowledge, session, messages = await _fixture()
        contexts = []

        async def factory(context):
            contexts.append(context)
            return _source()

        request = _request(
            sessions=sessions, knowledge=knowledge, session=session, messages=messages
        )
        policy = _policy(_registration(factory))
        first = await policy.build_with_checkpoint(request, checkpoint=None)
        second = await policy.build_with_checkpoint(request, checkpoint=first.checkpoint)
        assert len(contexts) == 1
        assert _manifest(first) == _manifest(second)
        assert "enterprise evidence" in _manifest(first)
        assert "Friday" in _manifest(first)  # Both built-in and custom retrieval survive.
        context = contexts[0]
        assert type(context) is AutomaticRecallSourceContext
        assert context.session_id == session.id
        assert context.knowledge_store is knowledge
        assert context.session_store is sessions
        assert (
            context.knowledge_access_scope
            == RecallSituation(
                query="scope comparison", knowledge_access_scope=knowledge.bound_access_scope()
            ).knowledge_access_scope
        )
        assert context.knowledge_namespace == "project:cayu"
        assert not hasattr(context, "messages")
        with pytest.raises(FrozenInstanceError):
            context.session_id = "other-session"
        changed = _policy(_registration(factory, configuration_version="enterprise-v2"))
        await changed.build_with_checkpoint(request, checkpoint=second.checkpoint)
        assert len(contexts) == 2
        receipts = (
            await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=session.id))
        ).items
        assert len(receipts) == 2
        assert (
            receipts[0].source_configuration_fingerprint
            != receipts[1].source_configuration_fingerprint
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other"),
        ("channel_names", ("other",)),
        ("channel_names", ["enterprise_lexical"]),
        ("continuation_channels", ("enterprise_lexical",)),
        ("required", False),
        ("candidate_limit", True),
        ("candidate_limit", 3),
    ],
)
def test_factory_cannot_change_declared_metadata_before_retrieval(field, value):
    from test_automatic_recall_context import _fixture, _request

    async def run():
        sessions, knowledge, session, messages = await _fixture()

        async def factory(context):
            source = _source()
            setattr(source, field, value)

            async def must_not_retrieve(situation):
                pytest.fail("Mismatched metadata reached retrieval.")

            source.retrieve = must_not_retrieve
            return source

        policy = _policy(_registration(factory))
        with pytest.raises(ContextBuildError):
            await policy.build_with_checkpoint(
                _request(
                    sessions=sessions, knowledge=knowledge, session=session, messages=messages
                ),
                checkpoint=None,
            )

    asyncio.run(run())


def test_shared_policy_constructs_independent_sources_with_each_request_access_scope():
    from test_automatic_recall_context import _fixture, _request

    from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope

    async def run():
        contexts = []

        async def factory(context):
            contexts.append(context)
            return _source()

        policy = _policy(_registration(factory))

        async def build(label):
            sessions, knowledge, session, messages = await _fixture()
            knowledge = InMemoryKnowledgeStore(
                access_scope=KnowledgeAccessScope(
                    allowed_namespaces=["project:cayu"], required_labels={"tenant": label}
                )
            )
            request = _request(
                sessions=sessions, knowledge=knowledge, session=session, messages=messages
            )
            return await policy.build_with_checkpoint(request, checkpoint=None)

        await asyncio.gather(build("one"), build("two"))
        assert {
            context.knowledge_access_scope.required_labels["tenant"] for context in contexts
        } == {"one", "two"}
        assert contexts[0].session_store is not contexts[1].session_store
        assert contexts[0].knowledge_store is not contexts[1].knowledge_store

    asyncio.run(run())


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("failure", ["factory", "timeout", "metadata", "retrieve"])
def test_factory_and_retrieval_failures_obey_required_optional_contract(required, failure):
    from test_automatic_recall_context import _fixture, _request

    async def run():
        sessions, knowledge, session, messages = await _fixture()
        cancelled = False

        async def factory(context):
            nonlocal cancelled
            if failure == "factory":
                raise RuntimeError("private factory credentials must not leak")
            if failure == "timeout":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled = True
            return _source(
                required=required,
                name="mismatch" if failure == "metadata" else "enterprise",
                fail=failure == "retrieve",
            )

        policy = _policy(
            _registration(factory, required=required),
            engine_config=RecallEngineConfig(source_timeout_seconds=0.01),
        )
        request = _request(
            sessions=sessions, knowledge=knowledge, session=session, messages=messages
        )
        if required:
            with pytest.raises(ContextBuildError) as caught:
                await policy.build_with_checkpoint(request, checkpoint=None)
            assert "private factory" not in str(caught.value)
            assert not (
                await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=session.id))
            ).items
        else:
            result = await policy.build_with_checkpoint(request, checkpoint=None)
            assert "private factory" not in str(result.recall_telemetry)
            receipt = (
                await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=session.id))
            ).items[0]
            coverage = next(item for item in receipt.sources if item.source == "enterprise")
            assert coverage.failure_code == ("timeout" if failure == "timeout" else "failed")
        if failure == "timeout":
            assert cancelled

    asyncio.run(run())


def test_cancelling_factory_cannot_publish_receipt():
    from test_automatic_recall_context import _fixture, _request

    async def run():
        sessions, knowledge, session, messages = await _fixture()
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def factory(context):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        policy = _policy(_registration(factory))
        task = asyncio.create_task(
            policy.build_with_checkpoint(
                _request(
                    sessions=sessions, knowledge=knowledge, session=session, messages=messages
                ),
                checkpoint=None,
            )
        )
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert not (
            await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=session.id))
        ).items

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["deliver", "suppress", "bad_receipt", "bad_binding"])
@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite"])
def test_real_runtime_owns_custom_source_receipts_and_actual_delivery(tmp_path, mode, persistent):
    from test_automatic_recall_context import (
        _ContinueOnceBeforeStop,
        _fixture,
        _ReceiptDigestCorruptingAutomaticRecallPolicy,
        _ReceiptManifestBindingCorruptingAutomaticRecallPolicy,
        _SummarizeAndRemoveUserAnchor,
    )

    async def run():
        sessions, knowledge, _, _ = await _fixture()
        if persistent:
            from cayu.storage.sqlite import SQLiteSessionStore

            sessions = SQLiteSessionStore(tmp_path / "custom-recall.sqlite")
        calls = []

        async def factory(context):
            calls.append(context.session_id)
            return _source()

        policy = _policy(
            _registration(factory),
            base_policy=_SummarizeAndRemoveUserAnchor() if mode == "suppress" else None,
            policy_type={
                "bad_receipt": _ReceiptDigestCorruptingAutomaticRecallPolicy,
                "bad_binding": _ReceiptManifestBindingCorruptingAutomaticRecallPolicy,
            }.get(mode, AutomaticRecallContextPolicy),
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("Friday."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
                for _ in range(2)
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="custom-source-test-key-material-32",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
            loop_policies=[_ContinueOnceBeforeStop()],
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="custom-delivery",
                        messages=[Message.text("user", "Atlas enterprise release evidence?")],
                    )
                )
            ]
            assert calls == ["custom-delivery"]
            query = RecallEvidenceQuery(session_id="custom-delivery")
            receipts = (await sessions.list_recall_receipts(query)).items
            exposures = (await sessions.list_context_exposures(query)).items
            assert len(receipts) == 1
            if mode in {"bad_receipt", "bad_binding"}:
                assert events[-1].type is EventType.SESSION_FAILED
                assert not provider.requests
                assert not exposures
                return
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 2
            assert len(exposures) == 2
            assert all(item.state is ContextExposureState.COMPLETED for item in exposures)
            for exposure in exposures:
                items = await sessions.load_recall_item_exposures(
                    exposure.session_id, exposure.exposure_id
                )
                if mode == "suppress":
                    assert not items
                else:
                    assert exposure.receipt_ids == (receipts[0].receipt_id,)
                    assert len(items) >= 2
        finally:
            if persistent:
                await sessions.close()

    asyncio.run(run())
