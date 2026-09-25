"""Lose a committed planning acknowledgement, then recover in another interpreter."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

from tests.core.test_participant_identity import app, registration
from tests.core.test_prepared_admission_public import PreparationResolver
from tests.core.test_request_planning_public import complete_plan

from cayu.collaboration.planning import ConfiguredRequestPlanningPolicy, RequestPlanningRequest
from cayu.collaboration.prepared_admission import prepared_budget
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode


def resource_registration(material, policy, store):
    """Rebuild trusted fixture configuration, not private receiving readbacks."""
    from tests.artifacts.test_resources import LocalArtifactResourceOwner as SelectorOwner
    from tests.artifacts.test_resources import _MandateFixtureResolver
    from tests.core.test_request_planning_resource_creation import resource_environment

    from cayu.artifacts import LocalArtifactStore
    from cayu.artifacts.resources import (
        LocalArtifactResourceOwner,
        MandateResourcePreparationReader,
    )
    from cayu.collaboration._contracts import ObjectRef
    from cayu.collaboration.mandates import MandateAccessContext, MandateResolution
    from cayu.collaboration.participants import CollaborationInitialization
    from cayu.vaults.redaction import SecretRedactor

    configured = material["resource"]
    recipes = policy.default.resources
    artifacts = LocalArtifactStore(configured["artifact_root"])
    selector = SelectorOwner(
        configured["selector_journal"],
        owner=recipes[0].acquisition.source,
        artifact_store=artifacts,
    )
    reader = MandateResourcePreparationReader(
        owner=selector,
        resolver=_MandateFixtureResolver(
            MandateResolution.model_validate(configured["resolution"])
        ),
        context=MandateAccessContext.model_validate(configured["context"]),
        redactor=SecretRedactor(),
        registration=ObjectRef.model_validate(configured["registration"]),
        policy=recipes[0].acquisition.intent.policy,
        artifact_store=artifacts,
        collaboration_store=store,
        initialized=CollaborationInitialization.model_validate(configured["initialized"]),
        responsibilities=tuple((item.acquisition, item.acquisition_permit) for item in recipes),
        transfer_templates=tuple((item.transfer, item.transfer_permit) for item in recipes),
    )
    owner = LocalArtifactResourceOwner(
        configured["owner_journal"],
        owner=reader.owner,
        artifact_store=artifacts,
        preparation_reader=reader,
    )
    if material["mode"] == "crash_after_resource_transfer":
        transfer = owner.accept_transfer

        async def crash_after_transfer(*args, **kwargs):
            await transfer(*args, **kwargs)
            os._exit(19)

        owner.accept_transfer = crash_after_transfer
    else:

        async def forbid_new_pin(*args, **kwargs):
            raise AssertionError("Restart must adopt the already accepted resource transfer.")

        artifacts._pin_resource = forbid_new_pin
    return owner, resource_environment(artifacts)


async def main():
    material = json.loads(sys.stdin.read())
    expected = RequestPlanningRequest.model_validate(material["expected"])
    resolver = PreparationResolver(
        expected.expected.intent.request,
        expected.expected.intent.selection.recipient.reference,
    )
    store = (
        SQLiteCollaborationStore(material["address"])
        if material["backend"] == "sqlite"
        else PostgresCollaborationStore(material["address"], schema_mode=SchemaMode.CREATE)
    )
    crash = material["mode"] == "crash_after_intent"
    resource = material["mode"] in {"crash_after_resource_transfer", "recover_resource"}
    fresh = resource or material["mode"] in {"crash_after_creation_commit", "recover_fresh"}
    policy = (
        ConfiguredRequestPlanningPolicy.model_validate(material["policy"])
        if crash or fresh
        else None
    )
    sessions = None
    resource_owner = environment = None
    options = {}
    if fresh:
        from cayu.storage.postgres import PostgresSessionStore
        from cayu.storage.sqlite import SQLiteSessionStore

        assert policy is not None

        class BudgetReceiver:
            async def resolve_budget_binding(self, *, request):
                return prepared_budget(policy.default.preparation.budget_binding_json)

        sessions = (
            SQLiteSessionStore(material["session_address"])
            if material["backend"] == "sqlite"
            else PostgresSessionStore(material["address"], schema_mode=SchemaMode.CREATE)
        )
        options = {
            "session_store": sessions,
            "budget_binding_receiver": BudgetReceiver(),
            "enable_common_root_budget_binding": True,
        }
    if resource:
        resource_owner, environment = resource_registration(material, policy, store)
    application = app(
        store,
        registration(
            scope=expected.operation.application_scope, limits=expected.expected.intent.limits
        ),
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            max_ttl_ms=expected.expected.intent.request.ttl_ms,
            planning_policies=(policy,) if policy is not None else (),
            resource_owners=(resource_owner,) if resource_owner is not None else (),
            **(
                {
                    "prepared_admission": PreparedAdmissionRegistration(
                        receiver=policy.default.preparation.receiver
                    )
                }
                if fresh
                else {}
            ),
        ),
        **options,
    )
    if fresh:
        from cayu.agents import AgentSpec
        from cayu.evals.testing import ScriptedModelProvider

        provider = ScriptedModelProvider((), name="provider")
        application.register_provider(provider, default=True)
        application.register_agent(
            AgentSpec(name="reviewer", model="model", system_prompt="system")
        )
        if environment is not None:
            application.register_environment(environment, default=True)
        if material["mode"] == "crash_after_creation_commit":
            original_creation = sessions.create_participant_owned_session

            async def crash_after_creation(*args, **kwargs):
                await original_creation(*args, **kwargs)
                os._exit(19)

            sessions.create_participant_owned_session = crash_after_creation
        elif not resource:

            async def forbid_recreation(*args, **kwargs):
                raise AssertionError("Restart must reconcile the original native child.")

            application.create_recipient_session = forbid_recreation
    try:
        await application.initialize_collaboration()
        if crash:
            original = store._transaction

            @asynccontextmanager
            async def crash_after_commit(scope, *, write):
                async with original(scope, write=write) as tx:
                    yield tx
                if write:
                    os._exit(19)

            store._transaction = crash_after_commit
        result = await complete_plan(
            application,
            expected,
            resolver.recipient.context,
            reconcile=material["mode"] in {"recover", "recover_fresh", "recover_resource"},
        )
        assert material["mode"] not in {
            "crash_after_intent",
            "crash_after_creation_commit",
            "crash_after_resource_transfer",
        }
        if fresh:
            assert provider.requests == []
        print(result.model_dump_json())
    finally:
        await application.drain_collaboration_requests()
        if resource_owner is not None:
            await resource_owner.drain()
        await store.close()
        if sessions is not None:
            await sessions.close()


if __name__ == "__main__":
    asyncio.run(main())
