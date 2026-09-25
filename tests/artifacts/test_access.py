from __future__ import annotations

import asyncio

import pytest
from tests.artifacts.test_aws_s3 import _S3Client
from tests.core.test_session_access import rule

from cayu.artifacts import LocalArtifactStore, S3ArtifactStore
from cayu.artifacts.access import ScopedArtifactAccess
from cayu.sessions.access import SessionAccessDenied, SessionAccessScope


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_scoped_artifacts_have_durable_immutable_classification(tmp_path, backend):
    async def run():
        client = _S3Client()
        store = (
            LocalArtifactStore(tmp_path / "files")
            if backend == "local"
            else S3ArtifactStore("bucket", client=client)
        )
        scopes = {
            name: SessionAccessScope(
                read=(rule(organization=name),),
                create=(rule(organization=name),),
                delete=(rule(organization=name),),
            )
            for name in ("acme", "other")
        }

        async def acme_policy():
            return scopes["acme"]

        async def other_policy():
            return scopes["other"]

        acme = ScopedArtifactAccess(
            store, environment_name="files", admitted=scopes["acme"], resolve=acme_policy
        )
        other = ScopedArtifactAccess(
            store, environment_name="files", admitted=scopes["other"], resolve=other_policy
        )
        first = await acme.put_bytes(
            b"private", labels={"organization": "acme"}, filename="private.txt"
        )
        second = await other.put_bytes(
            b"foreign", labels={"organization": "other"}, filename="foreign.txt"
        )
        assert dict(first.labels) == {"organization": "acme"}
        assert (await acme.read_bytes(first.id)).content == b"private"
        before = len([call for call in client.get_calls if call["Key"].endswith("/content")])
        for operation in (
            lambda: acme.read_bytes(second.id),
            lambda: acme.read_range(second.id, offset=0, max_bytes=3),
            lambda: acme.delete(second.id),
        ):
            with pytest.raises(SessionAccessDenied):
                await operation()
        after = len([call for call in client.get_calls if call["Key"].endswith("/content")])
        assert before == after
        listed = await acme.list(limit=1)
        assert [item.id for item in listed.artifacts] == [first.id]
        assert listed.total_count == 1 and not listed.truncated
        with pytest.raises(SessionAccessDenied):
            await acme.put_bytes(b"bad", labels={"organization": "other"}, filename="bad.txt")
        if backend == "local":
            reconstructed = LocalArtifactStore(tmp_path / "files")
        else:
            reconstructed = S3ArtifactStore("bucket", client=client)
        recovered = ScopedArtifactAccess(
            reconstructed, environment_name="files", admitted=scopes["acme"], resolve=acme_policy
        )
        assert (await recovered.read_range(first.id, offset=1, max_bytes=2)).content == b"ri"
        scopes["acme"] = SessionAccessScope()
        with pytest.raises(SessionAccessDenied):
            await recovered.read_bytes(first.id)
        assert not (await recovered.list()).artifacts
        assert (await other.read_bytes(second.id)).content == b"foreign"

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_model_attachment_reuse_requires_artifact_classification(tmp_path, backend):
    from cayu import CayuApp, Environment, EnvironmentSpec, Message, RunRequest, SessionIdentity
    from cayu._resource_access_binding import ResourceExecutionBinding
    from cayu.artifacts import file_attachment
    from cayu.resource_access import ResourceAccessPolicy, encode_scope, execution_access
    from cayu.runtime._model_step_executor import _resolved_file_attachments

    async def run():
        store = (
            LocalArtifactStore(tmp_path / "files")
            if backend == "local"
            else S3ArtifactStore("bucket", client=_S3Client())
        )
        app = CayuApp(enable_logging=False)
        app.register_environment(
            Environment(EnvironmentSpec(name="files"), artifact_store=store), default=True
        )
        session = await app.session_store.create(
            RunRequest(
                agent_name="shared",
                session_id="reader",
                messages=[],
                labels={"organization": "acme"},
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        allowed = rule(organization="acme")
        current = SessionAccessScope(read=(allowed,), create=(allowed,), execute=(allowed,))

        class Policy(ResourceAccessPolicy):
            authority = "attachment-test"

            async def resolve(self, subject):
                return current

        policy = Policy()

        async def resolve():
            return current

        writer = ScopedArtifactAccess(
            store, environment_name="files", admitted=current, resolve=resolve
        )
        own = await writer.put_bytes(
            b"own",
            labels={"organization": "acme"},
            filename="own.pdf",
            content_type="application/pdf",
        )
        foreign_scope = SessionAccessScope(
            read=(rule(organization="other"),), create=(rule(organization="other"),)
        )

        async def foreign_resolve():
            return foreign_scope

        foreign_writer = ScopedArtifactAccess(
            store, environment_name="files", admitted=foreign_scope, resolve=foreign_resolve
        )
        foreign = await foreign_writer.put_bytes(
            b"foreign",
            labels={"organization": "other"},
            filename="foreign.pdf",
            content_type="application/pdf",
        )
        binding = ResourceExecutionBinding(
            authority=policy.authority, subject="alice", admitted_json=encode_scope(current)
        )

        async def resolve_attachment(artifact):
            attachment = file_attachment(
                artifact_id=artifact.id,
                kind="document",
                filename=artifact.filename,
                content_type=artifact.content_type,
                size_bytes=artifact.size_bytes,
            )
            return await _resolved_file_attachments(
                messages=[
                    Message.tool_result(
                        tool_call_id="call",
                        tool_name="files",
                        content="document",
                        artifacts=[attachment],
                    )
                ],
                session=session,
                registered_environment=app._environments["files"],
                max_file_attachment_bytes=100,
                max_total_file_attachment_bytes=100,
                max_file_attachments_per_request=1,
            )

        async with execution_access(binding, policy, session.labels):
            resolved, missing = await resolve_attachment(own)
            assert own.id in resolved and not missing
            with pytest.raises(SessionAccessDenied):
                await resolve_attachment(foreign)

    asyncio.run(run())
