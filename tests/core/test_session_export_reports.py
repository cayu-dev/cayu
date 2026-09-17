"""Qualified report projection through the public session export owner."""

import pytest
from examples.collaboration.session_export import ApprovedReportProjector
from tests.core.test_session_export_content_release import run_async
from tests.core.test_session_exports import AUDIENCE, CONTEXT, Projector, _ref, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.collaboration.exports import SessionExportConflict, SessionExportDenied
from cayu.messages import Message


@pytest.mark.parametrize(
    "variant",
    [
        "approved",
        "private_text",
        "extra_field",
        "attachment",
        "other_report",
        "other_revision",
        "other_owner",
        "other_incarnation",
        "boolean_as_integer",
        "boolean_revision",
        "float_revision",
    ],
)
@run_async
async def test_report_owner_approval_is_not_inferred_from_fields(backend, variant):
    async with harness(backend) as case:
        report = ApprovedReportProjector(
            reference=_ref("projector"),
            audience=AUDIENCE,
            report_reference=_ref("report"),
            passed=True,
        )
        app, store, _, _ = case.app(projectors=(report,))
        await case.create(store)
        data = report.approved_source()
        content = ""
        artifacts = []
        if variant == "extra_field":
            data["title"] = "Unapproved private report title"
        elif variant == "attachment":
            artifacts = [{"id": "private-attachment"}]
        elif variant == "other_report":
            data["report"]["object_id"] = "unrelated-report"
        elif variant == "other_revision":
            data["report"]["revision"] += 1
        elif variant == "other_owner":
            data["report"]["owner"]["owner_id"] = "unrelated-owner"
        elif variant == "other_incarnation":
            data["report"]["incarnation"] = "replacement"
        elif variant == "boolean_as_integer":
            data["passed"] = 1
        elif variant == "boolean_revision":
            data["report"]["revision"] = True
        elif variant == "float_revision":
            data["report"]["revision"] = 1.0
        elif variant == "private_text":
            content = "Private explanation copied next to an otherwise approved report"
        await store.append_transcript_messages(
            case.session_id,
            [
                Message.tool_result(
                    tool_call_id="report",
                    tool_name="report",
                    content=content,
                    structured=data,
                    artifacts=artifacts,
                )
            ],
        )
        request = (await case.request(app)).model_copy(update={"source_indices": (1,)})
        if variant in {"approved", "float_revision"}:
            # Transcript admission canonicalizes JSON integral numbers. Approval
            # binds that durable representation, not discarded Python syntax.
            # Unlike booleans, 1.0 and 1 have the same canonical JSON number.
            selected = await store.load_transcript_window(case.session_id, start_index=1, limit=1)
            assert (
                type(selected.records[0].message.content[0].structured["report"]["revision"]) is int
            )
            receipt = await app.export_session(request, context=CONTEXT)
            assert await app.read_session_export(request, context=CONTEXT) == data
            assert receipt.expected.intent.request.source_indices == (1,)
            reopened, _, _, _ = case.app(projectors=())
            assert await reopened.export_session(request, context=CONTEXT) == receipt
            assert await reopened.read_session_export(request, context=CONTEXT) == data
            assert len(await published(store, case.session_id)) == 1
        else:
            with pytest.raises(SessionExportDenied):
                await app.export_session(request, context=CONTEXT)
            assert not await published(store, case.session_id)
            assert (await app.lookup_session_export(request, context=CONTEXT)).status == "not_found"


@run_async
async def test_validator_cannot_approve_a_different_boolean_number_representation(backend):
    class RewritingProjector(Projector):
        def project(self, source):
            self.calls += 1
            return {"approved_count": True}

        def validate(self, source, output, audience):
            output["approved_count"] = 1
            return type(output["approved_count"]) is int

    async with harness(backend) as case:
        projector = RewritingProjector()
        app, store, _, _ = case.app(projectors=(projector,))
        await case.create(store)
        request = await case.request(app)
        with pytest.raises(SessionExportConflict):
            await app.export_session(request, context=CONTEXT)
        assert projector.calls == 1
        assert not await published(store, case.session_id)
        assert (await app.lookup_session_export(request, context=CONTEXT)).status == "not_found"
