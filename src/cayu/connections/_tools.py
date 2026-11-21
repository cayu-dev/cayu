"""Connector tools Cayu Cloud serves from a connector package: one generic adapter.

``tools("ramp")`` reads the toolset's catalog (``cayu.connections._tool_catalog``) and answers
one ``ConnectorPackageTool`` per tool. Each declares its slot, so registering it declares the
slot, and asks for approval itself (``Tool.approval_reason``) when it has an effect. Running one
posts its arguments to Cayu Cloud, which runs the package's code on the slot's binding.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from typing import Any

from cayu.connections import _tool_catalog
from cayu.connections._tool_catalog import (
    Catalog,
    CredentialUnavailable,
    Loaded,
    Origin,
    RunTargetUnavailable,
    ToolDescriptor,
)
from cayu.connections.client import ConnectionCallError
from cayu.connections.slots import (
    CONNECTION_SLOT_PATTERN,
    MAX_CONNECTION_SLOT_DESCRIPTION_LENGTH,
    ConnectionSlot,
)
from cayu.connectors import _common
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec

#: The largest arguments one run sends, as Cayu Cloud accepts them.
MAX_ARGUMENTS_BYTES = 1024 * 1024
_MAX_TOOL_NAME = 64
_RESTART = "The tool changed in Cayu Cloud; restart the Agent to use its new version."
_ADVICE = {
    "tool_changed": _RESTART + " Don't retry.",
    "tool_unavailable": "Cayu Cloud no longer offers this tool; don't retry it.",
    "connector_unavailable": "Cayu Cloud no longer offers these tools; don't retry them.",
    "connector_mismatch": (
        "The slot {slot} is bound to a connection these tools don't serve; an admin must bind "
        "it to one they do."
    ),
    "tool_timed_out": _common.TEMPORARY,
    "connection_failed": _common.TEMPORARY,
    "agent_ambiguous": (
        "Name the Agent by its id in this project's cayu-cloud.toml (`application`), then restart."
    ),
    "agent_not_found": (
        "Cayu Cloud has no active Agent of the name this project's cayu-cloud.toml gives "
        "(`application`): name a deployed Agent there, or deploy it, then restart."
    ),
    "tool_failed": "The tool failed in Cayu Cloud; don't retry it unchanged.",
}
_CHECK = "check whether it did"
_IDENTITY_PREFIX = "cayu.connector-tools/"
_OFFLINE = "This process has no connection to Cayu Cloud, which runs these tools."
_OFFLINE_ADVICE = (
    "These tools were loaded from a saved catalog (CAYU_CONNECTOR_TOOLS=snapshot or offline); "
    "sign in with `cayu cloud login` and link the Agent in cayu-cloud.toml to run them, or run "
    "the Agent in Cayu Cloud."
)
#: A call id as Cloud keeps it: 1-128 printable ASCII characters, no spaces.
_CALL_ID = re.compile(r"[!-~]{1,128}")
#: A run whose credential couldn't be had, by where the tools come from.
_NO_CREDENTIAL = {
    "agent": (
        "agent_credential_invalid",
        "The Agent's own connections credential was refused; redeploy the Agent. Don't retry.",
    ),
    "member": (
        "cloud_login_required",
        "Sign in again with `cayu cloud login` (or check CAYU_CLOUD_API_KEY or the selected "
        "context), then retry.",
    ),
}


def tools(*toolsets: str, slot: str | None = None, writes: bool = False) -> list[Tool]:
    """The tools of each named toolset, as Cayu Cloud offers them, on ``slot`` (default: each
    toolset's own ``default_slot``; ``slot`` names one toolset only); with ``writes=True`` also
    those with an effect, which ask for a person's approval themselves. Provisional.

    Call it where the app registers its agents: it reads each catalog once, synchronously, with
    a client of its own, so it never uses the caller's event loop and works at import, in a
    factory, or while one runs, as any import doing I/O does.

    The catalog comes from Cayu Cloud as the Agent, in Cloud; as the developer signed in with
    ``cayu cloud``, locally; or offline from the project's saved snapshot or the last catalog
    read here, whose tools answer ``no_connection_client`` when run. A toolset Cloud no longer
    offers gives no tools and declares nothing.
    """

    if not toolsets:
        raise ValueError("Name at least one toolset, such as tools('ramp').")
    if slot is not None and len(set(toolsets)) > 1:
        raise ValueError("slot= names one toolset's slot; call tools() once per toolset.")
    if slot is not None and not CONNECTION_SLOT_PATTERN.fullmatch(slot):
        raise ValueError("A slot is 1-63 lowercase letters, digits or interior hyphens.")
    if type(writes) is not bool:
        raise TypeError("writes is True or False.")
    chosen: list[Tool] = []
    for toolset in dict.fromkeys(toolsets):
        loaded = _tool_catalog.load(toolset)
        if loaded.catalog is not None:
            chosen.extend(_package_tools(loaded, loaded.catalog, slot, writes=writes))
    return chosen


def _package_tools(
    loaded: Loaded, catalog: Catalog, slot: str | None, *, writes: bool
) -> list[ConnectorPackageTool]:
    on = slot or catalog.default_slot
    chosen = [tool for tool in catalog.tools if writes or tool.effect == "none"]
    declared = ConnectionSlot(
        name=on,
        connectors=catalog.connectors,
        scopes=tuple(sorted({scope for tool in chosen for scope in tool.scopes})),
        description=_common.shortened(
            " ".join(catalog.description.split()), MAX_CONNECTION_SLOT_DESCRIPTION_LENGTH
        )[0],
    )
    return [
        ConnectorPackageTool(
            descriptor,
            catalog=catalog,
            origin=loaded.origin,
            slot=on,
            name=_name(catalog, descriptor.name, on),
            declared=declared,
        )
        for descriptor in chosen
    ]


def _name(catalog: Catalog, name: str, slot: str) -> str:
    """The tool's name on ``slot``: the package's on its default slot, else the slot's prefix
    in place of the toolset's (``ramp-eu``: ``ramp_eu_list_bills``)."""

    if slot == catalog.default_slot:
        return name
    renamed = slot.replace("-", "_") + name.removeprefix(catalog.toolset)
    if len(renamed) > _MAX_TOOL_NAME:
        raise ValueError(
            f"The slot {slot} makes the tool name {renamed} longer than {_MAX_TOOL_NAME} "
            "characters; give a shorter slot."
        )
    return renamed


class ConnectorPackageTool(Tool):
    """One tool of a connector package, run by Cayu Cloud on the slot's binding. Provisional."""

    def __init__(
        self,
        descriptor: ToolDescriptor,
        *,
        catalog: Catalog,
        origin: Origin,
        slot: str,
        name: str,
        declared: ConnectionSlot,
    ) -> None:
        super().__init__(
            ToolSpec(
                name=name,
                description=descriptor.description,
                input_schema=dict(descriptor.input_schema),
                effect=ToolEffect(descriptor.effect),
                max_terminal_payload_bytes=_common.MAX_TERMINAL_PAYLOAD_BYTES,
                # Stable across restarts and package versions; Cloud's digest of the tool's
                # effect, scopes and approval changes it, so eval baselines compare only runs
                # whose tool behaved alike.
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name=f"{_IDENTITY_PREFIX}{catalog.package}/{catalog.toolset}/{descriptor.name}",
                    behavior_version=descriptor.digest,
                    implementation_version=descriptor.digest,
                ),
            )
        )
        self.descriptor = descriptor
        self.slot = slot
        self.toolset = catalog.toolset
        #: The connector package that serves the tool, and its version, as Cloud names them.
        self.package = catalog.package
        self.version = catalog.version
        self._origin = origin
        self._changed_logged = False
        self.connection_slots = (declared,)
        if descriptor.approval is not None:
            self.approval_reason = ApprovalTemplate(descriptor.approval)

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        origin = self._origin
        write = self.descriptor.effect != "none"
        if origin.run_url is None or origin.bearer is None:
            return _common.error_result("no_connection_client", _OFFLINE, _OFFLINE_ADVICE)
        try:
            base = await asyncio.to_thread(origin.run_url)
        except CredentialUnavailable as error:
            return self._no_credential(error)
        except RunTargetUnavailable as error:
            return _common.error_result(error.code, str(error), _ADVICE[error.code])
        if base is None:
            return _common.error_result(
                "connection_unbound",
                "This project isn't linked to an Agent in Cayu Cloud, so its slots have no "
                "bindings to run on.",
                "Name the Agent in cayu-cloud.toml (`cayu cloud init`), then bind the slot: "
                f"`cayu cloud connections bind {self.slot} CONNECTION --application AGENT`.",
            )
        body = {
            "arguments": args,
            "digest": self.descriptor.digest,
            "call_id": _call_id(ctx),
        }
        if len(json.dumps(body, ensure_ascii=False).encode()) > MAX_ARGUMENTS_BYTES:
            return _common.error_result(
                "request_too_large", "The arguments are too large.", "Send less."
            )
        url = f"{base}/{self.slot}/tools/{self.descriptor.name}"
        try:
            answer = await _tool_catalog.post(url, origin.bearer, body)
        except CredentialUnavailable as error:
            return self._no_credential(error)
        except ConnectionCallError as error:
            if error.code == "tool_changed" and not self._changed_logged:
                self._changed_logged = True
                _tool_catalog.logger.warning(
                    "tool_changed: %s changed in Cayu Cloud since this process read it; "
                    "restart to use its new version.",
                    self.name,
                )
            return _common.call_error_result(error, self.slot, _ADVICE, write=write, check=_CHECK)
        return _result(answer, write=write)

    def _no_credential(self, error: Exception) -> ToolResult:
        code, advice = _NO_CREDENTIAL["agent" if self._origin.kind == "agent" else "member"]
        return _common.error_result(
            code, f"The credential for Cayu Cloud couldn't be had: {error}", advice
        )


class ApprovalTemplate:
    """A package's approval reason: its template, quoting each argument it names. Its text is
    its identity, so an agent's execution profile stays the same across restarts."""

    def __init__(self, template: str) -> None:
        self.template = template
        self._names = _tool_catalog.placeholders(template)
        digest = hashlib.sha256(template.encode()).hexdigest()
        self.execution_profile_identity = ExecutionProfileBehaviorIdentity(
            name=f"{_IDENTITY_PREFIX}approval",
            behavior_version=digest,
            implementation_version=digest,
        )

    def __call__(self, arguments: Mapping[str, Any]) -> str:
        return self.template.format_map(
            {name: _common.reason_argument(arguments.get(name)) for name in self._names}
        )


def _call_id(ctx: ToolContext) -> str:
    """The runtime's id of the call, which Cloud keeps with the calls the run makes."""

    for value in (ctx.metadata.get("tool_call_id"), ctx.idempotency_key):
        if isinstance(value, str) and _CALL_ID.fullmatch(value):
            return value
    return str(uuid.uuid4())


def _result(answer: Mapping[str, Any], *, write: bool) -> ToolResult:
    """Cloud's answer to a run: its result, or the error the tool answered. A write's error
    that may have taken effect is never advised to be retried unchecked."""

    error = answer.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or "tool_error")
        message = str(error.get("message") or "The tool failed.")
        if write and error.get("ambiguous") is True:
            return _common.ambiguous_write_result(code, message, _CHECK)
        if isinstance(error.get("path"), str):
            message = f"{message} (at {error['path']})"
        details = {
            name: value
            for name, value in error.items()
            if name not in {"code", "message", "advice"}
        }
        return _common.error_result(
            code,
            message,
            error.get("advice") if isinstance(error.get("advice"), str) else None,
            **details,
        )
    result = answer.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        return _common.error_result(
            "tool_failed", "Cayu Cloud's answer is malformed.", "Don't retry it unchanged."
        )
    structured = result.get("structured")
    return ToolResult(
        content=result["content"],
        structured=structured if isinstance(structured, dict) else None,
    )
