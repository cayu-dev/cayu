"""TypeSafe System One decisions through Cayu's governed model runtime.

Configure ``AgentSpec.provider_options['typesafe']['questions']`` with native
Choice, Score or Noul questions. Text messages become labeled state; answers
are emitted as JSON, not invented chat text. No tool execution is supported.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from typing import Any

import httpx

from cayu.messages import Message, TextPart
from cayu.providers._api_keys import resolve_api_key
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    _preflight_provider_portable_messages,
)


def _questions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError("TypeSafe requires a nonempty questions mapping.")
    result = json.loads(json.dumps(value, allow_nan=False))
    for key, question in result.items():
        if not key.strip() or not isinstance(question, dict):
            raise ValueError("Invalid TypeSafe question.")
        if set(question) - {"type", "instructions", "criteria"}:
            raise ValueError("Unsupported TypeSafe question field.")
        if (
            not isinstance(question.get("instructions"), str)
            or not question["instructions"].strip()
        ):
            raise ValueError("TypeSafe questions require instructions.")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError("Choice requires at least two named criteria.")
            if any(
                not k.strip() or not isinstance(v, str) or not v.strip()
                for k, v in criteria.items()
            ):
                raise ValueError("Choice criteria must be nonempty strings.")
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or len(criteria) < 2
                or any(not isinstance(v, str) or not v.strip() for v in criteria)
            ):
                raise ValueError("Score requires at least two text levels.")
        elif kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict)
                or set(criteria) != {"true", "false"}
                or any(not isinstance(v, str) or not v.strip() for v in criteria.values())
            ):
                raise ValueError("Noul criteria must describe true and false.")
        else:
            raise ValueError("Unsupported TypeSafe question type.")
    return result


def _request_questions(request: ModelRequest) -> dict[str, Any]:
    native = request.options.get("typesafe")
    if not isinstance(native, dict) or set(native) != {"questions"}:
        raise ValueError("Configure typesafe.questions in provider_options.")
    return _questions(native["questions"])


def _number(value: Any, upper: float = 1.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
        raise ValueError("Invalid TypeSafe numeric result.")
    return float(value)


def validate_response(body: Any, questions: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("model"), str)
        or not body["model"].strip()
    ):
        raise ValueError("Missing TypeSafe model identity.")
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("TypeSafe answer keys do not match questions.")
    clean = {}
    for key, question in questions.items():
        answer = answers[key]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError("TypeSafe answer type mismatch.")
        if kind == "noul":
            clean[key] = {"type": kind, "noul": _number(answer.get("noul"))}
            continue
        expected = (
            set(question["criteria"])
            if kind == "choice"
            else {str(i) for i in range(len(question["criteria"]))}
        )
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != expected:
            raise ValueError("TypeSafe probability options mismatch.")
        probabilities = {k: _number(v) for k, v in probabilities.items()}
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.02):
            raise ValueError("TypeSafe probabilities do not sum to one.")
        item = {
            "type": kind,
            "probabilities": probabilities,
            "confidence": _number(answer.get("confidence")),
        }
        if kind == "choice":
            if answer.get("choice") not in expected:
                raise ValueError("TypeSafe returned an unknown choice.")
            item["choice"] = answer["choice"]
        else:
            item["score"] = _number(answer.get("score"), len(expected) - 1)
            item["legend"] = {str(i): v for i, v in enumerate(question["criteria"])}
        clean[key] = item
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("Invalid TypeSafe usage.")
    clean_usage = {}
    for key in ("input_tokens", "output_tokens"):
        if key in usage:
            if type(usage[key]) is not int or not 0 <= usage[key] <= 2**53 - 1:
                raise ValueError("Invalid TypeSafe token usage.")
            clean_usage[key] = usage[key]
    return {"model": body["model"], "answers": clean, "usage": clean_usage}


class TypeSafeProvider(ModelProvider):
    """Native decision provider; uses TYPESAFE_API_KEY and a fixed HTTPS origin.

    Does not claim chat, tools, images, native JSON-schema decoding, background
    reconnect, or exactly-once remote dispatch. Runtime owns session durability.
    """

    name = "typesafe"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite.")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport

    def preflight_portable_messages(
        self, *, model: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> None:
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=False,
            supports_tool_definitions=False,
            supports_file_attachments=False,
        )

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        return {"typesafe": {"questions": _request_questions(request)}}

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.preflight_portable_messages(
            model=request.model, messages=request.messages, tools=request.tools
        )
        if (
            request.hosted_tools
            or request.tool_discovery_projection
            or request.targeted_tool_projection
        ):
            raise ValueError("TypeSafe does not support tools or projections.")
        if request.options.get("structured_output") or request.options.get("thinking"):
            raise ValueError("TypeSafe requires native decision questions, not chat controls.")
        questions = _request_questions(request)
        state = "\n\n".join(
            f"{message.role.value}: "
            + "\n".join(part.text for part in message.content if isinstance(part, TextPart))
            for message in request.messages
        )
        key = resolve_api_key(
            api_key=self._api_key,
            env_var="TYPESAFE_API_KEY",
            provider_name="TypeSafe",
            missing_hint="set TYPESAFE_API_KEY",
        )
        # Never retain vendor error bodies, request headers or transport exceptions
        # in persisted error events; any of those can contain credentials.
        error = None
        result = None
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport, follow_redirects=False
            ) as client:
                response = await client.post(
                    "https://api.typesafe.ai/v1/systemone",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": request.model, "state": state, "questions": questions},
                )
                if response.status_code != 200:
                    error = ModelProviderError(
                        f"TypeSafe returned HTTP {response.status_code}.",
                        provider=self.name,
                        status_code=response.status_code,
                    )
                else:
                    result = validate_response(response.json(), questions)
        except httpx.TimeoutException:
            error = ModelProviderError(
                "TypeSafe request timed out.", provider=self.name, retryable=True
            )
        except (httpx.NetworkError, httpx.RemoteProtocolError):
            error = ModelProviderError(
                "TypeSafe connection error.", provider=self.name, retryable=True
            )
        except (httpx.HTTPError, ValueError, TypeError):
            error = ModelProviderError(
                "TypeSafe request failed or returned an invalid decision.",
                provider=self.name,
                retryable=False,
            )
        if error:
            yield ModelStreamEvent.error(str(error), cause=error)
            return
        assert result is not None
        yield ModelStreamEvent.text_delta(json.dumps(result, allow_nan=False))
        yield ModelStreamEvent.completed(
            {"finish_reason": "stop", "model": result["model"], "usage": result["usage"]}
        )
