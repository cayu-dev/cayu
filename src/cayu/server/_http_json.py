"""Shared JSON request decoding, schema metadata, and response rendering."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from fastapi import HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel


def _parse_json_without_duplicate_keys(body: bytes) -> object:
    """Parse one request body while rejecting non-portable JSON spellings."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number {value!r} is not supported.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object keys are not supported.")
            result[key] = value
        return result

    return json.loads(
        body,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _validated_model_json(value: BaseModel, model_type: type[BaseModel]) -> bytes:
    validated = model_type.model_validate(value.model_dump(mode="python"))
    return validated.model_dump_json().encode("utf-8")


def _render_utf8(renderer: Callable[[Any], str], value: Any) -> bytes:
    return renderer(value).encode("utf-8")


async def _model_json_response(
    value: BaseModel,
    model_type: type[BaseModel],
    *,
    status_code: int = 200,
) -> Response:
    """Serialize a bounded validated response without occupying the server loop."""

    content = await asyncio.to_thread(_validated_model_json, value, model_type)
    return Response(
        content=content,
        media_type="application/json",
        status_code=status_code,
    )


@dataclass(frozen=True)
class _PreparsedPrivateJsonBody:
    """One private JSON body parsed before FastAPI request validation."""

    value: object


_PREPARSED_PRIVATE_JSON_SCOPE_KEY = "cayu.preparsed_private_json_body"
_PrivateBodyModel = TypeVar("_PrivateBodyModel", bound=BaseModel)


async def _validated_private_json_body(
    request: Request,
    model_type: type[_PrivateBodyModel],
    *,
    invalid_detail: str,
) -> _PrivateBodyModel:
    content_type = request.headers.get("content-type")
    if content_type is not None:
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != "application/json" and not (
            media_type.startswith("application/") and media_type.endswith("+json")
        ):
            raise HTTPException(status_code=422, detail=invalid_detail)
    parsed_body = request.scope.get(_PREPARSED_PRIVATE_JSON_SCOPE_KEY)
    if not isinstance(parsed_body, _PreparsedPrivateJsonBody):
        raise HTTPException(status_code=422, detail=invalid_detail)
    try:
        return await asyncio.to_thread(model_type.model_validate, parsed_body.value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise HTTPException(status_code=422, detail=invalid_detail) from exc


def _json_request_openapi(model: str | type[BaseModel]) -> dict[str, Any]:
    if isinstance(model, str):
        schema = {"$ref": f"#/components/schemas/{model}"}
    else:
        generated = model.model_json_schema()
        definitions = generated.pop("$defs", {})

        def inline_definitions(value: Any, active: frozenset[str] = frozenset()) -> Any:
            if isinstance(value, list):
                return [inline_definitions(item, active) for item in value]
            if not isinstance(value, dict):
                return value
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
                if name in active or name not in definitions:
                    raise ValueError(
                        "Private request OpenAPI schema contains an invalid reference."
                    )
                replacement = inline_definitions(definitions[name], active | {name})
                return {
                    **replacement,
                    **{
                        key: inline_definitions(item, active)
                        for key, item in value.items()
                        if key != "$ref"
                    },
                }
            return {key: inline_definitions(item, active) for key, item in value.items()}

        schema = inline_definitions(generated)
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": schema}},
        }
    }
