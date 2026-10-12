# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: what a model provider said when a call to it failed.

A provider error (the provider answered 4xx/5xx, or sent an ``error`` chunk
inside a stream) is read into optional, structured fields that the error
objects carry next to ``code`` and ``message``:

- ``provider_status``: the provider's HTTP status, or the numeric code of an
  error inside a stream;
- ``provider``: the upstream provider an aggregator names (OpenRouter's
  ``metadata.provider_name``);
- ``provider_detail``: the provider's own words (OpenRouter's ``metadata.raw``,
  else its message), at most 1000 characters.

The fields are omitted when unset, so every other error is serialized as
before. The text is the provider's, uncleaned: a client that shows it to end
users cleans it first.
"""

import json
from typing import Any

from openai import APIError

from ogx_api.common.errors import OpenAIErrorDetail, OpenAIErrorResponse

DETAIL_MAX = 1000
PROVIDER_MAX = 200
GENERIC_MESSAGE = "The model provider returned an error."

PROVIDER_ERROR_FIELDS = ("provider_status", "provider", "provider_detail")


def _text(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict | list):
        return json.dumps(v, ensure_ascii=False)
    return ""


def provider_error_fields(exc: BaseException) -> dict[str, Any] | None:
    """The structured fields of a provider SDK error, or None for anything else.

    Only an openai.APIError with a JSON object body is a provider error: an
    APIStatusError (the provider answered 4xx/5xx) or the APIError the SDK
    raises for an ``error`` chunk inside a stream. A connection error has no
    body and is not one.
    """
    if not isinstance(exc, APIError):
        return None
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    err = body.get("error") if isinstance(body.get("error"), dict) else body

    status = getattr(exc, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        code = err.get("code")
        status = code if isinstance(code, int) and not isinstance(code, bool) else None
    if status is not None and not 400 <= status <= 599:
        status = None

    meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
    provider = _text(meta.get("provider_name"))[:PROVIDER_MAX] or None

    message = _text(err.get("message"))
    detail = (_text(meta.get("raw")) or message)[:DETAIL_MAX] or None

    if status is None and provider is None and detail is None:
        return None
    code = err.get("code")
    return {
        "provider_status": status,
        "provider": provider,
        "provider_detail": detail,
        "message": message or GENERIC_MESSAGE,
        "code": "rate_limit_exceeded" if status == 429 else (code if isinstance(code, str) and code else None),
        "type": err.get("type") if isinstance(err.get("type"), str) else None,
    }


class ProviderRequestError(Exception):
    """A Responses call without a stream that failed on a provider error.

    Raised instead of InvalidParameterError (400) / InternalServerError (500) so
    the server answers with the provider's status and the structured fields.
    It deliberately has no ``status_code``: the route wrapper would turn it into
    a plain-text HTTPException; it reaches the server's exception handler.
    """

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error.get("message") or GENERIC_MESSAGE)
        self.error = error


def provider_error_response(exc: BaseException) -> tuple[int, dict[str, Any]] | None:
    """The HTTP status and OpenAI-shaped error body for a provider error, or None."""
    if isinstance(exc, ProviderRequestError):
        e = exc.error
        fields = {k: e.get(k) for k in PROVIDER_ERROR_FIELDS}
        message, code, typ = e.get("message") or GENERIC_MESSAGE, e.get("code"), None
    else:
        pf = provider_error_fields(exc)
        if pf is None:
            return None
        fields = {k: pf[k] for k in PROVIDER_ERROR_FIELDS}
        message, code, typ = pf["message"], pf["code"], pf["type"]
    status = fields["provider_status"] if isinstance(fields["provider_status"], int) else 502
    detail = OpenAIErrorDetail(message=message, type=typ, code=code, **fields)
    return status, OpenAIErrorResponse(error=detail).to_dict()
