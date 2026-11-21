"""Thin wrapper around OpenRouter (OpenAI-compatible API), free models only.

structured_completion() asks for output that follows a JSON schema and
returns the parsed JSON. It first requests strict Structured Outputs, routed
only to endpoints that support them. If no free endpoint supports them, it
retries the same free model(s) in plain JSON mode, with the schema in the
prompt. Every failure becomes an LLMError with a message that can be shown
directly in the app.
"""

import json
import re

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from .config import OPENROUTER_BASE_URL

REQUEST_TIMEOUT_SECONDS = 120  # free endpoints can be slow
SDK_RETRIES = 2  # the SDK retries timeouts, 429s and 5xx with backoff


class LLMError(Exception):
    """A user-facing LLM failure.

    fatal=True means retrying other requests won't help (bad key, daily limit
    reached, unknown model), so the caller should stop instead of continuing.
    """

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


class _NoStructuredOutputs(Exception):
    """No free endpoint supports strict JSON schema output."""


def make_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=SDK_RETRIES,
        default_headers={"X-Title": "Fact-Check Agent"},
    )


def _extract_json(text: str):
    """Parse JSON from a model reply, tolerating reasoning tags, code fences or
    extra text around it. Returns whatever JSON value was found (object or list).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("No JSON value found", text, 0)


def _coerce_to_schema(result, schema: dict):
    """Fix the common ways free models bend a {"key": [items]} schema.

    - a bare list of items           -> {"key": list}
    - the list under a different key -> {"key": that list}
    - one item on its own            -> {"key": [item]}
    Anything else is returned unchanged.
    """
    required = schema.get("required", [])
    if len(required) != 1:
        return result
    key = required[0]
    item_keys = set(schema["properties"][key].get("items", {}).get("required", []))

    if isinstance(result, list):
        return {key: result}
    if not isinstance(result, dict) or isinstance(result.get(key), list):
        return result
    lists = [v for v in result.values() if isinstance(v, list)]
    if len(lists) == 1:
        return {key: lists[0]}
    if item_keys and len(item_keys & set(result)) >= len(item_keys) // 2:
        return {key: [result]}
    return result


def _call(client: OpenAI, models: list[str], messages: list[dict], response_format: dict,
          max_tokens: int, require_parameters: bool, temperature: float | None = None):
    extra_body = {}
    if len(models) > 1:
        extra_body["models"] = models[1:]  # free fallbacks, tried in order
    if require_parameters:
        extra_body["provider"] = {"require_parameters": True}
    optional = {} if temperature is None else {"temperature": temperature}

    try:
        return client.chat.completions.create(
            model=models[0],
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
            **optional,
        )
    except AuthenticationError as exc:
        raise LLMError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY in your secrets.", fatal=True) from exc
    except NotFoundError as exc:
        if require_parameters:
            raise _NoStructuredOutputs() from exc
        raise LLMError(
            "No free endpoint is available for this model. Check LLM_MODEL, and in OpenRouter "
            "settings make sure free model endpoints are allowed by your privacy settings.",
            fatal=True,
        ) from exc
    except PermissionDeniedError as exc:
        raise LLMError("OpenRouter refused the request (permission denied).", fatal=True) from exc
    except RateLimitError as exc:
        if "per-day" in str(exc) or "per day" in str(exc):
            raise LLMError(
                "The daily limit for OpenRouter free models has been reached. It resets at 00:00 UTC.",
                fatal=True,
            ) from exc
        raise LLMError("OpenRouter is rate-limiting requests right now. Please try again in a minute.") from exc
    except BadRequestError as exc:
        if require_parameters:
            raise _NoStructuredOutputs() from exc
        raise LLMError(f"OpenRouter rejected the request: {exc.message}", fatal=True) from exc
    except APIConnectionError as exc:  # includes timeouts
        raise LLMError("Couldn't reach OpenRouter (network error or timeout). Please try again.") from exc
    except APIStatusError as exc:
        if exc.status_code == 402:
            raise LLMError("OpenRouter refused the request because the account balance is negative.", fatal=True) from exc
        raise LLMError(f"OpenRouter returned an error (HTTP {exc.status_code}). Please try again.") from exc


def structured_completion(
    client: OpenAI,
    models: list[str],
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict,
    max_output_tokens: int = 12000,
    temperature: float | None = None,
    meta: dict | None = None,
) -> dict:
    """Return the parsed JSON reply.

    If `meta` is given, it is filled with the model OpenRouter actually used
    and the finish reason, so the app can show what happened on each call.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = _call(
            client, models, messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            max_tokens=max_output_tokens,
            require_parameters=True,
            temperature=temperature,
        )
    except _NoStructuredOutputs:
        # Same free model(s), plain JSON mode, schema spelled out in the prompt.
        messages[0]["content"] = (
            f"{system_prompt}\n\nReply with a single JSON object only, no other text. "
            f"It must match this JSON schema exactly:\n{json.dumps(schema)}"
        )
        response = _call(
            client, models, messages,
            response_format={"type": "json_object"},
            max_tokens=max_output_tokens,
            require_parameters=False,
            temperature=temperature,
        )
        if meta is not None:
            meta["mode"] = "json_object"

    if not response.choices:
        raise LLMError("The model returned an empty response. Please try again.")
    choice = response.choices[0]
    if meta is not None:
        meta.setdefault("mode", "json_schema")
        meta["model"] = getattr(response, "model", None) or models[0]
        meta["finish_reason"] = choice.finish_reason
    if getattr(choice.message, "refusal", None):
        raise LLMError("The model declined to process this text.")
    if choice.finish_reason == "length":
        raise LLMError("The model's answer was cut off because it was too long.")

    content = choice.message.content or ""
    try:
        result = _extract_json(content)
    except json.JSONDecodeError as exc:
        raise LLMError("The model returned output that wasn't valid JSON.") from exc
    result = _coerce_to_schema(result, schema)
    if not isinstance(result, dict):
        raise LLMError("The model returned JSON in an unexpected shape.")
    return result