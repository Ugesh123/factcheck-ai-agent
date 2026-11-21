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


def _extract_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences or extra text."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _call(client: OpenAI, models: list[str], messages: list[dict], response_format: dict,
          max_tokens: int, require_parameters: bool):
    extra_body = {}
    if len(models) > 1:
        extra_body["models"] = models[1:]  # free fallbacks, tried in order
    if require_parameters:
        extra_body["provider"] = {"require_parameters": True}

    try:
        return client.chat.completions.create(
            model=models[0],
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
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
) -> dict:
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
        )

    if not response.choices:
        raise LLMError("The model returned an empty response. Please try again.")
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise LLMError("The model declined to process this text.")
    if choice.finish_reason == "length":
        raise LLMError("The model's answer was cut off because it was too long.")

    content = choice.message.content or ""
    try:
        result = _extract_json(content)
    except json.JSONDecodeError as exc:
        raise LLMError("The model returned output that wasn't valid JSON.") from exc
    if not isinstance(result, dict):
        raise LLMError("The model returned JSON in an unexpected shape.")
    return result