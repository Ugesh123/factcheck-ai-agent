"""Settings and API keys.

Keys are read from Streamlit secrets (.streamlit/secrets.toml locally,
App settings -> Secrets on Streamlit Cloud), falling back to environment
variables. Nothing is hard-coded.

The app uses OpenRouter's free models only. Any model ID that isn't free
is rejected, so the app can never spend credits.
"""

import os

import streamlit as st

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"  # OpenRouter's router across its free models


class ConfigError(Exception):
    """Raised with a user-facing message when a required setting is missing."""


def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            return value or default
    except Exception:  # no secrets file present
        pass
    value = os.getenv(name, "").strip()
    return value or default


def llm_api_key() -> str:
    key = get_secret("OPENROUTER_API_KEY")
    if not key:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. Add it to .streamlit/secrets.toml for local "
            "runs, or to App settings -> Secrets on Streamlit Cloud."
        )
    return key

def tavily_api_key() -> str:
    key = get_secret("TAVILY_API_KEY")
    if not key:
        raise ConfigError(
            "TAVILY_API_KEY is not set. "
            "Add it to .streamlit/secrets.toml."
        )
    return key
def is_free_model(model_id: str) -> bool:
    return model_id.endswith(":free") or model_id == "openrouter/free"


def llm_models() -> list[str]:
    """Models to try, in order. LLM_MODEL may list several, comma-separated.

    OpenRouter moves on to the next model if one is unavailable. Every entry
    must be a free model.
    """
    raw = get_secret("LLM_MODEL", DEFAULT_MODEL)
    models = [m.strip() for m in raw.split(",") if m.strip()]
    paid = [m for m in models if not is_free_model(m)]
    if paid:
        raise ConfigError(
            f"Only free models are allowed, but LLM_MODEL includes: {', '.join(paid)}. "
            "Use model IDs ending in ':free' (or 'openrouter/free')."
        )
    return models or [DEFAULT_MODEL]