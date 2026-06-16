"""
Low-level LLM call helpers for PiSAs (used by the pipeline, eval, and demo).
"""

import json
import os
import time
from typing import Dict

import litellm

litellm.suppress_debug_info = True
litellm.set_verbose = False


# gpt-5.5 is newer than litellm's built-in model table; mirror gpt-5's config so
# litellm handles its params correctly. Best-effort: skipped if litellm differs.
try:
    litellm.register_model({"gpt-5.5": dict(litellm.get_model_info("gpt-5"))})
except Exception:
    pass


# Model → provider registry. Lists only the exceptions to the OpenRouter default,
# so bare names route without a prefix: "gpt-5" → native OpenAI, "gpt-oss-120b" →
# local server. An explicit "local/<model>" prefix still works (back-compat).
# Any model not listed here falls back to OpenRouter.
MODEL_PROVIDER = {
    "gpt-oss-120b": "local",
    "gpt-5":        "openai",
    "gpt-5.5":      "openai",
    "o4-mini":      "openai",
}


def make_openrouter_kwargs(model_name: str, api_key: str, temperature: float = 0.7) -> Dict:
    # Resolve provider: explicit local/ prefix wins, else the registry, else OpenRouter.
    if model_name.startswith("local/"):
        provider, alias = "local", model_name.split("/", 1)[1]
    else:
        provider, alias = MODEL_PROVIDER.get(model_name, "openrouter"), model_name

    if provider == "local":
        kwargs = {
            "model": f"openai/{alias}",
            "api_key": os.environ.get("LOCAL_MODEL_API_KEY", "sk-local"),
            "api_base": os.environ.get("LOCAL_MODEL_API_BASE", "http://localhost:8003/v1"),
            "temperature": temperature,
            "num_retries": 3,
            "timeout": 600,
        }
        extra_body_raw = os.environ.get("LOCAL_MODEL_EXTRA_BODY")
        if extra_body_raw:
            kwargs["extra_body"] = json.loads(extra_body_raw)
        return kwargs

    if provider == "openai":
        # Native OpenAI API (api.openai.com), keyed by OPENAI_API_KEY. OpenAI
        # reasoning models (gpt-5.x, o-series) only accept the default temperature,
        # so we omit it; drop_params strips any other unsupported args.
        kwargs = {
            "model": f"openai/{alias}",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "drop_params": True,
            "num_retries": 3,
            "timeout": 120,
        }
        api_base = os.environ.get("OPENAI_API_BASE")
        if api_base:
            kwargs["api_base"] = api_base
        return kwargs

    # Default: OpenRouter.
    kwargs = {
        "model": f"openrouter/{model_name}",
        "api_key": api_key,
        "api_base": "https://openrouter.ai/api/v1",
        "temperature": temperature,
        "num_retries": 3,
        "timeout": 120,
    }
    extra_body_raw = os.environ.get("REMOTE_MODEL_EXTRA_BODY")
    if extra_body_raw:
        kwargs["extra_body"] = json.loads(extra_body_raw)
        if kwargs["extra_body"].get("thinking", {}).get("type") == "enabled":
            kwargs["temperature"] = 1
    elif "qwen3" in model_name.lower():
        kwargs["extra_body"] = {"reasoning": {"effort": "none"}}
    return kwargs
