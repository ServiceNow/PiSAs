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


def make_openrouter_kwargs(model_name: str, api_key: str, temperature: float = 0.7) -> Dict:
    if model_name.startswith("local/"):
        alias = model_name.split("/", 1)[1]
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
