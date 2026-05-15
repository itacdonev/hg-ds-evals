import json
import pandas as pd
from typing import Callable
from openai import (
    APIError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential


async def _call_responses_create_with_refresh(client, params, semaphore):
    """One-shot ``client.responses.create`` with auth-refresh-and-retry.

    On long Databricks runs the OAuth access token can expire mid-stream,
    after which every subsequent call returns
    AuthenticationError (401) or PermissionDeniedError (403) "Invalid Token".
    When the client carries a ``_refresh_databricks_credentials`` callable
    (attached in ``api_client.get_api_client``), this wrapper catches the
    auth failure once, mints a fresh token via the cached OAuth refresh
    token, and retries the request a single time. Other failures (rate
    limit, timeout, etc.) are re-raised so the existing tenacity retry
    layer above this wrapper can handle them.
    """
    try:
        async with semaphore:
            return await client.responses.create(**params)
    except (AuthenticationError, PermissionDeniedError) as auth_err:
        refresher = getattr(client, "_refresh_databricks_credentials", None)
        if refresher is None:
            raise
        try:
            refresher(client)
        except Exception as refresh_err:
            print(f"⚠️ Databricks credential refresh failed: {refresh_err}")
            raise auth_err from refresh_err
        print("🔄 Databricks credentials refreshed; retrying request")
        async with semaphore:
            return await client.responses.create(**params)


def _normalise_model_name(model_name: str) -> str:
    return model_name.lower().replace(".", "-")


def _supports_temperature(model_name: str) -> bool:
    return _normalise_model_name(model_name) in {"gpt-4o", "gpt-4-1"}


def _supports_reasoning_effort(model_name: str) -> bool:
    model_name = _normalise_model_name(model_name)
    if model_name.startswith("databricks-"):
        model_name = model_name.removeprefix("databricks-")
    return model_name.startswith("gpt-5")


def build_api_params(system_prompt, 
                     user_prompt, 
                     config,
                     ):
    """Builds API request parameters."""

    endpoint_name = config["model"]["model_deployment_name"]
    max_output_tokens = config["model"]["max_output_tokens"]
    temperature = config["model"].get("temperature")
    reasoning_effort = config["model"].get("reasoning_effort")

    params = {
        "model": endpoint_name,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "max_output_tokens": max_output_tokens
    }
    
    if _supports_temperature(endpoint_name) and temperature is not None:
        params["temperature"] = temperature
    
    if _supports_reasoning_effort(endpoint_name):
        params["reasoning"] = {"effort": reasoning_effort or "medium"}
    
    return params

@retry(
    stop=stop_after_attempt(2),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=4, max=70)
)
async def async_call_llm_for_evaluation(
    row_dict,
    client,
    semaphore,
    system_prompt,
    config,
    user_prompt_builder: Callable[[dict], str] | None = None,
):
    """Async call to LLM for evaluation with retry logic.

    Args:
        row_dict: Row data as a dictionary.
        client: OpenAI-compatible async client.
        semaphore: Concurrency limiter.
        system_prompt: The system prompt string.
        config: Experiment config dictionary.
        user_prompt_builder: Callable that takes a row dict and returns
            the user prompt string.  When *None*, falls back to the
            legacy hardcoded fallback template.
    """
    if user_prompt_builder is not None:
        user_prompt = user_prompt_builder(row_dict)
    else:
        # Legacy fallback prompt builder — lazy import to avoid hard
        # dependency on the templates subpackage.
        from hg_ds_evals.prompts.templates.fallback.prompts import (
            build_eval_user_prompt as _legacy_build_user_prompt,
        )
        user_prompt = _legacy_build_user_prompt(row_dict)
    
    params = build_api_params(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config
    )
    
    try:
        response = await _call_responses_create_with_refresh(client, params, semaphore)
        return response
    except (RateLimitError, APITimeoutError):
        raise  # Let retry handler catch these
    except APIError as e:
        print(f"API Error: {type(e).__name__} - {e}")
        return {'error': str(e), 'error_type': type(e).__name__}
