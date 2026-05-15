import os

from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI

from hg_ds_evals.common.config_api import (
    get_azure_openai_config,
    get_databricks_serving_config,
)


def _get_azure_api_version(model_deployment_name: str) -> str:
    from ds_common.config.config import llm_models_config

    return llm_models_config[model_deployment_name]["api_version"]


def _get_databricks_auth_kwargs(
    workspace_host: str,
    token: str | None = None,
    profile: str | None = None,
) -> dict:
    resolved_token = token or os.getenv("DATABRICKS_TOKEN")
    default_headers = None

    try:
        from databricks.sdk import WorkspaceClient

        ws_kwargs = {"host": workspace_host}
        if profile:
            ws_kwargs["profile"] = profile
        workspace = WorkspaceClient(**ws_kwargs)
        config = getattr(workspace, "config", None)
        resolved_token = resolved_token or getattr(config, "token", None)

        authenticate = getattr(config, "authenticate", None)
        if callable(authenticate):
            try:
                auth_headers = dict(authenticate())
            except Exception:
                auth_headers = {}

            authorization = str(auth_headers.get("Authorization") or "")
            if not resolved_token and authorization.lower().startswith("bearer "):
                resolved_token = authorization.split(" ", 1)[1]
            elif auth_headers:
                default_headers = {
                    str(key): str(value) for key, value in auth_headers.items()
                }
    except Exception:
        # Outside Databricks, users can still authenticate with DATABRICKS_TOKEN.
        pass

    if not resolved_token:
        resolved_token = _get_databricks_notebook_context_token()

    if not resolved_token and not default_headers:
        raise RuntimeError(
            "Databricks serving auth could not be resolved. Run inside a "
            "Databricks notebook/job, or set DATABRICKS_TOKEN."
        )

    kwargs = {"api_key": resolved_token or "databricks"}
    if default_headers:
        kwargs["default_headers"] = default_headers
    return kwargs


def _create_databricks_credential_refresher(
    workspace_host: str,
    profile: str | None = None,
):
    """Return a callable that refreshes Databricks creds on an OpenAI client.

    Used by ``api_calls.py`` to recover from mid-run OAuth token expiry
    on long evaluation runs. The returned callable takes the OpenAI
    client as its argument and mutates ``client.api_key`` so subsequent
    requests use a freshly-minted token.

    The refresh is fully unattended as long as ``databricks auth login``
    was run for ``profile`` within the workspace's refresh-token TTL
    (typically 7-30 days). The Databricks SDK uses the cached refresh
    token to mint a new access token without browser interaction.

    Returns ``None`` if a refresher cannot be constructed (SDK not
    installed, missing workspace_host).
    """
    if not workspace_host:
        return None
    try:
        from databricks.sdk import WorkspaceClient  # noqa: F401
    except ImportError:
        return None

    def _refresh(client) -> None:
        from databricks.sdk import WorkspaceClient

        ws_kwargs = {"host": workspace_host}
        if profile:
            ws_kwargs["profile"] = profile
        workspace = WorkspaceClient(**ws_kwargs)
        config = workspace.config

        authenticate = getattr(config, "authenticate", None)
        headers = dict(authenticate()) if callable(authenticate) else {}
        token = getattr(config, "token", None) or os.getenv("DATABRICKS_TOKEN")
        if not token:
            authorization = str(headers.get("Authorization") or "")
            if authorization.lower().startswith("bearer "):
                token = authorization.split(" ", 1)[1]
        if not token:
            hint = f" --profile {profile}" if profile else ""
            raise RuntimeError(
                "Databricks credential refresh produced no token. "
                f"Re-run `databricks auth login --host {workspace_host}{hint}`."
            )
        client.api_key = token

    return _refresh


def _get_databricks_notebook_context_token() -> str | None:
    try:
        from databricks.sdk.runtime import dbutils

        context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        return context.apiToken().get()
    except Exception:
        pass

    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        return context.apiToken().get()
    except Exception:
        return None


def _get_databricks_openai_kwargs(
    endpoint_url: str | None = None,
    base_url: str | None = None,
    workspace_host: str | None = None,
    token: str | None = None,
    profile: str | None = None,
) -> dict:
    config = get_databricks_serving_config(
        endpoint_url=endpoint_url,
        base_url=base_url,
        workspace_host=workspace_host,
    )
    return {
        "base_url": config.base_url,
        **_get_databricks_auth_kwargs(
            workspace_host=config.workspace_host,
            token=token,
            profile=profile,
        ),
    }


def get_api_client(
    model_deployment_name: str,
    api_provider: str = "azure_async",
    databricks_endpoint_url: str | None = None,
    databricks_base_url: str | None = None,
    databricks_workspace_host: str | None = None,
    databricks_token: str | None = None,
    databricks_profile: str | None = None,
):
    """
    Define the API client for Azure OpenAI or Databricks serving endpoints.

    Args:
        model_deployment_name (str): Azure deployment or Databricks serving endpoint name.
        api_provider (str): "azure", "azure_async", "databricks", "databricks_async",
            or "databricks_sync".
        databricks_endpoint_url (str | None): Optional URL such as
            https://<workspace>/serving-endpoints/<endpoint>/invocations.
        databricks_base_url (str | None): Optional OpenAI-compatible base URL
            https://<workspace>/serving-endpoints.
        databricks_workspace_host (str | None): Optional workspace host override.
        databricks_token (str | None): Optional token override. Prefer notebook/job
            auth or DATABRICKS_TOKEN over passing this directly.
        databricks_profile (str | None): Optional Databricks CLI profile name
            (~/.databrickscfg) used both for the initial connection and for
            unattended OAuth refresh during long runs. When None, the SDK
            uses the default profile or ``DATABRICKS_CONFIG_PROFILE`` env var.
    """
    api_provider = api_provider.lower()

    if api_provider in {"azure", "azure_sync"}:
        azure_config = get_azure_openai_config()
        client = AzureOpenAI(
            azure_endpoint=azure_config.endpoint,
            api_key=azure_config.api_key,
            api_version=_get_azure_api_version(model_deployment_name),
        )
    elif api_provider == "azure_async":
        azure_config = get_azure_openai_config()
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_config.endpoint,
            api_key=azure_config.api_key,
            api_version=_get_azure_api_version(model_deployment_name),
        )
    elif api_provider in {"databricks", "databricks_async", "databricks_sync"}:
        serving_config = get_databricks_serving_config(
            endpoint_url=databricks_endpoint_url,
            base_url=databricks_base_url,
            workspace_host=databricks_workspace_host,
        )
        auth_kwargs = _get_databricks_auth_kwargs(
            workspace_host=serving_config.workspace_host,
            token=databricks_token,
            profile=databricks_profile,
        )
        client_kwargs = {"base_url": serving_config.base_url, **auth_kwargs}
        if api_provider == "databricks_sync":
            client = OpenAI(**client_kwargs)
        else:
            client = AsyncOpenAI(**client_kwargs)
        # Attach an unattended credential refresher so that long runs can
        # survive OAuth token expiry without manual re-auth. Picked up by
        # ``api_calls.py`` on AuthenticationError / PermissionDeniedError.
        client._refresh_databricks_credentials = _create_databricks_credential_refresher(
            workspace_host=serving_config.workspace_host,
            profile=databricks_profile,
        )
    else:
        raise NotImplementedError(f"Endpoint host {api_provider} is not supported.")

    return client
