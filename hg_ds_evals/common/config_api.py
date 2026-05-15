# HEY GEORGE - EVALS
# Configs for the API calls.

# config_api.py
#=============================================================
import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse


# Azure OpenAI Configs
AZURE_SECRET_SCOPE = "key-vault"
AZURE_ENDPOINT_SECRET_KEY = "AZURE-OPENAI-ENDPOINT"
AZURE_API_KEY_SECRET_KEY = "AZURE-OPENAI-KEY"

# Databricks Serving Endpoint Configs
DATABRICKS_DEFAULT_ENDPOINT_URL = (
    "https://adb-3174992876438447.7.azuredatabricks.net/"
    "serving-endpoints/gpt-5-1/invocations"
)
DATABRICKS_DEFAULT_WORKSPACE_HOST = "https://adb-3174992876438447.7.azuredatabricks.net"
DATABRICKS_DEFAULT_SERVING_BASE_URL = (
    f"{DATABRICKS_DEFAULT_WORKSPACE_HOST}/serving-endpoints"
)

# Backward-compatible name for existing imports. This is the OpenAI-compatible
# serving base URL, not a hostname.
DATABRICKS_HOSTNAME = DATABRICKS_DEFAULT_SERVING_BASE_URL


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str


@dataclass(frozen=True)
class DatabricksServingConfig:
    base_url: str
    workspace_host: str
    endpoint_url: str | None = None


def _get_dbutils():
    try:
        from databricks.sdk.runtime import dbutils

        return dbutils
    except Exception:
        pass

    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        return DBUtils(spark)
    except Exception as exc:
        raise RuntimeError(
            "Azure OpenAI credentials are not available from environment "
            "variables and Databricks dbutils could not be initialized."
        ) from exc


def get_azure_openai_config() -> AzureOpenAIConfig:
    """Return Azure OpenAI config without reading Databricks secrets at import time."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if endpoint and api_key:
        return AzureOpenAIConfig(endpoint=endpoint, api_key=api_key)

    dbutils = _get_dbutils()
    return AzureOpenAIConfig(
        endpoint=dbutils.secrets.get(
            scope=AZURE_SECRET_SCOPE,
            key=AZURE_ENDPOINT_SECRET_KEY,
        ),
        api_key=dbutils.secrets.get(
            scope=AZURE_SECRET_SCOPE,
            key=AZURE_API_KEY_SECRET_KEY,
        ),
    )


def _workspace_host_from_url(url: str) -> str:
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")


def _current_databricks_workspace_host() -> str | None:
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        host = spark.conf.get("spark.databricks.workspaceUrl", None)
    except Exception:
        return None

    if not host:
        return None
    if "://" not in host:
        host = f"https://{host}"
    return host.rstrip("/")


def _serving_base_url_from_endpoint_url(endpoint_url: str) -> str:
    if "://" not in endpoint_url:
        endpoint_url = f"https://{endpoint_url}"
    parsed = urlparse(endpoint_url)
    path_parts = [part for part in parsed.path.split("/") if part]

    if "serving-endpoints" not in path_parts:
        return f"{_workspace_host_from_url(endpoint_url)}/serving-endpoints"

    serving_index = path_parts.index("serving-endpoints")
    base_path = "/" + "/".join(path_parts[: serving_index + 1])
    return urlunparse((parsed.scheme, parsed.netloc, base_path, "", "", "")).rstrip("/")


def get_databricks_serving_config(
    endpoint_url: str | None = None,
    base_url: str | None = None,
    workspace_host: str | None = None,
) -> DatabricksServingConfig:
    """Return the OpenAI-compatible Databricks serving URL config."""
    endpoint_url = (
        endpoint_url
        or os.getenv("DATABRICKS_SERVING_ENDPOINT_URL")
        or os.getenv("DATABRICKS_ENDPOINT_URL")
    )
    base_url = (
        base_url
        or os.getenv("DATABRICKS_SERVING_BASE_URL")
        or os.getenv("DATABRICKS_OPENAI_BASE_URL")
    )
    workspace_host = (
        workspace_host
        or os.getenv("DATABRICKS_HOST")
        or os.getenv("DATABRICKS_HOSTNAME")
    )

    if base_url:
        if "://" not in base_url:
            base_url = f"https://{base_url}"
        base_url = base_url.rstrip("/")
        workspace_host = workspace_host or _workspace_host_from_url(base_url)
    elif endpoint_url:
        base_url = _serving_base_url_from_endpoint_url(endpoint_url)
        workspace_host = workspace_host or _workspace_host_from_url(endpoint_url)
    else:
        workspace_host = (
            workspace_host
            or _current_databricks_workspace_host()
            or DATABRICKS_DEFAULT_WORKSPACE_HOST
        )
        if "://" not in workspace_host:
            workspace_host = f"https://{workspace_host}"
        workspace_host = workspace_host.rstrip("/")
        base_url = f"{workspace_host}/serving-endpoints"
        endpoint_url = DATABRICKS_DEFAULT_ENDPOINT_URL

    if workspace_host and "://" not in workspace_host:
        workspace_host = f"https://{workspace_host}"

    return DatabricksServingConfig(
        base_url=base_url,
        workspace_host=workspace_host.rstrip("/"),
        endpoint_url=endpoint_url,
    )


def __getattr__(name: str):
    """Preserve old constant imports while keeping secret lookup lazy."""
    if name == "AZURE_ENDPOINT":
        return get_azure_openai_config().endpoint
    if name == "AZURE_API_KEY":
        return get_azure_openai_config().api_key
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
