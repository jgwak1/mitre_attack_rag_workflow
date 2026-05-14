from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ModelProvider = Literal["litellm", "deterministic"]
RetrievalMethod = Literal["bm25", "tfidf", "vector", "hybrid"]


class ServerConfig(BaseModel):
    """
    Host/port config for optional local services.

    In this project, MCP can run as a local tool server process.
    The API server can run separately through FastAPI later.
    """

    host: str = "127.0.0.1"
    port: int


class AppSettings(BaseSettings):
    """
    Runtime settings for the MITRE ATT&CK RAG workflow.

    The important idea:
    - code stays in repo
    - secrets stay in .env / environment variables
    - model/tool routing reads from this settings object
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Run identity / trace output.
    run_id: str = ""
    task_id: str = "mitre-attack-threat-report-mapping"
    transcript_file: str = "out/transcript.json"

    # LiteLLM model routing.
    model_provider: ModelProvider = "litellm"
    fallback_provider: ModelProvider = "deterministic"

    # Example LiteLLM model strings:
    # - anthropic/claude-3-5-haiku-latest
    # - openai/gpt-4o-mini
    # - gemini/gemini-1.5-flash
    model: str = "anthropic/claude-3-5-haiku-latest"
    model_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # Retrieval defaults.
    default_retrieval_method: RetrievalMethod = "bm25"
    default_top_k: int = 5

    # Tool flags.
    enable_local_tools: bool = True
    enable_external_tools: bool = False
    web_search_api_key: str | None = Field(default=None, validation_alias="WEB_SEARCH_API_KEY")

    # MCP local server boundary.
    mcp_server_config: ServerConfig = ServerConfig(host="127.0.0.1", port=8080)

    # Runtime limits.
    max_report_chars: int = 20_000
    max_candidates_for_generation: int = 8

    def has_model_api_key(self) -> bool:
        """Return whether LiteLLM can call the configured model provider."""
        return bool(self.model_api_key and self.model_api_key.strip())

    def has_web_search_key(self) -> bool:
        """Return whether external web search can be attempted."""
        return bool(self.web_search_api_key and self.web_search_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return cached settings.

    This avoids reparsing .env every time planner.py, tools.py, generation.py,
    or api.py needs config.
    """
    return AppSettings()