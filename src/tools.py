from __future__ import annotations

import json
import re
import urllib.request
from urllib.parse import urlparse
from typing import Any

from src.config import AppSettings, get_settings
from src.data_sources import load_attack_techniques


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or a Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert Pydantic model or dict to plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {"value": str(obj)}


def lookup_attack_technique(technique_id: str) -> dict[str, Any]:
    """
    Local ATT&CK technique lookup tool.

    This is a real local tool.
    It does not need web access.

    Example:
        lookup_attack_technique("T1059.001")
    """
    technique_id = technique_id.strip()

    for technique in load_attack_techniques():
        mitre = _get(technique, "mitre_definition", {})
        current_id = _get(mitre, "technique_id")

        if current_id == technique_id:
            return {
                "ok": True,
                "tool": "lookup_attack_technique",
                "technique": _to_dict(technique),
            }

    return {
        "ok": False,
        "tool": "lookup_attack_technique",
        "error": f"Technique ID not found: {technique_id}",
    }


def fetch_official_source_url(
    url: str,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """
    Fetch a known official MITRE ATT&CK source URL.

    This is not general web search.

    Use case:
    - local attack_techniques.json already has source_url
    - planner.py decides source verification is useful
    - generation.py can receive a short official-source snippet

    Safety boundary:
    - external tools must be enabled
    - only https://attack.mitre.org URLs are allowed
    """
    settings = settings or get_settings()
    url = url.strip()

    if not settings.enable_external_tools:
        return {
            "ok": False,
            "tool": "fetch_official_source_url",
            "disabled": True,
            "reason": "External tools are disabled by config.",
            "url": url,
        }

    parsed = urlparse(url)

    if parsed.scheme != "https" or parsed.netloc != "attack.mitre.org":
        return {
            "ok": False,
            "tool": "fetch_official_source_url",
            "error": "Only https://attack.mitre.org URLs are allowed.",
            "url": url,
        }

    try:
        request = urllib.request.Request(
            url=url,
            headers={"User-Agent": "mitre-attack-rag-workflow/0.1"},
            method="GET",
        )


        # Fetch the official MITRE page.
        #
        # timeout=20 prevents the pipeline from hanging indefinitely if the
        # external site is slow or unavailable.
        #
        # errors="replace" keeps decoding robust if the page contains unusual
        # characters. Bad bytes are replaced instead of crashing the tool.
        with urllib.request.urlopen(request, timeout=20) as response:
            html = response.read().decode("utf-8", errors="replace")


        # Convert HTML into a rough text snippet for generation.py.
        #
        # This is intentionally simple for v1:
        # - remove script blocks
        # - remove style blocks
        # - remove all remaining HTML tags
        # - collapse repeated whitespace
        #
        # This is not a full web scraper. It is only a lightweight official-source
        # verification tool for allowlisted MITRE pages.
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return {
            "ok": True,
            "tool": "fetch_official_source_url",
            "provider": "mitre_attack",
            "url": url,
            "text_snippet": text[:2000],
        }

    except Exception as exc:
        return {
            "ok": False,
            "tool": "fetch_official_source_url",
            "url": url,
            "error": str(exc),
        }


def web_search(
    query: str,
    max_results: int = 5,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """
    External web search tool adapter.

    Current provider: Tavily-style search API.

    This tool is config-gated:
    - enable_external_tools must be true
    - WEB_SEARCH_API_KEY must exist in .env

    If disabled or missing a key, it returns a structured disabled result
    instead of crashing the pipeline.
    """
    settings = settings or get_settings()

    if not settings.enable_external_tools:
        return {
            "ok": False,
            "tool": "web_search",
            "disabled": True,
            "reason": "External tools are disabled by config.",
            "query": query,
        }

    if not settings.has_web_search_key():
        return {
            "ok": False,
            "tool": "web_search",
            "disabled": True,
            "reason": "WEB_SEARCH_API_KEY is not configured.",
            "query": query,
        }

    payload = {
        "api_key": settings.web_search_api_key,
        "query": query,
        "max_results": max_results,
    }

    request = urllib.request.Request(
        url="https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        return {
            "ok": True,
            "tool": "web_search",
            "provider": "tavily",
            "query": query,
            "results": data.get("results", []),
            "raw": data,
        }

    except Exception as exc:
        return {
            "ok": False,
            "tool": "web_search",
            "provider": "tavily",
            "query": query,
            "error": str(exc),
        }


def get_litellm_tool_schemas() -> list[dict[str, Any]]:
    """
    Return tool schemas in OpenAI/LiteLLM-compatible format.

    generation.py can pass this list to LiteLLM as the `tools` parameter.
    Later, the same underlying functions can be exposed through FastMCP.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_attack_technique",
                "description": "Look up a local MITRE ATT&CK technique by technique ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technique_id": {
                            "type": "string",
                            "description": "MITRE ATT&CK technique ID, for example T1059.001.",
                        }
                    },
                    "required": ["technique_id"],
                },
            },
        },

        {
            "type": "function",
            "function": {
                "name": "fetch_official_source_url",
                "description": (
                    "Fetch an allowlisted official MITRE ATT&CK source URL "
                    "for source verification."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": (
                                "Official MITRE ATT&CK URL, for example "
                                "https://attack.mitre.org/techniques/T1059/001/."
                            ),
                        }
                    },
                    "required": ["url"],
                },
            },
        },

        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the public web when local retrieval is insufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def execute_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | str | None,
) -> dict[str, Any]:
    """
    Execute a tool call by name.

    LiteLLM tool calls usually provide arguments as a JSON string.
    Local code may pass a dict directly.
    """
    if arguments is None:
        args: dict[str, Any] = {}
    elif isinstance(arguments, str):
        args = json.loads(arguments or "{}")
    else:
        args = arguments

    if tool_name == "lookup_attack_technique":
        return lookup_attack_technique(
            technique_id=str(args.get("technique_id", "")),
        )

    if tool_name == "fetch_official_source_url":
        return fetch_official_source_url(
            url=str(args.get("url", "")),
        )

    if tool_name == "web_search":
        return web_search(
            query=str(args.get("query", "")),
            max_results=int(args.get("max_results", 5)),
        )

    return {
        "ok": False,
        "tool": tool_name,
        "error": f"Unknown tool: {tool_name}",
    }