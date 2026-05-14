"""
FastMCP wrapper for ATT&CK analysis tools.

This module is not used by the default pipeline yet.
The current pipeline calls tools through src.tools inside generation.py, in the
same Python process as the main analysis workflow.

Later, this module can expose those same src.tools functions through a FastMCP
server, so the tools can be called through a separate MCP server process instead
of being called directly inside the pipeline process.

A separate MCP server is useful when tool execution needs to be shared by
multiple clients, isolated from the main API process, monitored separately, or
exposed through MCP's standard tool format, such as tool name, description,
input schema, and structured result.
"""

from __future__ import annotations

from typing import Any

from src.tools import (
    fetch_official_source_url as _fetch_official_source_url,
    lookup_attack_technique as _lookup_attack_technique,
    web_search as _web_search,
)


SERVER_NAME = "mitre-attack-tools"


def create_mcp_server() -> Any:
    """
    Create a FastMCP server exposing the existing analysis tools.

    FastMCP is imported lazily so the default pipeline can run without requiring
    the optional MCP dependency.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "fastmcp is required to run src.tools_mcp_server. "
            "Install it with: pip install fastmcp"
        ) from exc

    mcp = FastMCP(
        SERVER_NAME,
        instructions=(
            "MITRE ATT&CK tool server for local technique lookup, official "
            "source verification, and optional web search."
        ),
    )

    @mcp.tool()
    def lookup_attack_technique(technique_id: str) -> dict[str, Any]:
        """Return local ATT&CK technique details for a technique ID."""
        return _lookup_attack_technique(technique_id)

    @mcp.tool()
    def fetch_official_source_url(url: str) -> dict[str, Any]:
        """Fetch an allowlisted official MITRE ATT&CK source URL."""
        return _fetch_official_source_url(url)

    @mcp.tool()
    def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
        """Run the configured external web-search adapter."""
        return _web_search(query=query, max_results=max_results)

    return mcp


def main() -> None:
    """Run the MCP tool server."""
    create_mcp_server().run()


if __name__ == "__main__":
    main()
