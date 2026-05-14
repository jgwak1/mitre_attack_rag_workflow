from __future__ import annotations

from typing import Any, Literal

from src.config import AppSettings, get_settings


TaskType = Literal[
    "entity_extraction",
    "attack_mapping",
    "grounded_explanation",
]

RiskLevel = Literal["low", "normal", "high"]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Read a field from either a dict or a Pydantic model.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _candidate_source_url(candidate: dict[str, Any]) -> str | None:
    """
    Extract the official MITRE source_url from a candidate when available.

    Candidate shape reminder:
    candidates passed into planner.py are NOT web-search results.

    They come from:
        retrieval.py
        + candidate_expansion.py

    A candidate may contain:
        candidate["technique"]
            -> full technique object from attack_techniques.json

    Inside that technique object:
        mitre_definition.source_url
    """
    technique = _get(candidate, "technique", None)
    if not technique:
        return None

    mitre = _get(technique, "mitre_definition", None)
    if not mitre:
        return None

    source_url = _get(mitre, "source_url", None)
    if not source_url:
        return None

    return str(source_url)


def route_model(
    *,
    task_type: TaskType,
    risk_level: RiskLevel = "normal",
    candidate_count: int = 0,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """
    Decide which model path should handle the task.

    This is the model-routing layer.

    Model routing means:
        Do not hardcode one LLM path for every request.
        Choose the execution path based on task type, cost, latency, risk,
        and whether an API key is available.

    Examples:

    1. entity_extraction
       Meaning:
           Extract generic artifacts like IPs, domains, files, commands.
       Preferred path:
           deterministic local code.
       Why:
           This does not need an LLM.

    2. grounded_explanation
       Meaning:
           Generate the final ATT&CK mapping explanation from candidates,
           evidence, and tool results.
       Preferred path:
           LiteLLM model if key exists.
       Fallback:
           deterministic answer builder if key is missing.

    3. attack_mapping
       Meaning:
           Mapping behavior to candidate ATT&CK techniques.
       In this repo:
           retrieval.py and candidate_expansion.py already do most of this.
           LLM can be used later for final explanation or tie-breaking.
    """
    settings = settings or get_settings()

    if task_type == "entity_extraction":
        return {
            "provider": "deterministic",
            "model": None,
            "reason": (
                "Entity extraction is handled by local deterministic logic, "
                "so no LLM call is needed."
            ),
            "risk_level": risk_level,
            "candidate_count": candidate_count,
            "requires_groundedness_check": False,
        }

    if not settings.has_model_api_key():
        return {
            "provider": settings.fallback_provider,
            "model": None,
            "reason": (
                "No model API key is configured. Use deterministic fallback "
                "so the pipeline remains runnable and testable."
            ),
            "risk_level": risk_level,
            "candidate_count": candidate_count,
            "requires_groundedness_check": task_type == "grounded_explanation",
        }

    return {
        "provider": settings.model_provider,
        "model": settings.model,
        "reason": (
            "Use configured LiteLLM model for generation. "
            "LiteLLM keeps the model-provider boundary flexible."
        ),
        "risk_level": risk_level,
        "candidate_count": candidate_count,
        "requires_groundedness_check": (
            risk_level == "high" or task_type == "grounded_explanation"
        ),
    }


def plan_tool_use(
    *,
    report_text: str,
    candidates: list[dict[str, Any]],
    risk_level: RiskLevel = "normal",
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    """
    Decide which tools should be used or made available.

    Important input definitions:

    report_text:
        The threat report text that we are analyzing.
        In the full pipeline this is usually ingestion.body or cleaned text.
        It is NOT a web-search result.

    candidates:
        The technique candidates produced before generation.
        These come from:
            retrieval.py
            + candidate_expansion.py

        They are NOT lookup results.
        They are NOT web-search results.

        Example candidate:
            {
                "technique_id": "T1059.001",
                "rank": 2,
                "score": 9.51,
                "sources": ["retrieval", "analyst_heuristic_signature"],
                "heuristic_matches": [...]
            }

    Tool planning idea:
        1. If candidate T1059.001 exists,
           plan lookup_attack_technique("T1059.001").
           This gives generation.py the local technique description.

        2. If official MITRE source_url exists and source verification is useful,
           plan fetch_official_source_url(source_url).
           This is not general web search.
           This is official-source verification.

        3. If report asks for recent or external context,
           plan web_search(query).
           This is general web search.
    """
    settings = settings or get_settings()

    tool_plan: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # 1. Local ATT&CK lookup
    # ------------------------------------------------------------
    # This is the default tool path.
    #
    # Why:
    #   retrieval/candidate_expansion gives us technique IDs and scores.
    #   generation.py still benefits from full local technique details:
    #       name
    #       tactic
    #       description
    #       source_url
    #       analyst retrieval terms
    #
    # This tool is local, deterministic, and does not require web access.
    if settings.enable_local_tools:
        for candidate in candidates[: settings.max_candidates_for_generation]:
            technique_id = _get(candidate, "technique_id")
            if not technique_id:
                continue

            tool_plan.append(
                {
                    "tool": "lookup_attack_technique",
                    "arguments": {
                        "technique_id": str(technique_id),
                    },
                    "reason": (
                        "Candidate exists before generation. "
                        "Look up local ATT&CK details so the generator receives "
                        "more than just the technique ID."
                    ),

                    "required": False,   # required=False means this tool is useful enrichment, not a hard dependency.
                                         # If it fails, generation.py can still continue using retrieval/candidate data.
                                         # Use required=True only when the final answer should not be produced without
                                         # this tool result.
                }
            )

    # ------------------------------------------------------------
    # 2. Official source verification
    # ------------------------------------------------------------
    # This is different from web_search.
    #
    # web_search:
    #   Search unknown external information.
    #
    # fetch_official_source_url:
    #   Fetch a known official MITRE URL already stored in the local corpus.
    #
    # Example:
    #   source_url = https://attack.mitre.org/techniques/T1059/001/
    #
    # Use case:
    #   High-risk generation or source-verification mode.
    #
    # Current assumption:
    #   tools.py should expose fetch_official_source_url before generation.py
    #   actually executes this plan.
    should_verify_sources = risk_level == "high"

    if settings.enable_external_tools and should_verify_sources:
        for candidate in candidates[: settings.max_candidates_for_generation]:
            source_url = _candidate_source_url(candidate)
            if not source_url:
                continue

            # Allowlist official MITRE pages only.
            # This avoids arbitrary URL fetching.
            if not source_url.startswith("https://attack.mitre.org/"):
                continue

            tool_plan.append(
                {
                    "tool": "fetch_official_source_url",
                    "arguments": {
                        "url": source_url,
                    },
                    "reason": (
                        "High-risk generation requested. Fetch the official "
                        "MITRE source page for source verification."
                    ),
                    "required": False,
                }
            )

    # ------------------------------------------------------------
    # 3. General web search
    # ------------------------------------------------------------
    # This is only for external context that the local ATT&CK corpus cannot know.
    #
    # Example report text that should trigger web_search:
    #   "A recent vendor report says this campaign used WMI execution and
    #    credential dumping against financial institutions in late 2025."
    #
    # Why:
    #   Local ATT&CK lookup can explain T1047 and T1003.
    #   It cannot verify the recent vendor report or campaign context.
    #
    # This is gated by:
    #   enable_external_tools
    #   WEB_SEARCH_API_KEY
    lower_text = report_text.lower()


    # Simple v1 routing heuristic.
    # These cues indicate that the report may require external context beyond the
    # local ATT&CK corpus, such as recent campaigns, CVEs, vendor reports, or
    # public threat-actor reporting.
    #
    # This is not the same as analyst heuristic signatures.
    # Analyst heuristic signatures expand ATT&CK technique candidates.
    # These cues only decide whether web_search should be considered.
    #
    # Future Possible improvements:
    # - use retrieval confidence to trigger web_search only when local evidence is weak
    # - detect CVE IDs and years with regex instead of only substring cues
    # - include cost and latency policy before allowing external lookup
    external_lookup_cues = [
        "latest",
        "current",
        "recent",
        "cve-",
        "campaign",
        "threat actor",
        "vendor report",
        "public report",
        "financial institutions",
        "late 2025",
        "2026",
    ]

    needs_external_lookup = any(cue in lower_text for cue in external_lookup_cues)

    if (
        settings.enable_external_tools
        and settings.has_web_search_key()
        and needs_external_lookup
    ):
        tool_plan.append(
            {
                "tool": "web_search",
                "arguments": {
                    "query": report_text[:300],
                    "max_results": 5,
                },
                "reason": (
                    "The report contains recent or external-context cues. "
                    "Use web search to retrieve public context beyond the local "
                    "ATT&CK corpus."
                ),
                "required": False,
            }
        )

    return tool_plan


def build_execution_plan(
    *,
    report_text: str,
    candidates: list[dict[str, Any]],
    task_type: TaskType = "grounded_explanation",
    risk_level: RiskLevel = "normal",
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """
    Build the generation-time execution plan.

    This function does NOT execute tools.
    It only decides what should happen next.

    Inputs:

    report_text:
        The report body being analyzed.
        Usually from ingestion.py.

    candidates:
        Expanded technique candidates.
        Usually from:
            retrieval.py
            + candidate_expansion.py

    Output:
        A plan consumed by generation.py.

    Example:

        report_text:
            "The actor used encoded powershell."

        candidates:
            [
                {"technique_id": "T1059.001", ...},
                {"technique_id": "T1027", ...}
            ]

        plan:
            {
                "model_route": {
                    "provider": "litellm" or "deterministic",
                    ...
                },
                "tool_plan": [
                    {
                        "tool": "lookup_attack_technique",
                        "arguments": {"technique_id": "T1059.001"}
                    },
                    {
                        "tool": "lookup_attack_technique",
                        "arguments": {"technique_id": "T1027"}
                    }
                ]
            }

    Generation step after this:
        generation.py executes the tool plan,
        adds tool results to the prompt/context,
        then produces the final grounded answer.
    """
    settings = settings or get_settings()

    model_route = route_model(
        task_type=task_type,
        risk_level=risk_level,
        candidate_count=len(candidates),
        settings=settings,
    )

    tool_plan = plan_tool_use(
        report_text=report_text,
        candidates=candidates,
        risk_level=risk_level,
        settings=settings,
    )

    return {
        "task_type": task_type,
        "risk_level": risk_level,
        "model_route": model_route,
        "tool_plan": tool_plan,
        "external_tools_enabled": settings.enable_external_tools,
        "local_tools_enabled": settings.enable_local_tools,
    }