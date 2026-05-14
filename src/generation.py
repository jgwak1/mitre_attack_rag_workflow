from __future__ import annotations

import json
from typing import Any

from src.config import AppSettings, get_settings
from src.planner import RiskLevel, build_execution_plan
from src.tools import execute_tool_call, get_litellm_tool_schemas


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Read a field from either a dict or a Pydantic-style object.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """
    Convert a dict or Pydantic-style object into a plain dict.

    Why this exists:
        retrieval.py may preserve full Pydantic technique objects.
        generation.py wants JSON-serializable context.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {"value": str(obj)}


def _candidate_technique_id(candidate: dict[str, Any]) -> str:
    """
    Return candidate technique ID as a string.
    """
    return str(_get(candidate, "technique_id", ""))


def _candidate_sources(candidate: dict[str, Any]) -> list[str]:
    """
    Return candidate sources.

    Example:
        ["retrieval", "analyst_heuristic_signature"]
    """
    sources = _get(candidate, "sources", []) or []
    return [str(source) for source in sources]


def _candidate_heuristic_phrases(candidate: dict[str, Any]) -> list[str]:
    """
    Extract matched heuristic phrases from candidate_expansion.py output.

    Example:
        "encoded powershell"
    """
    matches = _get(candidate, "heuristic_matches", []) or []
    phrases: list[str] = []

    for match in matches:
        phrase = _get(match, "matched_phrase", None)
        if phrase:
            phrases.append(str(phrase))

    return phrases


def _technique_details_from_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """
    Try to get full technique details directly from the candidate.

    Retrieval candidates usually already contain:
        candidate["technique"]

    Heuristic-only candidates may not.

    Example heuristic-only case:
        retrieval returns T1059.001
        candidate_expansion sees "encoded powershell"
        heuristic adds T1027
        T1027 may only have technique_id until lookup_attack_technique fills details
    """
    technique = _get(candidate, "technique", None)
    if not technique:
        return None

    return _to_plain_dict(technique)


def _build_tool_result_map(tool_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Build technique_id -> technique details from lookup_attack_technique results.

    This supplements _technique_details_from_candidate().

    Priority later in _build_candidate_context():
        1. Use candidate["technique"] if it exists.
        2. Otherwise use lookup_attack_technique tool result.
        3. Otherwise keep only technique_id.

    Case 3 should be rare with the current data, but it is useful defensive code
    if a heuristic signature points to a technique missing from the local corpus.
    """
    result_map: dict[str, dict[str, Any]] = {}

    for result in tool_results:
        if not result.get("ok"):
            continue

        if result.get("tool") != "lookup_attack_technique":
            continue

        technique = result.get("technique")
        if not isinstance(technique, dict):
            continue

        mitre = technique.get("mitre_definition", {})
        technique_id = mitre.get("technique_id")

        if technique_id:
            result_map[str(technique_id)] = technique

    return result_map


def execute_tool_plan(tool_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Execute tools planned by planner.py.

    Tool plan entry example:
        {
            "tool": "lookup_attack_technique",
            "arguments": {"technique_id": "T1059.001"},
            "required": False,
            "reason": "Look up local ATT&CK details."
        }

    required=False:
        The tool is useful enrichment.
        If it fails, generation can still continue using retrieval candidates.

    required=True:
        The final answer should not continue without this tool result.
        We currently stop executing further tools after a required failure.
    """
    results: list[dict[str, Any]] = []

    for item in tool_plan:
        tool_name = str(item.get("tool", ""))
        arguments = item.get("arguments", {})
        required = bool(item.get("required", False))

        try:
            result = execute_tool_call(tool_name, arguments)
            result["required"] = required
            result["planned_reason"] = item.get("reason")
            result["tool_call_source"] = "planner_prefetch"
            results.append(result)

        except Exception as exc:
            error_result = {
               "ok": False,
               "tool": tool_name,
               "arguments": arguments,
               "required": required,
               "tool_call_source": "planner_prefetch",
               "error": str(exc),
               "planned_reason": item.get("reason"),
            }
            results.append(error_result)

            if required:
                break

    return results


def _build_candidate_context(
    candidates: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    """
    Build compact candidate context for deterministic generation or LiteLLM.

    Important:
        The candidate order is the retrieval/candidate-expansion order.
        The LLM should preserve this order rather than invent a new ranking.

    This context combines:
        retrieval rank and score
        candidate sources
        heuristic matched phrases
        full local ATT&CK details from either candidate or lookup tool
    """
    lookup_map = _build_tool_result_map(tool_results)
    context: list[dict[str, Any]] = []

    for candidate in candidates[:max_candidates]:
        technique_id = _candidate_technique_id(candidate)

        technique = (
            _technique_details_from_candidate(candidate)
            or lookup_map.get(technique_id)
            or {}
        )

        mitre = technique.get("mitre_definition", {}) if isinstance(technique, dict) else {}

        context.append(
            {
                "technique_id": technique_id,
                "input_rank": _get(candidate, "rank"),
                "score": _get(candidate, "score"),
                "retrieval_method": _get(candidate, "method"),
                "sources": _candidate_sources(candidate),
                "heuristic_phrases": _candidate_heuristic_phrases(candidate),
                "name": mitre.get("name"),
                "tactic": mitre.get("tactic"),
                "description": mitre.get("description"),
                "source_url": mitre.get("source_url"),
            }
        )

    return context


def _build_generation_messages(
    *,
    report_text: str,
    candidate_context: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    execution_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build messages for LiteLLM generation.

    Design choice:
        Retrieval and candidate expansion do the primary ranking.
        The LLM is not used as the main ranker.
        The LLM is used for grounded explanation, evidence wording,
        weak-evidence warnings, and at most one extra tool call.

    Tool-calling note:
        The instruction below is policy guidance.
        The actual structured tool-call capability comes from passing
        tools=... and tool_choice="auto" to LiteLLM.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are a security AI assistant. Map threat-report behavior to "
            "MITRE ATT&CK techniques using only the provided report, candidate "
            "techniques, and tool results. Do not invent facts. If evidence is "
            "weak, say so. Preserve the input candidate order unless a candidate "
            "is clearly unsupported. You may request one additional tool call if "
            "the provided candidate context is insufficient. Prefer the provided "
            "context when it is enough. Do not call tools unnecessarily."
        ),
    }

    user_payload = {
        "task": "Generate grounded MITRE ATT&CK mappings.",
        "report_text": report_text,
        "candidate_context": candidate_context,
        "tool_results": tool_results,
        "execution_plan": execution_plan,
        "ordering_instruction": (
            "Preserve the input candidate order. Do not invent a new ranking. "
            "Use the explanation to clarify evidence strength and weak-evidence "
            "warnings, not to arbitrarily reorder candidates."
        ),
        "confidence_instruction": (
            "Use confidence_label as the main confidence signal: high, medium, or low. "
            "If you include confidence_score, treat it as a heuristic score, not a "
            "calibrated probability."
        ),
        "output_requirements": {
            "format": "JSON only",
            "fields": {
                "mappings": [
                    {
                        "input_rank": "integer or null, copied from candidate_context",
                        "technique_id": "string",
                        "name": "string",
                        "tactic": "string or list",
                        "confidence_label": "high, medium, or low",
                        "confidence_score": "float from 0 to 1, heuristic not calibrated probability",
                        "evidence": "short phrase from report or matched heuristic",
                        "rationale": "brief grounded explanation",
                    }
                ],
                "summary": "brief answer summary",
                "warnings": ["unsupported or weak-evidence notes"],
            },
        },
    }

    return [
        system_message,
        {
            "role": "user",
            "content": json.dumps(user_payload, indent=2),
        },
    ]

def _message_content_to_text(content: Any) -> str:
    """
    Convert LiteLLM message.content into plain text.

    Some providers return content as a string.
    Some return content blocks.
    JSON parsing should operate on plain text.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
            else:
                parts.append(str(block))

        return "\n".join(parts)

    return str(content)


def _strip_json_fence(text: str) -> str:
    """
    Remove markdown JSON fences.

    Example:
        ```json
        {"ok": true}
        ```
    """
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()

    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    return cleaned


def _extract_json_object(text: str) -> str:
    """
    Extract the first JSON object from model output.

    Handles:
        Here is the JSON:
        {"mappings": [...]}
    """
    cleaned = _strip_json_fence(text)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return cleaned

    return cleaned[start : end + 1]


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """
    Try to parse model output as JSON.

    Handles:
    - strict JSON
    - markdown fenced JSON
    - short prose before/after JSON
    """
    candidate = _extract_json_object(text)

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def _confidence_from_sources(sources: list[str]) -> tuple[str, float]:
    """
    Convert candidate evidence sources into deterministic fallback confidence.

    These values are not calibrated probabilities.
    They only express relative evidence strength for deterministic generation.

    User-facing signal:
        confidence_label

    Secondary machine-readable signal:
        confidence_score

    Later improvement:
        Replace this heuristic with calibrated scoring using retrieval scores,
        heuristic evidence, evaluation labels, and human-reviewed examples.
    """
    if "retrieval" in sources and "analyst_heuristic_signature" in sources:
        return "high", 0.75

    if "retrieval" in sources:
        return "medium", 0.65

    if "analyst_heuristic_signature" in sources:
        return "medium", 0.60

    return "low", 0.55


def _deterministic_mapping_from_context(item: dict[str, Any]) -> dict[str, Any]:
    """
    Build one deterministic mapping from candidate context.

    This fallback is conservative.
    It preserves candidate order and does not claim more than the available
    retrieval, heuristic, and tool evidence supports.
    """
    heuristic_phrases = item.get("heuristic_phrases", []) or []
    evidence = heuristic_phrases[0] if heuristic_phrases else "retrieved candidate"

    sources = item.get("sources", []) or []
    score = item.get("score") or 0.0

    confidence_label, confidence_score = _confidence_from_sources(sources)

    rationale = (
        f"Candidate was selected by {item.get('retrieval_method')} retrieval "
        f"with score {round(float(score), 3)}"
    )

    if heuristic_phrases:
        rationale += f" and matched heuristic phrase(s): {', '.join(heuristic_phrases)}."
    else:
        rationale += "."

    return {
        "input_rank": item.get("input_rank"),
        "technique_id": item.get("technique_id"),
        "name": item.get("name"),
        "tactic": item.get("tactic"),
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
        "evidence": evidence,
        "rationale": rationale,
        "sources": sources,
        "source_url": item.get("source_url"),
    }


def _candidate_order_map(candidates: list[dict[str, Any]]) -> dict[str, int]:
    """
    Build technique_id -> order index from candidate order.

    Used to keep LLM-generated mappings aligned with retrieval/candidate order.
    """
    order: dict[str, int] = {}

    for idx, candidate in enumerate(candidates):
        technique_id = _candidate_technique_id(candidate)
        if technique_id and technique_id not in order:
            order[technique_id] = idx

    return order


def _normalize_generated_mappings(
    *,
    parsed_output: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Normalize LLM mappings.

    Why this exists:
        The LLM is not the primary ranker.
        Retrieval/candidate order is the primary ordering signal.

    This function:
        keeps only mappings with technique_id
        warns if a generated technique is outside the candidate pool
        sorts mappings by original candidate order
        fills input_rank when possible
    """
    mappings = parsed_output.get("mappings", [])
    if not isinstance(mappings, list):
        return [], ["Model output field 'mappings' was not a list."]

    order_map = _candidate_order_map(candidates)
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []

    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue

        technique_id = mapping.get("technique_id")
        if not technique_id:
            continue

        technique_id = str(technique_id)

        if technique_id not in order_map:
            warnings.append(
                f"Generated technique {technique_id} was not in the candidate pool."
            )

        if mapping.get("input_rank") is None and technique_id in order_map:
            mapping["input_rank"] = order_map[technique_id] + 1

        normalized.append(mapping)

    normalized.sort(
        key=lambda item: order_map.get(str(item.get("technique_id")), 10_000)
    )

    return normalized, warnings


def generate_deterministic_response(
    *,
    report_text: str,
    candidates: list[dict[str, Any]],
    execution_plan: dict[str, Any],
    tool_results: list[dict[str, Any]],
    settings: AppSettings,
) -> dict[str, Any]:
    """
    Deterministic fallback generation.

    Used when:
        no LLM API key exists
        LiteLLM call fails
        regression tests need stable output

    This is not a replacement for LLM generation.
    It keeps the pipeline runnable and testable.
    """
    candidate_context = _build_candidate_context(
        candidates=candidates,
        tool_results=tool_results,
        max_candidates=settings.max_candidates_for_generation,
    )

    mappings = [
        _deterministic_mapping_from_context(item)
        for item in candidate_context
    ]

    return {
        "ok": True,
        "provider": "deterministic",
        "model": None,
        "summary": (
            "Generated deterministic ATT&CK candidate mappings from retrieval, "
            "candidate expansion, and tool results."
        ),
        "mappings": mappings,
        "warnings": [
            "Deterministic fallback used. No external LLM generation was performed."
        ],
        "execution_plan": execution_plan,
        "tool_results": tool_results,
        "raw_model_output": None,
    }


def _message_to_dict(message: Any) -> dict[str, Any]:
    """
    Convert LiteLLM message object into a dict for message history.

    model_dump is not a key inside the LLM output.
    It is a Pydantic-style method that serializes a Python object to a dict.

    We check hasattr(message, "model_dump") because LiteLLM may return a
    Pydantic-style message object. If not, we fall back to dict(message).
    """
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)

    if isinstance(message, dict):
        return message

    return dict(message)


def generate_with_litellm(
    *,
    report_text: str,
    candidates: list[dict[str, Any]],
    execution_plan: dict[str, Any],
    tool_results: list[dict[str, Any]],
    settings: AppSettings,
) -> dict[str, Any]:
    """
    Generate final answer with LiteLLM.

    There are two tool paths in this module.

    1. Planner-driven prefetch:
        planner.py decides useful tool calls before the LLM runs.
        Those tool results are already included in prompt context.

    2. Model-driven tool call:
        LiteLLM receives tool schemas through tools=...
        With tool_choice="auto", a tool-capable model may return structured
        tool calls in message.tool_calls.
        We execute the requested tool, append the tool result, and call the
        model once more.

    Important:
        This is not prompt-based JSON parsing.
        We are not asking the model to print a key named "tool_calls".
        tools=... and tool_choice="auto" are API-level parameters.
        LiteLLM normalizes provider-specific tool-call responses into an
        OpenAI-compatible message.tool_calls field when supported.

    We allow only one extra tool-call round to keep cost, latency, and control bounded.

    This is not a full autonomous agent loop.

    v1 behavior:
        - planned tools are executed once before generation
        - the model may request at most one additional tool-call round
        - after that, the model must produce the final answer

    This keeps the workflow predictable for regression tests, latency control, and demo reliability.    
    """
    try:
        import litellm
    except ImportError as exc:
        raise ImportError(
            "litellm is required for LLM generation. Install it with: pip install litellm"
        ) from exc

    candidate_context = _build_candidate_context(
        candidates=candidates,
        tool_results=tool_results,
        max_candidates=settings.max_candidates_for_generation,
    )

    messages = _build_generation_messages(
        report_text=report_text,
        candidate_context=candidate_context,
        tool_results=tool_results,
        execution_plan=execution_plan,
    )

    response = litellm.completion(
        model=settings.model,
        messages=messages,
        tools=get_litellm_tool_schemas(),
        tool_choice="auto",
        api_key=settings.model_api_key,
        temperature=0.0,
        max_tokens=2000,
        num_retries=2,
        timeout=60,
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        messages.append(_message_to_dict(message))

        extra_tool_results: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            result = execute_tool_call(tool_name, arguments)
            result["tool_call_source"] = "model_requested"
            result["tool_call_id"] = tool_call.id
            extra_tool_results.append(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        tool_results = tool_results + extra_tool_results

        response = litellm.completion(
            model=settings.model,
            messages=messages,
            tools=get_litellm_tool_schemas(),
            tool_choice="auto",
            api_key=settings.model_api_key,
            temperature=0.0,
            max_tokens=2000,
            num_retries=2,
            timeout=60,
        )

        message = response.choices[0].message

    raw_content = getattr(message, "content", "") or ""
    raw_text = _message_content_to_text(raw_content)
    parsed = _try_parse_json(raw_text)

    if parsed is None:
        return {
            "ok": True,
            "provider": "litellm",
            "model": settings.model,
            "summary": "LiteLLM returned non-JSON output.",
            "mappings": [],
            "warnings": ["Model output could not be parsed as JSON."],
            "execution_plan": execution_plan,
            "tool_results": tool_results,
            "raw_model_output": raw_text,
        }

    normalized_mappings, normalization_warnings = _normalize_generated_mappings(
        parsed_output=parsed,
        candidates=candidates,
    )

    parsed_warnings = parsed.get("warnings", [])
    if not isinstance(parsed_warnings, list):
        parsed_warnings = [str(parsed_warnings)]

    return {
        "ok": True,
        "provider": "litellm",
        "model": settings.model,
        "summary": parsed.get("summary", ""),
        "mappings": normalized_mappings,
        "warnings": parsed_warnings + normalization_warnings,
        "execution_plan": execution_plan,
        "tool_results": tool_results,
        "raw_model_output": raw_text,
    }


def generate_answer(
    *,
    report_text: str,
    candidates: list[dict[str, Any]],
    risk_level: RiskLevel = "normal",
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """
    Main generation entrypoint.

    Input:
        report_text:
            threat report body text

        candidates:
            expanded candidates from retrieval.py + candidate_expansion.py

    Steps:
        1. planner.py builds model route and tool plan
        2. tools.py executes planned tools
        3. LiteLLM generates final output if API key exists
        4. deterministic fallback runs otherwise

    Output:
        stable dict with provider, mappings, tool_results, and execution_plan
    """
    settings = settings or get_settings()

    execution_plan = build_execution_plan(
        report_text=report_text,
        candidates=candidates,
        task_type="grounded_explanation",
        risk_level=risk_level,
        settings=settings,
    )

    tool_results = execute_tool_plan(execution_plan.get("tool_plan", []))
    model_route = execution_plan.get("model_route", {})

    if model_route.get("provider") != "litellm":
        return generate_deterministic_response(
            report_text=report_text,
            candidates=candidates,
            execution_plan=execution_plan,
            tool_results=tool_results,
            settings=settings,
        )

    try:
        return generate_with_litellm(
            report_text=report_text,
            candidates=candidates,
            execution_plan=execution_plan,
            tool_results=tool_results,
            settings=settings,
        )

    except Exception as exc:
        fallback = generate_deterministic_response(
            report_text=report_text,
            candidates=candidates,
            execution_plan=execution_plan,
            tool_results=tool_results,
            settings=settings,
        )

        fallback["warnings"].append(
            f"LiteLLM generation failed, fallback used: {type(exc).__name__}: {exc}"
        )

        return fallback
