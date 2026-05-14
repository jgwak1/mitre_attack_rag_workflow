"""
Trace utilities for recording analysis runs.

This module only records pipeline outputs. It does not decide which pipeline
steps to run.

The actual runner should call retrieval, candidate expansion, planning, tools,
and generation, then pass those outputs into build_trace().

The trace can be saved as JSON now and later persisted through database.py.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return str(uuid.uuid4())


def _safe_json(value: Any) -> Any:
    """
    Convert arbitrary pipeline objects into JSON-serializable values.

    This is needed because some outputs may be Pydantic models, LiteLLM message
    objects, tuples, or custom tool-result objects instead of plain dicts.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [_safe_json(v) for v in value]

    if isinstance(value, tuple):
        return [_safe_json(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}

    if hasattr(value, "model_dump"):
        return _safe_json(value.model_dump())

    if hasattr(value, "__dict__"):
        return _safe_json(value.__dict__)

    return str(value)


def _get_mitre_definition(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the nested MITRE definition block from a retrieval/expanded candidate."""
    technique = candidate.get("technique") or {}
    if not isinstance(technique, dict):
        return {}

    mitre_definition = technique.get("mitre_definition") or {}
    if not isinstance(mitre_definition, dict):
        return {}

    return mitre_definition


def summarize_candidates(candidates: list[Any]) -> list[dict[str, Any]]:
    """
    Create a compact trace-friendly summary of retrieval or expanded candidates.

    Retrieval candidates contain the ATT&CK record under:
        candidate["technique"]["mitre_definition"]

    Expanded candidates may additionally contain:
        candidate["sources"]
        candidate["heuristic_matches"]
    """

    summary = []

    for fallback_rank, candidate in enumerate(candidates, start=1):
        c = _safe_json(candidate)

        if not isinstance(c, dict):
            summary.append({"rank": fallback_rank, "raw": c})
            continue

        mitre = _get_mitre_definition(c)

        heuristic_matches = c.get("heuristic_matches") or []
        matched_phrases = []
        if isinstance(heuristic_matches, list):
            matched_phrases = [
                match.get("matched_phrase")
                for match in heuristic_matches
                if isinstance(match, dict) and match.get("matched_phrase")
            ]

        summary.append(
            {
                "rank": c.get("rank", fallback_rank),
                "technique_id": c.get("technique_id") or mitre.get("technique_id"),
                "name": mitre.get("name"),
                "tactic": mitre.get("tactic"),
                "score": c.get("score"),
                "method": c.get("method"),
                "sources": c.get("sources"),
                "matched_phrases": matched_phrases,
                "heuristic_matches": heuristic_matches,
                "source_url": mitre.get("source_url"),
            }
        )

    return summary


def build_trace(
    *,
    report_text: str,
    ingestion_output: Any | None = None,
    retrieval_method: str | None = None,
    retrieved_candidates: list[Any] | None = None,
    expanded_candidates: list[Any] | None = None,
    execution_plan: Any | None = None,
    tool_results: list[Any] | None = None,
    generation_output: Any | None = None,
    evaluation_output: Any | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    generation = _safe_json(generation_output)

    trace = {
        "run_id": run_id or new_run_id(),
        "created_at": utc_now_iso(),
        "input": {
            "report_text": report_text,
            "report_char_count": len(report_text),
        },
        "ingestion": _safe_json(ingestion_output),
        "retrieval": {
            "method": retrieval_method,
            "retrieved_candidates": summarize_candidates(retrieved_candidates or []),
        },
        "candidate_expansion": {
            "expanded_candidates": summarize_candidates(expanded_candidates or []),
        },
        "planning": _safe_json(execution_plan),
        "tools": {
            "tool_results": _safe_json(tool_results or []),
        },
        "generation": generation,
        "evaluation": _safe_json(evaluation_output),
    }

    return trace


def save_trace_json(trace: dict[str, Any], output_dir: str | Path = "traces") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = trace.get("run_id") or new_run_id()
    file_path = output_path / f"{run_id}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    return file_path