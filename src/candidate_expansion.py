"""
Candidate expansion using analyst heuristic signatures.

This module scans ingested threat-report text for analyst-curated behavior
signatures and adds candidate ATT&CK technique IDs to the candidate pool.

It does not modify the report text.
It does not make the final ATT&CK mapping decision.
It only adds heuristic candidates that retrieval and generation can consider later.

Example:
    "encoded powershell" -> ["T1059.001", "T1027"]
"""

from __future__ import annotations

from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Read a field from either a dict or a Pydantic model.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def find_heuristic_candidates(
    report_text: str,
    heuristic_data: Any,
) -> list[dict[str, Any]]:
    """
    Scan report text for analyst-curated heuristic signatures.

    Supports both shapes:
    1. Raw JSON shape:
       {
         "signatures": {
           "encoded powershell": {
             "candidate_techniques": ["T1059.001", "T1027"],
             "note": "..."
           }
         }
       }

    2. Loaded list/Pydantic shape:
       [
         {
           "phrase": "encoded powershell",
           "candidate_techniques": ["T1059.001", "T1027"],
           "note": "..."
         }
       ]
    """
    if not report_text.strip():
        return []

    text = report_text.lower()
    candidates: list[dict[str, Any]] = []

    raw_signatures = _get(heuristic_data, "signatures", None)

    # Case 1: raw JSON dictionary where phrase is the dict key.
    if isinstance(raw_signatures, dict):
        iterable = [
            (
                phrase,
                _get(info, "candidate_techniques", []) or [],
                _get(info, "note", "") or "",
                info,
            )
            for phrase, info in raw_signatures.items()
        ]

    # Case 2: loader already returned a list of signature records.
    elif isinstance(heuristic_data, list):
        iterable = [
            (
                _get(sig, "phrase", None)
                or _get(sig, "signature", None)
                or _get(sig, "behavior_phrase", None)
                or _get(sig, "match_text", None),
                _get(sig, "candidate_techniques", None)
                or _get(sig, "technique_ids", None)
                or _get(sig, "candidate_technique_ids", None)
                or [],
                _get(sig, "note", "") or "",
                sig,
            )
            for sig in heuristic_data
        ]

    else:
        iterable = []

    for phrase, technique_ids, note, signature_info in iterable:
        if not phrase:
            continue

        phrase_text = str(phrase).strip()
        if not phrase_text:
            continue

        if phrase_text.lower() not in text:
            continue

        for technique_id in technique_ids:
            candidates.append(
                {
                    "technique_id": str(technique_id),
                    "source": "analyst_heuristic_signature",
                    "matched_phrase": phrase_text,
                    "reason": note
                    or f'Matched analyst heuristic phrase: "{phrase_text}"',
                    "signature": signature_info,
                }
            )

    return candidates


def merge_candidates(
    retrieval_candidates: list[dict[str, Any]],
    heuristic_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge retrieval candidates and heuristic candidates by technique_id.

    Retrieval candidates come from retrieval.py.
    Heuristic candidates come from analyst heuristic signatures.

    The merged result preserves:
    - retrieval rank and score
    - retrieval method
    - full technique object when available
    - matched heuristic phrases
    - candidate sources
    """
    merged: dict[str, dict[str, Any]] = {}

    # First add retrieval candidates.
    # These already have rank, score, method, technique, searchable_text.
    for candidate in retrieval_candidates:
        technique_id = str(candidate["technique_id"])

        merged[technique_id] = {
            **candidate,
            "sources": ["retrieval"],
            "retrieval": candidate,
            "heuristic_matches": [],
        }

    # Then add heuristic candidates.
    # If the technique already came from retrieval, attach the heuristic evidence.
    # If not, create a heuristic-only candidate.
    for candidate in heuristic_candidates:
        technique_id = str(candidate["technique_id"])

        if technique_id not in merged:
            merged[technique_id] = {
                "technique_id": technique_id,
                "rank": None,
                "score": 0.0,
                "method": None,
                "technique": None,
                "searchable_text": None,
                "sources": [],
                "retrieval": None,
                "heuristic_matches": [],
            }

        if "analyst_heuristic_signature" not in merged[technique_id]["sources"]:
            merged[technique_id]["sources"].append("analyst_heuristic_signature")

        merged[technique_id]["heuristic_matches"].append(candidate)

    def sort_key(candidate: dict[str, Any]) -> tuple[int, float]:
        """
        Keep retrieval-ranked candidates first.

        Heuristic-only candidates are still included, but appear after
        retrieval-ranked candidates because they do not have retrieval scores.
        """
        rank = candidate.get("rank")
        score = float(candidate.get("score") or 0.0)

        if rank is None:
            return (10_000, -score)

        return (int(rank), -score)

    return sorted(merged.values(), key=sort_key)


def expand_candidates(
    report_text: str,
    retrieval_candidates: list[dict[str, Any]],
    heuristic_data: Any,
) -> list[dict[str, Any]]:
    """
    Candidate expansion stage.

    Input:
    - report_text
    - retrieval candidates from retrieval.py
    - analyst heuristic signature data from data_sources.py

    Output:
    - merged candidate pool
    - deduped by technique_id
    - retrieval evidence preserved
    - heuristic evidence preserved
    """
    heuristic_candidates = find_heuristic_candidates(
        report_text=report_text,
        heuristic_data=heuristic_data,
    )

    return merge_candidates(
        retrieval_candidates=retrieval_candidates,
        heuristic_candidates=heuristic_candidates,
    )