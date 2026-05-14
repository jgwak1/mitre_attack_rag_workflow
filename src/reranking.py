"""
Optional reranking step for ATT&CK candidate techniques.

The current pipeline does not use this file yet.

Current behavior:
    retrieval.py and candidate_expansion.py produce the candidate order.
    generation.py preserves that order and adds grounded explanations.

Why reranking may be useful later:
    Initial retrieval is usually optimized for recall. It tries to recover the
    right techniques somewhere in the candidate pool, even if a few noisy
    candidates are included.

    Reranking is a second pass over that candidate pool. Its job is to improve
    precision by moving better-supported candidates higher and pushing weak or
    accidental matches lower.

Possible future reranking signals:
    - cross-encoder similarity between report text and technique text
    - LLM-based tie-breaking for close candidates
    - retrieval score plus heuristic evidence
    - whether matched evidence appears directly in the report
    - learned ranking from labeled evaluation examples

For now, rerank_candidates() returns candidates unchanged. This avoids changing
ranking behavior before reranking has its own evaluation and regression checks.
"""

from __future__ import annotations

from typing import Any


def rerank_candidates(
    candidates: list[dict[str, Any]],
    *,
    report_text: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return candidates unchanged.

    This keeps the future reranking step visible without affecting the current
    pipeline. Once implemented, this function should take the initial retrieved
    candidate pool and return the same candidates in a better-supported order.
    """
    return candidates