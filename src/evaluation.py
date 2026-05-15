from __future__ import annotations

import math
from typing import Any


"""
Evaluation utilities for the ATT&CK mapping workflow.

The current implementation focuses on two evaluation layers, with additional
tool-use, reranking, semantic-grounding, and judge/rubric-based evaluation left
as future extensions.


1. Retrieval / candidate-pool evaluation
   Goal:
       Check whether retrieval and candidate expansion recover expected ATT&CK
       techniques into the top-k candidate pool before generation.

   Why this matters:
       This is a candidate-first TTP mapping pipeline. The generator is expected
       to reason over retrieved and expanded candidates. If retrieval drops the
       correct technique, that should be counted as a retrieval/candidate-pool
       failure, not as a generation failure.

   Main metric:
       Recall@k:
           Measures whether expected techniques were recovered into the
           candidate pool. This is the primary retrieval-stage signal because
           retrieval's first job is to avoid dropping the correct technique.

   Other metrics:
       Precision@k:
           Measures how much noise is included in the candidate pool.

       Hit Rate@k:
           Measures whether at least one expected technique was recovered.

       MRR@k:
           Measures how early the first correct technique appears.

       NDCG@k:
           Measures whether correct techniques are ranked near the top.

       Coverage@k:
           Measures whether the retriever filled the requested top-k slots.
           This is a pipeline-health signal, not a correctness signal.

2. Generated-output evaluation
   Goal:
       Check the generated mapping as structured output, component by component.

   Example generated mappings:
       [
           {
               "technique_id": "T1059.001",
               "confidence_label": "high",
               "confidence_score": 0.95,
               "evidence": "encoded powershell",
               "rationale": "The report describes encoded PowerShell execution."
           },
           {
               "technique_id": "T1027",
               "confidence_label": "high",
               "confidence_score": 0.90,
               "evidence": "encoded powershell",
               "rationale": "The encoded command suggests obfuscation."
           }
       ]

   Components checked:
       Technique IDs:
           Generated technique IDs should stay inside the retrieved/expanded
           candidate pool. If the generator outputs a technique outside the
           pool, that is treated as a candidate-pool contract violation.

       Expected technique selection:
           Generation is evaluated against the candidate pool it received. If an
           expected technique (ground-truth technique of report) 
           is missing from the candidate pool, that is a retrieval failure. 
           If it is present in the candidate pool but absent
           from generated mappings, that is a generation selection failure.

       Confidence labels:
           confidence_label should be one of high, medium, or low.
           confidence_score may be present, but it is treated as a heuristic
           score, not a calibrated probability.

       Evidence grounding:
           For high-confidence mappings, the evidence field should be grounded
           when possible. This check currently uses a deterministic proxy:
           evidence text appears in the report text or in a heuristic phrase
           already matched from the report.

           This does not mean every valid mapping must have exact-string
           evidence. Some evidence is semantic. Those deeper cases are left for
           a later judge/rubric evaluator.

       Rationale presence:
           Each mapping should include a non-empty rationale. This does not prove
           the rationale is correct, but it makes the mapping inspectable during
           regression debugging.

       Warning/tool metadata:
           Warning count and tool-result count are recorded so regressions can
           track weak-evidence handling and tool execution behavior.

   Current scope:
       This is a deterministic regression check. It catches structural failures,
       candidate-pool violations, missing generated selections for candidates
       that were available, invalid confidence labels, and obvious ungrounded
       high-confidence evidence.

       It does not fully judge semantic explanation quality. A later evaluator
       can add a domain-specific human rubric or LLM-as-judge for:
           - rationale faithfulness
           - technique correctness
           - semantic evidence alignment
           - analyst usefulness

       That judge-based layer is kept separate because it is slower, more
       expensive, and less deterministic than the regression checks here.
"""


VALID_CONFIDENCE_LABELS = {"high", "medium", "low"}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or a Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ordered_unique(values: list[str]) -> list[str]:
    """Return values in first-seen order without duplicates."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def _normalize_text(text: Any) -> str:
    """Normalize text for simple string-based grounding checks."""
    return " ".join(str(text).lower().split())


def extract_candidate_ids(
    candidates: list[dict[str, Any]],
    k: int | None = None,
) -> list[str]:
    """
    Extract ordered, deduplicated ATT&CK technique IDs from candidates.
    """
    selected = candidates if k is None else candidates[:k]

    ids = [
        str(_get(candidate, "technique_id"))
        for candidate in selected
        if _get(candidate, "technique_id")
    ]

    return _ordered_unique(ids)


def _dcg(relevance: list[int]) -> float:
    """Discounted cumulative gain. Correct items near the top get more credit."""
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevance)
    )


def evaluate_retrieval_at_k(
    candidates: list[dict[str, Any]],
    expected_techniques: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """
    Evaluate retrieval / candidate-expansion quality.

    This evaluates the candidate pool before generation. For TTP mapping,
    Recall@k is the key metric because retrieval first needs to recover the
    expected technique somewhere in the top-k candidates.

    Precision@k and ranking metrics are still useful because they show how much
    noise the generator receives and whether correct candidates appear early.
    """
    expected = [str(t) for t in expected_techniques]
    expected_set = set(expected)

    predicted = extract_candidate_ids(candidates, k=k)
    predicted_set = set(predicted)

    matched = sorted(expected_set & predicted_set)
    missing = sorted(expected_set - predicted_set)

    num_expected = len(expected_set)
    num_predicted = len(predicted)
    num_matched = len(matched)

    recall_at_k = num_matched / num_expected if num_expected else 1.0
    precision_at_k = num_matched / num_predicted if num_predicted else 0.0
    hit_rate_at_k = 1.0 if num_matched else 0.0
    coverage_at_k = num_predicted / k if k > 0 else 0.0

    first_relevant_rank: int | None = None
    for idx, technique_id in enumerate(predicted, start=1):
        if technique_id in expected_set:
            first_relevant_rank = idx
            break

    mrr_at_k = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    relevance = [
        1 if technique_id in expected_set else 0
        for technique_id in predicted
    ]

    ideal_relevance = [1] * min(num_expected, k)
    ideal_dcg = _dcg(ideal_relevance)
    ndcg_at_k = _dcg(relevance) / ideal_dcg if ideal_dcg > 0 else 0.0

    return {
        "eval_type": "retrieval",
        "k": k,
        "expected_techniques": expected,
        "predicted_techniques": predicted,
        "matched_techniques": matched,
        "missing_techniques": missing,
        "num_expected": num_expected,
        "num_predicted": num_predicted,
        "num_matched": num_matched,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "hit_rate_at_k": hit_rate_at_k,
        "mrr_at_k": mrr_at_k,
        "ndcg_at_k": ndcg_at_k,
        "coverage_at_k": coverage_at_k,
        "passed": len(missing) == 0,
    }


def evaluate_report_candidates(
    report_id: str,
    candidates: list[dict[str, Any]],
    expected_techniques: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate one report's retrieved / expanded candidate set."""
    result = evaluate_retrieval_at_k(
        candidates=candidates,
        expected_techniques=expected_techniques,
        k=k,
    )

    return {
        "report_id": report_id,
        **result,
    }


def _extract_generated_mappings(
    generation_output: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract mapping objects from generation output."""
    mappings = _get(generation_output, "mappings", []) or []
    return [mapping for mapping in mappings if isinstance(mapping, dict)]


def _extract_mapping_ids(mappings: list[dict[str, Any]]) -> list[str]:
    """Extract ordered technique IDs from generated mappings."""
    ids = [
        str(_get(mapping, "technique_id"))
        for mapping in mappings
        if _get(mapping, "technique_id")
    ]

    return _ordered_unique(ids)


def _extract_heuristic_phrases(candidates: list[dict[str, Any]]) -> list[str]:
    """
    Extract heuristic phrases attached during candidate expansion.

    These are secondary grounding signals. They are not required for every
    evidence phrase, but they are valid grounding support when they were matched
    from the report text.
    """
    phrases: list[str] = []

    for candidate in candidates:
        for phrase in _get(candidate, "matched_phrases", []) or []:
            if phrase:
                phrases.append(str(phrase))

        for match in _get(candidate, "heuristic_matches", []) or []:
            phrase = _get(match, "matched_phrase")
            if phrase:
                phrases.append(str(phrase))

    return _ordered_unique(phrases)


def _evidence_is_grounded(
    evidence: Any,
    *,
    report_text: str,
    heuristic_phrases: list[str],
) -> bool:
    """
    Check whether a generated evidence phrase is grounded.

    Strongest case:
        evidence appears directly in the report text.

    Secondary case:
        evidence matches a heuristic phrase that was already matched from the
        report by candidate expansion.

    This is only a deterministic faithfulness proxy. It does not judge whether
    the full rationale is technically correct.
    """
    if evidence is None:
        return False

    evidence_text = _normalize_text(evidence)
    if not evidence_text:
        return False

    if evidence_text in _normalize_text(report_text):
        return True

    return evidence_text in {_normalize_text(p) for p in heuristic_phrases}


def _confidence_score_is_valid(score: Any) -> bool:
    """
    Check whether confidence_score is numeric and in [0, 1].

    confidence_score is treated as a heuristic score, not a calibrated
    probability.
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False

    return 0.0 <= value <= 1.0


def evaluate_generated_output(
    *,
    report_id: str,
    report_text: str,
    generation_output: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    expected_techniques: list[str] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Evaluate generated ATT&CK mappings as structured output.

    This checks generation only against the candidate pool it received.

    If an expected technique is missing from the candidate pool, that is a
    retrieval/candidate-pool failure.

    If an expected technique is present in the candidate pool but missing from
    generated mappings, that is a generation selection failure.

    Evidence grounding here is tracked as a deterministic proxy. It checks exact
    evidence phrases against the report text and matched heuristic phrases.

    This proxy is diagnostic rather than a hard pass/fail condition, because valid
    evidence can be semantic instead of an exact string match. Full semantic
    evidence alignment should be handled by a later rubric or LLM-as-judge layer.
    """
    expected = [str(t) for t in (expected_techniques or [])]
    expected_set = set(expected)

    candidate_ids = extract_candidate_ids(candidate_pool, k=None)
    candidate_id_set = set(candidate_ids)

    mappings_raw = _get(generation_output, "mappings", None)
    mappings = _extract_generated_mappings(generation_output)
    generated_ids = _extract_mapping_ids(mappings)
    generated_id_set = set(generated_ids)

    expected_in_candidate_pool = sorted(expected_set & candidate_id_set)
    missing_expected_from_candidate_pool = sorted(expected_set - candidate_id_set)
    matched_expected_techniques = sorted(expected_set & generated_id_set)

    missing_expected_from_generation = sorted(
        set(expected_in_candidate_pool) - generated_id_set
    )

    out_of_pool_ids = sorted(generated_id_set - candidate_id_set)
    heuristic_phrases = _extract_heuristic_phrases(candidate_pool)

    missing_required_field_indices: list[int] = []
    invalid_confidence_ids: list[str] = []
    missing_confidence_score_ids: list[str] = []
    invalid_confidence_score_ids: list[str] = []
    high_confidence_without_grounded_evidence: list[str] = []
    empty_rationale_ids: list[str] = []

    for idx, mapping in enumerate(mappings):
        technique_id = _get(mapping, "technique_id")
        confidence_label = _get(mapping, "confidence_label")
        confidence_score = _get(mapping, "confidence_score")
        evidence = _get(mapping, "evidence")
        rationale = _get(mapping, "rationale")

        if not technique_id or not confidence_label:
            missing_required_field_indices.append(idx)
            continue

        technique_id = str(technique_id)

        if confidence_label not in VALID_CONFIDENCE_LABELS:
            invalid_confidence_ids.append(technique_id)

        if confidence_score is None:
            missing_confidence_score_ids.append(technique_id)
        elif not _confidence_score_is_valid(confidence_score):
            invalid_confidence_score_ids.append(technique_id)

        if not rationale or not str(rationale).strip():
            empty_rationale_ids.append(technique_id)

        if confidence_label == "high":
            grounded = _evidence_is_grounded(
                evidence,
                report_text=report_text,
                heuristic_phrases=heuristic_phrases,
            )

            if not grounded:
                high_confidence_without_grounded_evidence.append(technique_id)

    warnings = _get(generation_output, "warnings", []) or []
    warning_count = len(warnings) if isinstance(warnings, list) else 0

    if tool_results is None:
        tool_results = _get(generation_output, "tool_results", []) or []

    tool_result_count = len(tool_results) if isinstance(tool_results, list) else 0

    schema_valid = (
        isinstance(generation_output, dict)
        and isinstance(mappings_raw, list)
        and len(missing_required_field_indices) == 0
    )

    passed = (
        schema_valid
        and len(out_of_pool_ids) == 0
        and len(missing_expected_from_generation) == 0
        and len(invalid_confidence_ids) == 0
        and len(invalid_confidence_score_ids) == 0
        and len(empty_rationale_ids) == 0
    )

    return {
        "report_id": report_id,
        "eval_type": "generation",
        "expected_techniques": expected,
        "candidate_techniques": candidate_ids,
        "generated_techniques": generated_ids,
        "expected_in_candidate_pool": expected_in_candidate_pool,
        "missing_expected_from_candidate_pool": missing_expected_from_candidate_pool,
        "matched_expected_techniques": matched_expected_techniques,
        "missing_expected_from_generation": missing_expected_from_generation,
        "out_of_pool_techniques": out_of_pool_ids,
        "schema_valid": schema_valid,
        "mappings_count": len(mappings),
        "missing_required_field_indices": missing_required_field_indices,
        "invalid_confidence_ids": invalid_confidence_ids,
        "missing_confidence_score_ids": missing_confidence_score_ids,
        "invalid_confidence_score_ids": invalid_confidence_score_ids,
        "high_confidence_without_grounded_evidence": (
            high_confidence_without_grounded_evidence
        ),
        "empty_rationale_ids": empty_rationale_ids,
        "warning_count": warning_count,
        "tool_result_count": tool_result_count,
        "passed": passed,
    }


def summarize_evaluations(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate report-level evaluations for regression.py.

    Retrieval averages are computed from retrieval eval records only.
    Generation checks are summarized by pass/fail counts.
    """
    total_reports = len(evaluations)
    passed_reports = sum(1 for result in evaluations if result.get("passed"))

    retrieval_results = [
        result for result in evaluations
        if result.get("eval_type") == "retrieval"
    ]

    generation_results = [
        result for result in evaluations
        if result.get("eval_type") == "generation"
    ]

    def avg(metric: str) -> float:
        values = [
            float(result.get(metric, 0.0))
            for result in retrieval_results
            if metric in result
        ]
        return sum(values) / len(values) if values else 0.0

    failed_reports = [
        {
            "report_id": result.get("report_id"),
            "eval_type": result.get("eval_type"),
            "missing_techniques": (
                result.get("missing_techniques")
                or result.get("missing_expected_from_generation", [])
            ),
            "missing_expected_from_candidate_pool": result.get(
                "missing_expected_from_candidate_pool",
                [],
            ),
            "predicted_techniques": (
                result.get("predicted_techniques")
                or result.get("generated_techniques", [])
            ),
            "out_of_pool_techniques": result.get("out_of_pool_techniques", []),
            "invalid_confidence_ids": result.get("invalid_confidence_ids", []),
            "invalid_confidence_score_ids": result.get(
                "invalid_confidence_score_ids",
                [],
            ),
            "high_confidence_without_grounded_evidence": result.get(
                "high_confidence_without_grounded_evidence",
                [],
            ),
            "empty_rationale_ids": result.get("empty_rationale_ids", []),
        }
        for result in evaluations
        if not result.get("passed")
    ]

    generation_failures = [
        result for result in generation_results
        if not result.get("passed")
    ]

    return {
        "total_reports": total_reports,
        "passed_reports": passed_reports,
        "failed_reports": failed_reports,
        "average_recall_at_k": avg("recall_at_k"),
        "average_precision_at_k": avg("precision_at_k"),
        "average_hit_rate_at_k": avg("hit_rate_at_k"),
        "average_mrr_at_k": avg("mrr_at_k"),
        "average_ndcg_at_k": avg("ndcg_at_k"),
        "average_coverage_at_k": avg("coverage_at_k"),
        "generation_eval_count": len(generation_results),
        "generation_failure_count": len(generation_failures),
        "passed": len(failed_reports) == 0,
    }