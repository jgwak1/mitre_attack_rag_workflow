from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.candidate_expansion import expand_candidates
from src.data_sources import (
    load_analyst_heuristic_signatures,
    load_attack_techniques,
    load_sample_reports,
)
from src.evaluation import (
    evaluate_generated_output,
    evaluate_report_candidates,
    summarize_evaluations,
)
from src.indexing import build_index_documents
from src.pipeline import analyze_report
from src.retrieval import retrieve_top_k


DEFAULT_BASELINE_PATH = Path("baselines/regression_bm25_k5.json")


"""
Regression suites for the ATT&CK mapping workflow.

This file evaluates labeled sample reports through the retrieval and generation
paths so changes can be checked against known expected ATT&CK labels and the
expected structure of generated mappings, such as technique IDs, confidence
fields, evidence, rationales, warnings, and tool metadata.

There are two comparison modes:

1. Fixed-case regression guardrail
   This does not require a saved baseline JSON. The sample reports already have
   expected ATT&CK labels, so the suite can check whether known behavior still
   holds after code, retrieval, prompt, tool, or pipeline changes.

   Example:
       r1 is expected to recover T1059.001, T1105, and T1027.
       If a later retrieval change drops one of them from top-k, the suite fails.

2. Saved-baseline comparison
   If a baseline JSON exists, the suite also compares current summary metrics
   against the accepted previous result.

   Example:
       current average_recall_at_k should stay >= baseline average_recall_at_k
       current generation failure counts should stay <= baseline failure counts

The retrieval suite exercises ingestion as part of the retrieval path:
    report text -> ingestion body -> retrieval -> candidate expansion -> eval

It does not unit-test ingestion normalization by itself. If ingestion breaks the
text that retrieval receives, the regression should surface it through candidate
recovery failures.

The generation suite runs the shared pipeline and evaluates structured LLM
output:
    report text -> pipeline.analyze_report() -> evaluate_generated_output()

Retrieval metrics are quantitative scores such as Recall@k, Precision@k, MRR@k,
NDCG@k, and Coverage@k.

Generation metrics are count-based checks such as missing expected techniques
from generated mappings, out-of-pool generated techniques, invalid confidence
fields, exact-grounding misses for high-confidence evidence, empty rationales,
and tool-result counts. Exact-grounding misses are tracked for review but are
not treated as hard failures because evidence can be semantic.
"""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or a Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_nested(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Read a nested dict value using a tuple path."""
    current: Any = obj

    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def _ingest_report_body(report_text: str) -> str:
    """
    Run ingestion.py and return the body text used for retrieval.

    This regression path tests ingestion indirectly. It checks whether the text
    produced by ingestion still lets retrieval recover the expected techniques.
    It is not a standalone ingestion unit test.
    """
    try:
        import src.ingestion as ingestion
    except ImportError:
        return report_text

    ingest_fn = None

    for fn_name in ("ingest_report", "ingest_threat_report", "ingest"):
        candidate_fn = getattr(ingestion, fn_name, None)
        if callable(candidate_fn):
            ingest_fn = candidate_fn
            break

    if ingest_fn is None:
        return report_text

    ingested = ingest_fn(report_text)

    body = _get(ingested, "body")
    if body:
        return str(body)

    cleaned_text = _get(ingested, "cleaned_text")
    if cleaned_text:
        return str(cleaned_text)

    if isinstance(ingested, str):
        return ingested

    return report_text


def run_one_report_retrieval_regression(
    report: Any,
    *,
    index_documents: list[dict[str, Any]],
    heuristic_data: Any,
    retrieval_method: str = "bm25",
    k: int = 5,
) -> dict[str, Any]:
    """
    Run one sample report through retrieval regression.

    This checks whether ingestion, retrieval, and candidate expansion recover
    expected ATT&CK techniques into the top-k candidate pool.
    """
    report_id = str(_get(report, "report_id", "unknown_report"))
    report_text = str(_get(report, "report_text", ""))
    expected_techniques = list(_get(report, "expected_techniques", []) or [])

    body_text = _ingest_report_body(report_text)

    retrieval_candidates = retrieve_top_k(
        query_text=body_text,
        index_documents=index_documents,
        k=k,
        method=retrieval_method,
    )

    expanded_candidates = expand_candidates(
        report_text=body_text,
        retrieval_candidates=retrieval_candidates,
        heuristic_data=heuristic_data,
    )

    evaluation = evaluate_report_candidates(
        report_id=report_id,
        candidates=expanded_candidates,
        expected_techniques=expected_techniques,
        k=k,
    )

    return {
        **evaluation,
        "retrieval_method": retrieval_method,
        "candidate_count": len(expanded_candidates),
        "expanded_candidates": [
            {
                "rank": _get(candidate, "rank"),
                "technique_id": _get(candidate, "technique_id"),
                "score": _get(candidate, "score"),
                "method": _get(candidate, "method"),
                "sources": _get(candidate, "sources", []),
                "heuristic_matches": [
                    _get(match, "matched_phrase")
                    for match in _get(candidate, "heuristic_matches", [])
                ],
            }
            for candidate in expanded_candidates
        ],
    }


def run_retrieval_regression_suite(
    *,
    retrieval_method: str = "bm25",
    k: int = 5,
) -> dict[str, Any]:
    """
    Run retrieval regression over fixed sample reports.

    Output is quantitative:
        Recall@k, Precision@k, Hit Rate@k, MRR@k, NDCG@k, Coverage@k
    """
    techniques = load_attack_techniques()
    reports = load_sample_reports()
    heuristic_data = load_analyst_heuristic_signatures()
    index_documents = build_index_documents(techniques)

    evaluations = [
        run_one_report_retrieval_regression(
            report,
            index_documents=index_documents,
            heuristic_data=heuristic_data,
            retrieval_method=retrieval_method,
            k=k,
        )
        for report in reports
    ]

    return {
        "suite": "retrieval_sample_reports",
        "retrieval_method": retrieval_method,
        "k": k,
        "summary": summarize_evaluations(evaluations),
        "reports": evaluations,
    }


def run_regression_suite(
    *,
    retrieval_method: str = "bm25",
    k: int = 5,
) -> dict[str, Any]:
    """
    Backward-compatible alias for existing tests.

    Existing pytest tests call run_regression_suite(), so this keeps that name
    mapped to the retrieval regression suite.
    """
    return run_retrieval_regression_suite(
        retrieval_method=retrieval_method,
        k=k,
    )


def _extract_candidates_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Read expanded candidates from a pipeline trace."""
    candidate_expansion = trace.get("candidate_expansion") or {}
    candidates = candidate_expansion.get("expanded_candidates") or []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def run_one_report_generation_regression(
    report: Any,
    *,
    retrieval_method: str = "bm25",
    k: int = 5,
) -> dict[str, Any]:
    """
    Run one sample report through generation regression.

    This runs the full pipeline but disables JSON/SQLite persistence so the
    regression suite does not create run artifacts.
    """
    report_id = str(_get(report, "report_id", "unknown_report"))
    report_text = str(_get(report, "report_text", ""))
    expected_techniques = list(_get(report, "expected_techniques", []) or [])

    pipeline_result = analyze_report(
        report_text=report_text,
        retrieval_method=retrieval_method,
        k=k,
        save_json_trace=False,
        save_db_trace=False,
    )

    generation_output = pipeline_result.get("generation") or {}
    trace = pipeline_result.get("trace") or {}
    candidate_pool = _extract_candidates_from_trace(trace)

    tool_results = []
    if isinstance(generation_output, dict):
        tool_results = generation_output.get("tool_results", []) or []

    evaluation = evaluate_generated_output(
        report_id=report_id,
        report_text=report_text,
        generation_output=generation_output,
        candidate_pool=candidate_pool,
        expected_techniques=expected_techniques,
        tool_results=tool_results,
    )

    return {
        **evaluation,
        "retrieval_method": retrieval_method,
        "candidate_count": len(candidate_pool),
        "provider": (
            generation_output.get("provider")
            if isinstance(generation_output, dict)
            else None
        ),
        "model": (
            generation_output.get("model")
            if isinstance(generation_output, dict)
            else None
        ),
    }


def _add_generation_summary_counts(
    summary: dict[str, Any],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Add count-based generation metrics.

    These counts make generation regression comparable across runs without
    requiring exact natural-language equality.
    """
    total_reports = len(reports)
    passed_reports = sum(1 for report in reports if report.get("passed"))

    summary["generation_pass_rate"] = (
        passed_reports / total_reports if total_reports else 0.0
    )

    summary["total_missing_expected_from_candidate_pool"] = sum(
        len(report.get("missing_expected_from_candidate_pool", []))
        for report in reports
    )

    summary["total_missing_expected_from_generation"] = sum(
        len(report.get("missing_expected_from_generation", []))
        for report in reports
    )

    summary["total_out_of_pool_techniques"] = sum(
        len(report.get("out_of_pool_techniques", []))
        for report in reports
    )

    summary["total_invalid_confidence_ids"] = sum(
        len(report.get("invalid_confidence_ids", []))
        for report in reports
    )

    summary["total_invalid_confidence_score_ids"] = sum(
        len(report.get("invalid_confidence_score_ids", []))
        for report in reports
    )

    summary["total_ungrounded_high_confidence"] = sum(
        len(report.get("high_confidence_without_grounded_evidence", []))
        for report in reports
    )

    summary["total_empty_rationale_ids"] = sum(
        len(report.get("empty_rationale_ids", []))
        for report in reports
    )

    summary["total_tool_result_count"] = sum(
        int(report.get("tool_result_count", 0))
        for report in reports
    )

    return summary


def run_generation_regression_suite(
    *,
    retrieval_method: str = "bm25",
    k: int = 5,
) -> dict[str, Any]:
    """
    Run generation regression over fixed sample reports.

    Output is count-based rather than exact text matching:
        generation pass rate
        missing expected selections
        out-of-pool mappings
        invalid confidence fields
        ungrounded high-confidence evidence
        empty rationales
        tool-result counts
    """
    reports = load_sample_reports()

    evaluations = [
        run_one_report_generation_regression(
            report,
            retrieval_method=retrieval_method,
            k=k,
        )
        for report in reports
    ]

    summary = summarize_evaluations(evaluations)
    summary = _add_generation_summary_counts(summary, evaluations)

    return {
        "suite": "generation_sample_reports",
        "retrieval_method": retrieval_method,
        "k": k,
        "summary": summary,
        "reports": evaluations,
    }


BASELINE_METRICS: dict[tuple[str, ...], str] = {
    ("retrieval_suite", "summary", "average_recall_at_k"): "higher",
    ("retrieval_suite", "summary", "average_precision_at_k"): "higher",
    ("retrieval_suite", "summary", "average_mrr_at_k"): "higher",
    ("retrieval_suite", "summary", "average_ndcg_at_k"): "higher",
    ("retrieval_suite", "summary", "average_coverage_at_k"): "higher",
    ("generation_suite", "summary", "generation_pass_rate"): "higher",
    ("generation_suite", "summary", "generation_failure_count"): "lower",
    ("generation_suite", "summary", "total_missing_expected_from_generation"): "lower",
    ("generation_suite", "summary", "total_out_of_pool_techniques"): "lower",
    ("generation_suite", "summary", "total_invalid_confidence_ids"): "lower",
    ("generation_suite", "summary", "total_invalid_confidence_score_ids"): "lower",
    ("generation_suite", "summary", "total_ungrounded_high_confidence"): "lower",
    ("generation_suite", "summary", "total_empty_rationale_ids"): "lower",
}


def compare_against_baseline(
    current_result: dict[str, Any],
    *,
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """
    Compare current metrics against a saved baseline JSON.

    Baseline comparison is optional. Without the file, fixed-case checks still
    run, but before/after metric comparison is marked unavailable.
    """
    path = Path(baseline_path)

    if not path.exists():
        return {
            "available": False,
            "baseline_path": str(path),
            "passed": True,
            "reason": "baseline file not found",
            "metrics": {},
        }

    baseline = json.loads(path.read_text(encoding="utf-8"))
    metric_results: dict[str, dict[str, Any]] = {}

    for metric_path, direction in BASELINE_METRICS.items():
        current_value = _get_nested(current_result, metric_path)
        baseline_value = _get_nested(baseline, metric_path)
        metric_name = ".".join(metric_path)

        if current_value is None or baseline_value is None:
            metric_results[metric_name] = {
                "direction": direction,
                "baseline": baseline_value,
                "current": current_value,
                "delta": None,
                "passed": True,
                "available": False,
            }
            continue

        current_float = float(current_value)
        baseline_float = float(baseline_value)
        delta = current_float - baseline_float

        if direction == "higher":
            passed = current_float >= baseline_float - tolerance
        elif direction == "lower":
            passed = current_float <= baseline_float + tolerance
        else:
            raise ValueError(f"unknown metric direction: {direction}")

        metric_results[metric_name] = {
            "direction": direction,
            "baseline": baseline_float,
            "current": current_float,
            "delta": delta,
            "passed": passed,
            "available": True,
        }

    return {
        "available": True,
        "baseline_path": str(path),
        "passed": all(item["passed"] for item in metric_results.values()),
        "metrics": metric_results,
    }


def run_all_regression_suites(
    *,
    retrieval_method: str = "bm25",
    k: int = 5,
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """
    Run retrieval and generation regression suites.

    The result includes current metrics plus optional baseline comparison.
    """
    retrieval_suite = run_retrieval_regression_suite(
        retrieval_method=retrieval_method,
        k=k,
    )

    generation_suite = run_generation_regression_suite(
        retrieval_method=retrieval_method,
        k=k,
    )

    result = {
        "suite": "all_regression_suites",
        "retrieval_method": retrieval_method,
        "k": k,
        "retrieval_suite": retrieval_suite,
        "generation_suite": generation_suite,
    }

    baseline_comparison = compare_against_baseline(
        result,
        baseline_path=baseline_path,
        tolerance=tolerance,
    )

    passed = (
        retrieval_suite["summary"]["passed"]
        and generation_suite["summary"]["passed"]
        and baseline_comparison["passed"]
    )

    result["baseline_comparison"] = baseline_comparison
    result["summary"] = {
        "passed": passed,
        "retrieval_passed": retrieval_suite["summary"]["passed"],
        "generation_passed": generation_suite["summary"]["passed"],
        "baseline_comparison_available": baseline_comparison["available"],
        "baseline_comparison_passed": baseline_comparison["passed"],
    }

    return result


def write_baseline(
    result: dict[str, Any],
    *,
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
) -> str:
    """
    Save the current regression result as the accepted baseline.

    Only use this after confirming that the current behavior is worth preserving.
    """
    path = Path(baseline_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    baseline = dict(result)
    baseline.pop("baseline_comparison", None)

    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["retrieval", "generation", "all"],
        default="all",
    )
    parser.add_argument("--retrieval-method", default="bm25")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--write-baseline", action="store_true")

    args = parser.parse_args()

    if args.suite == "retrieval":
        result = run_retrieval_regression_suite(
            retrieval_method=args.retrieval_method,
            k=args.k,
        )
    elif args.suite == "generation":
        result = run_generation_regression_suite(
            retrieval_method=args.retrieval_method,
            k=args.k,
        )
    else:
        result = run_all_regression_suites(
            retrieval_method=args.retrieval_method,
            k=args.k,
            baseline_path=args.baseline_path,
            tolerance=args.tolerance,
        )

    if args.write_baseline:
        if args.suite != "all":
            raise SystemExit("--write-baseline requires --suite all")

        baseline_path = write_baseline(
            result,
            baseline_path=args.baseline_path,
        )
        result["written_baseline_path"] = baseline_path

    print(json.dumps(result, indent=2))

    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()