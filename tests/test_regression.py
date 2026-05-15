"""
Pytest coverage for the ATT&CK regression suites.

The regression implementation lives in src.regression. It loads labeled sample
reports, runs the retrieval and generation paths, applies evaluation.py checks,
and returns JSON-style suite summaries.

This file keeps the pytest layer thin. It does not duplicate regression logic.
It calls the suite runners and asserts the stable behavior expected from the
labeled sample set.

Execution modes:
    python -m src.regression
        CLI path for inspecting full regression JSON output.

    pytest -q
        Automated pass/fail path for checking that regression suites still pass
        after code changes.

pytest.ini configures test discovery and imports:
    testpaths = tests
        pytest collects tests from the tests/ directory.

    pythonpath = .
        pytest can import src.* modules from the project root.
"""

from src.regression import (
    run_generation_regression_suite,
    run_regression_suite,
)


def test_retrieval_regression_suite_passes():
    """
    Retrieval regression checks candidate recovery before generation.

    The expected ATT&CK techniques for the labeled sample reports should still
    appear in the top-k retrieved/expanded candidates.
    """
    result = run_regression_suite(
        retrieval_method="bm25",
        k=5,
    )

    assert result["summary"]["passed"], result["summary"]["failed_reports"]
    assert result["summary"]["total_reports"] > 0
    assert result["summary"]["average_recall_at_k"] == 1.0


def test_r1_expected_techniques_are_found_in_retrieval_candidates():
    """
    Keep one concrete sample-level guardrail.

    r1 is expected to recover all of its labeled ATT&CK techniques in the
    retrieval/candidate-expansion path.
    """
    result = run_regression_suite(
        retrieval_method="bm25",
        k=5,
    )

    r1 = next(
        report
        for report in result["reports"]
        if report["report_id"] == "r1"
    )

    assert r1["passed"], r1
    assert set(r1["expected_techniques"]).issubset(
        set(r1["predicted_techniques"])
    )


def test_generation_regression_suite_passes_and_keeps_required_structure():
    """
    Generation regression checks structured LLM output after the full pipeline.

    This does not judge full semantic explanation quality. It checks deterministic
    output guarantees: schema validity, candidate-pool containment, expected
    selection when available, confidence fields, and rationale presence.
    """
    result = run_generation_regression_suite(
        retrieval_method="bm25",
        k=5,
    )

    assert result["summary"]["passed"], result["summary"]["failed_reports"]
    assert result["summary"]["generation_eval_count"] > 0
    assert result["summary"]["generation_failure_count"] == 0
    assert result["summary"]["generation_pass_rate"] == 1.0

    for report in result["reports"]:
        assert report["schema_valid"], report
        assert report["out_of_pool_techniques"] == [], report
        assert report["missing_expected_from_generation"] == [], report
        assert report["invalid_confidence_ids"] == [], report
        assert report["invalid_confidence_score_ids"] == [], report
        assert report["empty_rationale_ids"] == [], report