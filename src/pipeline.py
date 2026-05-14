"""
Pipeline runner for the ATT&CK analysis workflow.

This module contains the reusable orchestration path for one analysis run:

    raw report text
    -> ingestion
    -> retrieval
    -> candidate expansion
    -> grounded generation
    -> trace construction
    -> optional persistence

The runner is shared by smoke scripts, API handlers, tests, and future batch
jobs so the pipeline is assembled in one place instead of being duplicated
across entrypoints.

The orchestration is kept explicit rather than hidden behind a workflow
framework. The same steps could later be wrapped with LangChain, LlamaIndex, or
a graph/workflow framework, but keeping the control flow visible makes retrieval,
tool use, generation, tracing, and persistence easier to inspect directly.

Not every src module is imported here. Some modules are used through lower-level
pipeline components. For example, generation.py owns planner/tool execution, so
this runner calls generate_answer() instead of calling planner.py or tools.py
directly.

Evaluation and regression code are quality-check layers. They can call this
runner, but they are not part of the default single-request analysis path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.candidate_expansion import expand_candidates
from src.config import AppSettings, get_settings
from src.data_sources import load_analyst_heuristic_signatures, load_attack_techniques
from src.database import save_trace as save_trace_to_db
from src.generation import generate_answer
from src.indexing import build_index_documents
from src.ingestion import extract_entities, ingest_report
from src.planner import RiskLevel
from src.retrieval import retrieve_top_k
from src.trace import build_trace, save_trace_json


def _get_generation_field(
    generation_output: dict[str, Any],
    field_name: str,
    default: Any,
) -> Any:
    """
    Read a field from generation output with a defensive fallback.

    generate_answer() is expected to return a stable dict containing fields like:
        provider
        model
        mappings
        warnings
        execution_plan
        tool_results

    This helper keeps analyze_report() readable and avoids repeating
    isinstance(..., dict) checks.
    """
    if not isinstance(generation_output, dict):
        return default

    return generation_output.get(field_name, default)


def _build_ingestion_record(
    *,
    ingested_report: Any,
    extracted_entities: Any,
) -> dict[str, Any]:
    """
    Build the ingestion section saved into trace.

    ingestion.py returns structured objects. trace.py can serialize those objects
    directly, but this small wrapper makes the trace easier to read because it
    shows both:
        - normalized report structure
        - generic extracted artifacts

    These artifacts are not ATT&CK decisions. They are evidence/features that
    later stages can inspect.
    """
    return {
        "ingested_report": ingested_report,
        "extracted_entities": extracted_entities,
    }


def analyze_report(
    report_text: str,
    *,
    retrieval_method: str | None = None,
    k: int | None = None,
    risk_level: RiskLevel = "normal",
    save_json_trace: bool = True,
    save_db_trace: bool = True,
    trace_output_dir: str | Path = "traces",
    db_path: str | Path = "runs.db",
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """
    Run one end-to-end ATT&CK analysis.

    This is the reusable runner for:
        - manual smoke scripts
        - FastAPI handlers
        - future regression tests
        - future batch jobs

    It intentionally does not hide the pipeline behind a framework abstraction.
    Each step is visible and inspectable.

    Args:
        report_text:
            Raw threat-report text or report excerpt.

        retrieval_method:
            Retrieval mode. Current options come from retrieval.py:
                bm25
                tfidf
                vector
                hybrid

            If omitted, the default from config.py is used.

        k:
            Number of retrieval candidates before candidate expansion.
            If omitted, the default from config.py is used.

        risk_level:
            Passed into generation.py. generation.py uses it when building the
            execution plan through planner.py.

        save_json_trace:
            Save a human-readable trace JSON file under trace_output_dir.

        save_db_trace:
            Save the same trace into SQLite through database.py.

        trace_output_dir:
            Directory for JSON trace files.

        db_path:
            SQLite file path for operational run history.

        settings:
            Optional AppSettings override. Normal runtime uses get_settings(), which
            reads config defaults and .env. Test code or local smoke scripts can provide
            a custom settings object to control model/tool behavior explicitly.

    Returns:
        A dict with:
            run_id
            generation
            trace
            trace_path
            db_saved
            high-level counts

    Design choice:
        The LLM is not the main ranker. Retrieval and candidate expansion build
        the candidate pool first. generation.py then uses planner/tool context to
        produce grounded explanations and warnings.
    """
    settings = settings or get_settings()

    method = retrieval_method or settings.default_retrieval_method
    top_k = k or settings.default_top_k

    if not report_text or not report_text.strip():
        raise ValueError("report_text must be a non-empty string")

    if top_k <= 0:
        raise ValueError("k must be greater than 0")

    # ------------------------------------------------------------
    # 1. Ingestion
    # ------------------------------------------------------------
    # ingestion.py performs generic report normalization and artifact extraction.
    # It does not make ATT&CK mapping decisions.
    #
    # The body field is the text used for retrieval. If the report has sections,
    # ingestion.py joins section bodies. Otherwise it uses the cleaned full text.
    ingested_report = ingest_report(report_text)
    extracted_entities = extract_entities(ingested_report)

    body_text = ingested_report.body

    # ------------------------------------------------------------
    # 2. Static reference loading and indexing
    # ------------------------------------------------------------
    # Static ATT&CK/reference data stays in data/*.json and is loaded here.
    #
    # build_index_documents() turns each ATT&CK technique into the retrieval
    # document shape expected by retrieval.py:
    #     technique_id
    #     full technique object
    #     searchable_text
    techniques = load_attack_techniques()
    heuristic_signatures = load_analyst_heuristic_signatures()
    index_documents = build_index_documents(techniques)

    # ------------------------------------------------------------
    # 3. Retrieval
    # ------------------------------------------------------------
    # retrieve_top_k() returns candidates in a stable shape across BM25, TF-IDF,
    # vector, and hybrid retrieval.
    #
    # The output is still only a candidate set. It is not the final mapping.
    retrieved_candidates = retrieve_top_k(
        query_text=body_text,
        index_documents=index_documents,
        k=top_k,
        method=method,
    )

    # ------------------------------------------------------------
    # 4. Candidate expansion
    # ------------------------------------------------------------
    # candidate_expansion.py scans the report body for analyst heuristic phrases.
    # It merges heuristic candidates with retrieval candidates by technique_id.
    #
    # Example:
    #     "encoded powershell"
    #         -> T1059.001
    #         -> T1027
    #
    # The final LLM step can then see both retrieval evidence and heuristic
    # evidence for each candidate.
    expanded_candidates = expand_candidates(
        report_text=body_text,
        retrieval_candidates=retrieved_candidates,
        heuristic_data=heuristic_signatures,
    )

    # ------------------------------------------------------------
    # 5. Grounded generation
    # ------------------------------------------------------------
    # generate_answer() owns the generation-time planner/tool path.
    #
    # Internally, generation.py:
    #     - calls planner.py to build an execution plan
    #     - executes planner-prefetch tools such as lookup_attack_technique
    #     - calls LiteLLM if a model key is configured
    #     - falls back to deterministic generation otherwise
    #     - allows at most one model-requested tool-call round
    #
    # pipeline.py does not duplicate that logic. It treats generation.py as
    # the owner of model routing, tool execution, and answer normalization.
    generation_output = generate_answer(
        report_text=body_text,
        candidates=expanded_candidates,
        risk_level=risk_level,
        settings=settings,
    )

    execution_plan = _get_generation_field(
        generation_output,
        "execution_plan",
        default=None,
    )

    tool_results = _get_generation_field(
        generation_output,
        "tool_results",
        default=[],
    )

    # ------------------------------------------------------------
    # 6. Trace construction
    # ------------------------------------------------------------
    # trace.py records what happened during the run. It does not execute the
    # pipeline by itself.
    #
    # The trace is useful for debugging:
    #     what text was retrieved against
    #     which candidates were returned
    #     which heuristic phrases matched
    #     which tools ran
    #     which model generated the final answer
    #     which warnings were produced
    ingestion_record = _build_ingestion_record(
        ingested_report=ingested_report,
        extracted_entities=extracted_entities,
    )

    trace = build_trace(
        report_text=report_text,
        ingestion_output=ingestion_record,
        retrieval_method=method,
        retrieved_candidates=retrieved_candidates,
        expanded_candidates=expanded_candidates,
        execution_plan=execution_plan,
        tool_results=tool_results,
        generation_output=generation_output,
    )

    # ------------------------------------------------------------
    # 7. Optional persistence
    # ------------------------------------------------------------
    # JSON trace:
    #     easy to open by hand during local debugging
    #
    # SQLite:
    #     structured operational run history
    #     useful for API-backed runs, recent-run views, and later evaluation
    #     history
    trace_path: Path | None = None
    if save_json_trace:
        trace_path = save_trace_json(trace, output_dir=trace_output_dir)

    db_saved = False
    if save_db_trace:
        save_trace_to_db(trace, db_path=db_path)
        db_saved = True

    # ------------------------------------------------------------
    # 8. Return compact result plus full trace
    # ------------------------------------------------------------
    # API handlers can choose to return only the compact fields.
    # Debug scripts can inspect the full trace.
    return {
        "run_id": trace["run_id"],
        "retrieval_method": method,
        "k": top_k,
        "risk_level": risk_level,
        "retrieved_count": len(retrieved_candidates),
        "expanded_count": len(expanded_candidates),
        "generation": generation_output,
        "trace": trace,
        "trace_path": str(trace_path) if trace_path else None,
        "db_saved": db_saved,
    }