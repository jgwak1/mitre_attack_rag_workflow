"""
HTTP boundary for the ATT&CK analysis pipeline.

This module exposes the reusable pipeline runner through FastAPI routes.

Runtime path:
    HTTP client
    -> uvicorn
    -> app.py
    -> src.api route
    -> src.pipeline.analyze_report()

The API layer is intentionally thin:
    - validate request shape
    - call the pipeline runner
    - translate failures into HTTP errors
    - return JSON responses

The analysis workflow itself stays in src.pipeline.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.database import get_trace, list_runs
from src.pipeline import analyze_report


RetrievalMethod = Literal["bm25", "tfidf", "vector", "hybrid"]
RiskLevel = Literal["low", "normal", "high"]


class AnalyzeRequest(BaseModel):
    """
    Request body for POST /analyze.

    This starts a new analysis run, so it can create new artifacts such as a
    trace file, SQLite run record, tool results, and model output.
    """

    report_text: str = Field(
        ...,
        min_length=1,
        description="Raw threat-report text or report excerpt.",
    )

    retrieval_method: RetrievalMethod = Field(
        default="bm25",
        description="Initial retrieval method before candidate expansion.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of initial retrieval candidates.",
    )

    risk_level: RiskLevel = Field(
        default="normal",
        description="Risk setting passed to planner/generation logic.",
    )

    save_json_trace: bool = Field(
        default=True,
        description="Whether to write a local JSON trace file.",
    )

    save_db_trace: bool = Field(
        default=True,
        description="Whether to persist the trace in SQLite.",
    )

    include_trace: bool = Field(
        default=False,
        description=(
            "Include the full trace in this response. Usually false because "
            "the trace can be large. Use GET /runs/{run_id} for later inspection."
        ),
    )


class AnalyzeResponse(BaseModel):
    """
    Response body for POST /analyze.

    The response includes the generation output and run identifiers. The full
    trace is optional to keep normal responses compact.
    """

    run_id: str
    retrieval_method: str
    k: int
    risk_level: str
    retrieved_count: int
    expanded_count: int
    generation: dict[str, Any] | None
    trace_path: str | None
    db_saved: bool
    trace: dict[str, Any] | None = None


app = FastAPI(
    title="MITRE ATT&CK RAG Workflow API",
    version="0.1.0",
    description=(
        "API for mapping threat-report behavior to MITRE ATT&CK techniques "
        "using retrieval, heuristic expansion, tool-grounded generation, "
        "tracing, and persistence."
    ),
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """
    Redirect the browser root to FastAPI's built-in docs UI.

    This is only a local convenience route. It is not part of the analysis API.
    """
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    """
    Return a lightweight process health check.

    This confirms that the API process is alive. It does not run retrieval,
    tools, generation, or database writes.
    """
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run one ATT&CK analysis request.

    FastAPI validates the incoming JSON against AnalyzeRequest before this
    function runs. If validation succeeds, this handler delegates the actual
    workflow to src.pipeline.analyze_report().
    """
    try:
        result = analyze_report(
            report_text=request.report_text,
            retrieval_method=request.retrieval_method,
            k=request.top_k,
            risk_level=request.risk_level,
            save_json_trace=request.save_json_trace,
            save_db_trace=request.save_db_trace,
        )

    except ValueError as exc:
        # Invalid user input, such as an empty report body.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        # Pipeline/runtime failure after the request passed validation.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    generation = result.get("generation")
    generation_dict = generation if isinstance(generation, dict) else None

    return AnalyzeResponse(
        run_id=str(result["run_id"]),
        retrieval_method=str(result["retrieval_method"]),
        k=int(result["k"]),
        risk_level=str(result["risk_level"]),
        retrieved_count=int(result["retrieved_count"]),
        expanded_count=int(result["expanded_count"]),
        generation=generation_dict,
        trace_path=result.get("trace_path"),
        db_saved=bool(result.get("db_saved")),
        trace=result.get("trace") if request.include_trace else None,
    )


@app.get("/runs")
def recent_runs(limit: int = 20) -> list[dict[str, Any]]:
    """
    Return recent saved analysis runs.

    This endpoint reads compact run metadata from SQLite rather than loading
    every full trace blob.
    """
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be greater than 0")

    return list_runs(limit=limit)


@app.get("/runs/{run_id}")
def read_run(run_id: str) -> dict[str, Any]:
    """
    Return the full saved trace for one run.

    This is the inspection path after POST /analyze returns a run_id.
    """
    trace = get_trace(run_id)

    if trace is None:
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")

    return trace