"""
SQLite persistence for operational analysis records.

This module stores dynamic outputs from analysis runs, such as traces,
tool results, generation outputs, warnings, and evaluation summaries.

Static reference inputs, including ATT&CK techniques and analyst heuristic
signatures, remain in data/*.json so their changes are version-controlled,
reviewable, and reproducible.

The current retrieval layer uses local JSON reference data, BM25-style keyword
retrieval, and an in-process FAISS index for semantic retrieval. SQLite is used
only for operational records, not as the source of truth for ATT&CK content.

In a larger deployment, the static ATT&CK corpus could be indexed into
Elasticsearch/OpenSearch for keyword, boolean, and fielded retrieval over
technique names, tactics, descriptions, and analyst enrichment terms. The local
FAISS index could be replaced or wrapped by a persistent vector store such as
pgvector, OpenSearch vector search, Milvus, Pinecone, or Weaviate. SQLite could
be replaced by Postgres for run history, evaluation records, audit metadata,
and structured querying over pipeline outputs.
"""


from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = "runs.db"


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open a SQLite connection for operational run storage.

    A normal path such as runs.db creates a file-backed database. The special
    SQLite path ":memory:" can still be used by tests when an in-memory database
    is preferred.
    """
    path = Path(db_path)

    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """
    Initialize tables used for analysis run history.

    The complete trace is stored as JSON text. A small set of summary columns is
    duplicated from the trace so recent runs can be listed without parsing the
    full trace payload.
    """
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                warning_count INTEGER NOT NULL DEFAULT 0,
                tool_result_count INTEGER NOT NULL DEFAULT 0,
                trace_json TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at
            ON analysis_runs(created_at)
            """
        )


def _extract_run_summary(
    trace: dict[str, Any],
) -> tuple[str | None, str | None, int, int]:
    """
    Pull out small run-level metadata fields from the full trace.

    Metadata here means fields that describe the run, not the main result.
    The main result stays in trace_json. These columns just make it easy to
    list recent runs and quickly see which model ran, whether warnings were
    produced, and whether tools were used.

    Returned fields:
    - provider: generation provider, for example litellm
    - model: generation model name
    - warning_count: number of warnings in the generation output
    - tool_result_count: number of tool results recorded for the run
    """
    generation = trace.get("generation") or {}
    tools = trace.get("tools") or {}

    if isinstance(generation, dict):
        provider = generation.get("provider")
        model = generation.get("model")
        warnings = generation.get("warnings") or []
    else:
        provider = None
        model = None
        warnings = []

    if isinstance(tools, dict):
        tool_results = tools.get("tool_results") or []
    else:
        tool_results = []

    warning_count = len(warnings) if isinstance(warnings, list) else 0
    tool_result_count = len(tool_results) if isinstance(tool_results, list) else 0

    return provider, model, warning_count, tool_result_count


def save_trace(
    trace: dict[str, Any],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    status: str = "completed",
) -> str:
    """
    Persist one analysis trace.

    The full trace is stored as JSON. Summary fields are stored separately for
    recent-run views, debugging, and simple operational checks.

    Saving the same run_id again updates the existing row. This keeps local
    smoke tests idempotent and avoids duplicate records for the same run.
    """
    run_id = trace.get("run_id")
    created_at = trace.get("created_at")

    if not run_id:
        raise ValueError("trace must contain run_id")

    if not created_at:
        raise ValueError("trace must contain created_at")

    provider, model, warning_count, tool_result_count = _extract_run_summary(trace)
    trace_json = json.dumps(trace, ensure_ascii=False)

    init_db(db_path)

    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                run_id,
                created_at,
                status,
                provider,
                model,
                warning_count,
                tool_result_count,
                trace_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                created_at = excluded.created_at,
                status = excluded.status,
                provider = excluded.provider,
                model = excluded.model,
                warning_count = excluded.warning_count,
                tool_result_count = excluded.tool_result_count,
                trace_json = excluded.trace_json
            """,
            (
                run_id,
                created_at,
                status,
                provider,
                model,
                warning_count,
                tool_result_count,
                trace_json,
            ),
        )

    return str(run_id)


def get_trace(
    run_id: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """
    Load the full trace for one run_id.

    Returns None when the run_id is not present.
    """
    init_db(db_path)

    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT trace_json
            FROM analysis_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return json.loads(row["trace_json"])


def list_runs(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Return recent run summaries without loading full trace JSON blobs.

    This is intended for lightweight debug views, API status views, and quick
    checks of provider, model, warning count, and tool usage.
    """
    init_db(db_path)

    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                run_id,
                created_at,
                status,
                provider,
                model,
                warning_count,
                tool_result_count
            FROM analysis_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]