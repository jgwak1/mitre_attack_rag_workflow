from __future__ import annotations

from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Read a field from either a dict or a Pydantic model.

    Our data is loaded as validated Pydantic objects, but keeping this helper
    flexible makes the indexing code easy to smoke-test with plain dicts too.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _join_nonempty(parts: list[Any]) -> str:
    """
    Join only non-empty text fields.

    Each field is placed on its own line so the retrieval document remains
    readable during debugging.
    """
    return "\n".join(str(p).strip() for p in parts if p and str(p).strip())


def build_searchable_text(technique: Any) -> str:
    """
    Build retrieval text for one ATT&CK technique record.

    This function only prepares text for search.

    It does not:
    - retrieve candidates
    - score candidates
    - expand candidates from analyst heuristics
    - generate the final answer

    The searchable text combines:
    - MITRE technique fields: ID, name, tactic, and description
    - analyst-curated retrieval terms

    The retrieval terms are useful because real threat reports often contain
    operational words like "wmic", "-enc", "lsass", or "cmd /c" rather than
    the exact official MITRE technique description.
    """

    mitre = _get(technique, "mitre_definition", {})
    enrichment = _get(technique, "analyst_enrichment", {})

    retrieval_terms = _get(enrichment, "retrieval_terms", [])
    if retrieval_terms is None:
        retrieval_terms = []

    # We intentionally include technique_id in searchable_text.
    # This lets exact mentions like "T1059.001" match directly during retrieval.
    #
    # We intentionally exclude source_url.
    # URLs are useful as metadata, but they usually add noise to lexical scoring.
    return _join_nonempty(
        [
            _get(mitre, "technique_id"),
            _get(mitre, "name"),
            _get(mitre, "tactic"),
            _get(mitre, "description"),
            " ".join(retrieval_terms),
        ]
    )


def build_index_documents(techniques: list[Any]) -> list[dict[str, Any]]:
    """
    Convert ATT&CK technique records into retrieval documents.

    Output shape:
    {
      "technique_id": "...",
      "technique": full technique object from attack_techniques.json,
      "searchable_text": text used by retrieval.py
    }

    The full technique object is kept so later stages can return metadata
    after retrieval, not just the flattened searchable text.
    """

    documents: list[dict[str, Any]] = []

    for technique in techniques:
        mitre = _get(technique, "mitre_definition", {})
        technique_id = _get(mitre, "technique_id")

        documents.append(
            {

               # Technique-ID copied out for easy lookup, output, and tracing.
               # Example: "T1059.001"
               "technique_id": technique_id,

               # Full technique object loaded from attack_techniques.json.
               # Keep this so later stages can access fields such as:
               # - MITRE name / tactic / description / source_url
               # - analyst retrieval terms
               # Retrieval uses searchable_text, keep this so later stages can access the fields.
               "technique": technique,

               # Flattened text used for BM25 / TF-IDF style retrieval.
               "searchable_text": build_searchable_text(technique),
            }
        )

    return documents