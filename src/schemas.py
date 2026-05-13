from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# -----------------------------
# MITRE ATT&CK technique corpus
# -----------------------------
# Matches data/attack_techniques.json.
# Each technique is split into:
# 1. mitre_definition: MITRE-derived technique information
# 2. analyst_enrichment: internal retrieval terms, not official MITRE fields


class MitreDefinition(BaseModel):
    technique_id: str
    name: str
    tactic: str | list[str]
    description: str
    source_url: str | None = None


class AnalystEnrichment(BaseModel):
    retrieval_terms: list[str] = Field(default_factory=list)


class AttackTechniqueRecord(BaseModel):
    mitre_definition: MitreDefinition
    analyst_enrichment: AnalystEnrichment = Field(default_factory=AnalystEnrichment)


# -----------------------------
# Alias resolver
# -----------------------------
# Matches data/alias_resolver.json.
# The phrase itself is the dictionary key.
# Example:
# "encoded powershell": {
#     "candidate_techniques": ["T1059.001", "T1027"],
#     "note": "May indicate PowerShell execution and command obfuscation."
# }


class AliasResolverMetadata(BaseModel):
    description: str | None = None
    source_type: str | None = None
    official_mitre_source: bool = False


class AliasResolverEntry(BaseModel):
    candidate_techniques: list[str]
    note: str | None = None


class AliasResolverConfig(BaseModel):
    metadata: AliasResolverMetadata = Field(default_factory=AliasResolverMetadata)
    aliases: dict[str, AliasResolverEntry] = Field(default_factory=dict)


# -----------------------------
# Sample reports
# -----------------------------
# Matches data/sample_reports.json.
# expected_techniques are labels for evaluation only.
# Retrieval and generation code should not use expected_techniques.


class ThreatReport(BaseModel):
    report_id: str
    query: str | None = None
    report_text: str
    expected_techniques: list[str] = Field(default_factory=list)


# -----------------------------
# Permissions
# -----------------------------
# Matches data/permissions.json.
# The JSON currently has users at the top level:
# {
#   "default_user": {
#       "allowed_sources": [...],
#       "allowed_tools": [...],
#       "can_use_external_lookup": false
#   }
# }
#
# The validator wraps that into:
# {
#   "users": {
#       "default_user": {...}
#   }
# }
#
# This keeps the internal Python shape cleaner while still accepting the current JSON.


class UserPermissions(BaseModel):
    allowed_sources: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    can_use_external_lookup: bool = False


class PermissionsConfig(BaseModel):
    users: dict[str, UserPermissions] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_top_level_user_mapping(cls, data: Any) -> Any:
        if isinstance(data, dict) and "users" not in data:
            return {"users": data}
        return data

# -----------------------------
# API request / response models
# -----------------------------
# These models define the data contract between pipeline stages.
# They do not implement ingestion, retrieval, generation, or API logic.
# The actual logic will be added later in ingestion.py, retrieval.py,
# generation.py, api.py, and trace.py.


class AnalyzeRequest(BaseModel):
    # Input object for POST /analyze_report.
    # report_text is the raw threat report provided by the user.
    # user_id controls permission checks.
    # top_k controls how many candidate techniques to return.
    # use_tools allows the planner to enable or skip local lookup tools.
    report_text: str
    user_id: str = "default_user"
    top_k: int = 5
    use_tools: bool = True


class IngestedReport(BaseModel):
    # Output from ingestion.py.
    # cleaned_text is normalized report text.
    # sections stores detected report sections such as Summary or Indicators.
    # body is the main text used when no clear sections are found.
    cleaned_text: str
    sections: dict[str, str] = Field(default_factory=dict)
    body: str


class ExtractedEntities(BaseModel):
    # Output from the lightweight entity/phrase extraction step.
    # These fields help retrieval and alias resolution.
    # This is intentionally simple for now.
    commands: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    ips: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    behavior_phrases: list[str] = Field(default_factory=list)


class EvidenceSpan(BaseModel):
    # A short text span from the report that supports a technique mapping.
    # start_char and end_char are optional because early versions may not
    # compute exact character offsets.
    text: str
    start_char: int | None = None
    end_char: int | None = None
    source_section: str | None = None


class TechniqueCandidate(BaseModel):
    # A candidate MITRE ATT&CK technique returned by retrieval, reranking,
    # alias resolution, or local lookup.
    # score is the retrieval or reranking score.
    # confidence is the final calibrated confidence, if generation.py sets it.
    # evidence stores report spans supporting the mapping.
    # retrieval_sources records where the candidate came from.
    technique_id: str
    name: str
    tactic: str | list[str]
    score: float
    confidence: float | None = None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    retrieval_sources: list[str] = Field(default_factory=list)


class TraceInfo(BaseModel):
    # Debug and audit information for one analysis run.
    # Later trace.py or database.py can persist this to SQLite.
    # steps can store pipeline events such as ingestion, retrieval, tool use,
    # reranking, and generation.
    trace_id: str | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    # Final response object returned by POST /analyze_report.
    # mappings are the predicted ATT&CK technique mappings.
    # entities are extracted from the report for visibility/debugging.
    # trace explains what the backend did.
    mappings: list[TechniqueCandidate]
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    trace: TraceInfo = Field(default_factory=TraceInfo)