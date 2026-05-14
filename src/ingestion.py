from __future__ import annotations

import re

from src.schemas import ExtractedEntities, IngestedReport


# Generic section names seen in threat reports.
# This is report-structure parsing, not ATT&CK technique logic.
SECTION_NAMES = [
    "Summary",
    "Observed Activity",
    "Indicators",
    "Analyst Note",
    "Incident note",
]


def clean_report_text(report_text: str) -> str:
    # Normalize whitespace but keep paragraph boundaries.
    text = report_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_sections(cleaned_text: str) -> dict[str, str]:
    # Split known report sections when headers are present.
    # If no headers are found, ingestion falls back to the full body text.
    section_pattern = "|".join(re.escape(name) for name in SECTION_NAMES)
    pattern = re.compile(rf"(?P<header>{section_pattern})\s*:\s*", re.IGNORECASE)

    matches = list(pattern.finditer(cleaned_text))
    if not matches:
        return {}

    sections: dict[str, str] = {}

    for i, match in enumerate(matches):
        header = match.group("header").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        sections[header] = cleaned_text[start:end].strip()

    return sections


def ingest_report(report_text: str) -> IngestedReport:
    # Main report ingestion step.
    # Output body becomes the main retrieval query text.
    cleaned_text = clean_report_text(report_text)
    sections = detect_sections(cleaned_text)

    body = "\n\n".join(sections.values()) if sections else cleaned_text

    return IngestedReport(
        cleaned_text=cleaned_text,
        sections=sections,
        body=body,
    )



def extract_entities(ingested_report: IngestedReport) -> ExtractedEntities:
    # Generic artifact extraction only.
    #
    # Important:
    # This function does NOT decide MITRE ATT&CK techniques.
    # It only extracts observable artifacts from the report text.
    #
    # Example:
    # "download payload.exe from c2.example.org"
    #   file   -> payload.exe
    #   domain -> c2.example.org
    #
    # Mapping that behavior to T1105 happens later in retrieval.py and candidate_expansion.py.

    text = ingested_report.body

    # File artifacts.
    # Examples: payload.exe, run.ps1, helper.dll, script.bat
    # These are useful as evidence, but a filename alone should not directly
    # determine an ATT&CK technique.
    files = sorted(
        set(
            re.findall(
                r"\b[\w.-]+\.(?:exe|dll|ps1|bat|cmd)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    )

    # IP address artifacts.
    # Examples: 10.0.0.5, 192.168.1.20
    # In v1, these are mainly kept for evidence/trace.
    ips = sorted(
        set(
            re.findall(
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                text,
            )
        )
    )

    # Domain artifacts.
    # Examples: evil.com, c2.example.org, download.attacker.net
    #
    # We filter out filenames like payload.exe because they also contain a dot,
    # but they are files, not network domains.
    raw_domains = re.findall(r"\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", text)
    file_set = {f.lower() for f in files}

    domains = sorted(
        {
            domain
            for domain in raw_domains
            if domain.lower() not in file_set
            and not domain.lower().endswith((".exe", ".dll", ".ps1", ".bat", ".cmd"))
        }
    )

    # Conservative command-line-like spans.
    # Examples:
    #   cmd.exe /c
    #   script.ps1 -enc
    #
    # Bare filenames such as payload.exe are already captured under files.
    # We do not duplicate them as commands unless they appear with CLI-style args.
    command_like_spans = sorted(
        set(
            re.findall(
                r"\b[\w.-]+\.(?:exe|cmd|bat|ps1)(?:\s+(?:[-/][^\s,.;]+))+",
                text,
                flags=re.IGNORECASE,
            )
        )
    )

    return ExtractedEntities(
        commands=command_like_spans,
        tools=[],
        files=files,
        ips=ips,
        domains=domains,
        behavior_phrases=[],
    )

def ingest_and_extract(report_text: str) -> tuple[IngestedReport, ExtractedEntities]:
    # Convenience wrapper for later pipeline stages.
    ingested = ingest_report(report_text)
    entities = extract_entities(ingested)
    return ingested, entities