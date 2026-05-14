"""
Candidate expansion using analyst heuristic signatures.

This module scans ingested threat-report text for analyst-curated behavior
signatures and adds candidate ATT&CK technique IDs to the candidate pool.

It does not modify the report text.
It does not make the final ATT&CK mapping decision.
It only adds heuristic candidates that retrieval and generation can consider later.

Example:
    "encoded powershell" -> ["T1059.001", "T1027"]
"""