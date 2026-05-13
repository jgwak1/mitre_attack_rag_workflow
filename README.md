# MITRE ATT&CK Threat Report Assistant

A compact RAG backend for mapping threat-report behavior to MITRE ATT&CK techniques.

The system takes threat-report text as input, retrieves relevant ATT&CK techniques, resolves aliases and entities, optionally uses lookup tools, and returns evidence-based technique mappings with confidence scores and trace logs.

## Workflow

Data sources → ingestion → indexing and entity layer → retrieval and reranking → tool use → grounded output → evaluation and regression → production considerations

## Tech Stack

- FastAPI: API layer
- Pydantic: input and output schemas
- SQLite: metadata, evaluation, and trace store
- FAISS: vector search
- rank-bm25: keyword search
- JSON: seed ATT&CK data
- logging: trace and debug logs
- pytest: regression tests