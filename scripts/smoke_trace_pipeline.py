from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sources import load_attack_techniques, load_analyst_heuristic_signatures
from src.indexing import build_index_documents
from src.retrieval import retrieve_top_k
from src.candidate_expansion import expand_candidates
from src.generation import generate_answer
from src.trace import build_trace, save_trace_json

report = "The actor used encoded powershell and then downloaded payload.exe."

techniques = load_attack_techniques()
signatures = load_analyst_heuristic_signatures()
docs = build_index_documents(techniques)

retrieved = retrieve_top_k(report, docs, k=5, method="bm25")
expanded = expand_candidates(report, retrieved, signatures)

generation = generate_answer(
    report_text=report,
    candidates=expanded,
)

tool_results = []
if isinstance(generation, dict):
    tool_results = generation.get("tool_results", [])

trace = build_trace(
    report_text=report,
    retrieval_method="bm25",
    retrieved_candidates=retrieved,
    expanded_candidates=expanded,
    tool_results=tool_results,
    generation_output=generation,
)

path = save_trace_json(trace)

print("saved:", path)
print("retrieved_count:", len(trace["retrieval"]["retrieved_candidates"]))
print("expanded_count:", len(trace["candidate_expansion"]["expanded_candidates"]))
print("generation_present:", trace["generation"] is not None)

print("top_retrieved:")
for c in trace["retrieval"]["retrieved_candidates"][:3]:
    print(c.get("rank"), c.get("technique_id"), c.get("name"), c.get("sources"))

print("top_expanded:")
for c in trace["candidate_expansion"]["expanded_candidates"][:3]:
    print(
        c.get("rank"),
        c.get("technique_id"),
        c.get("name"),
        c.get("sources"),
        c.get("matched_phrases"),
    )

if isinstance(generation, dict):
    print("generation_provider:", generation.get("provider"))
    print("generation_model:", generation.get("model"))
    print("tool_result_count:", len(generation.get("tool_results", [])))

    print("tool_results:")
    for r in generation.get("tool_results", []):
        print(r.get("tool"), r.get("tool_call_source"), r.get("ok"))