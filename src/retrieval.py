from __future__ import annotations

import re
from typing import Any, Callable, Literal


# Retrieval modes:
#
# bm25:
#   Keyword-search retriever. Primary local v1 path.
#
# tfidf:
#   Keyword-search baseline for comparison against BM25.
#
# vector:
#   FAISS-based vector-search retriever using embedding cosine similarity.
#
# hybrid:
#   True hybrid retrieval:
#   keyword-search BM25 + vector-search FAISS, fused by rank.
RetrievalMethod = Literal["bm25", "tfidf", "vector", "hybrid"]

# Replace this later with an internal embedding model if needed.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Custom embedding function shape:
# input: list of strings
# output: 2D array-like object with shape [num_texts, embedding_dim]
EmbeddingFn = Callable[[list[str]], Any]


def tokenize(text: str) -> list[str]:
    """
    Tokenize threat-report text and ATT&CK searchable text.

    This tokenizer is intentionally command-line friendly.

    It keeps useful tokens such as:
    - powershell
    - cmd.exe
    - -enc
    - T1059.001
    - /c
    - C:\\Windows\\Temp\\payload.exe

    Generic NLP tokenizers often split or discard these strings, which hurts
    retrieval for security reports and host-log evidence.
    """
    return re.findall(r"[a-z0-9_.:/\\-]+", text.lower())


def _format_result(
    *,
    rank: int,
    score: float,
    method: str,
    document: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return one retrieval result in a stable format.

    Keeping BM25, TF-IDF, FAISS vector, and hybrid output in the same shape
    makes evaluation, regression, tracing, reranking, and generation easier.
    """
    result = {
        "rank": rank,
        "technique_id": document["technique_id"],
        "score": float(score),
        "method": method,

        # Full technique object loaded from attack_techniques.json.
        # Later stages can access name, tactic, description, source_url,
        # and analyst enrichment terms from this object.
        "technique": document["technique"],

        # Flattened retrieval text built by indexing.py.
        # Useful for debugging why this technique was retrieved.
        "searchable_text": document["searchable_text"],
    }

    if details:
        result["details"] = details

    return result


def _rank_scores(
    *,
    scores: list[float],
    index_documents: list[dict[str, Any]],
    k: int,
    method: str,
) -> list[dict[str, Any]]:
    """
    Sort document scores descending and format the top-k results.
    """
    ranked = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    return [
        _format_result(
            rank=i + 1,
            score=score,
            method=method,
            document=index_documents[doc_idx],
        )
        for i, (doc_idx, score) in enumerate(ranked)
    ]


def retrieve_top_k_bm25(
    query_text: str,
    index_documents: list[dict[str, Any]],
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k ATT&CK techniques using BM25 keyword search.

    BM25 is the primary local retriever because threat reports often contain
    exact operational terms:
    - powershell
    - -enc
    - cmd.exe
    - lsass
    - wmic
    - mimikatz
    - payload.exe
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 is required. Install it with: pip install rank-bm25"
        ) from exc

    if not query_text.strip() or not index_documents:
        return []

    corpus_tokens = [
        tokenize(document["searchable_text"])
        for document in index_documents
    ]
    query_tokens = tokenize(query_text)

    if not query_tokens:
        return []

    bm25 = BM25Okapi(corpus_tokens)
    scores = [float(score) for score in bm25.get_scores(query_tokens)]

    return _rank_scores(
        scores=scores,
        index_documents=index_documents,
        k=k,
        method="bm25",
    )


def retrieve_top_k_tfidf(
    query_text: str,
    index_documents: list[dict[str, Any]],
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k ATT&CK techniques using TF-IDF cosine similarity.

    This is a keyword-search baseline, not vector-search.

    TF-IDF meaning:
    - TF: how often a term appears in a document
    - IDF: how rare that term is across all documents
    - cosine similarity: similarity between query vector and document vector
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required. Install it with: pip install scikit-learn"
        ) from exc

    if not query_text.strip() or not index_documents:
        return []

    corpus_texts = [
        document["searchable_text"]
        for document in index_documents
    ]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=True,
    )

    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_vector = vectorizer.transform([query_text])

    scores = cosine_similarity(query_vector, doc_matrix).flatten()
    scores = [float(score) for score in scores]

    return _rank_scores(
        scores=scores,
        index_documents=index_documents,
        k=k,
        method="tfidf",
    )


def _default_embedding_fn(
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingFn:
    """
    Build the default local embedding function.

    This uses sentence-transformers for v1 local development.

    Production replacement options:
    - internal embedding service
    - OpenAI / Anthropic / Cohere embedding endpoint
    - self-hosted embedding model
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for vector retrieval. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(embedding_model_name)

    def embed(texts: list[str]) -> Any:
        return model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    return embed


def _to_normalized_float32_matrix(embeddings: Any) -> Any:
    """
    Convert embeddings to normalized float32 matrix for FAISS.

    FAISS IndexFlatIP uses inner product.
    If vectors are L2-normalized first, inner product is equivalent to cosine
    similarity.
    """
    import numpy as np

    matrix = np.asarray(embeddings, dtype="float32")

    if matrix.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix with shape [num_texts, embedding_dim]."
        )

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    return matrix / norms


def retrieve_top_k_vector(
    query_text: str,
    index_documents: list[dict[str, Any]],
    k: int = 5,
    embedding_fn: EmbeddingFn | None = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k ATT&CK techniques using FAISS vector search.

    Vector-search flow:
    1. Embed each ATT&CK searchable_text document.
    2. Embed the query report text.
    3. Normalize embeddings.
    4. Search with FAISS IndexFlatIP.
    5. Because vectors are normalized, inner product equals cosine similarity.

    This is the implemented vector-search backend for v1.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for vector retrieval. "
            "Install it with: pip install faiss-cpu"
        ) from exc

    if not query_text.strip() or not index_documents:
        return []

   
    using_custom_embedding_fn = embedding_fn is not None

    if embedding_fn is None:
        embedding_fn = _default_embedding_fn(embedding_model_name)

    corpus_texts = [
        document["searchable_text"]
        for document in index_documents
    ]

    doc_embeddings = _to_normalized_float32_matrix(
        embedding_fn(corpus_texts)
    )
    query_embedding = _to_normalized_float32_matrix(
        embedding_fn([query_text])
    )

    embedding_dim = doc_embeddings.shape[1]

    # IndexFlatIP is exact search, not approximate search.
    # Good for this small v1 corpus.
    # Later, this can move to IVF/HNSW/managed vector DB for larger corpora.
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(doc_embeddings)

    top_k = min(k, len(index_documents))
    scores, doc_indices = index.search(query_embedding, top_k)

    results: list[dict[str, Any]] = []

    for i, doc_idx in enumerate(doc_indices[0]):
        if doc_idx < 0:
            continue

        results.append(
            _format_result(
                rank=i + 1,
                score=float(scores[0][i]),
                method="vector",
                document=index_documents[int(doc_idx)],

                details={
                  "backend": "faiss",
                  "similarity": "cosine",
                  "embedding_source": "custom_embedding_fn" if using_custom_embedding_fn else "sentence_transformers",
                  "embedding_model": None if using_custom_embedding_fn else embedding_model_name,
                },

            )
        )

    return results


def retrieve_top_k_hybrid(
    query_text: str,
    index_documents: list[dict[str, Any]],
    k: int = 5,
    embedding_fn: EmbeddingFn | None = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k ATT&CK techniques using true hybrid retrieval.

    Hybrid here means:
    - BM25 keyword-search results
    - FAISS vector-search results
    - rank fusion using Reciprocal Rank Fusion

    RRF score:
        1 / (rrf_k + bm25_rank) + 1 / (rrf_k + vector_rank)

    Why rank fusion:
    - BM25 and vector similarity scores have different numeric scales.
    - Combining ranks is more stable than averaging raw scores.
    """
    if not query_text.strip() or not index_documents:
        return []

    bm25_results = retrieve_top_k_bm25(
        query_text=query_text,
        index_documents=index_documents,
        k=len(index_documents),
    )

    vector_results = retrieve_top_k_vector(
        query_text=query_text,
        index_documents=index_documents,
        k=len(index_documents),
        embedding_fn=embedding_fn,
        embedding_model_name=embedding_model_name,
    )

    bm25_rank_by_id = {
        result["technique_id"]: result["rank"]
        for result in bm25_results
    }
    vector_rank_by_id = {
        result["technique_id"]: result["rank"]
        for result in vector_results
    }

    document_by_id = {
        document["technique_id"]: document
        for document in index_documents
    }

    fused_scores: dict[str, float] = {}

    for technique_id in document_by_id:
        score = 0.0

        if technique_id in bm25_rank_by_id:
            score += 1.0 / (rrf_k + bm25_rank_by_id[technique_id])

        if technique_id in vector_rank_by_id:
            score += 1.0 / (rrf_k + vector_rank_by_id[technique_id])

        fused_scores[technique_id] = score

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    return [
        _format_result(
            rank=i + 1,
            score=score,
            method="hybrid",
            document=document_by_id[technique_id],

            details={
               "fusion": "rrf",
               "keyword_backend": "bm25",
               "vector_backend": "faiss",
               "bm25_rank": bm25_rank_by_id.get(technique_id),
               "vector_rank": vector_rank_by_id.get(technique_id),
               "embedding_source": "custom_embedding_fn" if embedding_fn is not None else "sentence_transformers",
               "embedding_model": None if embedding_fn is not None else embedding_model_name,
            },


        )
        for i, (technique_id, score) in enumerate(ranked)
    ]


def retrieve_top_k(
    query_text: str,
    index_documents: list[dict[str, Any]],
    k: int = 5,
    method: RetrievalMethod = "bm25",
    embedding_fn: EmbeddingFn | None = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """
    Public retrieval entrypoint.

    Parameters:
    - query_text:
        Threat-report text, usually ingestion.body_text.
    - index_documents:
        Output from indexing.py build_index_documents().
    - k:
        Number of candidate techniques to return.
    - method:
        bm25, tfidf, vector, or hybrid.
    - embedding_fn:
        Optional custom embedding function for vector or hybrid retrieval.
    - embedding_model_name:
        Local sentence-transformers model used if embedding_fn is not supplied.

    Default is BM25 because it runs locally without embedding dependencies.
    """
    if method == "bm25":
        return retrieve_top_k_bm25(
            query_text=query_text,
            index_documents=index_documents,
            k=k,
        )

    if method == "tfidf":
        return retrieve_top_k_tfidf(
            query_text=query_text,
            index_documents=index_documents,
            k=k,
        )

    if method == "vector":
        return retrieve_top_k_vector(
            query_text=query_text,
            index_documents=index_documents,
            k=k,
            embedding_fn=embedding_fn,
            embedding_model_name=embedding_model_name,
        )

    if method == "hybrid":
        return retrieve_top_k_hybrid(
            query_text=query_text,
            index_documents=index_documents,
            k=k,
            embedding_fn=embedding_fn,
            embedding_model_name=embedding_model_name,
        )

    raise ValueError(f"Unsupported retrieval method: {method}")