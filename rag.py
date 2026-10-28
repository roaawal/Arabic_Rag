"""
Core retrieval logic (hybrid dense + BM25, fused via Reciprocal Rank
Fusion) plus the plain "rag" route's end-to-end answer function.

Final-answer prompting/LLM-calling now lives in generation.py and is
shared across every route (direct, rag, advanced_rag) -- see
that module's docstring.
"""
import re

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import requests

import config
import generation

_embed_model = None
_collection = None
_bm25_index = None
_bm25_ids = None
_bm25_doc_lookup = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model '{config.EMBED_MODEL_NAME}' ...")
        _embed_model = SentenceTransformer(config.EMBED_MODEL_NAME)
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        try:
            _collection = chroma_client.get_collection(config.COLLECTION_NAME)
        except chromadb.errors.NotFoundError as e:
            raise RuntimeError(
                f"Index collection '{config.COLLECTION_NAME}' was not found in "
                f"'{config.CHROMA_DIR}'. Run `python ingestion.py` first."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "The index exists but could not be opened. This usually means "
                "the ChromaDB client version does not match the one used to "
                "build the index. Run Streamlit with the project virtual "
                "environment and reinstall requirements if needed."
            ) from e
    return _collection


def get_known_metadata_values():
    """
    Returns {"sources": [...], "sections": [...]} -- the distinct values
    actually present in the index. self_query.py uses this so it only
    ever proposes filters against metadata that genuinely exists, instead
    of hallucinating a source filename or section name that isn't real.
    """
    collection = _get_collection()
    all_data = collection.get(include=["metadatas"])
    sources, sections = set(), set()
    for meta in all_data["metadatas"]:
        if meta.get("source"):
            sources.add(meta["source"])
        if meta.get("section"):
            sections.add(meta["section"])
    return {"sources": sorted(sources), "sections": sorted(sections)}


def _tokenize(text: str):
    """Simple Unicode-aware word tokenizer -- works for Arabic and Latin
    script alike, which is all BM25 needs (it just counts term overlap)."""
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def _get_bm25():
    """
    Builds (and caches) a BM25 keyword index over every chunk already
    stored in Chroma. No re-indexing required -- this just reads the same
    chunk text that ingestion.py already embedded and stored.
    """
    global _bm25_index, _bm25_ids, _bm25_doc_lookup
    if _bm25_index is None:
        collection = _get_collection()
        all_data = collection.get(include=["documents", "metadatas"])
        _bm25_ids = all_data["ids"]
        texts = all_data["documents"]
        metadatas = all_data["metadatas"]

        _bm25_doc_lookup = {
            doc_id: {
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "section": meta.get("section") or None,
            }
            for doc_id, text, meta in zip(_bm25_ids, texts, metadatas)
        }

        tokenized_corpus = [_tokenize(t) for t in texts]
        _bm25_index = BM25Okapi(tokenized_corpus)
    return _bm25_index, _bm25_ids, _bm25_doc_lookup


def _bm25_ids_matching_filter(where: dict) -> set:
    """
    BM25 has no native metadata filter -- rank_bm25 just scores a
    tokenized corpus. To honor a self_query `where` filter for the BM25
    side too, we ask Chroma which ids match the filter (it's good at
    that) and intersect with BM25's own ranking afterwards.
    """
    collection = _get_collection()
    matched = collection.get(where=where, include=[])
    return set(matched["ids"])


def _dense_search(question: str, pool: int, where: dict = None):
    """Returns a list of chunk ids ranked by dense (embedding) similarity,
    optionally restricted to a Chroma metadata `where` filter (self_query)."""
    model = _get_embed_model()
    collection = _get_collection()
    query_embedding = model.encode([question], normalize_embeddings=True)[0].tolist()
    kwargs = {"query_embeddings": [query_embedding], "n_results": pool}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    return results["ids"][0]


def _bm25_search(question: str, pool: int, where: dict = None):
    """Returns a list of chunk ids ranked by BM25 keyword overlap,
    optionally restricted to a self_query `where` filter."""
    bm25, ids, _ = _get_bm25()
    scores = bm25.get_scores(_tokenize(question))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if where:
        allowed_ids = _bm25_ids_matching_filter(where)
        ranked_indices = [i for i in ranked_indices if ids[i] in allowed_ids]

    ranked_indices = ranked_indices[:pool]
    return [ids[i] for i in ranked_indices]


def _dense_only_retrieve(question: str, top_k: int, where: dict = None):
    """Pure dense retrieval -- kept as a fallback if USE_HYBRID_RETRIEVAL is
    turned off in config.py, and used as the
    baseline retrieval method."""
    model = _get_embed_model()
    collection = _get_collection()
    query_embedding = model.encode([question], normalize_embeddings=True)[0].tolist()

    kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)

    hits = []
    for doc_id, text, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "id": doc_id,
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "section": meta.get("section") or None,
            "score": 1 - dist,
        })
    return hits


def rrf_fuse(ranked_id_lists: list, k: int = None) -> dict:
    """
    Generic Reciprocal Rank Fusion over any number of ranked id lists --
    used both for dense+BM25 fusion within retrieve(), and (by
    advanced_rag.py) for fusing results across several query variants
    from multi-query/decomposition. Returns {id: fused_score}, higher is
    better; caller sorts and truncates.
    """
    k = k if k is not None else config.RRF_K
    scores = {}
    for id_list in ranked_id_lists:
        for rank, doc_id in enumerate(id_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve(question: str, top_k: int = None, where: dict = None):
    """
    Hybrid dense+BM25 retrieval for a single query string, fused via RRF.
    `where` is an optional Chroma metadata filter (e.g.
    {"section": "..."} or {"page": {"$gte": 5, "$lte": 10}}) produced by
    self_query.py -- pass None for the normal unfiltered case.
    """
    top_k = top_k or config.TOP_K

    if not config.USE_HYBRID_RETRIEVAL:
        return _dense_only_retrieve(question, top_k, where=where)

    pool = config.HYBRID_CANDIDATE_POOL
    dense_ids = _dense_search(question, pool, where=where)
    bm25_ids = _bm25_search(question, pool, where=where)
    _, _, doc_lookup = _get_bm25()  # already loaded by _bm25_search, just fetching the lookup

    rrf_scores = rrf_fuse([dense_ids, bm25_ids])
    ranked_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

    hits = []
    for doc_id in ranked_ids:
        info = doc_lookup[doc_id]
        hits.append({
            "id": doc_id,
            "text": info["text"],
            "source": info["source"],
            "page": info["page"],
            "section": info.get("section"),
            "score": rrf_scores[doc_id],  # RRF score, not a 0-1 similarity -- higher is still better
        })
    return hits


def answer_question_full(question: str, top_k: int = None) -> dict:
    """
    Same pipeline as answer_question(), but returns the full structured
    record (question, chunk_ids, context, answer, tokens, cost, latency)
    in the schema evaluate.py expects.
    """
    import time
    import uuid

    total_start = time.perf_counter()
    hits = retrieve(question, top_k=top_k)
    context = "\n\n---\n\n".join(h["text"] for h in hits)

    prompt = generation.build_grounded_prompt(question, hits)
    try:
        result = generation.call_llm(prompt, arabic=generation.is_arabic(question))
    except requests.exceptions.ConnectionError:
        raise RuntimeError(generation.connection_error_message())

    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    cost = generation.compute_cost(config.GENERATION_MODEL, input_tokens, output_tokens)

    return {
        "record_id": str(uuid.uuid4()),
        "approach": "hybrid_rag" if config.USE_HYBRID_RETRIEVAL else "dense_rag",
        "question": question,
        "executed_steps": ["router", "retrieval", "generation"],
        "retrieval_stats": {
            "num_candidates": len(hits),
            "num_after_reranking": None,
            "num_after_compression": None,
        },
        "chunk_ids": [h["id"] for h in hits],
        "retrieved_chunks": [
            {
                "id": h["id"],
                "text": h["text"],
                "source": h["source"],
                "page": h["page"],
                "section": h.get("section"),
                "score": h["score"],
            }
            for h in hits
        ],
        "context": context,
        "answer": generation.strip_inline_sources(result["text"]),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "latency_seconds": result["latency_seconds"],
        "total_latency_seconds": time.perf_counter() - total_start,
    }
