"""
Advanced RAG orchestration.

Applies the query-understanding techniques the router actually selected,
then always runs retrieval, reranking, contextual compression, and CRAG,
in that fixed order:

  1. self_query      -> metadata filter + cleaned question. Driven
     directly by the router's `has_metadata_constraints` flag, NOT by
     `techniques` -- a metadata filter isn't a phrasing choice, so it
     runs whenever the router flagged a metadata constraint, regardless
     of which (if any) of the four techniques below were also picked.
  2. rewriting        -> cleaned question, disambiguated
  3. decomposition OR multi_query -> a list of search queries
     (decomposition takes priority if the router picked both, since it
     means the question genuinely has separate information needs;
     multi_query is about phrasing the SAME need several ways)
  4. hyde              -> one extra hypothetical-passage query added to
     the search-query list
  5. hybrid retrieval, once per search query, fused across queries via RRF
  6. reranking          -> keep the LLM-rescored top few
  7. contextual_compression -> trim the surviving chunks to relevant
     sentences
  8. crag               -> grade the (reranked + compressed) evidence;
     one corrective re-retrieval -> rerank -> compression attempt if
     graded ambiguous/incorrect
  9. final grounded generation (generation.py -- same module every route
     uses)

If none of the four query-understanding techniques were selected, the
raw (or self_query-cleaned) question is used as the single search query.
Retrieval, reranking, compression, and CRAG are mandatory for every
advanced_rag question.

HyDE passages, rewrites, decomposed sub-questions, and multi-query
variants exist ONLY to build search queries for retrieval -- they are
never passed to the final LLM as evidence. Only chunks actually returned
by rag.retrieve() ever reach generation.build_grounded_prompt().
"""
import time
import uuid

import requests

import config
import generation
import rag
import query_transform
import retrieval_improve


def _multi_query_retrieve(queries: list, where: dict, pool_top_k: int):
    """Runs hybrid retrieve() once per query string, then RRF-fuses the
    results across queries (on top of the dense+BM25 fusion retrieve()
    already does internally for each individual query)."""
    id_lists = []
    hit_by_id = {}
    for q in queries:
        hits = rag.retrieve(q, top_k=pool_top_k, where=where)
        id_lists.append([h["id"] for h in hits])
        for h in hits:
            hit_by_id.setdefault(h["id"], h)

    fused_scores = rag.rrf_fuse(id_lists)
    ranked_ids = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)
    return [hit_by_id[i] for i in ranked_ids if i in hit_by_id]


def answer_advanced(question: str, decision: dict, top_k: int = None) -> dict:
    """
    Runs the advanced_rag route for one question.

    `decision` is the FULL router decision dict from router.classify_route()
    (not just the `techniques` list) -- advanced_rag needs
    `has_metadata_constraints` too, since self_query is driven by that
    flag rather than being one of the router-selectable techniques.

    Returns the same record schema as rag.answer_question_full(), plus advanced-route-specific fields:
    techniques_used, executed_steps, retrieval_stats, search_queries,
    self_query_filter, crag_grade, crag_corrected.
    """
    techniques = list(decision.get("techniques") or [])
    top_k = top_k or config.TOP_K
    total_start = time.perf_counter()
    executed_steps = ["router"]

    # 1. self_query -- gated on has_metadata_constraints, independent of
    # the `techniques` list.
    where_filter = None
    cleaned_question = question
    if decision.get("has_metadata_constraints"):
        print("[pipeline] executing: self_query")
        executed_steps.append("self_query")
        where_filter, cleaned_question = query_transform.self_query(question)

    # 2. rewriting
    if "rewriting" in techniques:
        print("[pipeline] executing: rewriting")
        executed_steps.append("rewriting")
        cleaned_question = query_transform.rewrite_query(cleaned_question)

    # 3. decomposition OR multi_query
    if "decomposition" in techniques:
        print("[pipeline] executing: decomposition")
        executed_steps.append("decomposition")
        search_queries = query_transform.decompose_query(cleaned_question)
    elif "multi_query" in techniques:
        print("[pipeline] executing: multi_query")
        executed_steps.append("multi_query")
        search_queries = query_transform.multi_query(cleaned_question)
    else:
        search_queries = [cleaned_question]

    # 4. hyde -- one extra hypothetical-passage query, based on the
    # (possibly rewritten) primary question, not per sub-question, to
    # keep the extra LLM call bounded regardless of how many search
    # queries decomposition/multi_query produced.
    if "hyde" in techniques:
        print("[pipeline] executing: hyde")
        executed_steps.append("hyde")
        hyde_passage = query_transform.hyde_answer(cleaned_question)
        search_queries = search_queries + [hyde_passage]

    # 5. retrieval, fused across every search query
    print("[pipeline] executing: retrieval")
    executed_steps.append("retrieval")
    retrieval_pool = config.RERANK_CANDIDATE_POOL
    hits = _multi_query_retrieve(search_queries, where_filter, retrieval_pool)
    num_candidates = len(hits)

    # 6. reranking
    print("[pipeline] executing: reranking")
    executed_steps.append("reranking")
    hits = retrieval_improve.rerank(cleaned_question, hits, keep_top_k=top_k)
    num_after_reranking = len(hits)

    # 7. contextual compression
    print("[pipeline] executing: contextual_compression")
    executed_steps.append("contextual_compression")
    hits = retrieval_improve.contextual_compression(cleaned_question, hits)
    num_after_compression = len(hits)

    # 8. CRAG -- grades the reranked + compressed evidence. A corrective
    # attempt re-runs retrieval -> rerank -> compression with a widened
    # pool, so a correction's output has gone through the exact same
    # steps as the first attempt before CRAG re-grades it.
    print("[pipeline] executing: crag")
    executed_steps.append("crag")

    def _corrective_retrieve():
        wider_pool = retrieval_pool * config.CRAG_CORRECTED_POOL_MULTIPLIER
        corrected = _multi_query_retrieve(search_queries, where_filter, wider_pool)
        corrected = retrieval_improve.rerank(cleaned_question, corrected, keep_top_k=top_k)
        corrected = retrieval_improve.contextual_compression(cleaned_question, corrected)
        return corrected

    crag_result = retrieval_improve.crag_evaluate_and_correct(
        cleaned_question, hits, _corrective_retrieve, top_k
    )
    hits = crag_result["hits"]
    crag_grade = crag_result["grade"]
    crag_corrected = crag_result["corrected"]
    if crag_corrected:
        # The corrected hits already passed through rerank + compression
        # inside _corrective_retrieve, so the "after" counts reflect them.
        num_after_reranking = len(hits)
        num_after_compression = len(hits)

    # 9. final grounded generation -- same shared module every route uses.
    # `hits` here are the only things that ever reach the prompt as
    # evidence; search_queries/hyde_passage/rewrites never do.
    print("[pipeline] executing: generation")
    executed_steps.append("generation")
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    prompt = generation.build_grounded_prompt(cleaned_question, hits)
    try:
        result = generation.call_llm(prompt, arabic=generation.is_arabic(cleaned_question))
    except requests.exceptions.ConnectionError:
        raise RuntimeError(generation.connection_error_message())

    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    cost = generation.compute_cost(config.GENERATION_MODEL, input_tokens, output_tokens)

    techniques_used = sorted(
        set(techniques)
        | config.MANDATORY_ADVANCED_STAGES
        | ({"self_query"} if decision.get("has_metadata_constraints") else set())
    )

    return {
        "record_id": str(uuid.uuid4()),
        "approach": "advanced_rag",
        "question": question,
        "cleaned_question": cleaned_question,
        "techniques_used": techniques_used,
        "executed_steps": executed_steps,
        "retrieval_stats": {
            "num_candidates": num_candidates,
            "num_after_reranking": num_after_reranking,
            "num_after_compression": num_after_compression,
        },
        "search_queries": search_queries,
        "self_query_filter": where_filter,
        "crag_grade": crag_grade,
        "crag_corrected": crag_corrected,
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
