"""
Retrieval-improvement techniques for the advanced_rag route: reranking,
contextual compression, and CRAG. Each takes the hits already produced by
rag.retrieve()/rrf_fuse() and returns an improved hit list; none of them
re-embed or re-query on their own except CRAG's corrective step, which
calls back into rag.retrieve() explicitly.
"""
import json
import re

import requests

import config
import generation


def _score_llm_json(system_prompt: str, user_prompt: str, max_tokens: int, fallback, stage: str,
                     model: str = None):
    try:
        result = generation.call_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            stage=stage,
            model=model,
            timeout=60,
        )
        raw = result["text"]
        match = re.search(r"[\{\[].*[\}\]]", raw, re.DOTALL)
        if not match:
            return fallback
        return json.loads(match.group(0))
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
_RERANK_SYSTEM = """You are scoring how relevant each retrieved passage is to a question, \
on a 1-10 integer scale (10 = directly and fully answers it, 1 = unrelated). \
Respond with ONLY a JSON object mapping each given id to its integer score, \
no explanation, e.g. {"id1": 8, "id2": 3}"""


def rerank(question: str, hits: list, keep_top_k: int = None) -> list:
    """
    LLM-based pointwise reranking (no cross-encoder dependency needed --
    reuses the same local LLM already running for generation). Scores
    each of the (already RRF-fused) candidate hits for relevance to the
    question, then keeps the top `keep_top_k` by that score.

    Caps the candidate pool at config.RERANK_CANDIDATE_POOL before
    scoring, since pointwise LLM scoring is O(n) LLM calls worth of
    tokens in one batched prompt -- an unbounded candidate pool would
    make reranking cost more than the final answer itself.
    """
    keep_top_k = keep_top_k or config.RERANK_KEEP_TOP_K
    if not hits:
        return hits

    candidates = hits[: config.RERANK_CANDIDATE_POOL]
    passages = "\n\n".join(f'id="{h["id"]}":\n{h["text"][:1000]}' for h in candidates)
    user_prompt = f"Question: {question}\n\nPassages:\n{passages}"

    fallback_scores = {h["id"]: len(candidates) - i for i, h in enumerate(candidates)}
    scores = _score_llm_json(_RERANK_SYSTEM, user_prompt, max_tokens=600, fallback=fallback_scores,
                              stage="reranking", model=config.RERANKER_MODEL)
    if not isinstance(scores, dict):
        scores = fallback_scores

    def _score_of(h):
        try:
            return float(scores.get(h["id"], 0))
        except (TypeError, ValueError):
            return 0.0

    ranked = sorted(candidates, key=_score_of, reverse=True)
    return ranked[:keep_top_k]


# ---------------------------------------------------------------------------
# Contextual Compression
# ---------------------------------------------------------------------------
def contextual_compression(question: str, hits: list) -> list:
    """
    For each hit, asks the LLM to extract ONLY the sentences relevant to
    the question, dropping unrelated content from the same chunk (e.g. a
    chunk that drifted across a section boundary, or carries a mix of
    rules only one of which matters here). A hit whose compressed text
    comes back empty/near-empty is dropped entirely -- it wasn't
    actually relevant. Falls back to the ORIGINAL uncompressed text for
    any hit where compression fails, rather than risking silently
    dropping a genuinely relevant chunk due to an LLM/parse hiccup.
    """
    arabic = generation.is_arabic(question)
    system = (
        "استخرج فقط الجمل ذات الصلة بالسؤال من المقطع التالي، دون تلخيص أو إضافة أي "
        "معلومة جديدة، ودون تغيير الصياغة الأصلية. إذا لم يكن هناك شيء ذو صلة، أعد "
        "سلسلة نصية فارغة."
        if arabic else
        "Extract ONLY the sentences relevant to the question from the passage below, "
        "verbatim -- don't summarize, don't add anything, don't reword. If nothing in "
        "the passage is relevant, return an empty string."
    )

    compressed_hits = []
    for h in hits:
        user_prompt = f"Question: {question}\n\nPassage:\n{h['text']}"
        try:
            result = generation.call_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
                temperature=0.0,
                stage="contextual_compression",
                model=config.COMPRESSION_MODEL,
                timeout=60,
            )
            compressed_text = result["text"]
        except Exception:
            compressed_text = h["text"]  # fail safe: keep original rather than drop

        if not compressed_text:
            continue  # LLM explicitly found nothing relevant in this chunk
        new_hit = dict(h)
        new_hit["text"] = compressed_text
        compressed_hits.append(new_hit)

    return compressed_hits if compressed_hits else hits


# ---------------------------------------------------------------------------
# CRAG (Corrective RAG)
# ---------------------------------------------------------------------------
_CRAG_GRADE_SYSTEM = """You are grading whether retrieved context is sufficient to answer a question, \
for a document QA system. Respond with ONLY a JSON object:
{"grade": "correct|ambiguous|incorrect", "reason": "<one short sentence>"}
"correct": the context clearly contains what's needed to answer.
"ambiguous": the context is partially relevant or incomplete.
"incorrect": the context is not actually about what the question asks."""


def _grade_context(question: str, hits: list) -> dict:
    if not hits:
        return {"grade": "incorrect", "reason": "No context was retrieved at all."}
    context_preview = "\n\n".join(h["text"][:400] for h in hits)
    user_prompt = f"Question: {question}\n\nRetrieved context:\n{context_preview}"
    fallback = {"grade": "ambiguous", "reason": "Grading call failed; treating as ambiguous to be safe."}
    result = _score_llm_json(_CRAG_GRADE_SYSTEM, user_prompt, max_tokens=150, fallback=fallback,
                              stage="crag_evaluator", model=config.CRAG_MODEL)
    if not isinstance(result, dict) or result.get("grade") not in {"correct", "ambiguous", "incorrect"}:
        return fallback
    return result


def crag_evaluate_and_correct(question: str, hits: list, retrieve_fn, top_k: int) -> dict:
    """
    Grades the retrieved context's quality and, if it's graded
    "ambiguous" or "incorrect", triggers ONE corrective re-retrieval
    (config.CRAG_MAX_CORRECTIONS) with a widened candidate pool -- this
    pipeline has no external web-search fallback (fully local/offline),
    so "correction" here means retrying retrieval wider/differently
    rather than falling back to the web, which is the other half of the
    original CRAG paper's design.

    `retrieve_fn` is a zero-arg callable the caller builds (a closure
    over whatever query/where-filter it was already using) that returns
    a fresh list of hits with a wider pool -- kept generic so
    advanced_rag.py can pass in whatever retrieval call it was already
    making, single- or multi-query, filtered or not.

    Returns {"hits": [...], "grade": "...", "corrected": bool} -- the
    final grade and whether a correction attempt actually ran, both
    surfaced in the pipeline's returned record for transparency.
    """
    grade_result = _grade_context(question, hits)
    grade = grade_result["grade"]
    corrected = False

    attempts = 0
    while grade in {"ambiguous", "incorrect"} and attempts < config.CRAG_MAX_CORRECTIONS:
        attempts += 1
        new_hits = retrieve_fn()
        if new_hits:
            hits = new_hits
            corrected = True
        grade_result = _grade_context(question, hits)
        grade = grade_result["grade"]

    return {"hits": hits, "grade": grade, "grade_reason": grade_result["reason"], "corrected": corrected}
