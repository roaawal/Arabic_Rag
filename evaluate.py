"""
Per-question evaluation: scores any RAG record (from rag.answer_question_full()
or advanced_rag.answer_advanced()) against four metrics, using ONE fixed
rubric so scores are comparable across approaches:

  - Context Relevance : did retrieval return useful evidence?
  - Faithfulness       : is the answer supported by the retrieved context?
  - Answer Relevance   : does the answer actually answer the question?
  - Correctness        : matches a trusted reference answer (only scored
                          when a reference is supplied -- otherwise null)

CAVEAT: the judge (config.JUDGE_MODEL) is, by default, the same Gemini
model that writes the answers, and LLM judges tend to favour their own
output. Treat the scores as directional -- good for spotting regressions
between approaches, not as ground truth. For results you need to trust
more, point config.JUDGE_MODEL at a stronger model (e.g.
gemini-3.1-pro-preview; it costs about 2.7x more per token and its
mandatory thinking makes each judgement slower and pricier).
"""
import json
import re
import time
import uuid

import config
import generation


_RUBRIC = """You are a strict evaluator for a document question-answering system.
Score the given ANSWER on these four metrics, each on a 0-5 integer scale:

1. context_relevance (0-5): Does the CONTEXT contain information that is
   actually relevant to answering the QUESTION? 5 = highly relevant,
   direct evidence for what's needed; 3 = partially relevant, mixed with
   irrelevant material; 0 = completely irrelevant (or no context at all).

2. faithfulness (0-5): Is every important claim in the ANSWER actually
   supported by the CONTEXT (no hallucination, no outside knowledge)?
   5 = all important claims clearly supported; 3 = mostly grounded but
   some unsupported claims or extrapolation; 0 = completely unsupported
   or fabricated.

3. answer_relevance (0-5): Does the ANSWER directly and completely
   address what the QUESTION actually asked, without padding or going
   off-topic? 5 = directly and completely answers it; 3 = partially
   answers or includes significant irrelevant content; 0 = does not
   address the question at all.

4. correctness (0-5 or null): ONLY if a REFERENCE ANSWER is provided below,
   compare the ANSWER's key facts against it. 5 = fully correct, matches
   the reference's key facts; 3 = partially correct; 0 = completely
   incorrect/contradicts the reference. If no REFERENCE ANSWER is
   provided, output null for this field to mark correctness as
   unavailable -- do not guess a score or pretend it was verified.

Respond with ONLY a single JSON object, no markdown fences, no extra text:
{"context_relevance": <0-5>, "faithfulness": <0-5>, "answer_relevance": <0-5>, "correctness": <0-5 or null>, "notes": "<one short sentence explaining the weakest score>"}
"""


def _extract_json(text: str) -> dict:
    """Judge models sometimes wrap JSON in markdown fences or add stray
    text around it -- pull out the first {...} block and parse that."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:200]!r}")
    return json.loads(match.group(0))


def _call_judge(question: str, context: str, answer: str, reference: str = None) -> dict:
    reference_block = (
        f"\nREFERENCE ANSWER:\n{reference}\n" if reference else "\n(No reference answer supplied.)\n"
    )
    user_prompt = f"""QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
{reference_block}
Score the ANSWER now."""

    result = generation.call_completion(
        messages=[
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.JUDGE_MAX_NEW_TOKENS,
        temperature=config.JUDGE_TEMPERATURE,
        stage="evaluation_judge",
        model=config.JUDGE_MODEL,
        timeout=180,
    )
    raw_text = result["text"]

    try:
        scores = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as e:
        scores = {
            "context_relevance": None,
            "faithfulness": None,
            "answer_relevance": None,
            "correctness": None,
            "notes": f"JUDGE PARSE FAILURE: {e}. Raw: {raw_text[:300]!r}",
        }
    return scores


def evaluate_record(rag_record: dict, reference: str = None) -> dict:
    """
    Scores one RAG record (as produced by rag.answer_question_full() or
    advanced_rag.answer_advanced() -- both share the same schema) against
    the fixed four-metric rubric, and returns the record with the scores
    added.
    """
    start = time.perf_counter()
    try:
        scores = _call_judge(
            question=rag_record["question"],
            context=rag_record["context"],
            answer=rag_record["answer"],
            reference=reference,
        )
    except Exception as exc:
        scores = {
            "context_relevance": None,
            "faithfulness": None,
            "answer_relevance": None,
            "correctness": None,
            "notes": f"JUDGE CALL FAILURE: {type(exc).__name__}: {exc}",
        }
    judge_latency = time.perf_counter() - start

    combined = dict(rag_record)
    if not rag_record.get("context"):
        scores["context_relevance"] = None
        scores["faithfulness"] = None
    if not reference:
        scores["correctness"] = None
    judge_notes = scores.get("notes")
    if not isinstance(judge_notes, str) or not judge_notes.strip():
        scored_metrics = {
            name: scores.get(name)
            for name in ("context_relevance", "faithfulness", "answer_relevance", "correctness")
            if isinstance(scores.get(name), (int, float))
        }
        if scored_metrics:
            weakest_metric = min(scored_metrics, key=scored_metrics.get)
            judge_notes = (
                f"The judge omitted notes; the weakest available score was "
                f"{weakest_metric} ({scored_metrics[weakest_metric]}/5)."
            )
        else:
            judge_notes = "The judge did not return an explanatory note or usable scores."
    combined.update({
        "eval_record_id": str(uuid.uuid4()),
        "reference_answer": reference,
        "context_relevance": scores.get("context_relevance"),
        "faithfulness": scores.get("faithfulness"),
        "answer_relevance": scores.get("answer_relevance"),
        "correctness": scores.get("correctness"),
        "judge_notes": judge_notes,
        "judge_model": config.JUDGE_MODEL,
        "judge_latency_seconds": judge_latency,
        "evaluation_rubric": "fixed_0_to_5_context_faithfulness_relevance_correctness_v2",
        "reference_available": bool(reference),
    })

    return combined
