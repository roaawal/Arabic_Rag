"""
Routes a question BEFORE any retrieval happens.

Output shape (always -- a fully validated, machine-readable decision):
    {
      "complexity": "simple" | "complex",
      "requires_documents": bool,
      "multiple_information_needs": bool,
      "multiple_search_perspectives": bool,
      "has_metadata_constraints": bool,
      "is_vague_or_conversational": bool,
      "needs_clarification": bool,
      "route": "direct" | "rag" | "advanced_rag" | "clarify",
      "techniques": [...],
      "reason": "...",
    }

`techniques` is only meaningful for route == "advanced_rag" and may only
contain the four query-UNDERSTANDING techniques the router actively
selects: "rewriting", "multi_query", "decomposition", "hyde".
`has_metadata_constraints` is intentionally NOT one of those four -- it
isn't a phrasing choice, it's a metadata filter, so advanced_rag.py acts
on that boolean directly (via query_transform.self_query()) regardless
of which of the four techniques were also picked. Advanced RAG always
adds reranking at execution time; the router does not need to select it.

The LLM is the primary classifier (it's the only thing that can actually
read intent), but a small local model won't always return clean JSON, so
_heuristic_classify() is a keyword/structure-based fallback that fires
whenever the LLM's output can't be parsed -- the router always returns
something usable, never raises.
"""
import json
import re

import requests

import config
import generation

_ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a document question-answering system over an Arabic bank internal-procedures manual. For each user question, classify it along these dimensions:

- complexity: "simple" (a quick fact/definition/greeting) or "complex" (multi-part, needs reasoning)?
- requires_documents: does answering it require information from the indexed procedure documents at all, or is it general/conversational (e.g. greetings, small talk, asking about your own capabilities, general knowledge unrelated to the bank's procedures)?
- multiple_information_needs: does it bundle MULTIPLE distinct information needs (e.g. "what is the annual inventory procedure AND who approves asset write-offs")?
- multiple_search_perspectives: would it benefit from searching with several different phrasings/angles because the wording is abstract or could be described several ways?
- has_metadata_constraints: does it carry a METADATA constraint -- a specific section name, page number, source document, department, year/date/version -- that should narrow the search rather than just being extra keywords?
- is_vague_or_conversational: is the wording vague, conversational, or dependent on unstated context (e.g. "what about the second one", "and if it's damaged?") such that it needs to be rewritten into a clear standalone query before searching?
- needs_clarification: is it about the manuals but so broad that several different procedures could be meant, and nothing in the question says which (e.g. "how are inventory cases handled?", when there are separate inventories for stationery warehouses, fixed assets, and loan files)? A question that names the specific procedure, or asks for something all of them share, does not need clarification.

Then decide the route:
- "direct": requires_documents is false (greetings, meta questions about the assistant, or anything outside the bank's procedures, such as sports, politics, or general knowledge). This route declines out-of-scope questions, so send them here rather than searching the manuals.
- "clarify": needs_clarification is true. Takes priority over "rag" and "advanced_rag".
- "rag": requires_documents is true, but it is a single, clear, well-formed, single-need question with no metadata constraint and no ambiguity -- plain hybrid retrieval + generation is enough.
- "advanced_rag": requires_documents is true AND at least one of multiple_information_needs / multiple_search_perspectives / has_metadata_constraints / is_vague_or_conversational is true -- plain retrieval would likely handle it poorly.

If route is "advanced_rag", choose the SMALLEST set of techniques from this fixed list that actually addresses what's complicating this specific question -- do not include a technique "just in case", and do not include a technique the boolean flags above don't support:
- "rewriting": use when is_vague_or_conversational is true.
- "multi_query": use when multiple_search_perspectives is true.
- "decomposition": use when multiple_information_needs is true.
- "hyde": use when the question is conceptual/asks "why" or "how does X work" in a way where a hypothetical answer passage would embed closer to the real answer than the bare question would.
Do NOT put "self_query" in techniques -- metadata filtering is handled automatically from has_metadata_constraints, it is not a technique you select.
If route is "rag", "direct", or "clarify", techniques MUST be an empty list.

Respond with ONLY a single JSON object, no markdown fences, no extra text, matching exactly this shape:
{"complexity": "simple|complex", "requires_documents": true|false, "multiple_information_needs": true|false, "multiple_search_perspectives": true|false, "has_metadata_constraints": true|false, "is_vague_or_conversational": true|false, "needs_clarification": true|false, "route": "direct|rag|advanced_rag|clarify", "techniques": [], "reason": "one short sentence"}
The value of "route" must be exactly ONE of: "direct", "rag", "advanced_rag", "clarify" -- do not copy the pipe-separated placeholder into the value.
The "reason" value is required and must be one short, non-empty sentence explaining the route choice.
Example (a real decision, not the placeholders above):
{"complexity": "complex", "requires_documents": true, "multiple_information_needs": true, "multiple_search_perspectives": true, "has_metadata_constraints": false, "is_vague_or_conversational": false, "needs_clarification": false, "route": "advanced_rag", "techniques": ["decomposition", "multi_query"], "reason": "Multiple information needs require decomposition and multiple retrieval perspectives."}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in router response: {text[:200]!r}")
    return json.loads(match.group(0))


def _heuristic_classify(question: str, fallback_reason: str = None) -> dict:
    """
    Non-LLM fallback used only when the router's LLM call fails outright
    (connection error) or returns something unparseable. Deliberately
    simple and conservative -- it leans toward "rag" (safe middle ground)
    rather than guessing at advanced techniques it can't justify from a
    quick heuristic. Always returns the full validated decision shape.
    """
    q = question.strip()
    q_lower = q.lower()
    fallback_prefix = "Heuristic fallback"
    if fallback_reason:
        fallback_prefix += f" ({fallback_reason})"

    def _decision(**overrides):
        base = {
            "complexity": "simple",
            "requires_documents": False,
            "multiple_information_needs": False,
            "multiple_search_perspectives": False,
            "has_metadata_constraints": False,
            "is_vague_or_conversational": False,
            "needs_clarification": False,
            "route": "direct",
            "techniques": [],
            "reason": fallback_prefix,
        }
        base.update(overrides)
        return base

    greeting_markers = ["hello", "hi ", "hey", "thanks", "thank you", "مرحبا", "شكرا", "أهلا", "السلام عليكم"]
    if len(q) < 25 and any(m in q_lower for m in greeting_markers):
        return _decision(
            reason=f"{fallback_prefix}: short greeting/small-talk pattern, no document need detected.",
        )

    if not q.endswith("?") and not q.endswith("؟") and len(q.split()) < 4:
        return _decision(
            reason=f"{fallback_prefix}: very short, non-question fragment.",
        )

    metadata_markers = ["page", "صفحة", "section", "قسم", "الجزء", ".pdf", "version", "إصدار"]
    has_metadata = any(m in q_lower for m in metadata_markers)
    tokens = re.findall(r"[\w’']+", q_lower, re.UNICODE)
    interrogative_words = {
        "what", "who", "which", "when", "where", "why", "how",
        "ما", "ماذا", "من", "أي", "متى", "أين", "لماذا", "كيف", "هل",
    }
    interrogative_count = sum(
        token in interrogative_words
        or (token.startswith("و") and token[1:] in interrogative_words)
        for token in tokens
    )
    has_multi_need = interrogative_count >= 2
    is_long = len(q.split()) > 25

    techniques = []
    if has_multi_need:
        techniques.append("decomposition")
    if is_long and "decomposition" not in techniques:
        techniques.append("multi_query")

    if has_metadata or techniques:
        return _decision(
            complexity="complex" if (has_multi_need or is_long) else "simple",
            requires_documents=True,
            multiple_information_needs=has_multi_need,
            multiple_search_perspectives=is_long,
            has_metadata_constraints=has_metadata,
            route="advanced_rag",
            techniques=techniques,
            reason=f"{fallback_prefix}: detected multi-part wording and/or a metadata-like reference.",
        )

    return _decision(
        requires_documents=True,
        route="rag",
        reason=f"{fallback_prefix}: looks like a single clear document question.",
    )


def _looks_document_related(question: str) -> bool:
    """Recognize terms that indicate the question belongs to the indexed manual."""
    tokens = set(re.findall(r"[\w’']+", question.lower(), re.UNICODE))
    document_markers = {
        # Arabic terms found in, or strongly indicative of, procedure manuals.
        "وحدة", "الانذار", "الإنذار", "مركزي", "المركزي", "موجودات",
        "الموجودات", "ثابتة", "الجرد", "جرد", "إجراء", "إجراءات",
        "اعتماد", "اعتماد", "اعتمد", "شطب", "الأصناف", "مستودع",
        "مستودعات", "نموذج", "نماذج", "مسؤول", "المسؤول", "سياسة",
        "سياسات", "عملية", "عمليات", "البنك", "الفرع", "الفروع",
        # English equivalents for mixed-language questions.
        "procedure", "procedures", "inventory", "asset", "assets", "approval",
        "approve", "writeoff", "write-off", "warehouse", "branch", "manual",
    }
    return bool(tokens & document_markers)


def _validate(decision: dict, question: str) -> dict:
    """Clamps whatever the LLM returned to the actually-valid vocabulary,
    falling back to the heuristic if the structure is unusable."""
    if not isinstance(decision, dict):
        return _heuristic_classify(question, "classifier response was not an object")

    route = decision.get("route")
    if route not in config.VALID_ROUTES:
        return _heuristic_classify(question, f"invalid route {route!r}")

    def _bool(key):
        return bool(decision.get(key))

    complexity = decision.get("complexity")
    if complexity not in config.VALID_COMPLEXITY:
        complexity = "complex" if route == "advanced_rag" else "simple"

    requires_documents = _bool("requires_documents") or route in {"rag", "advanced_rag", "clarify"}
    multiple_information_needs = _bool("multiple_information_needs")
    multiple_search_perspectives = _bool("multiple_search_perspectives")
    has_metadata_constraints = _bool("has_metadata_constraints")
    is_vague_or_conversational = _bool("is_vague_or_conversational")
    needs_clarification = route == "clarify"

    techniques = decision.get("techniques") or []
    if not isinstance(techniques, list):
        techniques = []
    techniques = [
        technique
        for technique in techniques
        if isinstance(technique, str) and technique in config.VALID_TECHNIQUES
    ]

    # Consistency check: a technique is only allowed to survive if the
    # boolean flag that's supposed to justify it actually says so. Small
    # local models (esp. at temperature 0 with a single detailed few-shot
    # example in the prompt) can regress toward reproducing that example
    # -- echoing its `reason` text and technique choice -- rather than
    # reasoning about the actual question, even while getting their own
    # boolean flags right. Cross-checking against the flags here catches
    # that mismatch in code, independent of anything the model's `reason`
    # string claims.
    dropped_for_inconsistency = []
    if "decomposition" in techniques and not multiple_information_needs:
        techniques.remove("decomposition")
        dropped_for_inconsistency.append("decomposition (multiple_information_needs=False)")
    if "multi_query" in techniques and not multiple_search_perspectives:
        techniques.remove("multi_query")
        dropped_for_inconsistency.append("multi_query (multiple_search_perspectives=False)")
    if "rewriting" in techniques and not is_vague_or_conversational:
        techniques.remove("rewriting")
        dropped_for_inconsistency.append("rewriting (is_vague_or_conversational=False)")

    if route != "advanced_rag":
        techniques = []
    elif not techniques and not has_metadata_constraints:
        # advanced_rag was chosen but nothing survived validation and
        # there's no metadata constraint to act on either -- fall back to
        # plain rag rather than running an "advanced" pipeline that
        # would do nothing differently from it.
        route = "rag"

    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = f"Router model omitted a reason; using the validated {route!r} route."
    reason = reason.strip()
    if dropped_for_inconsistency:
        reason += (
            " [Corrected: dropped inconsistent technique(s) the model's own flags didn't "
            f"support: {', '.join(dropped_for_inconsistency)}.]"
        )

    if route == "direct" and _looks_document_related(question):
        route = "rag"
        requires_documents = True
        reason = (
            f"{reason} Overridden: document-related terms were detected, "
            "so the indexed manual must be searched."
        )

    return {
        "complexity": complexity,
        "requires_documents": requires_documents,
        "multiple_information_needs": multiple_information_needs,
        "multiple_search_perspectives": multiple_search_perspectives,
        "has_metadata_constraints": has_metadata_constraints,
        "is_vague_or_conversational": is_vague_or_conversational,
        "needs_clarification": needs_clarification,
        "route": route,
        "techniques": techniques,
        "reason": reason,
    }


def classify_route(question: str) -> dict:
    """
    The main entry point. Always returns a valid, fully-populated
    decision dict (see module docstring) -- never raises, falling back
    to _heuristic_classify() on any LLM/parse failure.
    """
    raw_text = ""
    try:
        result = generation.call_completion(
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ],
            max_tokens=config.ROUTER_MAX_NEW_TOKENS,
            temperature=config.ROUTER_TEMPERATURE,
            stage="router",
            model=config.ROUTER_MODEL,
            timeout=60,
        )
        raw_text = result["text"]
        decision = _extract_json(raw_text)
    except requests.exceptions.ConnectionError as exc:
        return _heuristic_classify(question, f"connection error: {exc}")
    except RuntimeError as exc:  # rate limit, overload, or other API rejection
        return _heuristic_classify(question, f"API error: {str(exc)[:240]}")
    except requests.exceptions.RequestException as exc:
        detail = f"request error: {exc}"
        if getattr(exc, "response", None) is not None:
            detail += f"; server response: {exc.response.text[:240]!r}"
        return _heuristic_classify(question, detail)
    except (ValueError, json.JSONDecodeError, KeyError, IndexError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if raw_text:
            detail += f"; raw response: {raw_text[:240]!r}"
        return _heuristic_classify(question, detail)

    return _validate(decision, question)
