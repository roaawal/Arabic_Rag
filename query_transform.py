"""
Query understanding & transformation techniques for the advanced_rag
route. Each function does ONE technique and is meant to be called only
when the router actually selected it (rewrite_query/multi_query/
decompose_query/hyde_answer -- gated on router.classify_route()'s
`techniques` list) or, for self_query, when the router's
`has_metadata_constraints` flag is set (self_query is a metadata filter,
not one of the four selectable techniques). advanced_rag.py is the only
caller, and it skips whatever wasn't selected/flagged.
"""
import json
import re

import requests

import config
import generation
import rag


def _call_json_llm(system_prompt: str, user_prompt: str, max_tokens: int, fallback, stage: str,
                    model: str = None):
    """Shared helper: call the LLM expecting JSON back, fall back to
    `fallback` (a plain value, not a callable) on any failure so a
    technique degrading gracefully never breaks the whole pipeline."""
    try:
        result = generation.call_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
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
# Rewriting
# ---------------------------------------------------------------------------
def rewrite_query(question: str) -> str:
    """
    Turns vague/conversational wording into a clear, standalone search
    query (resolves implicit references, drops filler, keeps it in the
    same language). Falls back to the original question untouched on any
    failure -- a failed rewrite is a no-op, not an error.
    """
    arabic = generation.is_arabic(question)
    system = (
        "أعد صياغة سؤال المستخدم ليصبح سؤالاً واضحاً ومستقلاً بذاته صالحاً للبحث، "
        "بنفس اللغة، دون إضافة معلومات جديدة أو حذف القصد الأصلي. أعد الصياغة فقط، بلا شرح."
        if arabic else
        "Rewrite the user's question into a clear, standalone, search-ready "
        "question in the same language, resolving vague references, without "
        "adding new information or changing the original intent. Return only "
        "the rewritten question, no explanation."
    )
    try:
        result = generation.call_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            max_tokens=150,
            temperature=0.2,
            stage="rewriter",
            model=config.REWRITER_MODEL,
            timeout=60,
        )
        rewritten = result["text"]
        return rewritten if rewritten else question
    except Exception:
        return question


# ---------------------------------------------------------------------------
# Multi-Query
# ---------------------------------------------------------------------------
def multi_query(question: str, n: int = None) -> list:
    """
    Generates `n` differently-phrased variants of the same information
    need (different angles/synonyms/specificity levels), for broader
    retrieval recall. Always includes the original question in the
    returned list. Falls back to just [question] on any failure.
    """
    n = n or config.MULTI_QUERY_N
    arabic = generation.is_arabic(question)
    system = (
        f"أنشئ {n} صياغات مختلفة لنفس السؤال بزوايا أو مرادفات مختلفة لتحسين نتائج البحث، "
        f"بنفس اللغة. أعد النتيجة كمصفوفة JSON من النصوص فقط، بلا أي شرح."
        if arabic else
        f"Generate {n} differently-phrased variants of the same question (different "
        f"angles, synonyms, or specificity) to broaden search recall, in the same "
        f"language. Return ONLY a JSON array of strings, no explanation."
    )
    result = _call_json_llm(system, question, max_tokens=300, fallback=[question], stage="multi_query",
                             model=config.MULTI_QUERY_MODEL)
    if isinstance(result, list) and result:
        variants = [str(v).strip() for v in result if str(v).strip()]
        if question not in variants:
            variants.append(question)
        return variants[: n + 1]
    return [question]


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
def decompose_query(question: str, max_subqs: int = None) -> list:
    """
    Splits a question that bundles multiple distinct information needs
    into atomic sub-questions. If the question turns out to be atomic
    already, returns [question] unchanged -- decomposition should never
    invent sub-questions that weren't actually asked.
    """
    max_subqs = max_subqs or config.DECOMPOSITION_MAX_SUBQS
    arabic = generation.is_arabic(question)
    system = (
        f"إذا كان سؤال المستخدم يحتوي على أكثر من احتياج معلوماتي منفصل، قسّمه إلى أسئلة "
        f"فرعية ذرية (بحد أقصى {max_subqs})، كل سؤال يغطي احتياجاً واحداً فقط. إذا كان "
        f"السؤال يغطي احتياجاً واحداً بالفعل، أعده كما هو دون تقسيم. أعد النتيجة كمصفوفة "
        f"JSON من النصوص فقط، بلا أي شرح."
        if arabic else
        f"If the user's question bundles more than one distinct information need, split "
        f"it into atomic sub-questions (at most {max_subqs}), each covering exactly one "
        f"need. If the question already covers a single need, return it unchanged, "
        f"un-split. Return ONLY a JSON array of strings, no explanation."
    )
    result = _call_json_llm(system, question, max_tokens=300, fallback=[question], stage="decomposition",
                             model=config.DECOMPOSITION_MODEL)
    if isinstance(result, list) and result:
        subqs = [str(v).strip() for v in result if str(v).strip()]
        return subqs[:max_subqs] if subqs else [question]
    return [question]


# ---------------------------------------------------------------------------
# HyDE (Hypothetical Document Embeddings)
# ---------------------------------------------------------------------------
def hyde_answer(question: str) -> str:
    """
    Generates a short hypothetical passage that WOULD answer the
    question, as if excerpted from the procedure manual -- used as an
    additional embedding query (its phrasing/vocabulary tends to sit
    closer, in embedding space, to a real matching chunk than the bare
    question does). Falls back to the question itself on failure, which
    just means HyDE degrades into a duplicate of the plain query.
    """
    arabic = generation.is_arabic(question)
    system = (
        "اكتب فقرة قصيرة افتراضية (3-4 جمل) كما لو أنها مقتطفة من دليل إجراءات بنكي، "
        "تجيب على سؤال المستخدم بأسلوب رسمي إجرائي. لا تذكر أنها افتراضية، ولا تضف أي "
        "تنبيه أو شرح -- فقط النص الافتراضي نفسه."
        if arabic else
        "Write a short hypothetical passage (3-4 sentences) as if excerpted from a "
        "bank procedures manual, answering the user's question in a formal "
        "procedural style. Don't mention that it's hypothetical or add any caveat -- "
        "just the hypothetical passage text itself."
    )
    try:
        result = generation.call_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            max_tokens=config.HYDE_MAX_NEW_TOKENS,
            temperature=0.4,
            stage="hyde",
            model=config.HYDE_MODEL,
            timeout=60,
        )
        passage = result["text"]
        return passage if passage else question
    except Exception:
        return question


# ---------------------------------------------------------------------------
# Self-Query (metadata extraction)
# ---------------------------------------------------------------------------
_PAGE_PATTERN = re.compile(r"(?:page|صفحة)\s*(\d{1,3})", re.IGNORECASE)
# A manual named in Arabic ("في دليل الإنذار المركزي") -> source-file prefix.
_MANUAL_PATTERNS = {
    "Central Alarm": re.compile(r"دليل\s+(?:\S+\s+){0,2}?(?:ال)?[إا]نذار"),
    "Central Mail": re.compile(r"دليل\s+(?:\S+\s+){0,2}?(?:ال)?بريد"),
    "Assets": re.compile(r"دليل\s+(?:\S+\s+){0,2}?(?:ال)?(?:موجودات|مستودعات)"),
}


def self_query(question: str) -> tuple:
    """
    Detects metadata constraints (a specific source file, section
    heading, or page number) in the question and returns
    (where_filter, cleaned_question):
      - where_filter: a Chroma `where` dict, or None if no constraint found
      - cleaned_question: the question with the metadata phrase stripped,
        for cleaner downstream retrieval/generation

    Page numbers and filenames are pulled out with plain regex (cheap,
    reliable, no LLM needed). Section names are matched against the
    section headings ACTUALLY present in the index (via
    rag.get_known_metadata_values()), so self_query can never hallucinate
    a filter against a section/source that doesn't exist -- an unmatched
    mention is simply left in the cleaned question as ordinary search
    text instead of becoming a (silently wrong) filter.
    """
    cleaned = question
    where_clauses = []

    page_match = _PAGE_PATTERN.search(question)
    if page_match:
        page_num = int(page_match.group(1))
        where_clauses.append({"page": page_num})
        cleaned = _PAGE_PATTERN.sub("", cleaned).strip()

    pdf_match = re.search(r"[\w\-]+\.pdf", question, re.IGNORECASE)
    if pdf_match:
        try:
            known = rag.get_known_metadata_values()
        except Exception:
            known = {"sources": [], "sections": []}
        mentioned = pdf_match.group(0)
        matched_source = next((s for s in known["sources"] if mentioned.lower() in s.lower()), None)
        if matched_source:
            where_clauses.append({"source": matched_source})
            cleaned = cleaned.replace(mentioned, "").strip()

    # A manual named in Arabic. Only one may match: a question naming two
    # manuals needs both, so it must not be narrowed to either.
    if not pdf_match:
        named = [p for p, pattern in _MANUAL_PATTERNS.items() if pattern.search(question)]
        if len(named) == 1:
            try:
                known_sources = rag.get_known_metadata_values()["sources"]
            except Exception:
                known_sources = []
            matched_source = next((s for s in known_sources if s.startswith(named[0])), None)
            if matched_source:
                where_clauses.append({"source": matched_source})

    # Section-name matching: a crude but safe containment check -- if a
    # meaningful chunk of a KNOWN section heading's words appear in the
    # question, treat it as a match. Cheap, no LLM call, and can't
    # hallucinate against a nonexistent section.
    try:
        known_sections = rag.get_known_metadata_values()["sections"]
    except Exception:
        known_sections = []
    for section in known_sections:
        section_words = [w for w in section.split() if len(w) > 2]
        if section_words and sum(1 for w in section_words if w in question) >= max(2, len(section_words) // 2):
            where_clauses.append({"section": section})
            break

    if not where_clauses:
        return None, question

    where_filter = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}
    return where_filter, (cleaned if cleaned else question)
