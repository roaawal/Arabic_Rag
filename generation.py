"""
Shared "final LLM call" logic.

Whichever route the router sends a question down (direct, rag,
or advanced_rag), it ends here: one grounded-answer prompt template, one
LLM call function. This is deliberate -- the grounding rules (use only
the supplied context, don't invent facts, say so if the context is
insufficient, cite chunk/page ids) are exactly the kind of thing that
quietly drifts out of sync if every route builds its own prompt. Define
them once, reuse everywhere.

"""
import os
import re
import sys
import time
from contextvars import ContextVar

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import requests

import config

_call_ledger = ContextVar("rag_call_ledger", default=None)


def start_call_tracking():
    """Start a per-question ledger and return its context token."""
    return _call_ledger.set([])


def stop_call_tracking(token):
    """Restore the previous ledger context and return this question's calls."""
    calls = list(_call_ledger.get() or [])
    _call_ledger.reset(token)
    return calls


_warned_unpriced = set()


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost of one call from config.MODEL_PRICING (today's price).
    Output tokens must include thinking tokens (the API bills them as
    output). Unknown models cost 0.0 and are flagged once on stderr."""
    price = config.price_per_million(model)
    if price is None:
        if model not in _warned_unpriced:
            _warned_unpriced.add(model)
            print(f"WARNING: no price for model {model!r} in config.MODEL_PRICING; "
                  f"its cost is counted as $0.", file=sys.stderr)
        return 0.0
    return input_tokens / 1_000_000 * price[0] + output_tokens / 1_000_000 * price[1]


def _record_call(stage: str, model: str, input_tokens: int, output_tokens: int,
                latency_seconds: float, success: bool = True, tokens_estimated: bool = False,
                reasoning_tokens: int = None):
    ledger = _call_ledger.get()
    if ledger is None:
        return
    ledger.append({
        "stage": stage,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": compute_cost(model, input_tokens, output_tokens),
        "latency_seconds": latency_seconds,
        "tokens_estimated": tokens_estimated,
        "success": success,
        "executed": True,
        "timestamp": time.time(),
    })


_RETRY_STATUS = {429, 500, 502, 503, 504}
# Models that rejected reasoning_effort: don't send it to them again.
_effort_rejected = set()


def missing_key_message() -> str:
    return ("GEMINI_API_KEY is not set. Create a key in Google AI Studio "
            "(https://aistudio.google.com/apikey) and set it in this terminal, e.g. "
            "PowerShell:  $env:GEMINI_API_KEY = \"your-key\"")


def call_completion(messages: list, max_tokens: int, temperature: float,
                    stage: str, model: str = None, timeout: int = 180) -> dict:
    """Call the Gemini API (OpenAI-compatible endpoint) and track this LLM
    call: tokens, USD cost (thinking tokens included) and latency, retries
    included."""
    model = model or config.GENERATION_MODEL
    if not config.GEMINI_API_KEY:
        raise RuntimeError(missing_key_message())

    payload = {
        "model": model,
        "messages": messages,
        # max_tokens also covers thinking tokens on Gemini; add headroom so
        # thinking can't leave the visible answer empty/truncated.
        "max_tokens": max_tokens + config.THINKING_TOKEN_HEADROOM,
        "temperature": temperature,
    }
    effort = config.REASONING_EFFORT.get(stage, config.DEFAULT_REASONING_EFFORT)
    if effort and model not in _effort_rejected:
        payload["reasoning_effort"] = effort

    url = f"{config.GEMINI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {config.GEMINI_API_KEY}",
               "Content-Type": "application/json"}

    start = time.perf_counter()
    attempt = 0
    while True:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException:
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_BACKOFF_SECONDS * (2 ** attempt))
                attempt += 1
                continue
            # Record the failed call (0 tokens, success=False) before letting
            # the caller's own error handling/fallback take over.
            _record_call(stage, model, 0, 0, time.perf_counter() - start, success=False)
            raise

        if (response.status_code == 400 and "reasoning_effort" in payload
                and any(w in response.text.lower() for w in ("reasoning", "thinking"))):
            # This model doesn't accept that effort level: drop it and retry.
            _effort_rejected.add(model)
            payload.pop("reasoning_effort")
            continue
        if response.status_code in _RETRY_STATUS and attempt < config.MAX_RETRIES:
            time.sleep(config.RETRY_BACKOFF_SECONDS * (2 ** attempt))
            attempt += 1
            continue
        break

    latency_seconds = time.perf_counter() - start
    if not response.ok:
        detail = response.text.strip()
        _record_call(stage, model, 0, 0, latency_seconds, success=False)
        raise RuntimeError(
            f"The Gemini API rejected the {stage} request with HTTP "
            f"{response.status_code} for model {model!r}. Server response: {detail[:1000]}"
        )

    data = response.json()
    choice = data["choices"][0]
    content = (choice.get("message", {}).get("content") or "").strip()
    if not content:
        print(f"WARNING: empty {stage} response from {model} "
              f"(finish_reason={choice.get('finish_reason')!r}).", file=sys.stderr)

    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

    input_tokens = prompt_tokens or estimate_tokens(
        "\n".join(str(m.get("content", "")) for m in messages)
    )
    output_tokens = completion_tokens or estimate_tokens(content)
    if prompt_tokens is not None and total_tokens is not None:
        # Everything generated beyond the prompt is billed as output,
        # including thinking tokens even if completion_tokens leaves them out.
        output_tokens = max(output_tokens, total_tokens - prompt_tokens)
    tokens_estimated = not (prompt_tokens and completion_tokens)

    _record_call(stage, model, input_tokens, output_tokens, latency_seconds,
                 success=True, tokens_estimated=tokens_estimated,
                 reasoning_tokens=reasoning_tokens)
    return {
        "data": data,
        "text": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency_seconds,
        "tokens_estimated": tokens_estimated,
    }


def is_arabic(text: str) -> bool:
    """True if the text contains Arabic script characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def estimate_tokens(text: str) -> int:
    """
    Rough fallback token estimate when the server doesn't report usage.
    Not exact -- Arabic and English tokenize differently -- but good
    enough for relative cost/size comparisons across approaches.
    """
    return max(1, len(text.split()))


_SOURCE_WORD = r"(?:المصدر|المصادر|المرجع|المراجع|source|sources|reference|references)"
_SOURCE_LINE_RE = re.compile(
    r"^[\s>*_#\-\u2022]*" + _SOURCE_WORD + r"\b\s*[:：،\-]?[^\n]*$", re.IGNORECASE)
_SOURCE_CUE_RE = re.compile(
    r"(صفحة|صفحه|page|chunk|مقتطف|معرّف|معرف|مُعرّف|مُعرِّف|مُعَرِّف|\.pdf|::p\d+::c\d+)",
    re.IGNORECASE)
_ID_PIECE = r"[A-Za-z0-9 _\-\.]+\.pdf::p\d+::c\d+"
_ID_ONLY_LINE_RE = re.compile(
    r"^[\s>*_\-\u2022\[\(]*" + _ID_PIECE + r"(?:[\s\]\)\.،,و]+" + _ID_PIECE + r")*[\s\]\)\.،,]*$")
_INLINE_ID_RE = re.compile(
    r"\s*[\[\(]\s*(?:chunk id\s*[:：]\s*)?[^\[\]\(\)\n]*?\.pdf::p\d+::c\d+[^\[\]\(\)\n]*[\]\)]",
    re.IGNORECASE)


def strip_inline_sources(text: str) -> str:
    """Safety net: remove 'Source: page N, Chunk ID: ...' lines (and bare chunk-id
    lines / bracketed ids) from an answer. Sources are shown in the UI's separate
    Sources section, so they must not appear inside the answer text. A line is
    only dropped when it STARTS with a source word AND carries a page/chunk cue,
    so ordinary sentences that happen to begin with 'المصدر' are kept."""
    if not text:
        return text
    kept = []
    for line in text.splitlines():
        if _SOURCE_LINE_RE.match(line) and _SOURCE_CUE_RE.search(line):
            continue
        if _ID_ONLY_LINE_RE.match(line):
            continue
        kept.append(_INLINE_ID_RE.sub("", line))
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_grounded_prompt(question: str, hits: list) -> str:
    """
    The RAG-route final prompt. Requirements this MUST keep enforcing,
    regardless of which route or which advanced techniques ran upstream:
      - use ONLY the supplied context for document-grounded claims
      - answer the user's actual question
      - never invent facts not supported by the context
      - explicitly say so if the context can't fully support an answer
      - NOT print source/chunk identifiers inside the answer text (the UI shows them
        in a separate Sources section; strip_inline_sources() removes any that slip through)
    """
    context_blocks = []
    for h in hits:
        section_label = f", Section: {h['section']}" if h.get("section") else ""
        context_blocks.append(
            f"[Chunk ID: {h['id']} | Source: {h['source']}, page {h['page']}{section_label}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(No context was retrieved.)"

    if is_arabic(question):
        instruction = """أجب على السؤال باستخدام المعلومات الواردة في المقتطفات أدناه فقط.

قواعد صارمة يجب اتباعها:
- استخدم فقط المعلومات المرفقة في السياق أدناه لأي ادعاء متعلق بالوثيقة، ولا تستخدم معرفة خارجية.
- أجب على السؤال الفعلي الذي طرحه المستخدم.
- لا تختلق أي معلومة غير مدعومة بالسياق.
- إذا كان السياق لا يدعم إجابة كاملة، اذكر ذلك بوضوح صراحةً بدلاً من التخمين.
- أجب باللغة العربية الفصحى فقط من أول كلمة إلى آخر كلمة. ممنوع منعًا باتًا استخدام أي كلمة أو حرف باللغة الإنجليزية.
- تجاهل تمامًا أي معلومات إدارية أو شكلية من رأس أو تذييل الوثيقة (رقم الإصدار، تاريخ الإصدار، "للاستخدام الداخلي").
- إذا كان المحتوى يصف إجراءً أو خطوات، اذكر كل خطوة مرقّمة بالترتيب، مع التفاصيل المهمة (من المسؤول، ما هي النماذج أو السجلات المطلوبة، التوقيت). لا تلخص في جملة واحدة قصيرة.
- السياق أدناه هو المقتطفات المسترجعة فعلياً من الوثائق فقط -- لا تعتبر أي إعادة صياغة للسؤال أو فقرة افتراضية أو استعلامات بحث مولّدة دليلاً على الإطلاق.
- إذا تعارضت المصادر المسترجعة مع بعضها البعض، اذكر هذا التعارض بوضوح في إجابتك بدلاً من اختيار جانب واحد بصمت.
- لا تذكر في إجابتك معرّف المقتطف ولا رقم الصفحة ولا اسم الملف ولا سطر "المصدر"؛ تُعرض المصادر تلقائياً في قسم منفصل أسفل الإجابة."""
    else:
        instruction = """Using ONLY the context excerpts below, give a complete, detailed \
    answer to the question. Treat the context as the complete source of truth: do not use \
    outside knowledge, typical industry practices, or facts remembered from training. Do not \
    introduce a domain term, abbreviation, process, role, or explanation unless it is directly \
    supported by the context. If the context does not fully support an answer, explicitly say \
    there is insufficient information rather than guessing. Do not summarize into a single short \
    sentence -- if the context describes a procedure, list every step in order, numbered, with \
    any relevant detail (who does it, what forms/records are involved, timing). Ignore \
    administrative letterhead/boilerplate (issue dates, revision numbers, "Internal Use" labels). \
    The context below consists only of passages actually retrieved from the documents -- never \
    treat a query rewrite, a hypothetical passage, or a generated search query as evidence. If the \
    retrieved sources disagree with each other, clearly point out the disagreement instead of \
    silently picking one side. Do not mention chunk ids, page numbers, file names, or a "Source" \
    line in the answer; the sources are displayed separately below the answer."""

    return f"""{instruction}

Context:
{context}

Question: {question}

Detailed answer:"""


def build_direct_prompt(question: str) -> str:
    """
    The "direct" route's prompt: no document context at all. Used
    for genuinely general/conversational questions the router decided
    don't need the corpus. Still answers in the question's language and
    stays honest about not consulting any document.
    """
    if is_arabic(question):
        instruction = ("أجب على السؤال التالي بإيجاز ووضوح باللغة العربية الفصحى فقط. "
                        "هذا سؤال عام لا يتطلب الرجوع إلى وثائق الشركة.")
    else:
        instruction = ("Answer the following question directly and concisely. "
                        "This is a general question that does not require consulting "
                        "any company documents.")
    return f"{instruction}\n\nQuestion: {question}\n\nAnswer:"


def call_llm(prompt: str, arabic: bool, system_message: str = None,
             max_tokens: int = None, temperature: float = None,
             stage: str = "final_generation", model: str = None) -> dict:
    """
    Returns {"text": answer, "input_tokens": int, "output_tokens": int,
    "latency_seconds": float}.
    """
    if system_message is None:
        if arabic:
            system_message = ("أنت مساعد دقيق يجيب فقط بناءً على السياق المُعطى عند توفره. "
                               "تجيب دائمًا باللغة العربية الفصحى فقط، دون أي كلمات إنجليزية، "
                               "وتقدّم إجابات كاملة ومفصّلة تعدّد كل خطوة من خطوات أي إجراء بدلاً "
                               "من تلخيصه في جملة واحدة، وتتجاهل أي معلومات شكلية من رأس أو تذييل "
                               "الوثيقة. لا تضف أي مصطلح أو اختصار أو معلومة غير موجودة في السياق، "
                               "ولا تستخدم معرفة عامة أو معرفة سابقة لسد أي فجوة؛ اذكر صراحةً أن "
                               "المعلومات غير كافية عند الحاجة.")
        else:
            system_message = ("You are a meticulous assistant that answers questions "
                               "grounded in any provided context. You always give "
                               "complete, detailed answers -- enumerating every step "
                               "of a procedure rather than summarizing it into one "
                               "sentence -- in the same language the question was "
                               "asked, and you ignore administrative letterhead or "
                               "boilerplate from any source document. Never add a "
                               "domain term, abbreviation, process, role, or fact that "
                               "is not directly supported by the supplied context; do "
                               "not use general knowledge to fill gaps, and explicitly "
                               "say when the context is insufficient.")

    result = call_completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens or config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE if temperature is None else temperature,
        stage=stage,
        model=model or config.GENERATION_MODEL,
    )

    return {
        "text": result["text"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "latency_seconds": result["latency_seconds"],
    }


def connection_error_message() -> str:
    return (
        "Couldn't reach the Gemini API. Check your internet connection or proxy, "
        "and that GEMINI_API_KEY is set and valid."
    )
