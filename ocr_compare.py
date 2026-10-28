"""
Bake-off of Gemini OCR models (or DPI / prompt tweaks) on Arabic PDFs.

    python ocr_compare.py data/manual.pdf --pages 8 12 21 28
    python ocr_compare.py data/manual.pdf --models gemini-3.5-flash-lite gemini-3.1-flash-lite
    python ocr_compare.py data/manual.pdf --pages 8 12 --gt ground_truth

Per model and page it saves the raw output to
    ocr_out/<pdf-name>/<model>/page_008.md
and prints a summary table: seconds per page, and (when a ground-truth file
exists as <gt dir>/page_008.md or .txt) character error rate, word error rate,
order-insensitive word F1 and an RTL-order warning.

Ground truth: type or hand-correct 2-3 pages by looking at the page IMAGE,
not at one candidate's output (correcting a candidate favours that candidate).

Scoring is on normalized text: Markdown syntax is stripped, tashkeel/tatweel
removed, alef variants unified, Arabic-Indic digits converted, punctuation and
bare list numbers (1, 2, ...) ignored. So it measures word accuracy and word
order, not formatting or where the step number sits in the line. Lower CER/WER is better; higher F1 is better.
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter

import ocr_backends as ob

# ---------------------------------------------------------------------------
# Normalization + metrics
# ---------------------------------------------------------------------------

_TASHKEEL = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # diacritics + tatweel
_STEP_MARK = re.compile(r"(?<!\S)\d{1,2}(?!\S)")  # standalone 1-2 digit token = list numbering
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


# Running page header/footer of these manuals. Ground truth leaves them out, and
# the structured Gemini prompt drops them, so they are removed from every
# candidate's output before scoring; otherwise engines that keep them are
# penalized for text nobody needs. Disable with --keep-furniture.
STRIP_FURNITURE = True
_FURNITURE = re.compile(
    r"(النسخة|الإصدار|الاصدار|مراجعة|دليل إجراءات|دليل اجراءات|اسم الاجراء|اسم الإجراء|"
    r"Internal Use|Housing Bank|بنك الإسكان|تطوير وهندسة العمليات|\b\d{1,3}\s*/\s*\d{1,3}\b)",
    re.IGNORECASE)


def drop_furniture(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not _FURNITURE.search(l))


def normalize(text: str, furniture: bool = False) -> str:
    if furniture and STRIP_FURNITURE:
        text = drop_furniture(ob.markdown_to_plain(text))
    text = unicodedata.normalize("NFKC", ob.markdown_to_plain(text))
    text = _TASHKEEL.sub("", text)
    text = re.sub("[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي").translate(_AR_DIGITS)
    text = re.sub(r"[^\w\s]", " ", text)          # punctuation, dashes, quotes: ignore
    text = _STEP_MARK.sub(" ", text)                # bare list numbers (1, 2, ... 11)
    return re.sub(r"\s+", " ", text).strip()


def _lev(a, b) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    prev = list(range(len(b) + 1))  # pure-Python fallback (slow on long pages)
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    return _lev(hyp, ref) / max(len(ref), 1)


def wer(hyp: str, ref: str) -> float:
    return _lev(hyp.split(), ref.split()) / max(len(ref.split()), 1)


def word_f1(hyp: str, ref: str) -> float:
    h, r = Counter(hyp.split()), Counter(ref.split())
    overlap = sum((h & r).values())
    if not overlap:
        return 0.0
    p, rc = overlap / sum(h.values()), overlap / sum(r.values())
    return 2 * p * rc / (p + rc)


def reverse_words_per_line(text: str) -> str:
    return "\n".join(" ".join(reversed(l.split())) for l in text.splitlines())


def rtl_suspect(raw_hyp: str, ref_norm: str) -> bool:
    """True if reversing word order per line matches the reference much better,
    which means the engine returned visual (left-to-right) order."""
    a = cer(normalize(raw_hyp, furniture=True), ref_norm)
    b = cer(normalize(reverse_words_per_line(ob.markdown_to_plain(raw_hyp)), furniture=True), ref_norm)
    return a > 0.3 and b < 0.7 * a


# Same idea as ingestion.estimate_text_corruption: common function words vs the
# malformed forms the corrupted font mapping produces. Needs no ground truth.
_PROBES = [(" من ", " ن "), (" على ", " عل "), (" التي ", " التم ")]


def probe_corruption(text: str) -> float:
    collapsed = re.sub(r"\s+", " ", text)
    padded = " " + collapsed + " "
    good = sum(padded.count(g) for g, _ in _PROBES)
    bad = sum(padded.count(b) for _, b in _PROBES)
    return bad / (good + bad) if good + bad else float("nan")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_gt(gt_dir: str, page_no: int):
    for ext in (".md", ".txt"):
        p = os.path.join(gt_dir, f"page_{page_no:03d}{ext}")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
    return None


def fmt(x, pct=False):
    if x is None or x != x:
        return "-"
    return f"{x * 100:.1f}%" if pct else f"{x:.1f}"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--pages", type=int, nargs="+", default=[8, 12, 21, 28],
                    help="1-indexed page numbers (default: 8 12 21 28)")
    ap.add_argument("--models", nargs="+", default=[ob.gemini_model()],
                    help="Gemini model ids (default: the one ingestion uses)")
    ap.add_argument("--gt", default="ground_truth", help="folder with page_008.md ...")
    ap.add_argument("--out", default="ocr_out")
    ap.add_argument("--keep-furniture", action="store_true",
                    help="do not strip page header/footer lines from candidate outputs")
    ap.add_argument("--reuse", action="store_true",
                    help="score outputs already saved in ocr_out/ instead of re-running OCR")
    ap.add_argument("--preview", type=int, default=0,
                    help="print the first N lines of each output")
    args = ap.parse_args()
    global STRIP_FURNITURE
    STRIP_FURNITURE = not args.keep_furniture

    stem = os.path.splitext(os.path.basename(args.pdf))[0]
    out_root = os.path.join(args.out, stem)
    gts = {p: load_gt(args.gt, p) for p in args.pages}
    have_gt = [p for p, g in gts.items() if g]
    print(f"PDF: {args.pdf} | pages: {args.pages} | ground truth for: {have_gt or 'none'}")

    summary = {}
    for name in args.models:
        rows, errors = [], []
        os.makedirs(os.path.join(out_root, name), exist_ok=True)
        for p in args.pages:
            t0 = time.time()
            saved = os.path.join(out_root, name, f"page_{p:03d}.md")
            if args.reuse and os.path.exists(saved):
                with open(saved, encoding="utf-8") as f:
                    raw = f.read()
                secs = float("nan")  # not re-timed
                hyp = normalize(raw, furniture=True)
                row = {"page": p, "secs": secs, "chars": len(hyp),
                       "probe_corruption": probe_corruption(raw),
                       "flags": ob.suspicious_words(raw)}
                if gts[p]:
                    ref = normalize(gts[p])
                    row.update(cer=cer(hyp, ref), wer=wer(hyp, ref),
                               f1=word_f1(hyp, ref), rtl=rtl_suspect(raw, ref))
                rows.append(row)
                print(f"  [{name}] page {p}: reused saved output")
                if row["flags"]:
                    print(f"      WARNING suspicious words: {row['flags'][:8]}")
                continue
            try:
                raw = ob.extract(args.pdf, p, model=name)
            except Exception as e:  # keep going: one failing model must not stop the rest
                errors.append(f"p{p}: {type(e).__name__}: {str(e)[:800]}")
                print(f"  [{name}] page {p} FAILED: {errors[-1]}")
                if os.environ.get("OCR_DEBUG"):
                    traceback.print_exc()
                if "GEMINI_API_KEY" in str(e):
                    break  # same error on every page; skip the rest
                continue
            secs = time.time() - t0
            with open(os.path.join(out_root, name, f"page_{p:03d}.md"), "w",
                      encoding="utf-8") as f:
                f.write(raw)
            hyp = normalize(raw, furniture=True)
            row = {"page": p, "secs": secs, "chars": len(hyp),
                   "probe_corruption": probe_corruption(raw),
                       "flags": ob.suspicious_words(raw)}
            if gts[p]:
                ref = normalize(gts[p])
                row.update(cer=cer(hyp, ref), wer=wer(hyp, ref), f1=word_f1(hyp, ref),
                           rtl=rtl_suspect(raw, ref))
            rows.append(row)
            print(f"  [{name}] page {p}: {secs:.1f}s, {len(hyp)} chars")
            if row["flags"]:
                print(f"      WARNING suspicious words: {row['flags'][:8]}")
            if args.preview:
                print("\n".join(raw.splitlines()[:args.preview]))
        summary[name] = {"rows": rows, "errors": errors}

    # ---- table ----
    def avg(rows, key):
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]
        return sum(vals) / len(vals) if vals else None

    print("\n" + "=" * 104)
    print(f"{'model':<24}{'pages':>6}{'s/page':>8}{'CER':>8}{'WER':>8}{'wordF1':>9}"
          f"{'RTL?':>6}{'probe-bad':>11}{'flagged':>9}")
    print("-" * 104)
    for name, s in summary.items():
        rows = s["rows"]
        if not rows:
            print(f"{name:<24}{'0':>6}   FAILED: {s['errors'][0] if s['errors'] else ''}")
            continue
        rtl = "YES" if any(r.get("rtl") for r in rows) else ("no" if have_gt else "-")
        print(f"{name:<24}{len(rows):>6}{fmt(avg(rows, 'secs')):>8}"
              f"{fmt(avg(rows, 'cer'), True):>8}{fmt(avg(rows, 'wer'), True):>8}"
              f"{fmt(avg(rows, 'f1'), True):>9}{rtl:>6}"
              f"{fmt(avg(rows, 'probe_corruption'), True):>11}"
              f"{sum(len(r.get('flags', [])) for r in rows):>9}")
    print("=" * 104)
    print("CER/WER: lower is better. wordF1: higher is better. RTL? YES = engine returned "
          "left-to-right word order.\nprobe-bad: share of sampled function words that "
          "are malformed (no ground truth needed).\nflagged: words mixing Arabic with other scripts "
          "(e.g. Cyrillic look-alikes) - open those pages.")
    if not have_gt:
        print(f"\nNo ground truth found in '{args.gt}/'. Add page_008.md etc. to get "
              "CER/WER/F1.")

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nOutputs + summary.json in {out_root}/")


if __name__ == "__main__":
    main()
