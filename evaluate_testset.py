"""
Runs a question set through the pipeline and reports how it did. This is where
the LLM judge belongs: chat.py no longer scores answers while the user waits.

    python evaluate_testset.py --retrieval-only     # local, no API calls
    python evaluate_testset.py --delay 15           # full run, uses the API key

The test set is a JSON list in ground_truth/ (git-ignored, because it quotes the
internal manuals):
    {"n": 1, "category": "...", "question": "...", "reference": "...",
     "expected_route": "direct" or null, "gold": [["Central Alarm", 5]]}
`gold` lists [source-file prefix, page] pairs the answer should draw on. Results
go to ground_truth/eval_results.json after every question, so a run that stops
early keeps what it finished.
"""
import argparse
import json
import os
import sys
import time

import config
import generation
import pipeline
import rag

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTSET = os.path.join(HERE, "ground_truth", "rag_testset.json")
DEFAULT_OUTPUT = os.path.join(HERE, "ground_truth", "eval_results.json")
# Stop a full run after this many failures in a row: the quota is probably gone
# and further calls would only use up what's left.
MAX_CONSECUTIVE_FAILURES = 2


def gold_found(row: dict, chunks: list) -> list:
    got = {(c["source"], c["page"]) for c in chunks}
    return [any(s.startswith(prefix) and p == page for s, p in got) for prefix, page in row["gold"]]


def page_recall(results: list):
    scored = [r for r in results if r.get("gold_found")]
    if not scored:
        return None
    return sum(sum(r["gold_found"]) / len(r["gold_found"]) for r in scored) / len(scored)


def retrieval_only(rows: list) -> list:
    results = []
    for row in rows:
        if not row["gold"]:
            continue
        hits = rag.retrieve(row["question"], top_k=config.TOP_K)
        found = gold_found(row, hits)
        results.append({**row, "gold_found": found,
                        "pages": [[h["source"][:13], h["page"]] for h in hits]})
        print(f"Q{row['n']:>2} {'OK  ' if all(found) else 'MISS'} gold={row['gold']} "
              f"got={sorted({h['page'] for h in hits})}")
    return results


def full_run(rows: list, delay: float, output: str) -> list:
    results, failures = [], 0
    for i, row in enumerate(rows):
        if i:
            time.sleep(delay)
        start = time.perf_counter()
        try:
            rec = pipeline.answer(row["question"], verbose=False, reference=row["reference"])
        except Exception as exc:
            failures += 1
            results.append({**row, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            print(f"Q{row['n']:>2} FAILED: {results[-1]['error']}")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"Stopping after {failures} failures in a row.")
                break
            continue
        failures = 0
        found = gold_found(row, rec.get("retrieved_chunks", []))
        results.append({
            **row,
            "route": rec.get("route"),
            "route_ok": row.get("expected_route") in (None, rec.get("route")),
            "gold_found": found,
            "answer": rec["answer"],
            "sources": [[c["source"], c["page"]] for c in rec.get("retrieved_chunks", [])],
            "generation_model": rec.get("generation_model"),
            "context_relevance": rec.get("context_relevance"),
            "faithfulness": rec.get("faithfulness"),
            "answer_relevance": rec.get("answer_relevance"),
            "correctness": rec.get("correctness"),
            "judge_notes": rec.get("judge_notes"),
            "answer_cost": rec.get("answer_cost"),
            "total_cost": rec.get("total_cost"),
            "answer_seconds": rec.get("answer_seconds"),
        })
        print(f"Q{row['n']:>2} route={rec.get('route'):<13} correctness={rec.get('correctness')} "
              f"gold={found} ${rec.get('total_cost', 0):.4f} {time.perf_counter() - start:.1f}s")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    return results


def summarize(results: list):
    recall = page_recall(results)
    if recall is not None:
        complete = sum(all(r["gold_found"]) for r in results if r.get("gold_found"))
        total = sum(1 for r in results if r.get("gold_found"))
        print(f"\nExpected pages found: {recall:.2f}; all pages found on {complete}/{total} questions")
    answered = [r for r in results if "answer" in r]
    if not answered:
        return
    routed = [r for r in answered if r.get("expected_route")]
    if routed:
        print(f"Expected route taken: {sum(r['route_ok'] for r in routed)}/{len(routed)}")
    scores = [r["correctness"] for r in answered if isinstance(r.get("correctness"), (int, float))]
    if scores:
        print(f"Mean correctness (LLM judge, 0-5): {sum(scores) / len(scores):.2f} on {len(scores)} questions")
    print(f"Answered {len(answered)}/{len(results)}; total cost ${sum(r.get('total_cost') or 0 for r in answered):.4f}")
    fallbacks = sum(1 for r in answered if r.get("generation_model") not in (None, config.GENERATION_MODEL))
    if fallbacks:
        print(f"Answered by the fallback model: {fallbacks}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", default=DEFAULT_TESTSET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--retrieval-only", action="store_true", help="local page check, no API calls")
    parser.add_argument("--delay", type=float, default=15, help="seconds between questions on a full run")
    parser.add_argument("--questions", type=int, nargs="*", help="only these question numbers")
    args = parser.parse_args()

    with open(args.testset, encoding="utf-8") as f:
        rows = json.load(f)
    if args.questions:
        rows = [r for r in rows if r["n"] in args.questions]

    if args.retrieval_only:
        results = retrieval_only(rows)
    else:
        if not config.GEMINI_API_KEY:
            raise SystemExit(generation.missing_key_message())
        results = full_run(rows, args.delay, args.output)
    summarize(results)


if __name__ == "__main__":
    main()
