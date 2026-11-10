"""Benchmark and hallucination evaluation script.

Run with:
    python run_eval.py --limit 5 --no-judge
    python run_eval.py
    python run_eval.py --inject-errors
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

from src.config import EVAL_RESULTS_DIR, ROOT
from src.eval_metrics import (
    answer_rate_answerable,
    compute_confusion_matrix,
    false_accept_rate,
    false_refuse_rate,
    inject_error,
    mean_confidence_by_category,
    mean_detector_scores_by_category,
    mean_latency_by_stage,
    refusal_rate_unanswerable,
)

logger = logging.getLogger("eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline accuracy & hallucination detection.")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json", help="Path to evaluation questions.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate.")
    parser.add_argument("--category", type=str, default=None, help="Evaluate only a specific category.")
    parser.add_argument("--no-judge", action="store_true", help="Disable the factuality judge LLM for faster runs.")
    parser.add_argument("--samples", type=int, default=3, help="Number of SelfCheck stochastic samples.")
    parser.add_argument("--out", type=Path, default=None, help="Custom output JSON path.")
    parser.add_argument("--inject-errors", action="store_true", help="Run error-injection sensitivity test.")
    return parser.parse_args()


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [q for q in data if "id" in q and "question" in q]


def print_summary(records: list[dict], error_stats: dict | None = None):
    print("\n" + "=" * 65)
    print("           RAG EVALUATION & HALLUCINATION SUMMARY")
    print("=" * 65)

    matrix = compute_confusion_matrix(records)
    print("\nDecision Confusion Matrix (Expected vs Actual):")
    print(f"{'Expected':<12} | {'SHOW':<8} | {'WARN':<8} | {'REFUSE':<8}")
    print("-" * 44)
    for exp in ("SHOW", "WARN", "REFUSE"):
        row = matrix.get(exp, {})
        print(f"{exp:<12} | {row.get('SHOW', 0):<8} | {row.get('WARN', 0):<8} | {row.get('REFUSE', 0):<8}")

    ans_rate = answer_rate_answerable(records)
    ref_rate = refusal_rate_unanswerable(records)
    far = false_accept_rate(records)
    frr = false_refuse_rate(records)

    print("\nAccuracy & Reliability Rates:")
    print(f"  • Answer Rate (Answerable)   : {ans_rate * 100:.1f}%")
    print(f"  • Refusal Rate (Unanswerable): {ref_rate * 100:.1f}%")
    print(f"  • False-Accept Rate (SHOW on unanswerable): {far * 100:.1f}%")
    print(f"  • False-Refuse Rate (REFUSE on answerable): {frr * 100:.1f}%")

    confs = mean_confidence_by_category(records)
    print("\nMean Confidence by Category:")
    for cat, val in confs.items():
        print(f"  • {cat:<14}: {val:.3f}")

    dscores = mean_detector_scores_by_category(records)
    print("\nMean Detector Scores:")
    for cat, d_map in dscores.items():
        print(f"  [{cat}]")
        for dname, score in d_map.items():
            print(f"    - {dname:<12}: {score:.3f}")

    latencies = mean_latency_by_stage(records)
    print("\nMean Latency per Stage:")
    for stage, secs in latencies.items():
        print(f"  • {stage:<12}: {secs:.2f}s")

    if error_stats:
        print("\nDetector Error-Injection Sensitivity:")
        print(f"  • Total Tested: {error_stats['total']}")
        print(f"  • Downgraded (SHOW -> WARN/REFUSE): {error_stats['downgraded']} ({error_stats['downgrade_rate'] * 100:.1f}%)")

    print("=" * 65 + "\n")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    questions = load_questions(args.questions)
    if args.category:
        questions = [q for q in questions if q.get("category") == args.category]
    if args.limit:
        questions = questions[: args.limit]

    print(f"Evaluating {len(questions)} questions...")

    # Load pipeline once
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline()

    records: list[dict] = []
    error_tests: list[dict] = []

    for item in questions:
        qid = item["id"]
        qtext = item["question"]
        cat = item.get("category", "unknown")
        expected = item.get("expected_decision", "SHOW")

        output = pipeline.ask(
            qtext,
            use_judge=not args.no_judge,
            num_samples=args.samples,
        )

        detector_scores = {}
        if output.final.report:
            for r in output.final.report.results:
                detector_scores[r.name] = r.score if not r.skipped else None

        record = {
            "id": qid,
            "question": qtext,
            "category": cat,
            "expected_decision": expected,
            "actual_decision": output.final.decision.value,
            "confidence": output.final.confidence,
            "answer": output.final.answer,
            "detector_scores": detector_scores,
            "timings": output.timings,
            "refused_early": output.refused_early,
        }
        records.append(record)

        # Error injection sensitivity test on answerable questions with SHOW
        if args.inject_errors and cat == "answerable" and output.final.decision.value == "SHOW":
            original_answer = output.final.generated.answer
            tampered_answer = inject_error(original_answer)

            # Re-run detection on the tampered answer
            from dataclasses import replace
            tampered_gen = replace(output.final.generated, answer=tampered_answer)
            tampered_final = pipeline._detector.run(tampered_gen, skip_judge=not (not args.no_judge))

            downgraded = tampered_final.decision.value in ("WARN", "REFUSE")
            error_tests.append({
                "id": qid,
                "original_decision": output.final.decision.value,
                "tampered_decision": tampered_final.decision.value,
                "downgraded": downgraded,
            })

    error_summary = None
    if error_tests:
        downgraded_count = sum(1 for e in error_tests if e["downgraded"])
        error_summary = {
            "total": len(error_tests),
            "downgraded": downgraded_count,
            "downgrade_rate": downgraded_count / len(error_tests),
        }

    # Save output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EVAL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / f"eval_{timestamp}.json")

    results_payload = {
        "timestamp": timestamp,
        "config": {
            "questions_count": len(questions),
            "no_judge": args.no_judge,
            "samples": args.samples,
        },
        "records": records,
        "error_injection": error_summary,
    }
    out_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"Results written to: {out_path}")

    print_summary(records, error_summary)
    sys.exit(0)


if __name__ == "__main__":
    main()

