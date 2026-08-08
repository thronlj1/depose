#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.submission_router import SubmissionDetectionError, SubmissionDetector
from runtime.llm_config import build_openai_client, load_llm_config


def calculate_metrics(expected, predicted):
    tp = sum(want and got for want, got in zip(expected, predicted))
    tn = sum(not want and not got for want, got in zip(expected, predicted))
    fp = sum(not want and got for want, got in zip(expected, predicted))
    fn = sum(want and not got for want, got in zip(expected, predicted))
    total = len(expected)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_cases(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GPT submission detection without scoring the four local intents."
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "tests" / "data" / "submission_detection.jsonl"),
    )
    parser.add_argument(
        "--skill",
        default=str(
            ROOT / "runtime" / "skills" / "detect-service-submission.md"
        ),
    )
    config = load_llm_config(load_env_file=True)
    parser.add_argument("--model", default=config.model)
    parser.add_argument(
        "--timeout",
        type=float,
        default=config.timeout_seconds,
    )
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_llm_config(load_env_file=True)
    if not config.api_key:
        print(
            "OPENAI_API_KEY is required for live submission evaluation.",
            file=sys.stderr,
        )
        return 2

    cases = load_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    detector = SubmissionDetector.from_skill(
        client=build_openai_client(
            type(config)(
                base_url=config.base_url,
                api_key=config.api_key,
                model=args.model,
                wire_api=config.wire_api,
                timeout_seconds=args.timeout,
                max_retries=config.max_retries,
                eval_concurrency=config.eval_concurrency,
            )
        ),
        model=args.model,
        skill_path=args.skill,
    )

    expected = []
    predicted = []
    mistakes = []
    for index, case in enumerate(cases, start=1):
        try:
            decision = detector.detect(case["text"])
        except SubmissionDetectionError as exc:
            print(f"Case {case['id']} failed: {exc}", file=sys.stderr)
            return 3
        want_submission = case["label"] == "service_submission"
        got_submission = decision.decision == "service_submission"
        expected.append(want_submission)
        predicted.append(got_submission)
        if want_submission != got_submission:
            mistakes.append(
                {
                    "id": case["id"],
                    "expected": case["label"],
                    "predicted": decision.decision,
                    "text": case["text"],
                    "reason": decision.reason,
                }
            )
        print(f"[{index}/{len(cases)}] {case['id']}: {decision.decision}")

    metrics = calculate_metrics(expected, predicted)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if mistakes:
        print("Misclassified cases:")
        print(json.dumps(mistakes, ensure_ascii=False, indent=2))

    if metrics["precision"] < args.min_precision:
        print(
            f"Precision {metrics['precision']:.4f} is below {args.min_precision:.4f}",
            file=sys.stderr,
        )
        return 1
    if metrics["recall"] < args.min_recall:
        print(
            f"Recall {metrics['recall']:.4f} is below {args.min_recall:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
