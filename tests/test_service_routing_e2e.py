import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests" / "data" / "service_routing_e2e.jsonl"


class RoutingFlowTests(unittest.TestCase):
    def test_non_submission_stops_before_service_resolver(self):
        from scripts.run_service_routing_e2e import run_routing_flow

        class IntentRouter:
            def route(self, text):
                return {"intent": "knowledge_process", "route_source": "local"}

        class Resolver:
            def resolve(self, text, *, language):
                raise AssertionError("resolver must not run for non-submission")

        result = run_routing_flow(
            "What documents are required?",
            "en",
            intent_router=IntentRouter(),
            resolver=Resolver(),
        )

        self.assertEqual("knowledge_process", result["intent_route"]["intent"])
        self.assertIsNone(result["service_resolution"])

    def test_submission_is_delegated_with_language(self):
        from scripts.run_service_routing_e2e import run_routing_flow

        calls = []

        class IntentRouter:
            def route(self, text):
                return {"intent": "service_submission", "route_source": "llm"}

        class Resolver:
            def resolve(self, text, *, language):
                calls.append((text, language))
                return {
                    "resolution_status": "semantic_match",
                    "service_code": "204",
                    "candidates": [],
                }

        result = run_routing_flow(
            "我要发行已经出版的电子书。",
            "zh",
            intent_router=IntentRouter(),
            resolver=Resolver(),
        )

        self.assertEqual("204", result["service_resolution"]["service_code"])
        self.assertEqual([("我要发行已经出版的电子书。", "zh")], calls)


class EndToEndDatasetTests(unittest.TestCase):
    def test_dataset_covers_every_path_in_english_chinese_and_arabic(self):
        cases = [
            json.loads(line)
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(15, len(cases))
        self.assertEqual({"en", "zh", "ar"}, {case["language"] for case in cases})
        coverage = defaultdict(set)
        for case in cases:
            coverage[case["language"]].add(case["category"])
        expected = {"203", "204", "ambiguous", "unsupported", "non_submission"}
        for language in ("en", "zh", "ar"):
            self.assertEqual(expected, coverage[language])
        self.assertEqual(len(cases), len({case["text"] for case in cases}))
        self.assertEqual(
            {5}, set(Counter(case["language"] for case in cases).values())
        )


if __name__ == "__main__":
    unittest.main()
