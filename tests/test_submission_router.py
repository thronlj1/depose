import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace


class FakeResponsesAPI:
    def __init__(self, output=None, output_text="", error=None):
        self.output = output or []
        self.output_text = output_text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output=self.output, output_text=self.output_text)


class FakeOpenAIClient:
    def __init__(self, output=None, output_text="", error=None):
        self.responses = FakeResponsesAPI(
            output=output, output_text=output_text, error=error
        )


def function_call(arguments, name="classify_service_submission"):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
    )


class SubmissionDetectorTests(unittest.TestCase):
    def make_detector(self, output_text="", error=None):
        from runtime.submission_router import SubmissionDetector

        client = FakeOpenAIClient(output_text=output_text, error=error)
        detector = SubmissionDetector(
            client=client,
            model="gpt-5.6",
            instructions="Test submission instructions",
        )
        return detector, client

    def test_returns_service_submission_from_strict_json_schema(self):
        detector, client = self.make_detector(
            output_text=json.dumps(
                {
                    "decision": "service_submission",
                    "reason": "The user explicitly asks to start service 203.",
                }
            )
        )

        decision = detector.detect("I want to apply for service 203")

        self.assertEqual(decision.decision, "service_submission")
        self.assertIn("start service 203", decision.reason)
        request = client.responses.calls[0]
        self.assertEqual(request["model"], "gpt-5.6")
        self.assertEqual(request["input"][0]["content"], "I want to apply for service 203")
        self.assertFalse(request["store"])
        self.assertNotIn("tools", request)
        output_format = request["text"]["format"]
        self.assertEqual("json_schema", output_format["type"])
        self.assertEqual("classify_service_submission", output_format["name"])
        self.assertTrue(output_format["strict"])
        self.assertFalse(output_format["schema"]["additionalProperties"])
        self.assertEqual(
            output_format["schema"]["properties"]["decision"]["enum"],
            ["service_submission", "delegate_to_existing_model"],
        )

    def test_returns_delegation_decision(self):
        detector, _ = self.make_detector(
            output_text=json.dumps(
                {
                    "decision": "delegate_to_existing_model",
                    "reason": "The user asks about required documents.",
                }
            )
        )

        decision = detector.detect("What documents are required for service 203?")

        self.assertEqual(decision.decision, "delegate_to_existing_model")

    def test_rejects_missing_structured_output(self):
        from runtime.submission_router import SubmissionDetectionError

        detector, _ = self.make_detector(output_text="")

        with self.assertRaisesRegex(SubmissionDetectionError, "structured output"):
            detector.detect("Start service 203")

    def test_rejects_malformed_structured_output(self):
        from runtime.submission_router import SubmissionDetectionError

        detector, _ = self.make_detector(output_text="not-json")

        with self.assertRaisesRegex(SubmissionDetectionError, "invalid JSON"):
            detector.detect("Start service 203")

    def test_rejects_unknown_decision(self):
        from runtime.submission_router import SubmissionDetectionError

        detector, _ = self.make_detector(
            output_text=json.dumps({"decision": "maybe", "reason": "unclear"})
        )

        with self.assertRaisesRegex(SubmissionDetectionError, "unsupported decision"):
            detector.detect("Maybe help me with a service")

    def test_wraps_openai_errors_without_fallback(self):
        from runtime.submission_router import SubmissionDetectionError

        detector, _ = self.make_detector(error=RuntimeError("network down"))

        with self.assertRaisesRegex(SubmissionDetectionError, "OpenAI request failed"):
            detector.detect("Start service 203")

    def test_loads_skill_instructions_from_file(self):
        from runtime.submission_router import SubmissionDetector

        with tempfile.TemporaryDirectory() as directory:
            skill_path = Path(directory) / "detect-service-submission.md"
            skill_path.write_text(
                "Delegate requirements questions. Detect explicit submissions.",
                encoding="utf-8",
            )
            client = FakeOpenAIClient(
                output_text=json.dumps(
                    {
                        "decision": "delegate_to_existing_model",
                        "reason": "Informational request.",
                    }
                )
            )
            detector = SubmissionDetector.from_skill(
                client=client,
                model="gpt-5.6",
                skill_path=skill_path,
            )

            detector.detect("What is service 203?")

        self.assertIn(
            "Delegate requirements questions",
            client.responses.calls[0]["instructions"],
        )

    def test_rejects_blank_skill_file(self):
        from runtime.submission_router import (
            SubmissionConfigurationError,
            SubmissionDetector,
        )

        with tempfile.TemporaryDirectory() as directory:
            skill_path = Path(directory) / "blank.md"
            skill_path.write_text("  \n", encoding="utf-8")

            with self.assertRaisesRegex(SubmissionConfigurationError, "blank"):
                SubmissionDetector.from_skill(
                    client=FakeOpenAIClient(),
                    model="gpt-5.6",
                    skill_path=skill_path,
                )


class UnifiedIntentRouterTests(unittest.TestCase):
    def test_submission_bypasses_local_model(self):
        from runtime.submission_router import SubmissionDecision, UnifiedIntentRouter

        class Detector:
            model = "gpt-5.6"

            def detect(self, text):
                return SubmissionDecision(
                    decision="service_submission",
                    reason="Explicit request to submit an application.",
                )

        local_calls = []

        def local_predictor(text):
            local_calls.append(text)
            raise AssertionError("local model must not run for submissions")

        router = UnifiedIntentRouter(Detector(), local_predictor)

        result = router.route("Please submit service 203")

        self.assertEqual(result["intent"], "service_submission")
        self.assertEqual(result["route_source"], "llm_submission_detector")
        self.assertIsNone(result["score"])
        self.assertEqual(result["predictions"], [])
        self.assertEqual(local_calls, [])

    def test_non_submission_delegates_to_local_model_once(self):
        from runtime.submission_router import SubmissionDecision, UnifiedIntentRouter

        class Detector:
            model = "gpt-5.6"

            def detect(self, text):
                return SubmissionDecision(
                    decision="delegate_to_existing_model",
                    reason="This is an informational question.",
                )

        local_calls = []

        def local_predictor(text):
            local_calls.append(text)
            return {
                "intent": "knowledge_process",
                "score": 0.91,
                "predictions": [
                    {"intent": "knowledge_process", "score": 0.91}
                ],
                "model_version": "v1.16.0",
            }

        router = UnifiedIntentRouter(Detector(), local_predictor)

        result = router.route("What documents are required for service 203?")

        self.assertEqual(result["intent"], "knowledge_process")
        self.assertEqual(result["route_source"], "local_intent_model")
        self.assertEqual(local_calls, ["What documents are required for service 203?"])
        self.assertEqual(result["model_version"], "v1.16.0")


class AppWiringTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.app_source = (self.root / "runtime" / "app.py").read_text(
            encoding="utf-8"
        )

    def test_app_exposes_unified_route_without_removing_predict(self):
        self.assertIn('@app.post("/predict")', self.app_source)
        self.assertIn('@app.post("/route")', self.app_source)
        self.assertIn("UnifiedIntentRouter", self.app_source)

    def test_app_fails_closed_when_llm_is_unavailable(self):
        self.assertIn("OPENAI_API_KEY", self.app_source)
        self.assertIn("status_code=503", self.app_source)
        self.assertIn("status_code=502", self.app_source)
        self.assertIn("SubmissionDetectionError", self.app_source)

    def test_container_packages_router_skill_and_openai_dependency(self):
        dockerfile = (self.root / "Dockerfile").read_text(encoding="utf-8")
        requirements = (self.root / "requirements.txt").read_text(encoding="utf-8")
        compose = (self.root / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("runtime/submission_router.py", dockerfile)
        self.assertIn("runtime/skills", dockerfile)
        self.assertIn("openai", requirements)
        self.assertIn("OPENAI_API_KEY", compose)
        self.assertIn("OPENAI_BASE_URL", compose)
        self.assertIn("CLAW_LLM_MODEL", compose)
        self.assertIn("OPENAI_WIRE_API", compose)
        self.assertIn("OPENAI_MAX_RETRIES", compose)
        self.assertIn("LLM_EVAL_CONCURRENCY", compose)


class LlmConfigurationTests(unittest.TestCase):
    def test_shared_environment_contract_is_parsed(self):
        from runtime.llm_config import load_llm_config

        environment = {
            "OPENAI_BASE_URL": "http://gateway.example:8080",
            "OPENAI_API_KEY": "test-key",
            "CLAW_LLM_MODEL": "gpt-5.6-sol",
            "OPENAI_WIRE_API": "responses",
            "OPENAI_TIMEOUT_SECONDS": "60",
            "OPENAI_MAX_RETRIES": "2",
            "LLM_EVAL_CONCURRENCY": "3",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = load_llm_config()

        self.assertEqual("http://gateway.example:8080", config.base_url)
        self.assertEqual("test-key", config.api_key)
        self.assertEqual("gpt-5.6-sol", config.model)
        self.assertEqual("responses", config.wire_api)
        self.assertEqual(60.0, config.timeout_seconds)
        self.assertEqual(2, config.max_retries)
        self.assertEqual(3, config.eval_concurrency)

    def test_non_responses_wire_api_is_rejected(self):
        from runtime.llm_config import LlmConfigurationError, load_llm_config

        with patch.dict(
            os.environ,
            {"OPENAI_WIRE_API": "chat_completions"},
            clear=True,
        ):
            with self.assertRaises(LlmConfigurationError):
                load_llm_config()

    def test_config_can_load_an_explicit_env_file_for_cli_evaluation(self):
        from runtime.llm_config import load_llm_config

        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=file-key\n"
                "CLAW_LLM_MODEL=gpt-5.6-sol\n"
                "OPENAI_WIRE_API=responses\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                config = load_llm_config(
                    load_env_file=True,
                    env_path=env_path,
                )

        self.assertEqual("file-key", config.api_key)
        self.assertEqual("gpt-5.6-sol", config.model)

    def test_client_receives_gateway_timeout_and_retry_settings(self):
        from runtime.llm_config import LlmConfig, build_openai_client

        captured = {}

        def client_factory(**kwargs):
            captured.update(kwargs)
            return object()

        config = LlmConfig(
            base_url="http://gateway.example:8080",
            api_key="test-key",
            model="gpt-5.6-sol",
            wire_api="responses",
            timeout_seconds=60.0,
            max_retries=2,
            eval_concurrency=1,
        )
        build_openai_client(config, client_factory=client_factory)

        self.assertEqual("http://gateway.example:8080", captured["base_url"])
        self.assertEqual("test-key", captured["api_key"])
        self.assertEqual(60.0, captured["timeout"])
        self.assertEqual(2, captured["max_retries"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset_path = (
            Path(__file__).resolve().parent / "data" / "submission_detection.jsonl"
        )

    def load_cases(self):
        return [
            json.loads(line)
            for line in self.dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_dataset_is_balanced_multilingual_and_large_enough(self):
        cases = self.load_cases()
        labels = [case["label"] for case in cases]
        languages = {case["language"] for case in cases}

        self.assertGreaterEqual(len(cases), 60)
        self.assertEqual(
            set(labels), {"service_submission", "non_submission"}
        )
        self.assertLessEqual(
            abs(labels.count("service_submission") - labels.count("non_submission")),
            2,
        )
        self.assertTrue({"en", "zh", "ar"}.issubset(languages))

    def test_dataset_has_unique_nonblank_text_and_required_scenarios(self):
        cases = self.load_cases()
        texts = [case["text"].strip() for case in cases]
        categories = {case["category"] for case in cases}

        self.assertTrue(all(texts))
        self.assertEqual(len(texts), len(set(texts)))
        self.assertTrue(
            {
                "explicit_submission",
                "attachment_submission",
                "requirements",
                "status_query",
                "analytics",
                "hypothetical",
            }.issubset(categories)
        )


class EvaluationMetricsTests(unittest.TestCase):
    def test_binary_metrics_count_submission_as_positive(self):
        from scripts.evaluate_submission_detection import calculate_metrics

        metrics = calculate_metrics(
            expected=[True, True, False, False],
            predicted=[True, False, True, False],
        )

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)


class EnvironmentConfigurationTests(unittest.TestCase):
    def test_gpt_configuration_is_available_in_ignored_env_file(self):
        root = Path(__file__).resolve().parents[1]
        env_text = (root / ".env").read_text(encoding="utf-8")
        gitignore = (root / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("OPENAI_BASE_URL=", env_text)
        self.assertIn("OPENAI_API_KEY=", env_text)
        self.assertIn("CLAW_LLM_MODEL=gpt-5.6-sol", env_text)
        self.assertIn("OPENAI_WIRE_API=responses", env_text)
        self.assertIn("OPENAI_TIMEOUT_SECONDS=60", env_text)
        self.assertIn("OPENAI_MAX_RETRIES=2", env_text)
        self.assertIn("LLM_EVAL_CONCURRENCY=1", env_text)
        self.assertIn(".env", gitignore.splitlines())


if __name__ == "__main__":
    unittest.main()
