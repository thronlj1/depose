import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SUBMISSION_TOOL_NAME = "classify_service_submission"
SUBMISSION_DECISIONS = {
    "service_submission",
    "delegate_to_existing_model",
}

SUBMISSION_OUTPUT_FORMAT = {
    "type": "json_schema",
    "name": SUBMISSION_TOOL_NAME,
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": [
                    "service_submission",
                    "delegate_to_existing_model",
                ],
                "description": (
                    "Use service_submission only for an explicit service application "
                    "action; otherwise delegate to the existing query intent model."
                ),
            },
            "reason": {
                "type": "string",
                "description": "A short explanation grounded in the user's request.",
            },
        },
        "required": ["decision", "reason"],
        "additionalProperties": False,
    },
}


class SubmissionDetectionError(RuntimeError):
    pass


class SubmissionConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SubmissionDecision:
    decision: str
    reason: str


class SubmissionDetector:
    def __init__(self, client, model: str, instructions: str):
        if not instructions.strip():
            raise ValueError("submission detector instructions must not be blank")
        self.client = client
        self.model = model
        self.instructions = instructions

    @classmethod
    def from_skill(cls, client, model: str, skill_path) -> "SubmissionDetector":
        path = Path(skill_path)
        try:
            instructions = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SubmissionConfigurationError(
                f"unable to read submission detector skill: {path}"
            ) from exc
        if not instructions.strip():
            raise SubmissionConfigurationError(
                f"submission detector skill is blank: {path}"
            )
        return cls(client=client, model=model, instructions=instructions)

    def detect(self, text: str) -> SubmissionDecision:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                text={"format": SUBMISSION_OUTPUT_FORMAT},
                store=False,
                input=[{"role": "user", "content": text}],
            )
        except Exception as exc:
            raise SubmissionDetectionError(
                f"OpenAI request failed: {type(exc).__name__}"
            ) from exc

        output_text = getattr(response, "output_text", "")
        if not isinstance(output_text, str) or not output_text.strip():
            raise SubmissionDetectionError(
                "OpenAI response did not include structured output"
            )

        try:
            arguments = json.loads(output_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SubmissionDetectionError(
                "OpenAI structured output contained invalid JSON"
            ) from exc

        decision = arguments.get("decision")
        reason = arguments.get("reason")
        if decision not in SUBMISSION_DECISIONS:
            raise SubmissionDetectionError(
                f"OpenAI function call returned unsupported decision: {decision!r}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise SubmissionDetectionError(
                "OpenAI function call returned a blank decision reason"
            )
        return SubmissionDecision(decision=decision, reason=reason.strip())


class UnifiedIntentRouter:
    def __init__(
        self,
        detector: SubmissionDetector,
        local_predictor: Callable[[str], dict],
    ):
        self.detector = detector
        self.local_predictor = local_predictor

    def route(self, text: str) -> dict:
        decision = self.detector.detect(text)
        if decision.decision == "service_submission":
            return {
                "intent": "service_submission",
                "score": None,
                "predictions": [],
                "model_version": self.detector.model,
                "route_source": "llm_submission_detector",
                "submission_reason": decision.reason,
            }
        if decision.decision != "delegate_to_existing_model":
            raise SubmissionDetectionError(
                f"unsupported routing decision: {decision.decision!r}"
            )

        result = dict(self.local_predictor(text))
        result["route_source"] = "local_intent_model"
        result["submission_reason"] = decision.reason
        return result
