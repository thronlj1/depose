from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from openai import OpenAI


class LlmConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlmConfig:
    base_url: str | None
    api_key: str
    model: str
    wire_api: str
    timeout_seconds: float
    max_retries: int
    eval_concurrency: int


def _positive_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError as exc:
        raise LlmConfigurationError(f"{name} must be a number") from exc
    if value <= 0:
        raise LlmConfigurationError(f"{name} must be greater than zero")
    return value


def _non_negative_int(name: str, default: str) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError as exc:
        raise LlmConfigurationError(f"{name} must be an integer") from exc
    if value < 0:
        raise LlmConfigurationError(f"{name} must not be negative")
    return value


def _positive_int(name: str, default: str) -> int:
    value = _non_negative_int(name, default)
    if value == 0:
        raise LlmConfigurationError(f"{name} must be greater than zero")
    return value


def load_llm_config(
    *,
    load_env_file: bool = False,
    env_path: str | Path | None = None,
) -> LlmConfig:
    if load_env_file:
        from dotenv import load_dotenv

        path = (
            Path(env_path)
            if env_path is not None
            else Path(__file__).resolve().parents[1] / ".env"
        )
        load_dotenv(path, override=False)

    wire_api = os.getenv("OPENAI_WIRE_API", "responses").strip().lower()
    if wire_api != "responses":
        raise LlmConfigurationError("OPENAI_WIRE_API must be responses")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    return LlmConfig(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        model=os.getenv("CLAW_LLM_MODEL", "gpt-5.6-sol").strip()
        or "gpt-5.6-sol",
        wire_api=wire_api,
        timeout_seconds=_positive_float("OPENAI_TIMEOUT_SECONDS", "60"),
        max_retries=_non_negative_int("OPENAI_MAX_RETRIES", "2"),
        eval_concurrency=_positive_int("LLM_EVAL_CONCURRENCY", "1"),
    )


def build_openai_client(
    config: LlmConfig,
    *,
    client_factory: Callable[..., object] = OpenAI,
):
    return client_factory(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )
