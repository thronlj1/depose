#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "data" / "service_routing_e2e.jsonl"
DEFAULT_SERVICE_ROUTER_ROOT = ROOT.parents[1] / "service-router-test"


class EndToEndRoutingError(RuntimeError):
    pass


class HttpIntentRouter:
    def __init__(self, base_url: str, *, timeout: float = 120.0):
        self.route_url = f"{base_url.rstrip('/')}/route"
        self.timeout = timeout

    def route(self, text: str) -> dict[str, Any]:
        request = urllib.request.Request(
            self.route_url,
            data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise EndToEndRoutingError(
                f"depose /route returned HTTP {exc.code}: {body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EndToEndRoutingError(f"depose /route failed: {exc}") from exc


def run_routing_flow(
    text: str,
    language: str,
    *,
    intent_router,
    resolver,
) -> dict[str, Any]:
    intent_route = intent_router.route(text)
    service_resolution = None
    if intent_route.get("intent") == "service_submission":
        service_resolution = resolver.resolve(text, language=language)
    return {
        "text": text,
        "language": language,
        "intent_route": intent_route,
        "service_resolution": service_resolution,
    }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_cases(cases, *, intent_router, resolver) -> dict[str, Any]:
    results = []
    failures = []
    for case in cases:
        result = run_routing_flow(
            case["text"],
            case["language"],
            intent_router=intent_router,
            resolver=resolver,
        )
        actual_intent = result["intent_route"].get("intent")
        expected_intent = case["expected_top_intent"]
        top_level_ok = (
            actual_intent != "service_submission"
            if expected_intent == "non_submission"
            else actual_intent == expected_intent
        )
        resolution = result["service_resolution"]
        service_ok = True
        if expected_intent == "service_submission":
            service_ok = (
                resolution is not None
                and resolution.get("resolution_status") == case["expected_status"]
                and resolution.get("service_code")
                == case["expected_service_code"]
            )
        elif resolution is not None:
            service_ok = False

        case_result = {
            "id": case["id"],
            "language": case["language"],
            "category": case["category"],
            "passed": top_level_ok and service_ok,
            "actual_intent": actual_intent,
            "actual_status": (
                resolution.get("resolution_status") if resolution else None
            ),
            "actual_service_code": (
                resolution.get("service_code") if resolution else None
            ),
        }
        results.append(case_result)
        if not case_result["passed"]:
            failures.append({**case_result, "text": case["text"]})

    return {
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run depose-to-service-resolver trilingual end-to-end checks."
    )
    parser.add_argument("--depose-url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--service-router-root",
        type=Path,
        default=Path(
            os.getenv("SERVICE_ROUTER_ROOT", str(DEFAULT_SERVICE_ROUTER_ROOT))
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")) * 2,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service_router_root = args.service_router_root.resolve()
    if str(service_router_root) not in sys.path:
        sys.path.insert(0, str(service_router_root))

    from service_resolver import ServiceResolver
    from service_resolver.config import build_openai_client, load_llm_config

    config = load_llm_config()
    if not config.api_key:
        print("OPENAI_API_KEY is required for end-to-end routing", file=sys.stderr)
        return 2
    resolver = ServiceResolver.from_bundle(
        client=build_openai_client(config),
        model=config.model,
        bundle_dir=service_router_root / "resolve-umc-service",
    )
    summary = evaluate_cases(
        load_cases(args.dataset),
        intent_router=HttpIntentRouter(args.depose_url, timeout=args.timeout),
        resolver=resolver,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
