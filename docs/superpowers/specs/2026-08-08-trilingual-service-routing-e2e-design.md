# Trilingual Service Routing End-to-End Design

## Goal

Connect the `depose` five-intent router to the standalone
`service-router-test` resolver as a Python integration flow. Both stages use the
same OpenAI-compatible Responses API configuration, and verification covers
English, Chinese, and Arabic.

## Boundaries

- `depose` remains the top-level intent router. It returns one of the original
  four intents or `service_submission`.
- `service-router-test` runs only when the upstream intent is
  `service_submission`.
- The resolver identifies service 203, service 204, an ambiguous request, or an
  unsupported service. It does not submit a form or call a UMC write API.
- The resolver remains a Python module and CLI test target. No HTTP service is
  added.

## Shared LLM Configuration

Both projects accept the following variables:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `CLAW_LLM_MODEL`
- `OPENAI_WIRE_API`, which must be `responses`
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_MAX_RETRIES`
- `LLM_EVAL_CONCURRENCY`

Secrets remain in ignored `.env` files. Runtime and test output must never print
the API key. The client passes the configured base URL, timeout, and retry count
to the OpenAI Python SDK.

## Resolver Architecture

The resolver first calls the existing deterministic prepass. Explicit service
numbers and exact aliases retain precedence. When the prepass returns `None`, a
new semantic resolver loads `resolve-umc-service/SKILL.md`,
`service_catalog.json`, and `output_schema.json`, then calls the Responses API
with strict JSON Schema output.

The semantic result is validated against catalog values and the output
contract. Invalid JSON, missing structured output, unknown service codes, schema
inconsistencies, gateway failures, and unsupported wire API settings fail
closed with a typed resolver error.

## End-to-End Flow

The integration runner accepts one user message and language:

1. Send the text to `depose` at `POST /route`.
2. If the result is not `service_submission`, return the top-level result and
   do not invoke the service resolver.
3. If it is `service_submission`, invoke the Python resolver with the same text
   and language.
4. Return a combined object containing the top-level route and service
   resolution result.

This mirrors the intended Flowise composition without introducing a second
HTTP deployment.

## Test Coverage

Offline unit tests use fake Responses clients and cover configuration,
structured output parsing, deterministic bypass, semantic resolution, failure
behavior, and top-level delegation.

Live evaluation covers English, Chinese, and Arabic for:

- service 203 submissions;
- service 204 submissions;
- ambiguous publication requests;
- unsupported service requests;
- non-submission requests that must stop after `depose`.

The existing 75-case resolver dataset remains the source of truth. Its 36
semantic-stage cases are evaluated through the LLM, with configurable
concurrency. End-to-end smoke cases are separate and intentionally small so
they can run before a demonstration.

## Acceptance Criteria

- Both projects use the seven-variable configuration contract.
- `depose` can reach the configured Responses API and preserve `/predict`.
- The service resolver performs real LLM semantic resolution after the
  deterministic prepass.
- Offline tests pass in both projects.
- Live internal checks succeed against the configured gateway.
- End-to-end English, Chinese, and Arabic cases reach the expected top-level
  and service routes.
