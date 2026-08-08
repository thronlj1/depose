# Trilingual Service Routing End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure `depose` and `service-router-test` for the same OpenAI-compatible Responses API, implement the missing LLM semantic service resolver, and verify the complete flow in English, Chinese, and Arabic.

**Architecture:** `depose` keeps its FastAPI five-intent route and gains a validated shared LLM configuration contract. `service-router-test` keeps its deterministic prepass and adds a dependency-injected Responses API semantic resolver. A Python end-to-end runner calls `depose /route` and invokes the resolver module only for `service_submission`.

**Tech Stack:** Python 3.11, FastAPI, OpenAI Python SDK Responses API, `python-dotenv`, standard-library `unittest`, Docker Compose for `depose`.

---

### Task 1: Align `depose` LLM Configuration

**Files:**
- Modify: `tests/test_submission_router.py`
- Modify: `runtime/app.py`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`

- [ ] Add tests requiring `OPENAI_BASE_URL`, `CLAW_LLM_MODEL`,
  `OPENAI_WIRE_API`, `OPENAI_MAX_RETRIES`, and `LLM_EVAL_CONCURRENCY` to be
  wired without exposing the key.
- [ ] Run the focused tests and verify they fail because the variables are not
  read or passed to the SDK.
- [ ] Add typed environment parsing, require `responses`, and construct
  `OpenAI(api_key=..., base_url=..., timeout=..., max_retries=...)`.
- [ ] Pass the complete configuration through Docker Compose and expose only
  non-secret settings from `/health`.
- [ ] Run the focused tests and full `depose` unit suite.

### Task 2: Implement the Semantic Service Resolver

**Files:**
- Create: `/Users/thron/Documents/odt/service-router-test/service_resolver/__init__.py`
- Create: `/Users/thron/Documents/odt/service-router-test/service_resolver/resolver.py`
- Create: `/Users/thron/Documents/odt/service-router-test/service_resolver/config.py`
- Create: `/Users/thron/Documents/odt/service-router-test/tests/test_llm_resolver.py`
- Create: `/Users/thron/Documents/odt/service-router-test/requirements.txt`

- [ ] Write tests for deterministic bypass, strict JSON Schema output,
  semantic 203/204 output, ambiguous and unsupported output, malformed output,
  unknown catalog services, and API errors.
- [ ] Run the tests and verify failure because `service_resolver` does not
  exist.
- [ ] Implement environment loading and validation for the seven shared
  variables.
- [ ] Implement `ServiceResolver.resolve(text, language)` so deterministic
  results return without an LLM call and semantic results call the Responses
  API using the Skill, catalog, and strict output contract.
- [ ] Run focused and complete resolver tests.

### Task 3: Add Live Trilingual Resolver Evaluation

**Files:**
- Modify: `/Users/thron/Documents/odt/service-router-test/scripts/evaluate_dataset.py`
- Modify: `/Users/thron/Documents/odt/service-router-test/tests/test_evaluator.py`
- Modify: `/Users/thron/Documents/odt/service-router-test/README.md`

- [ ] Add failing tests requiring an LLM evaluation mode for the 36 semantic
  cases and configurable concurrency.
- [ ] Run the evaluator tests and verify failure for the missing live mode.
- [ ] Add `--live-llm`, load `.env`, evaluate semantic records through
  `ServiceResolver`, and report per-language and overall accuracy without
  printing secrets.
- [ ] Run offline tests and a limited live case in each language before the
  complete 36-case evaluation.

### Task 4: Add the End-to-End Runner

**Files:**
- Create: `scripts/run_service_routing_e2e.py`
- Create: `tests/test_service_routing_e2e.py`
- Create: `tests/data/service_routing_e2e.jsonl`
- Modify: `README.md`
- Modify: `DEPLOYMENT.md`

- [ ] Add failing tests for non-submission short-circuiting and
  service-submission delegation to an injected resolver.
- [ ] Add a balanced smoke dataset containing English, Chinese, and Arabic
  examples for 203, 204, ambiguous, unsupported, and non-submission paths.
- [ ] Implement an injectable runner plus a CLI that calls `depose /route` and
  invokes the sibling resolver module only when required.
- [ ] Run unit tests, then execute the live smoke dataset against the running
  Docker service and configured gateway.

### Task 5: Final Verification

**Files:**
- Modify: `SHA256SUMS`
- Modify: `MANIFEST.json`

- [ ] Run both complete unit-test suites.
- [ ] Run Python compilation checks for all new runtime and test scripts.
- [ ] Rebuild and recreate the `depose` container with the shared `.env`.
- [ ] Verify `/health`, `/predict`, and `/route` against the configured gateway.
- [ ] Run all 36 semantic resolver cases and the complete trilingual end-to-end
  smoke dataset.
- [ ] Update delivery metadata and checksums without including either `.env`.
