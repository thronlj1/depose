# Five-Intent LLM Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing four-class intent model with a GPT-5.6 submission detector so the public router returns one of five intents while preserving the existing `/predict` contract.

**Architecture:** A pure-Python `SubmissionDetector` calls the OpenAI Responses API with one strict function tool and returns `service_submission` or `delegate_to_existing_model`. A dependency-injected `UnifiedIntentRouter` returns the fifth intent directly for submissions and otherwise delegates to the existing local predictor. FastAPI exposes this composition at `/route`; `/predict` remains unchanged.

**Tech Stack:** Python 3.11, FastAPI, OpenAI Python SDK Responses API, PyTorch/Transformers existing runtime, standard-library `unittest`.

---

### Task 1: Define Detector and Router Behavior with Failing Tests

**Files:**
- Create: `tests/test_submission_router.py`
- Create: `runtime/submission_router.py`

- [ ] **Step 1: Write the failing detector tests**

Cover strict function-call parsing, submission decisions, delegation decisions, malformed output, missing function calls, and API exceptions using an injected fake Responses client.

- [ ] **Step 2: Write the failing unified-router tests**

Assert that submission input never invokes the local model and returns `service_submission`; assert that non-submission input invokes the local predictor exactly once and preserves its four-class result.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_submission_router -v
```

Expected: FAIL because `runtime.submission_router` does not exist.

- [ ] **Step 4: Implement the minimal detector and router**

Create:

```python
class SubmissionDetectionError(RuntimeError):
    pass

class SubmissionDecision:
    decision: str
    reason: str

class SubmissionDetector:
    def detect(self, text: str) -> SubmissionDecision:
        ...

class UnifiedIntentRouter:
    def route(self, text: str) -> dict:
        ...
```

Use one strict function tool named `classify_service_submission` with a `decision` enum of `service_submission` and `delegate_to_existing_model`.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_submission_router -v
```

Expected: all detector and unified-router tests pass.

### Task 2: Add the Submission Detection Skill

**Files:**
- Create: `runtime/skills/detect-service-submission.md`
- Test: `tests/test_submission_router.py`

- [ ] **Step 1: Add a failing prompt-contract test**

Assert the detector loads the configured skill file and sends instructions containing the consultation/submission boundary and multilingual examples.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m unittest tests.test_submission_router.SubmissionDetectorTests.test_loads_skill_instructions -v
```

Expected: FAIL because skill loading is not implemented.

- [ ] **Step 3: Implement skill loading**

Load UTF-8 text from `SUBMISSION_SKILL_PATH`, defaulting to `runtime/skills/detect-service-submission.md` relative to the module. Raise a clear configuration error when the file is missing or blank.

- [ ] **Step 4: Write the skill instructions**

Define `service_submission` as an explicit request to start, continue, fill, upload for, or submit a UMC service application. Delegate informational questions, requirements, eligibility, fees, process duration, status checks, data queries, analytics, and hypothetical questions to the existing model. Bias ambiguous text toward delegation to avoid unsafe false-positive write routing.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_submission_router -v
```

Expected: all tests pass.

### Task 3: Expose the Five-Intent `/route` API

**Files:**
- Modify: `runtime/app.py`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add a failing source-level API wiring test**

Assert `runtime/app.py` defines `/route`, imports the unified router, and maps detector errors to HTTP 502/503 without silently invoking the local model.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m unittest tests.test_submission_router.AppWiringTests -v
```

Expected: FAIL because `/route` is absent.

- [ ] **Step 3: Implement FastAPI wiring**

Initialize `SubmissionDetector` from `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TIMEOUT_SECONDS`, and the skill path. Add `/route` using the existing `PredictRequest`. Preserve `/predict` unchanged. Extend `/health` with non-secret LLM configuration status.

- [ ] **Step 4: Update container packaging**

Copy `runtime/submission_router.py` and `runtime/skills/` into the image, add the OpenAI SDK dependency, and pass `OPENAI_API_KEY`/`OPENAI_MODEL` through Compose.

- [ ] **Step 5: Run tests and syntax checks**

Run:

```bash
python3 -m unittest tests.test_submission_router -v
python3 -m py_compile runtime/app.py runtime/submission_router.py
```

Expected: tests pass and compilation exits 0.

### Task 4: Add Binary Evaluation Data and Live GPT-5.6 Evaluation

**Files:**
- Create: `tests/data/submission_detection.jsonl`
- Create: `scripts/evaluate_submission_detection.py`
- Create: `.env.example`
- Test: `tests/test_submission_router.py`

- [ ] **Step 1: Add failing dataset validation tests**

Require at least 60 balanced cases, only `service_submission`/`non_submission` labels, unique nonblank texts, and English, Chinese, and Arabic coverage. Include adversarial pairs where the same service is mentioned in consultation, status, analytics, and submission contexts.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_submission_router.DatasetTests -v
```

Expected: FAIL because the dataset does not exist.

- [ ] **Step 3: Create the dataset**

Add explicit submissions, requests to start/continue applications, attachment-led submissions, informational questions, eligibility/requirements questions, status queries, analytics queries, and ambiguous/hypothetical requests.

- [ ] **Step 4: Implement the live evaluator**

Read JSONL, call `SubmissionDetector`, calculate TP/TN/FP/FN, accuracy, precision, recall, and F1, print misclassified cases, and exit nonzero when configurable minimum precision or recall is not met.

- [ ] **Step 5: Add blank key configuration**

Create:

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.6
OPENAI_TIMEOUT_SECONDS=30
```

- [ ] **Step 6: Run offline dataset tests**

Run:

```bash
python3 -m unittest tests.test_submission_router.DatasetTests -v
```

Expected: all dataset validation tests pass.

### Task 5: Document and Verify the Delivery

**Files:**
- Modify: `README.md`
- Modify: `DEPLOYMENT.md`
- Modify: `MANIFEST.json`
- Modify: `SHA256SUMS`

- [ ] **Step 1: Document `/route` and environment variables**

Explain the five-intent public contract, four-class internal model, API-key handling, no-PDF-to-OpenAI boundary, failure behavior, and live evaluation command.

- [ ] **Step 2: Update manifest capabilities without changing model metrics**

Record that the package has a GPT-5.6 LLM wrapper, five public route labels, four local model labels, and is no longer fully offline when `/route` is used.

- [ ] **Step 3: Run the full offline verification suite**

Run:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile runtime/app.py runtime/submission_router.py scripts/evaluate_submission_detection.py
```

Expected: all tests pass and compilation exits 0.

- [ ] **Step 4: Verify live-eval precondition behavior without a key**

Run:

```bash
env -u OPENAI_API_KEY python3 scripts/evaluate_submission_detection.py
```

Expected: nonzero exit with a clear instruction to set `OPENAI_API_KEY`; no secret is printed.

- [ ] **Step 5: Regenerate checksums**

Update `SHA256SUMS` for delivered runtime, configuration, documentation, test, skill, and evaluation files while preserving the checksum entry for the separately delivered real model weight.
