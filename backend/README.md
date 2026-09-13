# PromptLens AI Backend

The backend is the independent Python foundation for PromptLens AI. It currently provides a FastAPI health check, deterministic prompt preprocessing, a spaCy-based NLP endpoint, a baseline TF-IDF intent classifier, and local semantic embeddings. Quality ML and provider integrations will be added in later steps.

## Setup

From the `backend` directory, create a local Python 3.13 virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the minimal backend dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the lightweight English spaCy model:

```powershell
python -m spacy download en_core_web_sm
```

Start the FastAPI development server:

```powershell
python -m uvicorn app.main:app --reload
```

The server runs at `http://127.0.0.1:8000` by default.

## Health Check

```text
GET http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "service": "PromptLens AI backend"
}
```

## Prompt Preprocessing

```text
POST http://127.0.0.1:8000/api/v1/preprocess
```

Request body:

```json
{
  "prompt": "Explain photosynthesis to a high school student."
}
```

The response contains deterministic text statistics, structure signals, placeholders, output-format indicators, detected sections, repeated phrases, and size flags. No NLP, ML, embedding, or provider integrations are used.

## NLP Analysis

```text
POST http://127.0.0.1:8000/api/v1/nlp/analyze
```

The endpoint uses the cached `en_core_web_sm` model to return linguistic statistics, part-of-speech counts, named entities, sentence complexity indicators, noun phrases, verb density, heuristic technical terms, stopword ratio, and punctuation statistics. If the model is unavailable, the endpoint returns HTTP 503 with the installation command.

## Intent Classification

The intent classifier supports these baseline labels:

`CODING`, `EDUCATION`, `RESEARCH`, `WRITING`, `SUMMARIZATION`, `TRANSLATION`, `BUSINESS`, `MARKETING`, `DATA_ANALYSIS`, `CREATIVE`, `IMAGE_GENERATION`, and `GENERAL`.

The starter dataset is stored at `backend/data/intent_dataset_v1.csv` with fields `text,intent`. It is a balanced baseline dataset with 96 manually authored starter examples, not a production-quality or production-representative corpus. Real deployment requires a larger manually validated dataset.

Train and evaluate both the Logistic Regression and calibrated Linear SVM baselines from the `backend` directory:

```powershell
.\.venv\Scripts\python.exe -m app.ml.intent.train `
  --dataset data/intent_dataset_v1.csv `
  --model-dir models/intent `
  --artifact-dir artifacts/intent
```

The pipeline uses a reproducible stratified 70%/15%/15% train/validation/test split, selects primarily by validation macro F1, and evaluates the selected model once on the held-out test set. The persisted model is stored under `backend/models/intent/` and evaluation metadata under `backend/artifacts/intent/`; both generated directories are ignored by Git.

The API endpoint is:

```text
POST http://127.0.0.1:8000/api/v1/intent/classify
```

Request:

```json
{
  "prompt": "Write a Python program to analyze a CSV file and calculate average sales."
}
```

The response contains the predicted intent, calibrated model confidence, model metadata, and a confidence methodology note. Confidence is model confidence, not certainty. The current starter-data evaluation is intentionally a baseline and must not be interpreted as production accuracy.

## Local Embeddings

PromptLens uses the local `sentence-transformers/all-MiniLM-L6-v2` model for reusable semantic vectors. The first embedding request downloads the lightweight model into the local Hugging Face cache; later requests reuse a process-level cached model. CPU is the default and does not require a GPU or API key.

Embeddings are 384-dimensional and L2-normalized. Similarity uses cosine similarity. Long prompts are split at paragraph and sentence boundaries, then tokenizer-aware windows are used for oversized units. Chunk embeddings are combined with token-count weighting and the aggregate is normalized again.

```text
POST http://127.0.0.1:8000/api/v1/embeddings/generate
POST http://127.0.0.1:8000/api/v1/embeddings/similarity
```

The service supports batch encoding internally for future semantic search and duplicate-detection work. Near-duplicate thresholds are configurable utilities only; calibration requires real PromptLens data.

Embedding similarity is a model-dependent semantic similarity estimate, not a measure of factual correctness, prompt quality, or model output quality.

The model is not stored in this repository. Install dependencies from `requirements.txt`; the first request performs the local model download when needed.

## Prompt Quality ML Model

PromptLens incorporates a dedicated baseline ML regression model that estimates overall prompt quality on a continuous 0–100 scale and maps it to five heuristic quality categories:

- `0–39`: **POOR**
- `40–59`: **FAIR**
- `60–74`: **GOOD**
- `75–89`: **STRONG**
- `90–100`: **EXCELLENT**

### Dataset

The versioned dataset is located at `backend/data/prompt_quality_dataset_v1.csv` containing 530 balanced, heuristically annotated examples across 15 prompt categories (coding, education, research, writing, business, marketing, data analysis, creative, image generation, general, summarization, translation, comparison, analysis, and roleplay).

The dataset includes 12 annotated dimension scores (clarity, specificity, context, goal definition, constraints, output format, role persona, audience, ambiguity, completeness, actionability, consistency) for analytical documentation. To prevent data leakage, these dimension scores are **never** passed into feature engineering or model training.

### Feature Engineering

30 structured numeric features are extracted using existing PromptLens engines:
- **Preprocessing Engine (Step 8)**: character count, word count, sentence count, paragraph count, average word length, vocabulary diversity, question count, instruction count, bullet count, heading count, code block presence, URL presence, placeholder count, output format count, section count, repeated phrase count, near-empty flag.
- **NLP Engine (Step 9)**: unique token count, noun count, verb count, adjective count, noun phrase count, verb density, stopword ratio, average dependency depth, subordinate clause count, technical term count, named entity count, average sentence length.
- **Derived**: question-to-instruction ratio.

### Benchmarked Algorithms & Selection

Four regressors were benchmarked using a stratified 70% / 15% / 15% train/validation/test split (`random_state=42`):
- `RandomForestRegressor`: validation MAE = 2.9233, RMSE = 4.9299, R² = 0.9700
- `GradientBoostingRegressor`: validation MAE = **2.8913**, RMSE = **4.9158**, R² = **0.9702**
- `HistGradientBoostingRegressor`: validation MAE = 3.7349, RMSE = 5.7078, R² = 0.9598
- `Ridge`: validation MAE = 6.3942, RMSE = 9.5594, R² = 0.8872

`GradientBoostingRegressor` was selected based on the lowest validation MAE.

Held-out test set performance:
- **MAE**: 3.3959
- **RMSE**: 6.6441
- **R²**: 0.9386
- **Pearson r**: 0.9691
- **Secondary Classification Accuracy**: 95.0% (Macro F1: 0.9496)

Train the quality baseline model:

```powershell
.\.venv\Scripts\python.exe -m app.ml.quality.train `
  --dataset data/prompt_quality_dataset_v1.csv `
  --model-dir models/quality `
  --artifact-dir artifacts/quality
```

### Quality Prediction API

```text
POST http://127.0.0.1:8000/api/v1/quality/predict
```

Request:

```json
{
  "prompt": "Explain machine learning to a beginner using a simple example."
}
```

Response:

```json
{
  "score": 43.58,
  "label": "FAIR",
  "model": {
    "name": "promptlens-quality-model",
    "version": "0.1.0"
  }
}
```

### Limitations & Disclaimer

> "The quality model is a learned baseline estimate and does not represent ground-truth prompt quality."
>
> "Production performance requires a substantially larger, manually validated dataset and ongoing evaluation."

## Scoring & Fusion Engine (Step 13)

PromptLens implements a centralized Scoring & Fusion Engine that synthesizes multiple upstream analysis signals into a unified prompt quality assessment, complete with an overall score (0–100), 12 quality dimensions, findings, and actionable recommendations.

### Architecture & Signal Weighting

The fusion architecture combines four primary signals with the following configured weights:

| Signal | Source | Configured Weight | Effective Weight (Step 13) |
|---|---|:---:|:---:|
| **Rules** | Deterministic preprocessing (Step 8) | 25% | **41.67%** (25/60) |
| **NLP** | Linguistic spaCy analysis (Step 9) | 15% | **25.00%** (15/60) |
| **Quality ML** | Regression model prediction (Step 12) | 20% | **33.33%** (20/60) |
| **LLM** | Local LLM provider (Future step) | 40% | *0.0% (Unavailable)* |

> **Missing LLM Handling**: The local LLM signal is not yet implemented in Step 13. To avoid penalizing prompts with a false zero, effective weights are dynamically renormalized across the three available signals ($\text{sum} = 60$). When an LLM provider is integrated in a later step, the weights automatically rebalance.

Supporting signals:
- **Intent Classification (Step 10)**: Surface detected intent and confidence as contextual metadata.
- **Semantic Embeddings (Step 11)**: Provide vector representation metadata and semantic search capability.

### 12 Quality Dimensions

Every evaluation assesses 12 explicit quality dimensions with bounded scores (0–100), categories (`POOR`, `FAIR`, `GOOD`, `STRONG`, `EXCELLENT`), deterministic reasons, and recommendations:

1. `clarity`: Phrasing simplicity, sentence length, and absence of repetitive clutter.
2. `specificity`: Concrete parameters, technical terms, numerical targets, and placeholders.
3. `context`: Sufficient domain framing, background information, and entity coverage.
4. `goal_definition`: Clear, actionable task objective with direct imperative verbs.
5. `constraints`: Explicit limitations, negative constraints, and operating boundaries.
6. `output_format`: Defined deliverable format (JSON, markdown table, code, or bullet list).
7. `role_persona`: Explicit expert perspective or domain framing where helpful.
8. `audience`: Explicit target reader proficiency level or stakeholder group.
9. `ambiguity`: Inverted ambiguity assessment (higher score = lower ambiguity).
10. `completeness`: Presence of core task building blocks (goal + context + criteria).
11. `actionability`: Direct, executable instructions with active verbs.
12. `consistency`: Absence of contradictory requirements and excessive repetition.

### Quality Categories

Scores are mapped to PromptLens quality categories:
- `0–39`: **POOR**
- `40–59`: **FAIR**
- `60–74`: **GOOD**
- `75–89`: **STRONG**
- `90–100`: **EXCELLENT**

### Scoring Endpoint

```text
POST http://127.0.0.1:8000/api/v1/score
```

Request:

```json
{
  "prompt": "Explain machine learning to a beginner using a simple example."
}
```

Response structure:

```json
{
  "overall_score": {
    "score": 61.61,
    "category": "GOOD",
    "status": "GOOD",
    "explanation": "Fused score 61.61/100 (GOOD) derived from Rules (53.0 × 41.67%), NLP (100.0 × 25.00%), and Quality ML (43.58 × 33.33%). LLM signal is currently unavailable."
  },
  "dimensions": {
    "clarity": { "score": 90.0, "status": "EXCELLENT", "reason": "...", "recommendation": "..." },
    "specificity": { "score": 55.0, "status": "FAIR", "reason": "...", "recommendation": "..." },
    "...": "..."
  },
  "signals": {
    "rules": { "source": "rules", "raw_score": 53.0, "configured_weight": 0.25, "effective_weight": 0.416667, "available": true },
    "nlp": { "source": "nlp", "raw_score": 100.0, "configured_weight": 0.15, "effective_weight": 0.25, "available": true },
    "quality_ml": { "source": "quality_ml", "raw_score": 43.58, "configured_weight": 0.2, "effective_weight": 0.333333, "available": true },
    "llm": { "source": "llm", "raw_score": null, "configured_weight": 0.4, "effective_weight": 0.0, "available": false }
  },
  "intent": { "intent": "GENERAL", "confidence": 0.1028, "model": "promptlens-intent-classifier" },
  "embeddings": { "dimension": 384, "normalized": true, "model": "sentence-transformers/all-MiniLM-L6-v2", "available": true },
  "findings": [
    { "type": "missing_output_format", "severity": "INFO", "dimension": "output_format", "message": "No explicit deliverable format specified (e.g. JSON, table, or markdown)." }
  ],
  "recommendations": [
    "Specify the desired output format (e.g., JSON, markdown table, bullet list, or CSV).",
    "Define explicit constraints (e.g. length limits, forbidden methods, or required libraries)."
  ],
  "metadata": {
    "scoring_version": "0.1.0",
    "llm_available": false,
    "signals_evaluated": 3
  }
}
```

### Limitations & Disclaimer

> "The Step 13 score is an AI-assisted heuristic quality estimate, not ground truth."

## Local LLM Integration (Step 14)

PromptLens integrates a local, open-source LLM integration layer designed for CPU-first execution without reliance on commercial or paid inference APIs (no OpenAI, Gemini, or Anthropic keys required).

### Architecture & Provider Abstraction

The LLM subsystem is isolated behind an abstract `LLMProvider` interface (`backend/app/llm/provider.py`):
```text
Prompt + Upstream Context (Steps 8–13)
  ↓
LLM Service (Lazy loading, context budgeting, retry loop)
  ↓
LLMProvider (TransformersLocalProvider | MockLLMProvider)
  ↓
Local Model Runtime (Hugging Face Transformers / PyTorch on CPU)
  ↓
Raw Generated Text
  ↓
Defensive JSON Extraction (markdown fence stripping, regex boundary search)
  ↓
Strict Pydantic Validation (PromptAnalysisOutput schema)
  ↓
Targeted Error-Correction Retry (up to LLM_MAX_RETRIES)
  ↓
Validated LLM Analysis Response
```

### Selected Local Model
- **Model Name**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Model Format**: Hugging Face Safetensors (`AutoModelForCausalLM`, `AutoTokenizer`)
- **Download Size**: ~980 MB
- **RAM Footprint**: ~1.2 GB on CPU (float32, low memory usage enabled)
- **Supported Context Length**: 32,768 tokens (configured default: 2,048 tokens)
- **Runtime**: Local PyTorch (`torch 2.6.0`) + Hugging Face `transformers 5.17.0` on CPU
- **First-Download Behavior**: Weights are downloaded from Hugging Face Hub to the standard cache (`~/.cache/huggingface/hub`) only when `LLM_ENABLED=true` and inference is first requested. Model binaries are never committed to Git.

### Configuration Variables

Configured via environment variables with safe defaults:

| Environment Variable | Default Value | Description |
|---|:---:|---|
| `LLM_ENABLED` | `false` | Master toggle to prevent accidental memory consumption |
| `LLM_PROVIDER` | `transformers` | Provider backend (`transformers` or `mock`) |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Local Hugging Face model identifier |
| `LLM_CONTEXT_LENGTH`| `2048` | Total token window budget |
| `LLM_MAX_TOKENS` | `768` | Maximum new tokens to generate |
| `LLM_TEMPERATURE` | `0.0` | Near-deterministic greedy decoding |
| `LLM_TIMEOUT` | `60.0` | Timeout threshold in seconds |
| `LLM_MAX_RETRIES` | `2` | Number of targeted error-correction retries |

### Structured Output Schema

The model returns strict validated JSON matching the `PromptAnalysisOutput` schema:
```json
{
  "interpreted_goal": "A concise summary of the primary task the user wants to accomplish.",
  "context": "Summary of background context, scenario assumptions, or domain framing.",
  "ambiguities": ["List of ambiguous terms, vague requirements, or open-ended phrasing."],
  "missing_information": ["List of unstated inputs, missing parameter bounds, or omitted constraints."],
  "contradictions": ["List of conflicting instructions or incompatible output formats."],
  "instruction_quality": {
    "score": 82.0,
    "reason": "Direct active imperative phrasing with clear scope."
  },
  "strengths": ["List of identified strengths in prompt construction."],
  "weaknesses": ["List of identified deficits or weaknesses."],
  "recommendations": ["List of concrete, actionable advice to strengthen the prompt."]
}
```

### JSON Extraction & Targeted Error-Correction Retries
1. **Defensive Extraction**: Strips markdown fences (` ```json ... ``` `) and extracts JSON from conversational prose without using unsafe `eval()`.
2. **Schema Enforcement**: Validates that all fields exist and `instruction_quality.score` is in `[0.0, 100.0]`.
3. **Correction Loop**: If parsing or validation fails, builds a targeted prompt quoting the exact validation failure and requests a corrected JSON object (up to `LLM_MAX_RETRIES` attempts).

### Health & Availability States
`GET /api/v1/llm/health` reports the operational state:
- `LLM_DISABLED`: `LLM_ENABLED=false` (safe default).
- `MODEL_NOT_DOWNLOADED`: Model enabled, but weights not yet in local Hugging Face cache.
- `MODEL_LOADING`: Model weights actively being transferred to RAM.
- `MODEL_AVAILABLE`: Model loaded or cached and ready for inference.
- `MODEL_ERROR`: Runtime or out-of-memory error occurred.

### Deterministic Mock Provider for Testing
The automated test suite uses `MockLLMProvider` to test schema validation, fence extraction, malformed recovery, retry exhaustion, and endpoint error codes deterministically without downloading multi-gigabyte models or requiring network access.

### Endpoints

```text
GET  http://127.0.0.1:8000/api/v1/llm/health
POST http://127.0.0.1:8000/api/v1/llm/analyze
```

### Limitations & Disclaimer

> "The local LLM is an assistive analysis component and does not constitute ground truth."
>
> "Prompt optimization and prompt rewriting are intentionally outside Step 14 (reserved for Step 16)."

Run the backend tests from the `backend` directory with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```





---

## Step 15 - AI Analyzer

Step 15 adds a dedicated AI Analyzer combining deterministic pipeline evidence (Steps 8-13) with optional local LLM reasoning (Step 14).

### Package layout

`
backend/app/analyzer/
    __init__.py       - Package declaration
    config.py         - ANALYZER_VERSION, confidence thresholds, token limits
    context.py        - AnalyzerContext dataclass; build_analyzer_context()
    evidence.py       - Evidence generation from deterministic and LLM signals
    prompts.py        - PROMPTLENS_ANALYZER_V1 system prompt; input builder
    schemas.py        - Pydantic schemas (EvidenceItem, AnalyzerAnalysis, ...)
    analyzer.py       - Core PromptAnalyzer class
backend/app/services/analyzer_service.py  - AnalyzerService singleton
backend/app/api/analyzer.py               - POST /api/v1/analyzer/analyze
`

### Endpoint

`
POST /api/v1/analyzer/analyze
`

### Analysis modes

| analysis_mode | Condition |
|---------------|-----------|
| LOCAL_LLM     | LLM_ENABLED=true, TransformersLocalProvider active |
| MOCK          | MockLLMProvider active |
| UNAVAILABLE   | LLM disabled, model missing, or retries exhausted |

### Design invariants

- Step 13 overall quality score is NEVER replaced by instruction_quality.score
- Evidence priority: preprocessing > nlp > intent > quality_ml > scoring > llm
- The analyzer NEVER rewrites prompts (optimization belongs to Step 16)
- Production NEVER silently uses mock output -- analysis_mode is always accurate
- confidence values are model-assisted estimates (0.0-1.0), NOT calibrated probabilities

### Test count (cumulative)

| Step | New tests | Total |
|------|-----------|-------|
| 8-13 | 126 | 126 |
| 14 (LLM) | 40 | 166 |
| 15 (Analyzer) | 97 | 263 |



---

## Step 16 - AI Optimizer

Step 16 adds a dedicated AI Optimizer that accepts:
1. Original user prompt
2. Optional Step 15 Analyzer result
3. Optimization mode (`balanced`, `analytical`, `creative`, `expert`)

and generates a structured, improved prompt preserving the user's intent, domain, scope, and stated constraints.

### Package layout

```
backend/app/optimizer/
    __init__.py       - Package declaration
    config.py         - OPTIMIZER_VERSION, context token limits, placeholder constants
    prompts.py        - Mode-specific system prompts (balanced, analytical, creative, expert)
    schemas.py        - Pydantic models (ChangeRecord, PreservedRequirement, OptimizerResult, ...)
    optimizer.py      - Core PromptOptimizer class
backend/app/services/optimizer_service.py  - OptimizerService singleton
backend/app/api/optimizer.py               - POST /api/v1/optimizer/optimize
```

### Endpoint

```
POST /api/v1/optimizer/optimize
```

### Optimization Modes

- `balanced`: Proportional improvement across clarity, specificity, structure, and completeness.
- `analytical`: Prioritizes numbered constraints, precise boundaries, and structured criteria.
- `creative`: Enriches expression and context while strictly retaining original scope.
- `expert`: Domain-specific technical depth, rigorous terminology, and explicit evaluation criteria.

### Design Invariants

- Never fabricates requirements or facts.
- Placeholders such as `[WEB_FRAMEWORK]` or `[TARGET AUDIENCE]` are inserted when information is genuinely missing.
- When LLM is unavailable (`optimizer_mode: "UNAVAILABLE"`), returns the original prompt unchanged with explicit explanatory metadata.
- Accurate reporting of provider states (`LOCAL_LLM`, `MOCK`, `UNAVAILABLE`).

### Test count (cumulative)

| Step | Tests | Total |
|------|-------|-------|
| 8-13 | 126 | 126 |
| 14 (LLM) | 40 | 166 |
| 15 (Analyzer) | 97 | 263 |
| 16 (Optimizer) | 62 | **325** |


---

## Step 17 - AI Critic

Step 17 introduces the PromptLens AI Critic, an independent evaluation engine that determines whether an optimized prompt is genuinely better than the original prompt.

### Core Principle

> "Optimization is NOT automatically improvement."

The Critic evaluates the original prompt against the proposed optimized prompt across multiple dimensions (intent preservation, requirement retention, semantic similarity, quality score deltas, verbosity, contradictions, and unsupported assumptions) to produce a `PASS`, `FAIL`, or `NEEDS_REVIEW` verdict. The Critic never rewrites or optimizes prompts.

### Architecture

```text
Original Prompt ──┐
                  ├──> Critic Context ──> Deterministic Checks ──> Local LLM Critic ──> Structured Critique ──> Final Decision
Optimized Prompt ─┘                                                                                           (PASS / FAIL / NEEDS_REVIEW)
```

### Package Layout

```
backend/app/critic/
    __init__.py       - Package declaration
    config.py         - CRITIC_VERSION, documented score and similarity thresholds
    context.py        - CriticContext dataclass; comparative signal assembly
    checks.py         - Deterministic rules (intent, format, constraints, score regression, verbosity, contradictions)
    evidence.py       - Prioritized evidence generation with semantic similarity disclaimer
    prompts.py        - PROMPTLENS_CRITIC_V1 system prompt, input builder, correction prompt
    schemas.py        - Strict Pydantic models (CriticRequest, CriticResponse, CriticEvaluation, CriticIssue, ...)
    critic.py         - Core PromptCritic class coordinating checks, LLM, and arbitration
backend/app/services/critic_service.py  - CriticService singleton with test isolation
backend/app/api/critic.py               - POST /api/v1/critic/evaluate
```

### Endpoint

```
POST /api/v1/critic/evaluate
```

### Decision System

- **`FAIL`**:
  - Original intent materially changed (intent regression)
  - Critical output format or negative constraint dropped
  - Newly introduced contradictions
  - Critical quality regression (overall score drop > 15.0 pts)
  - Excessive verbosity expansion (> 7.0x original)
- **`PASS`**:
  - Intent preserved
  - Critical requirements preserved
  - No serious contradictions or unsupported introduced stack requirements
  - Quality score improves or remains strong
- **`NEEDS_REVIEW`**:
  - Borderline score changes
  - Multiple minor warnings or moderate verbosity expansion
  - Semantic preservation cannot be confidently confirmed

### Disclaimers & Invariants

- *"The AI Critic is an assistive evaluation component and does not constitute ground truth."*
- *"Semantic similarity is supporting evidence and is not, by itself, proof of intent preservation."*
- Deterministic safety checks take precedence: if a critical requirement is lost or an intent regression occurs, the decision is `FAIL` regardless of model suggestions.
- Analysis mode is accurately reported as `LOCAL_LLM`, `MOCK`, or `UNAVAILABLE`. Production never silently uses mock output.

### Test Count (Cumulative)

| Step | Tests | Total |
|------|-------|-------|
| 8-13 | 126 | 126 |
| 14 (LLM) | 40 | 166 |
| 15 (Analyzer) | 97 | 263 |
| 16 (Optimizer) | 62 | 325 |
| **17 (Critic)** | **56** | **381** |


---

## Step 18 - AI Validator

Step 18 introduces the PromptLens AI Validator, the final deterministic/semantic gatekeeper in the optimization pipeline.

### Core Principles

> *"The Validator is an acceptance gate, not a prompt generator."*
> *"Optimization must be validated before it is accepted."*
> *"Optimization is not automatically improvement."*
> *"Semantic similarity is supporting evidence and is not, by itself, proof of intent preservation."*

The Validator does NOT rewrite or optimize prompts. Its sole purpose is to verify that an optimized prompt is structurally valid, intent-preserving, constraint-respecting, free from introduced contradictions or bloat, and safe to continue toward eventual execution.

### Architecture

```text
Original Prompt ──┐
                  ├──> Validator Context ──> Deterministic Checks ──> Local LLM Validator ──> Final Decision
Optimized Prompt ─┤                                                                             (PASS / FAIL / NEEDS_REVIEW)
Critic Result ────┘                                                                           is_valid: True/False
```

### Package Layout

```
backend/app/validator/
    __init__.py       - Package declaration
    config.py         - VALIDATOR_VERSION, documented safety and regression thresholds
    context.py        - ValidatorContext dataclass; comparative signal & Critic integration
    checks.py         - Deterministic validation rules (schema, intent, requirements, negative constraints, format, contradictions, tech additions, score, bloat, critic consistency)
    evidence.py       - Traceable validation evidence with semantic similarity disclaimer
    prompts.py        - PROMPTLENS_VALIDATOR_V1 system prompt, input builder, correction prompt
    schemas.py        - Strict Pydantic models (ValidatorRequest, ValidatorResponse, ValidationResult, ValidationIssue, ...)
    validator.py      - Core PromptValidator class coordinating deterministic checks, LLM, and arbitration
backend/app/services/validator_service.py  - ValidatorService singleton with test isolation
backend/app/api/validator.py               - POST /api/v1/validator/validate
```

### Endpoint

```
POST /api/v1/validator/validate
```

### Decision Hierarchy

The Validator enforces strict decision arbitration where deterministic safety checks override model suggestions:
1. **Fatal deterministic violation** (field out of range, empty prompt) → `FAIL`
2. **Critical requirement loss** (omitted explicit must/include requirements) → `FAIL`
3. **Negative constraint loss** (dropped "avoid", "must not", "never") → `FAIL`
4. **Format regression** (omitted requested JSON, CSV, table, etc.) → `FAIL`
5. **Introduced contradiction** (e.g., "JSON only" combined with narrative explanation requests) → `FAIL`
6. **Strong intent regression** (high-confidence shift in primary intent or extreme semantic divergence) → `FAIL`
7. **Critic failure consistency** (unresolved Step 17 Critic rejection) → `FAIL`
8. **Unresolved semantic uncertainty or multiple warnings** → `NEEDS_REVIEW`
9. **Clean validation** (all deterministic & semantic checks pass) → `PASS` (`is_valid: True`)

### Disclaimers & Invariants

- *"The Validator is an acceptance gate, not a prompt generator."*
- *"Semantic similarity is supporting evidence and is not, by itself, proof of intent preservation."*
- When the local LLM is disabled or unavailable (`validation_mode: "UNAVAILABLE"`), deterministic validation runs independently and produces a complete, sound `ValidationResult`.
- Production never silently uses mock output; validation mode is accurately reported as `LOCAL_LLM`, `MOCK`, or `UNAVAILABLE`.

### Test Count (Cumulative)

| Step | Tests | Total |
|------|-------|-------|
| 8-13 | 126 | 126 |
| 14 (LLM) | 40 | 166 |
| 15 (Analyzer) | 97 | 263 |
| 16 (Optimizer) | 62 | 325 |
| 17 (Critic) | 56 | 381 |
| **18 (Validator)** | **60** | **441** |


---

## Step 19 - Agent Loop

Step 19 introduces the PromptLens AI Agent Loop, an orchestration layer that coordinates the Analyzer (Step 15), Optimizer (Step 16), Critic (Step 17), and Validator (Step 18) in a bounded, iterative feedback cycle.

### Core Principles

> *"Optimization is iterative, but bounded."*  
> *"The Agent Loop never accepts an unvalidated prompt as successfully optimized."*  
> *"PASS terminates the loop."*  
> *"NEEDS_REVIEW terminates the loop."*  
> *"Semantic similarity is supporting evidence and is not, by itself, proof of correctness."*

The Agent Loop ensures that an optimized prompt is only accepted when validation explicitly indicates success. It enforces a strict hard cap of **3 optimization attempts** (`MAX_OPTIMIZATION_ITERATIONS = 3`), feeds structured diagnostic feedback (lost requirements, lost negative constraints, contradictions, and score warnings) into subsequent optimization attempts, and automatically detects non-improving or degenerating loops early to prevent unnecessary computation.

### Architecture

```text
Original Prompt
      │
      ▼
   Analyzer (Step 15)
      │
      ├────────────────────────────────────────┐
      ▼                                        │ (Loop cap: max 3 attempts)
┌──────────────┐                               │
│  Optimizer   │ ◄─── Diagnostic Feedback      │
│  (Step 16)   │      (lost reqs/constraints)  │
└──────┬───────┘                               │
       │ candidate                             │
       ▼                                       │
┌──────────────┐                               │
│    Critic    │                               │
│  (Step 17)   │                               │
└──────┬───────┘                               │
       │ critique                              │
       ▼                                       │
┌──────────────┐                               │
│  Validator   │                               │
│  (Step 18)   │                               │
└──────┬───────┘                               │
       │                                       │
       ├───── PASS ────────────────────────────┴──► COMPLETED (is_validated: True)
       │
       ├───── NEEDS_REVIEW ───────────────────────► NEEDS_REVIEW (is_validated: False)
       │
       └───── FAIL ───► attempt < 3 ? ─────────────► (Retry with structured feedback)
                         │
                         └─► attempt == 3 or ─────► MAX_ITERATIONS_REACHED / FAILED
                             non-improving?         (is_validated: False)
```

### Package Layout

```
backend/app/agent/
    __init__.py       - Package declaration and public exports
    config.py         - AGENT_VERSION, MAX_OPTIMIZATION_ITERATIONS=3, thresholds, disclaimers
    schemas.py        - Strict Pydantic models (AgentRequest, AgentResponse, AgentIteration, AgentMetrics, ...)
    errors.py         - Typed exception hierarchy (AgentException, AgentServiceUnavailableError, ...)
    state.py          - Immutable AgentState with copy-on-write transitions and full chronological history
    context.py        - Bounded context builder formatting structured Critic and Validator feedback
    decisions.py      - Deterministic decision engine and non-improving loop detector
    orchestrator.py   - AgentOrchestrator coordinating Analyzer -> Optimizer -> Critic -> Validator
backend/app/services/agent_service.py  - AgentService process-level singleton with test isolation
backend/app/api/agent.py               - POST /api/v1/agent/run
```

### Endpoint

```
POST /api/v1/agent/run
```

### Decision System & Precedence

1. **`Validator PASS`**: Immediate termination with `AgentStatus.COMPLETED` and `AgentTerminationReason.VALIDATED`. The candidate prompt is selected as `final_prompt` and marked `is_validated = True`.
2. **`Validator NEEDS_REVIEW`**: Immediate safe termination with `AgentStatus.NEEDS_REVIEW` and `AgentTerminationReason.VALIDATION_REVIEW`. No automated modifications continue.
3. **Non-Improving Loop Detection**:
   - Identical candidate prompt produced.
   - Negligible prompt change (< 2% character edit difference with unresolved issues).
   - Repeated identical validation failure codes across consecutive iterations.
   - Consecutive score regressions (scores steadily dropping across iterations).
   - Severe prompt bloat (expansion ratio > 4.0x).
   - Semantic similarity collapse (< 0.40).
   When detected, terminates early with `AgentStatus.FAILED` and `AgentTerminationReason.VALIDATION_FAILED`.
4. **`Validator FAIL` at Attempt 3**: Terminated safely with `AgentStatus.MAX_ITERATIONS_REACHED` and `AgentTerminationReason.MAX_ITERATIONS`.
5. **`Validator FAIL` at Attempt < 3**: Context builder extracts structured feedback from Critic (lost requirements, assumptions) and Validator (negative constraints, contradictions) and passes it to the next optimization attempt.

### Disclaimers & Invariants

- *"Optimization is iterative, but bounded."*
- *"The Agent Loop never accepts an unvalidated prompt as successfully optimized."*
- *"PASS terminates the loop."*
- *"NEEDS_REVIEW terminates the loop."*
- *"Semantic similarity is supporting evidence and is not, by itself, proof of correctness."*
- All iterations are individually recorded with before/after prompts, scores, and decisions. Historical records are immutable.
- An unvalidated prompt is NEVER marked as successfully validated (`is_validated = False`).

| Step | Tests | Total |
|------|-------|-------|
| 8-13 | 126 | 126 |
| 14 (LLM) | 40 | 166 |
| 15 (Analyzer) | 97 | 263 |
| 16 (Optimizer) | 62 | 325 |
| 17 (Critic) | 56 | 381 |
| 18 (Validator) | 60 | 441 |
| 19 (Agent Loop) | 42 | 483 |
| 20 (Database) | 76 | 559 |
| **21 (Frontend & Product)** | **9** | **568** |

---

## Step 20 — Database & Persistence Layer

Step 20 introduces SQLite-backed persistence for all completed Agent Loop executions. Every `POST /api/v1/agent/run` request that produces a terminal `AgentResponse` is automatically persisted to a local SQLite database via SQLAlchemy 2.x.

### Architecture

```
backend/app/db/
├── __init__.py       — Public package exports
├── config.py         — Database URL configuration (env-var driven, SQLite default)
├── base.py           — SQLAlchemy DeclarativeBase
├── session.py        — Engine singleton, SessionLocal factory, init_db(), get_db()
├── models.py         — AgentRun ORM model
├── repositories.py   — AgentRunRepository CRUD layer
└── errors.py         — Typed exception hierarchy (DatabaseError, AgentRunNotFoundError, …)

backend/app/services/
└── agent_run_service.py  — AgentRunService (singleton, test-isolation reset)
```

**Persistence flow:**

```
POST /api/v1/agent/run
    ↓
AgentService.run()
    ↓
AgentOrchestrator.run()      ← unchanged (Steps 8-19)
    ↓
AgentResponse (terminal)
    ↓
AgentRunService.save_agent_run()
    ↓
AgentRunRepository.create()
    ↓
SQLite: backend/data/promptlens.db
    ↓
AgentResponse.run_id = "<uuid>"
AgentResponse.persistence_status = "persisted" | "failed"
```

**Failure isolation:** Persistence failure sets `persistence_status = "failed"` on the response and logs the error. The Agent result and its content are never affected by a database failure.

### Database

| Property | Value |
|---|---|
| Engine | SQLite (file-based) |
| ORM | SQLAlchemy 2.x |
| Schema migration | `SQLAlchemy metadata.create_all()` (idempotent, no Alembic) |
| Default path | `backend/data/promptlens.db` |
| Environment variable | `DATABASE_URL` (overrides default) |

The database file is ignored by Git (`.gitignore` patterns: `*.db`, `*.sqlite`, `*.sqlite3`, `backend/data/*.db`).

### AgentRun Model

| Column | Type | Notes |
|---|---|---|
| `id` | `String(36)` | UUID4, primary key, auto-generated |
| `created_at` | `DateTime` | UTC, auto-set on insert |
| `updated_at` | `DateTime` | UTC, auto-updated |
| `original_prompt` | `Text` | Original user prompt |
| `final_prompt` | `Text` | Terminal prompt (validated or best-effort) |
| `status` | `String(32)` | `COMPLETED`, `FAILED`, `NEEDS_REVIEW`, `MAX_ITERATIONS_REACHED` |
| `termination_reason` | `String(64)` | `VALIDATED`, `VALIDATION_FAILED`, `VALIDATION_REVIEW`, `MAX_ITERATIONS`, … |
| `is_validated` | `Boolean` | True only if Validator returned PASS |
| `iteration_count` | `Integer` | Number of optimization iterations executed |
| `final_score` | `Float` | Nullable — composite quality score |
| `agent_version` | `String(16)` | Semantic version of the Agent component |
| `mode` | `String(16)` | Optimization mode: `balanced`, `analytical`, `creative`, `expert` |
| `iterations` | `JSON` | Full serialized iteration history |
| `score_history` | `JSON` | List of `[score_before, score_after, …]` floats |
| `critic_result` | `JSON` | Nullable — final Critic result serialized |
| `final_validation` | `JSON` | Nullable — final ValidationResult serialized |
| `metrics` | `JSON` | AgentMetrics serialized |
| `run_metadata` | `JSON` | `{"disclaimers": […]}` |

**CHECK constraints:**
- `iteration_count >= 0`
- `final_score IS NULL OR (final_score >= 0.0 AND final_score <= 100.0)`

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/promptlens.db` | SQLAlchemy connection URL |

Example override:

```powershell
$env:DATABASE_URL = "sqlite:///./data/custom.db"
```

### AgentResponse Fields Added (Step 20)

Two optional fields added to `AgentResponse` (non-breaking, default `None`):

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "persistence_status": "persisted"
}
```

| Field | Type | Description |
|---|---|---|
| `run_id` | `string \| null` | UUID of the persisted `AgentRun` record |
| `persistence_status` | `string \| null` | `"persisted"` on success, `"failed"` if DB write fails |

### Test Isolation

All Step 20 tests use isolated in-memory SQLite databases:

```python
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
svc = AgentRunService(session_factory=factory)
# setUp: reset_db_engine() + set_agent_run_service(None)
# tearDown: Base.metadata.drop_all() + engine.dispose() + reset_db_engine()
```

### Step 20 Test Files

| File | Tests | Description |
|---|---|---|
| `test_db_config.py` | 6 | URL config, env-var override, SQLite detection |
| `test_db_session.py` | 6 | Engine singleton, session factory, init_db(), reset |
| `test_db_models.py` | 10 | AgentRun ORM field defaults, constraints, JSON fields |
| `test_db_repository.py` | 8 | CRUD operations, pagination, integrity error handling |
| `test_agent_run_service.py` | 35 | Service-level persistence, retrieval, list, delete, count, singleton |
| `test_db_integration.py` | 11 | End-to-end AgentService → AgentRunService → SQLite persistence |
| **Total** | **76** | |

---

## Step 21 — Final Frontend & Product Integration

Step 21 delivers a polished, production-ready AI SaaS user experience connecting the modern React frontend to the real backend Agent Loop pipeline and SQLite persistence layer.

### Product UX & Features

- **Product Identity:** PromptLens AI — *"See what your prompt is missing. Build prompts that work."*
- **Main Prompt Workspace:**
  - Responsive prompt editor with word and character counters.
  - Multi-mode optimization selector (`balanced`, `analytical`, `creative`, `expert`).
  - Bounded iteration control (max 1 to 3 loops).
  - Quick-start example prompt chips (Executive Summary, Python CLI Tool, Concept Explanation, Marketing Launch).
  - Live execution states with disable controls and graceful error banners.
- **Results Experience:**
  - **Quality Scorecard:** 0–100 overall composite quality score, semantic category badges (`POOR`, `FAIR`, `GOOD`, `STRONG`, `EXCELLENT`), validation status badge (`VALIDATED` / `NEEDS REVIEW` / `UNVALIDATED`), and execution latency / expansion metrics.
  - **Original vs. Optimized Prompt Comparison:** Side-by-side or stacked layout with monospace code typography, character counts, and instant copy feedback.
  - **Export Capabilities:** One-click Markdown export, JSON export, and shareable clipboard copy.
  - **Agent Loop Timeline:** Visual pipeline stepper (`Prompt In` → `Analyzer` → `Optimizer` → `Critic` → `Validator` → `Verdict`) with per-iteration inspection tabs showing score delta, before/after prompts, and optimizer change logs.
  - **Critic & Validator Findings:** Preserved strengths, lost requirements warnings, unsupported assumptions, recommendations, and individual verification invariant checks.
- **Persisted History View:**
  - Direct integration with Step 20 SQLite database via `GET /api/v1/runs`.
  - Real-time search by prompt text or mode.
  - One-click loading of past runs back into the active workspace.
  - Record deletion via `DELETE /api/v1/runs/{id}`.
- **Accessibility & Design:**
  - Dark/Light mode theme switching with localStorage persistence.
  - Visible focus indicators and accessible keyboard shortcuts (`Ctrl+Enter` to submit).
  - Backend connectivity pill indicator in the header.

### Endpoints Added (Step 21)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/runs` | List persisted AgentRuns with pagination (`skip`, `limit`) |
| `GET` | `/api/v1/runs/{run_id}` | Retrieve full details and serialized iteration artifacts of a specific run |
| `DELETE` | `/api/v1/runs/{run_id}` | Delete a persisted AgentRun record |

### Final Verification Summary

| Component | Check | Result |
|---|---|---|
| Backend | `compileall app tests -q` | **0 errors** ✅ |
| Backend | Unit & Integration Test Suite | **568/568 passed** ✅ |
| Frontend | `npm run lint` | **0 errors, 0 warnings** ✅ |
| Frontend | `npm run build` | **1873 modules transformed, 0 errors** ✅ |
| Persistence | SQLite database (`promptlens.db`) | **Persisted & Git-ignored** ✅ |

