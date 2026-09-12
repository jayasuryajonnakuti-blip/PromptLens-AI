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

Run the backend tests from the `backend` directory with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
