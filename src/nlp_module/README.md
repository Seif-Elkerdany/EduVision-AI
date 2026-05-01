# NLP Module

## Overview

The `nlp_module` is a small FastAPI service that classifies a spoken question into one of the supported academic topics using a zero-shot transformer model.

It receives text from an upstream speech-to-text step, predicts the most likely subject, and returns the predicted topic with a confidence score.

## What This Module Does

- Exposes an HTTP API for topic classification
- Accepts a text string and language code
- Classifies the text into one of the predefined subjects
- Returns the original text, language, predicted topic, and score
- Provides a lightweight health endpoint for service checks

## Supported Subjects

The classifier currently predicts one of these topics:

- Math
- Computer Vision
- Natural Language Processing
- Data Science
- Databases
- Algorithms
- Computer Networks
- Operating Systems

## Features

- FastAPI-based API
- Zero-shot classification with Hugging Face Transformers
- Lazy model loading
  The transformer pipeline is created only when the first classification request is made.
- Health check endpoint
  `GET /health` returns a simple readiness response without loading the model.
- Structured request and response models using Pydantic
- Unit tests with mocked inference
  Tests do not require downloading the real model

## API Endpoints

### `GET /health`

Checks whether the API is reachable.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /stt-result`

Classifies input text into one of the supported subjects.

Request body:

```json
{
  "text": "How does binary search work?",
  "language": "en"
}
```

Successful response:

```json
{
  "text": "How does binary search work?",
  "language": "en",
  "predicted_topic": "Algorithms",
  "score": 0.98
}
```

If required fields are missing or set to `null`, FastAPI validation returns `422 Unprocessable Entity`.

## Project Files

- `src/nlp_module/classifier.py`
  FastAPI app, lazy classifier loading, endpoints, and classification function
- `src/nlp_module/dom.py`
  Pydantic request and response models
- `tests/test_nlp_module_api.py`
  Unit tests for health, response time, valid responses, and invalid input handling

## How To Use This Module

### 1. Install dependencies

This repository does not currently include a Python dependency manifest, so install the needed packages manually:

```bash
python -m pip install fastapi transformers uvicorn pytest
```

### 2. Run the API

From the project root:

```bash
uvicorn nlp_module.classifier:app --app-dir src --reload
```

The API will start locally, usually at:

```text
http://127.0.0.1:8000
```

### 3. Check the health endpoint

```bash
curl http://127.0.0.1:8000/health
```

### 4. Send a classification request

```bash
curl -X POST http://127.0.0.1:8000/stt-result \
  -H "Content-Type: application/json" \
  -d '{"text":"How does binary search work?","language":"en"}'
```

## Unit Tests

The module currently has 9 unit tests in `tests/test_nlp_module_api.py`.

### What The Tests Cover

1. Health endpoint returns `200`
2. Health endpoint returns a non-null `status`
3. Health endpoint responds in under `200 ms`
4. Valid `/stt-result` request returns `200`
5. Valid `/stt-result` response contains non-null fields
6. Valid `/stt-result` returns the mocked topic and score
7. `/stt-result` rejects `null` values with `422`
8. `/stt-result` rejects missing required fields with `422`
9. `classify_question()` correctly maps classifier output to `(predicted_topic, score)`

### How To Run The Unit Tests

From the project root:

```bash
pytest tests/test_nlp_module_api.py
```

## Latest Test Result

Latest local run in this workspace:

```text
collected 9 items
tests/test_nlp_module_api.py .........                                   [100%]
9 passed in 3.16s
```

The exact execution time may vary by machine, but all tests passed in the latest run.

**All unit testing is done by Codex model**