import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlp_module import classifier as classifier_module


def build_mock_classifier(label: str = "Algorithms", score: float = 0.98):
    def mock_classifier(text, candidate_labels, multi_label):
        assert text
        assert candidate_labels == classifier_module.SUBJECTS
        assert multi_label is False
        return {"labels": [label], "scores": [score]}

    return mock_classifier


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        classifier_module,
        "get_classifier",
        lambda: build_mock_classifier(),
    )
    return TestClient(classifier_module.app)


def test_health_returns_ok_without_loading_classifier(monkeypatch):
    def fail_if_called():
        raise AssertionError("health check should not load the classifier")

    monkeypatch.setattr(classifier_module, "get_classifier", fail_if_called)

    with TestClient(classifier_module.app) as test_client:
        start = time.perf_counter()
        response = test_client.get("/health")
        elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["status"] is not None
    assert elapsed_ms < 200


def test_stt_result_returns_mocked_non_null_classification(client):
    payload = {"text": "How does binary search work?", "language": "en"}

    start = time.perf_counter()
    response = client.post("/stt-result", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == payload["text"]
    assert body["language"] == payload["language"]
    assert body["predicted_topic"] == "Algorithms"
    assert body["score"] == pytest.approx(0.98)
    assert all(body[field] is not None for field in ("text", "language", "predicted_topic", "score"))
    assert elapsed_ms < 200


@pytest.mark.parametrize(
    "payload",
    [
        {"text": None, "language": "en"},
        {"text": "What is NLP?", "language": None},
        {"text": None, "language": None},
    ],
)
def test_stt_result_rejects_null_values(client, payload):
    response = client.post("/stt-result", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"language": "en"},
        {"text": "What is a database index?"},
        {},
    ],
)
def test_stt_result_rejects_missing_required_fields(client, payload):
    response = client.post("/stt-result", json=payload)

    assert response.status_code == 422


def test_classify_question_maps_pipeline_output():
    predicted_topic, score = classifier_module.classify_question(
        "What is overfitting?",
        classifier_fn=build_mock_classifier(label="Data Science", score=0.87),
    )

    assert predicted_topic == "Data Science"
    assert score == pytest.approx(0.87)
