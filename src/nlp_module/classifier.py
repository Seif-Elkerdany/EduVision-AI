from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import sys
import warnings
from pathlib import Path

from fastapi import FastAPI

try:
    from .dom import ClassificationResult, STTResponse
except ImportError:
    MODULE_DIR = Path(__file__).resolve().parent
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))

    from dom import ClassificationResult, STTResponse


MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SUBJECTS = [
    "Math",
    "Computer Vision",
    "Natural Language Processing",
    "Data Science",
    "Databases",
    "Algorithms",
    "Computer Networks",
    "Operating Systems",
]

app = FastAPI()
_classifier = None


@contextmanager
def quiet_model_output():
    previous_verbosity = None
    try:
        from transformers.utils import logging as transformers_logging

        previous_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
    except Exception:
        transformers_logging = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                yield
    finally:
        if previous_verbosity is not None and transformers_logging is not None:
            transformers_logging.set_verbosity(previous_verbosity)


def get_classifier():
    """Lazily initializes the zero-shot classification pipeline."""
    global _classifier
    if _classifier is None:
        with quiet_model_output():
            from transformers import pipeline

            device = -1
            try:
                import torch

                if torch.cuda.is_available():
                    device = 0
            except Exception:
                pass

            _classifier = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                device=device,
            )
    return _classifier


def classify_question(text, classifier_fn=None):
    classifier_fn = classifier_fn or get_classifier()
    prediction = classifier_fn(text, candidate_labels=SUBJECTS, multi_label=False)
    return str(prediction["labels"][0]), float(prediction["scores"][0])


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/stt-result", response_model=ClassificationResult)
async def accept_stt_result(data: STTResponse):
    predicted_topic, score = classify_question(data.text)
    return ClassificationResult(
        text=data.text,
        language=data.language,
        predicted_topic=predicted_topic,
        score=score,
    )
