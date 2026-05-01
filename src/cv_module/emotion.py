from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CVConfig


LOGGER = logging.getLogger(__name__)


EMOTION_LABEL_MAP = {
    "anger": "angry",
    "angry": "angry",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
    "tired": "tired",
    "sleepy": "tired",
    "fatigue": "tired",
    "contempt": "unknown",
    "disgust": "unknown",
    "fear": "unknown",
}


@dataclass(slots=True)
class EmotionResult:
    mood_label: str
    mood_confidence: float
    raw_probabilities: dict[str, float]
    backend: str

    @classmethod
    def unknown(cls, backend: str = "unavailable") -> "EmotionResult":
        return cls(
            mood_label="unknown",
            mood_confidence=0.0,
            raw_probabilities={"unknown": 1.0},
            backend=backend,
        )


class EmotionEstimator:
    """Lightweight facial-expression impression module with a clean fallback path."""

    def __init__(self, config: CVConfig) -> None:
        self.config = config
        self._model = None
        self._load_attempted = False
        self.backend_name = "disabled" if not config.emotion_enabled else "pending"

    def _load_backend(self) -> None:
        if self._load_attempted or not self.config.emotion_enabled:
            return

        self._load_attempted = True

        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer

            # The ONNX backend in EmotiEffLib uses CPU execution internally, which keeps
            # this module lightweight and avoids spending VRAM on expression estimation.
            device = self.config.device if self.config.emotion_engine == "torch" else "cpu"
            self._model = EmotiEffLibRecognizer(
                engine=self.config.emotion_engine,
                model_name=self.config.emotion_model_name,
                device=device,
            )
            self.backend_name = f"emotiefflib:{self.config.emotion_engine}"
        except Exception as exc:
            self._model = None
            self.backend_name = "fallback_unknown"
            LOGGER.warning(
                "Emotion backend could not be initialized. The pipeline will emit unknown mood impressions. Reason: %s",
                exc,
            )

    def _map_probabilities(self, source_probabilities: dict[str, float]) -> dict[str, float]:
        mapped = {
            "happy": 0.0,
            "neutral": 0.0,
            "tired": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "unknown": 0.0,
        }

        for label, score in source_probabilities.items():
            target_label = EMOTION_LABEL_MAP.get(label.lower(), "unknown")
            mapped[target_label] += float(score)

        total = sum(mapped.values())
        if total <= 0:
            return {"unknown": 1.0}

        return {label: score / total for label, score in mapped.items() if score > 0}

    def predict(self, face_crop_bgr: np.ndarray | None) -> EmotionResult:
        if not self.config.emotion_enabled:
            return EmotionResult.unknown("disabled")
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return EmotionResult.unknown(self.backend_name)

        self._load_backend()
        if self._model is None:
            return EmotionResult.unknown(self.backend_name)

        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        try:
            labels, scores = self._model.predict_emotions(face_crop_rgb, logits=False)
            probability_vector = scores[0] if np.asarray(scores).ndim > 1 else np.asarray(scores)
            label_names = getattr(self._model, "idx_to_emotion_class", {})
            source_probabilities = {
                label_names[index]: float(probability_vector[index])
                for index in range(min(len(probability_vector), len(label_names)))
            }
            mapped_probabilities = self._map_probabilities(source_probabilities)
            mood_label = max(mapped_probabilities, key=mapped_probabilities.get)
            mood_confidence = float(mapped_probabilities[mood_label])
            return EmotionResult(
                mood_label=mood_label,
                mood_confidence=mood_confidence,
                raw_probabilities=mapped_probabilities,
                backend=self.backend_name,
            )
        except Exception as exc:
            LOGGER.warning("Emotion inference failed for the current crop: %s", exc)
            return EmotionResult.unknown(self.backend_name)
