from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import CVConfig


LOGGER = logging.getLogger(__name__)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if norm <= 0:
            raise ValueError("Embedding norm must be greater than zero.")
        return array / norm

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return array / norms


@dataclass(slots=True)
class StudentRecord:
    name: str
    raw_embeddings: np.ndarray
    mean_embedding: np.ndarray
    sample_paths: list[str]


@dataclass(slots=True)
class RecognitionResult:
    predicted_name: str
    similarity_score: float
    top_2_score: float
    margin_between_top_1_and_top_2: float
    is_unknown: bool

    @classmethod
    def unknown(cls) -> "RecognitionResult":
        return cls(
            predicted_name="Unknown",
            similarity_score=0.0,
            top_2_score=0.0,
            margin_between_top_1_and_top_2=0.0,
            is_unknown=True,
        )


def save_embedding_database(
    student_embeddings: Mapping[str, Sequence[np.ndarray]],
    output_path: Path | str,
    sample_paths: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Persist raw and mean embeddings for every enrolled student."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized_students: dict[str, dict[str, object]] = {}
    sample_paths = sample_paths or {}

    for student_name, embeddings in student_embeddings.items():
        normalized_embeddings = [l2_normalize(np.asarray(embedding)) for embedding in embeddings]
        if not normalized_embeddings:
            continue

        raw_embeddings = np.vstack(normalized_embeddings).astype(np.float32)
        mean_embedding = l2_normalize(raw_embeddings.mean(axis=0)).astype(np.float32)
        serialized_students[student_name] = {
            "raw_embeddings": raw_embeddings,
            "mean_embedding": mean_embedding,
            "sample_paths": list(sample_paths.get(student_name, [])),
            "num_samples": int(raw_embeddings.shape[0]),
        }

    payload: dict[str, object] = {
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "students": serialized_students,
    }

    with output_path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)

    return payload


def load_embedding_database(input_path: Path | str) -> dict[str, object]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Embedding database was not found: {input_path}")

    with input_path.open("rb") as file_handle:
        payload = pickle.load(file_handle)

    return payload


class FaceRecognizer:
    """Cosine-similarity based student recognizer using ArcFace-style embeddings."""

    def __init__(
        self,
        config: CVConfig,
        embeddings_path: Path | str | None = None,
        auto_load: bool = True,
    ) -> None:
        self.config = config
        self.embeddings_path = Path(embeddings_path or self.config.embeddings_path)
        self.records: dict[str, StudentRecord] = {}
        if auto_load:
            self.load_if_exists()

    def load_if_exists(self) -> None:
        if not self.embeddings_path.exists():
            LOGGER.warning(
                "Embedding database not found at %s. Recognition will return Unknown until enrollment runs.",
                self.embeddings_path,
            )
            return
        self.load()

    def load(self) -> None:
        payload = load_embedding_database(self.embeddings_path)
        students = payload.get("students", {})
        self.records = {}

        for student_name, serialized in students.items():
            raw_embeddings = l2_normalize(np.asarray(serialized["raw_embeddings"], dtype=np.float32))
            mean_embedding = l2_normalize(np.asarray(serialized["mean_embedding"], dtype=np.float32))
            sample_paths = list(serialized.get("sample_paths", []))
            self.records[student_name] = StudentRecord(
                name=student_name,
                raw_embeddings=raw_embeddings,
                mean_embedding=mean_embedding,
                sample_paths=sample_paths,
            )

    def recognize(self, embedding: np.ndarray | None) -> RecognitionResult:
        if embedding is None or not self.records:
            return RecognitionResult.unknown()

        query = l2_normalize(np.asarray(embedding, dtype=np.float32))
        scores: list[tuple[str, float]] = []

        for student_name, record in self.records.items():
            raw_similarity = float(np.max(record.raw_embeddings @ query))
            mean_similarity = float(record.mean_embedding @ query)
            student_score = max(raw_similarity, mean_similarity)
            scores.append((student_name, student_score))

        if not scores:
            return RecognitionResult.unknown()

        scores.sort(key=lambda item: item[1], reverse=True)
        top_name, top_score = scores[0]
        top_2_score = scores[1][1] if len(scores) > 1 else 0.0
        margin = top_score - top_2_score

        is_unknown = top_score < self.config.recognition_threshold
        if len(scores) > 1 and margin < self.config.top1_top2_margin_threshold:
            is_unknown = True

        return RecognitionResult(
            predicted_name="Unknown" if is_unknown else top_name,
            similarity_score=float(top_score),
            top_2_score=float(top_2_score),
            margin_between_top_1_and_top_2=float(margin),
            is_unknown=is_unknown,
        )
