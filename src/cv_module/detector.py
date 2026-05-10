from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import CVConfig, preload_onnxruntime_cuda_dlls


LOGGER = logging.getLogger(__name__)


def _normalize_embedding(embedding: np.ndarray | None) -> np.ndarray | None:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return None
    return vector / norm


def _clip_bbox(bbox: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(int).tolist()
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return (x1, y1, x2, y2)


def crop_from_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return a clipped BGR crop from a bounding box."""
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2].copy()


@dataclass(slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray | None
    detection_confidence: float
    aligned_crop: np.ndarray | None
    embedding: np.ndarray | None
    raw_face: Any | None = None

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)


class FaceDetector:
    """InsightFace-backed face detection, alignment, and embedding extraction."""

    def __init__(self, config: CVConfig) -> None:
        self.config = config.ensure_directories()
        self._app: Any | None = None
        self.runtime_device = self.config.device

    def _build_app(self, device: str, providers: tuple[str, ...]) -> Any:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace is not installed. Install requirements in the EduVision venv first."
            ) from exc

        if device == "cuda":
            preload_onnxruntime_cuda_dlls()

        app = FaceAnalysis(
            name=self.config.insightface_model_pack,
            root=str(self.config.model_root),
            allowed_modules=["detection", "recognition"],
            providers=list(providers),
        )
        ctx_id = 0 if device == "cuda" else -1
        app.prepare(ctx_id=ctx_id, det_size=self.config.detector_input_size)
        return app

    def _get_app(self) -> Any:
        if self._app is not None:
            return self._app

        try:
            self._app = self._build_app(self.config.device, self.config.ort_providers or ())
            self.runtime_device = self.config.device
        except Exception as exc:
            if self.config.device != "cuda":
                raise RuntimeError(
                    "Failed to initialize InsightFace. Check model download access and ONNX Runtime."
                ) from exc

            LOGGER.warning("InsightFace CUDA initialization failed, falling back to CPU: %s", exc)
            self._app = self._build_app("cpu", ("CPUExecutionProvider",))
            self.runtime_device = "cpu"

        return self._app

    def _build_aligned_crop(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray | None,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        if landmarks is not None and landmarks.shape == (5, 2):
            try:
                from insightface.utils import face_align

                return face_align.norm_crop(
                    frame,
                    landmarks.astype(np.float32),
                    image_size=self.config.alignment_size,
                )
            except Exception as exc:
                LOGGER.debug("Landmark alignment failed, using bbox crop instead: %s", exc)

        crop = crop_from_bbox(frame, bbox)
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.config.alignment_size, self.config.alignment_size))

    def _extract_embedding(self, face: Any) -> np.ndarray | None:
        if getattr(face, "normed_embedding", None) is not None:
            return _normalize_embedding(face.normed_embedding)
        if getattr(face, "embedding", None) is not None:
            return _normalize_embedding(face.embedding)
        return None

    def detect(self, frame: np.ndarray, apply_filters: bool = True) -> list[DetectedFace]:
        """Run detection and return aligned face metadata for the current frame."""
        if frame is None or frame.size == 0:
            return []

        faces = self._get_app().get(frame)
        detected_faces: list[DetectedFace] = []

        for face in faces:
            bbox = _clip_bbox(np.asarray(face.bbox, dtype=np.float32), frame.shape)
            landmarks = None
            if getattr(face, "kps", None) is not None:
                landmarks = np.asarray(face.kps, dtype=np.float32)

            detected_face = DetectedFace(
                bbox=bbox,
                landmarks=landmarks,
                detection_confidence=float(getattr(face, "det_score", 0.0)),
                aligned_crop=self._build_aligned_crop(frame, landmarks, bbox),
                embedding=self._extract_embedding(face),
                raw_face=face,
            )

            if apply_filters:
                if detected_face.detection_confidence < self.config.detection_confidence_threshold:
                    continue
                if detected_face.min_side < self.config.min_face_size:
                    continue

            detected_faces.append(detected_face)

        return detected_faces
