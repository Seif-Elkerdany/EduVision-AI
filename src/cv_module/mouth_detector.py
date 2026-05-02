from __future__ import annotations

import contextlib
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
from urllib.request import urlopen


MEDIAPIPE_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


class MouthDetector:
    """Detects talking/mouth movement using MediaPipe Face Landmarker Tasks API."""

    def __init__(self, threshold: float = 0.015, consecutive_frames: int = 2) -> None:
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "data" / "models" / "mediapipe" / "face_landmarker.task"
        if not model_path.exists():
            self._download_model(model_path)

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.talking_frames = 0
        self.is_talking = False

    def _download_model(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"MediaPipe model not found. Downloading to {model_path} ...")
        try:
            with contextlib.closing(urlopen(MEDIAPIPE_FACE_LANDMARKER_URL, timeout=60)) as response:
                model_path.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(
                f"Missing MediaPipe model at {model_path}, and automatic download failed. "
                f"Download it manually from {MEDIAPIPE_FACE_LANDMARKER_URL}"
            ) from exc

    def process(self, frame: np.ndarray) -> bool:
        """Process a BGR frame and return True if any person is talking."""
        if frame is None or frame.size == 0:
            return self.is_talking

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)

        talking_detected_in_frame = False

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # Landmark points for MAR calculation
                # 13: inner top lip, 14: inner bottom lip
                # 78: left inner corner, 308: right inner corner
                top_lip = face_landmarks[13]
                bottom_lip = face_landmarks[14]
                left_corner = face_landmarks[78]
                right_corner = face_landmarks[308]

                # Vertical distance
                v_dist = abs(bottom_lip.y - top_lip.y)
                # Horizontal distance
                h_dist = abs(right_corner.x - left_corner.x)

                # Use Mouth Aspect Ratio (MAR) to filter out wide smiles
                # Speech usually has a higher vertical-to-horizontal ratio than a static smile
                mar = v_dist / h_dist if h_dist > 0 else 0

                # Also normalize vertical distance by face height for scale invariance
                top_head = face_landmarks[10]
                bottom_chin = face_landmarks[152]
                face_height = abs(bottom_chin.y - top_head.y)
                v_ratio = v_dist / face_height if face_height > 0 else 0

                # Trigger if mouth is open significantly (v_ratio) AND 
                # it's not just a wide/thin shape (mar)
                # Increasing v_ratio threshold to 0.025 to be less sensitive
                if v_ratio > 0.025 and mar > 0.2:
                    talking_detected_in_frame = True
                    break

        # State machine for stability
        if talking_detected_in_frame:
            self.talking_frames += 1
        else:
            self.talking_frames = max(0, self.talking_frames - 1)

        if self.talking_frames >= self.consecutive_frames:
            self.is_talking = True
            # Cap to prevent unbounded growth
            self.talking_frames = self.consecutive_frames
        elif self.talking_frames == 0:
            self.is_talking = False

        return self.is_talking

    def release(self) -> None:
        """Release MediaPipe resources."""
        self.detector.close()
