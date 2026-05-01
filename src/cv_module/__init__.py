"""Computer vision pipeline package for EduVision-AI."""

from .attendance import AttendanceDecision, AttendanceManager
from .config import CVConfig
from .detector import DetectedFace, FaceDetector, crop_from_bbox
from .emotion import EmotionEstimator, EmotionResult
from .logger import SessionLogger
from .pipeline import FrameFaceResult, SmartClassroomPipeline
from .recognizer import (
    FaceRecognizer,
    RecognitionResult,
    StudentRecord,
    load_embedding_database,
    save_embedding_database,
)
from .tracker import FaceTracker, TrackAssignment, TrackState

__all__ = [
    "AttendanceDecision",
    "AttendanceManager",
    "CVConfig",
    "DetectedFace",
    "EmotionEstimator",
    "EmotionResult",
    "FaceDetector",
    "FaceRecognizer",
    "FaceTracker",
    "FrameFaceResult",
    "RecognitionResult",
    "SessionLogger",
    "SmartClassroomPipeline",
    "StudentRecord",
    "TrackAssignment",
    "TrackState",
    "crop_from_bbox",
    "load_embedding_database",
    "save_embedding_database",
]
