from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def generate_session_id() -> str:
    """Create a session id that is safe to use in logs and filenames."""
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def preload_onnxruntime_cuda_dlls() -> None:
    """Best-effort preload for CUDA/cuDNN DLLs used by ONNX Runtime on Windows."""
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    try:
        import onnxruntime as ort

        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload()
            except Exception:
                preload(directory="")
    except Exception:
        pass


def detect_device() -> str:
    """Prefer CUDA when ONNX Runtime or Torch reports GPU support."""
    preload_onnxruntime_cuda_dlls()

    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in set(ort.get_available_providers()):
            return "cuda"
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def default_ort_providers(device: str) -> tuple[str, ...]:
    if device == "cuda":
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


@dataclass(slots=True)
class CVConfig:
    """Central configuration for the computer vision pipeline."""

    insightface_model_pack: str = "buffalo_l"
    device: str = field(default_factory=detect_device)
    detector_input_size: tuple[int, int] = (640, 640)
    detection_confidence_threshold: float = 0.60
    recognition_threshold: float = 0.45
    top1_top2_margin_threshold: float = 0.05
    min_face_size: int = 60
    alignment_size: int = 112
    frame_skip: int = 0
    tracking_iou_threshold: float = 0.30
    tracking_centroid_threshold: float = 120.0
    max_missing_frames: int = 10
    smoothing_window: int = 10
    min_confirmed_frames: int = 6
    emotion_enabled: bool = True
    emotion_engine: str = "onnx"
    emotion_model_name: str = "enet_b0_8_best_vgaf"
    save_face_crops: bool = False
    blur_variance_threshold: float = 100.0
    enrollment_min_detection_score: float = 0.70
    enrollment_min_face_size: int = 80
    model_root: Path = DATA_DIR / "models" / "insightface"
    students_dir: Path = DATA_DIR / "students"
    logs_dir: Path = DATA_DIR / "logs"
    embeddings_path: Path = DATA_DIR / "embeddings" / "student_embeddings.pkl"
    session_id: str = field(default_factory=generate_session_id)
    display_window_name: str = "EduVision Smart Classroom"
    video_codec: str = "mp4v"
    webcam_index: int = 0
    ort_providers: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.model_root = Path(self.model_root)
        self.students_dir = Path(self.students_dir)
        self.logs_dir = Path(self.logs_dir)
        self.embeddings_path = Path(self.embeddings_path)
        self.device = self.device.lower()
        self.emotion_engine = self.emotion_engine.lower()
        if self.insightface_model_pack != "buffalo_l":
            raise ValueError(
                "This project is configured to use only the InsightFace buffalo_l model pack."
            )
        if self.ort_providers is None:
            self.ort_providers = default_ort_providers(self.device)

    @property
    def session_log_path(self) -> Path:
        return self.logs_dir / "session_log.jsonl"

    @property
    def attendance_log_path(self) -> Path:
        return self.logs_dir / "attendance_log.csv"

    @property
    def face_crops_dir(self) -> Path:
        return self.logs_dir / "face_crops" / self.session_id

    def ensure_directories(self) -> "CVConfig":
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.students_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_face_crops:
            self.face_crops_dir.mkdir(parents=True, exist_ok=True)
        return self
