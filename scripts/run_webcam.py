from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module import CVConfig, SmartClassroomPipeline  # noqa: E402
from cv_module.config import detect_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the classroom CV pipeline on a webcam.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--embeddings-path", type=Path, default=PROJECT_ROOT / "data" / "embeddings" / "student_embeddings.pkl")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "data" / "logs")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--frame-skip", type=int, default=0)
    parser.add_argument("--recognition-threshold", type=float, default=0.45)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--disable-emotion", action="store_true")
    parser.add_argument("--save-face-crops", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CVConfig(
        device=args.device or detect_device(),
        embeddings_path=args.embeddings_path,
        logs_dir=args.logs_dir,
        frame_skip=args.frame_skip,
        recognition_threshold=args.recognition_threshold,
        top1_top2_margin_threshold=args.margin_threshold,
        emotion_enabled=not args.disable_emotion,
        save_face_crops=args.save_face_crops,
        webcam_index=args.camera_index,
    )
    pipeline = SmartClassroomPipeline(config)

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera_index}")

    summary = pipeline.run_capture(capture, display=not args.headless)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
