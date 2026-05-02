from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module import CVConfig, SmartClassroomPipeline
from cv_module.config import detect_device
from speech_module.Speech_recognition import push_to_talk_pipeline

def speech_worker(trigger_event: threading.Event, mouth_active_event: threading.Event) -> None:
    """Background thread to run the speech pipeline continuously."""
    print("🎙️ Speech worker thread started.")
    while True:
        try:
            push_to_talk_pipeline(trigger_event, mouth_active_event)
        except Exception as e:
            print(f"❌ Speech worker error: {e}")
            import time
            time.sleep(2)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the combined Smart Classroom CV and Speech pipeline.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--frame-skip", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    
    # 1. Initialize shared trigger events
    speech_trigger = threading.Event()
    mouth_active_event = threading.Event()
    
    # 2. Start speech module in a background thread
    speech_thread = threading.Thread(
        target=speech_worker, 
        args=(speech_trigger, mouth_active_event), 
        daemon=True
    )
    speech_thread.start()
    
    # 3. Initialize CV Pipeline
    config = CVConfig(
        device=args.device or detect_device(),
        webcam_index=args.camera_index,
        frame_skip=args.frame_skip,
    )
    pipeline = SmartClassroomPipeline(config)

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera_index}")

    print("🚀 Starting Smart Classroom Pipeline... (Press ESC or 'q' to exit)")
    
    # 4. Run the CV loop in the main thread (blocks until window closed)
    try:
        pipeline.run_capture(
            capture, 
            display=not args.headless, 
            speech_trigger=speech_trigger,
            mouth_active_event=mouth_active_event
        )
    finally:
        print("👋 Shutting down...")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
