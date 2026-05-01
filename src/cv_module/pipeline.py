from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

import cv2

from .attendance import AttendanceDecision, AttendanceManager
from .config import CVConfig
from .detector import FaceDetector, crop_from_bbox
from .emotion import EmotionEstimator, EmotionResult
from .logger import SessionLogger
from .recognizer import FaceRecognizer, RecognitionResult
from .tracker import FaceTracker, TrackState
from .mouth_detector import MouthDetector


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameFaceResult:
    frame_id: int
    track_id: int
    bbox: tuple[int, int, int, int]
    display_name: str
    recognition: RecognitionResult
    mood: EmotionResult
    attendance: AttendanceDecision


class SmartClassroomPipeline:
    """End-to-end local computer vision pipeline for classroom attendance demos."""

    def __init__(
        self,
        config: CVConfig | None = None,
        detector: FaceDetector | None = None,
        recognizer: FaceRecognizer | None = None,
        emotion_estimator: EmotionEstimator | None = None,
        tracker: FaceTracker | None = None,
        attendance_manager: AttendanceManager | None = None,
        session_logger: SessionLogger | None = None,
    ) -> None:
        self.config = (config or CVConfig()).ensure_directories()
        self.detector = detector or FaceDetector(self.config)
        self.recognizer = recognizer or FaceRecognizer(self.config)
        self.emotion_estimator = emotion_estimator or EmotionEstimator(self.config)
        self.tracker = tracker or FaceTracker(self.config)
        self.attendance_manager = attendance_manager or AttendanceManager(self.config)
        self.session_logger = session_logger or SessionLogger(self.config)
        self.mouth_detector = MouthDetector()
        self.frame_index = 0
        self._last_results: list[FrameFaceResult] = []
        self._finalized = False
        self._summary_cache: dict[str, object] | None = None
        self._display_supported: bool | None = None

    def _make_timestamp(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _has_highgui_support(self) -> bool:
        if self._display_supported is not None:
            return self._display_supported

        try:
            build_info = cv2.getBuildInformation()
            self._display_supported = "GUI:                           NONE" not in build_info
        except Exception:
            self._display_supported = False

        return self._display_supported

    def _smooth_emotion(self, track: TrackState) -> EmotionResult:
        if not track.emotion_history:
            return EmotionResult.unknown("smoothed")

        combined: dict[str, float] = {}
        for emotion in track.emotion_history:
            for label, score in emotion.raw_probabilities.items():
                combined[label] = combined.get(label, 0.0) + float(score)

        if combined:
            count = max(1, len(track.emotion_history))
            averaged = {label: score / count for label, score in combined.items()}
            mood_label = max(averaged, key=averaged.get)
            return EmotionResult(
                mood_label=mood_label,
                mood_confidence=float(averaged[mood_label]),
                raw_probabilities=averaged,
                backend="temporal_average",
            )

        labels = [
            emotion.mood_label
            for emotion in track.emotion_history
            if emotion.mood_label != "unknown"
        ]
        if not labels:
            return EmotionResult.unknown("smoothed")
        mood_label, count = Counter(labels).most_common(1)[0]
        return EmotionResult(
            mood_label=mood_label,
            mood_confidence=count / max(1, len(track.emotion_history)),
            raw_probabilities={mood_label: count / max(1, len(track.emotion_history))},
            backend="majority_vote",
        )

    def _display_name(
        self,
        recognition: RecognitionResult,
        attendance: AttendanceDecision,
    ) -> str:
        if attendance.attendance_status in {"present_marked", "present"}:
            return attendance.stable_name
        return recognition.predicted_name

    def process_frame(self, frame) -> tuple[object, list[FrameFaceResult]]:
        frame_id = self.frame_index
        self.frame_index += 1

        if self.config.frame_skip and frame_id % (self.config.frame_skip + 1) != 0:
            annotated = self.draw_overlay(frame.copy(), self._last_results)
            return annotated, list(self._last_results)

        timestamp = self._make_timestamp()
        detections = self.detector.detect(frame)
        assignments = self.tracker.update(detections, frame_id)
        frame_results: list[FrameFaceResult] = []

        for assignment in assignments:
            track = assignment.track
            detection = assignment.detection
            recognition = self.recognizer.recognize(detection.embedding)
            track.append_recognition(recognition)

            face_crop = (
                detection.aligned_crop
                if detection.aligned_crop is not None
                else crop_from_bbox(frame, detection.bbox)
            )
            emotion = self.emotion_estimator.predict(face_crop)
            track.append_emotion(emotion)
            smoothed_emotion = self._smooth_emotion(track)

            attendance = self.attendance_manager.evaluate_track(track, timestamp)
            display_name = self._display_name(recognition, attendance)

            result = FrameFaceResult(
                frame_id=frame_id,
                track_id=track.track_id,
                bbox=detection.bbox,
                display_name=display_name,
                recognition=recognition,
                mood=smoothed_emotion,
                attendance=attendance,
            )
            frame_results.append(result)

            self.session_logger.log_detection(
                timestamp=timestamp,
                frame_id=frame_id,
                track_id=track.track_id,
                predicted_student=display_name,
                recognition_score=recognition.similarity_score,
                recognition_margin=recognition.margin_between_top_1_and_top_2,
                attendance_status=attendance.attendance_status,
                mood_label=smoothed_emotion.mood_label,
                mood_confidence=smoothed_emotion.mood_confidence,
                bbox=detection.bbox,
            )
            self.session_logger.save_face_crop(
                face_crop=detection.aligned_crop,
                frame_id=frame_id,
                track_id=track.track_id,
                predicted_student=display_name,
            )

        self._last_results = frame_results
        annotated = self.draw_overlay(frame.copy(), frame_results)
        return annotated, frame_results

    def draw_overlay(self, frame, results: list[FrameFaceResult]):
        for result in results:
            x1, y1, x2, y2 = result.bbox
            if result.attendance.attendance_status in {"present_marked", "present"}:
                color = (60, 180, 75)
            elif result.recognition.is_unknown:
                color = (0, 165, 255)
            else:
                color = (66, 135, 245)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            top_line = f"#{result.track_id} {result.display_name}"
            middle_line = (
                f"rec {result.recognition.similarity_score:.2f} | "
                f"{result.attendance.attendance_status}"
            )
            bottom_line = f"mood {result.mood.mood_label} {result.mood.mood_confidence:.2f}"

            cv2.putText(frame, top_line, (x1, max(20, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, middle_line, (x1, max(20, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            cv2.putText(frame, bottom_line, (x1, min(frame.shape[0] - 10, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        return frame

    def _build_writer(self, capture, output_path: str | None):
        if not output_path:
            return None

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output = cv2.VideoWriter(
            str(output_file),
            cv2.VideoWriter_fourcc(*self.config.video_codec),
            capture.get(cv2.CAP_PROP_FPS) or 20.0,
            (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )
        return output

    def run_capture(self, capture, display: bool = True, output_path: str | None = None, speech_trigger=None, mouth_active_event=None) -> dict[str, object]:
        if capture is None or not capture.isOpened():
            raise RuntimeError("Could not open the requested video source.")

        if display and not self._has_highgui_support():
            LOGGER.warning(
                "OpenCV was installed without GUI support. Continuing in headless mode. "
                "Use --headless explicitly or reinstall a GUI-enabled OpenCV build if you need a live window."
            )
            display = False

        writer = self._build_writer(capture, output_path)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                is_talking = self.mouth_detector.process(frame)
                
                # Signal events
                if is_talking:
                    if speech_trigger is not None:
                        speech_trigger.set()
                    if mouth_active_event is not None:
                        mouth_active_event.set()
                elif mouth_active_event is not None:
                    mouth_active_event.clear()

                annotated_frame, _ = self.process_frame(frame)
                
                if is_talking:
                    cv2.putText(annotated_frame, "VOICE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                if writer is not None:
                    writer.write(annotated_frame)

                if display:
                    try:
                        cv2.imshow(self.config.display_window_name, annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                    except cv2.error as exc:
                        LOGGER.warning(
                            "OpenCV display is unavailable at runtime. Continuing in headless mode: %s",
                            exc,
                        )
                        display = False
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            self.mouth_detector.release()

        return self.finalize()

    def finalize(self) -> dict[str, object]:
        if self._finalized and self._summary_cache is not None:
            return self._summary_cache

        summary_rows = self.attendance_manager.get_summary_rows()
        self.session_logger.write_attendance_summary(summary_rows)
        self._summary_cache = {
            "session_id": self.config.session_id,
            "attendance_count": len(summary_rows),
            "students_present": [row["student_name"] for row in summary_rows],
            "session_log_path": str(self.config.session_log_path),
            "attendance_log_path": str(self.config.attendance_log_path),
        }
        self._finalized = True
        return self._summary_cache
