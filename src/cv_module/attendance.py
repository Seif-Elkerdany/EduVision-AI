from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .config import CVConfig
from .tracker import TrackState


@dataclass(slots=True)
class AttendanceDecision:
    attendance_status: str
    stable_name: str
    confirmed_frames: int
    window_size: int
    already_present: bool = False


class AttendanceManager:
    """Session-level attendance confirmation with temporal smoothing."""

    def __init__(self, config: CVConfig) -> None:
        self.config = config
        self.present_students: dict[str, dict[str, object]] = {}

    def evaluate_track(self, track: TrackState, timestamp: str) -> AttendanceDecision:
        recognized_names = [
            result.predicted_name
            for result in track.recognition_history
            if not result.is_unknown and result.predicted_name != "Unknown"
        ]

        if not recognized_names:
            return AttendanceDecision(
                attendance_status="unknown",
                stable_name="Unknown",
                confirmed_frames=0,
                window_size=len(track.recognition_history),
            )

        stable_name, confirmed_frames = Counter(recognized_names).most_common(1)[0]
        if confirmed_frames < self.config.min_confirmed_frames:
            return AttendanceDecision(
                attendance_status="observing",
                stable_name=stable_name,
                confirmed_frames=confirmed_frames,
                window_size=len(track.recognition_history),
            )

        if stable_name in self.present_students:
            return AttendanceDecision(
                attendance_status="present",
                stable_name=stable_name,
                confirmed_frames=confirmed_frames,
                window_size=len(track.recognition_history),
                already_present=True,
            )

        self.present_students[stable_name] = {
            "student_name": stable_name,
            "attendance_status": "Present",
            "confirmed_at": timestamp,
            "first_track_id": track.track_id,
            "session_id": self.config.session_id,
        }
        return AttendanceDecision(
            attendance_status="present_marked",
            stable_name=stable_name,
            confirmed_frames=confirmed_frames,
            window_size=len(track.recognition_history),
        )

    def get_summary_rows(self) -> list[dict[str, object]]:
        return [
            self.present_students[student_name]
            for student_name in sorted(self.present_students)
        ]
