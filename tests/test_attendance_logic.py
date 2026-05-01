import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module.attendance import AttendanceManager
from cv_module.config import CVConfig
from cv_module.recognizer import RecognitionResult
from cv_module.tracker import TrackState


def make_result(name: str, unknown: bool = False) -> RecognitionResult:
    return RecognitionResult(
        predicted_name=name,
        similarity_score=0.85 if not unknown else 0.0,
        top_2_score=0.20 if not unknown else 0.0,
        margin_between_top_1_and_top_2=0.65 if not unknown else 0.0,
        is_unknown=unknown,
    )


def test_attendance_marks_present_after_temporal_confirmation():
    config = CVConfig(min_confirmed_frames=3, smoothing_window=5)
    manager = AttendanceManager(config)
    track = TrackState(track_id=1, bbox=(0, 0, 10, 10), history_size=config.smoothing_window)

    track.append_recognition(make_result("Alice"))
    track.append_recognition(make_result("Unknown", unknown=True))
    track.append_recognition(make_result("Alice"))
    track.append_recognition(make_result("Alice"))

    decision = manager.evaluate_track(track, "2026-04-21T22:00:00+02:00")

    assert decision.attendance_status == "present_marked"
    assert decision.stable_name == "Alice"
    assert "Alice" in manager.present_students


def test_attendance_does_not_duplicate_same_student_within_session():
    config = CVConfig(min_confirmed_frames=2, smoothing_window=4)
    manager = AttendanceManager(config)
    track = TrackState(track_id=1, bbox=(0, 0, 10, 10), history_size=config.smoothing_window)

    track.append_recognition(make_result("Bob"))
    track.append_recognition(make_result("Bob"))

    first_decision = manager.evaluate_track(track, "2026-04-21T22:00:00+02:00")
    second_decision = manager.evaluate_track(track, "2026-04-21T22:00:05+02:00")

    assert first_decision.attendance_status == "present_marked"
    assert second_decision.attendance_status == "present"
    assert len(manager.present_students) == 1
