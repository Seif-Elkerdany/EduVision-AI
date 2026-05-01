import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module.config import CVConfig
from cv_module.detector import DetectedFace
from cv_module.tracker import FaceTracker


def make_detection(bbox):
    return DetectedFace(
        bbox=bbox,
        landmarks=None,
        detection_confidence=0.95,
        aligned_crop=None,
        embedding=None,
    )


def test_tracker_reuses_track_id_for_close_detections():
    tracker = FaceTracker(CVConfig(smoothing_window=5, max_missing_frames=2))

    first_assignments = tracker.update([make_detection((10, 10, 50, 50))], frame_index=0)
    second_assignments = tracker.update([make_detection((12, 11, 52, 51))], frame_index=1)

    assert len(first_assignments) == 1
    assert len(second_assignments) == 1
    assert first_assignments[0].track.track_id == second_assignments[0].track.track_id


def test_tracker_removes_track_after_missing_too_many_frames():
    tracker = FaceTracker(CVConfig(smoothing_window=5, max_missing_frames=1))

    tracker.update([make_detection((10, 10, 50, 50))], frame_index=0)
    tracker.update([], frame_index=1)
    tracker.update([], frame_index=2)

    assert tracker.tracks == {}
