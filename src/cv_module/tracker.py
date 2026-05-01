from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot

from .config import CVConfig
from .detector import DetectedFace
from .emotion import EmotionResult
from .recognizer import RecognitionResult


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection
    return intersection / union


def centroid_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    return hypot(ax - bx, ay - by)


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    history_size: int
    age: int = 1
    missing_count: int = 0
    last_seen_frame: int = 0
    recognition_history: deque[RecognitionResult] = field(init=False)
    emotion_history: deque[EmotionResult] = field(init=False)
    last_recognition: RecognitionResult = field(init=False)
    last_emotion: EmotionResult = field(init=False)

    def __post_init__(self) -> None:
        self.recognition_history = deque(maxlen=self.history_size)
        self.emotion_history = deque(maxlen=self.history_size)
        self.last_recognition = RecognitionResult.unknown()
        self.last_emotion = EmotionResult.unknown("tracker")

    def append_recognition(self, result: RecognitionResult) -> None:
        self.last_recognition = result
        self.recognition_history.append(result)

    def append_emotion(self, result: EmotionResult) -> None:
        self.last_emotion = result
        self.emotion_history.append(result)


@dataclass(slots=True)
class TrackAssignment:
    track: TrackState
    detection: DetectedFace


class FaceTracker:
    """Lightweight IoU and centroid based tracker for short classroom scenes."""

    def __init__(self, config: CVConfig) -> None:
        self.config = config
        self.tracks: dict[int, TrackState] = {}
        self._next_track_id = 1

    def _new_track(self, detection: DetectedFace, frame_index: int) -> TrackState:
        track = TrackState(
            track_id=self._next_track_id,
            bbox=detection.bbox,
            history_size=self.config.smoothing_window,
            last_seen_frame=frame_index,
        )
        self.tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _candidate_matches(self, detections: list[DetectedFace]) -> list[tuple[float, float, int, int]]:
        candidates: list[tuple[float, float, int, int]] = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                iou = bbox_iou(track.bbox, detection.bbox)
                distance = centroid_distance(track.bbox, detection.bbox)
                if iou >= self.config.tracking_iou_threshold or distance <= self.config.tracking_centroid_threshold:
                    candidates.append((iou, -distance, track_id, detection_index))
        candidates.sort(reverse=True)
        return candidates

    def update(self, detections: list[DetectedFace], frame_index: int) -> list[TrackAssignment]:
        assignments_by_detection: dict[int, TrackAssignment] = {}
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for _iou, _distance, track_id, detection_index in self._candidate_matches(detections):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue

            track = self.tracks[track_id]
            detection = detections[detection_index]
            track.bbox = detection.bbox
            track.age += 1
            track.missing_count = 0
            track.last_seen_frame = frame_index
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments_by_detection[detection_index] = TrackAssignment(track=track, detection=detection)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = self._new_track(detection, frame_index)
            assignments_by_detection[detection_index] = TrackAssignment(track=track, detection=detection)
            matched_tracks.add(track.track_id)

        expired_tracks: list[int] = []
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue
            track.missing_count += 1
            if track.missing_count > self.config.max_missing_frames:
                expired_tracks.append(track_id)

        for track_id in expired_tracks:
            self.tracks.pop(track_id, None)

        return [assignments_by_detection[index] for index in sorted(assignments_by_detection)]
