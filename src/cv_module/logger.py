from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from .config import CVConfig


class SessionLogger:
    """Append-only logger for per-frame events and per-session attendance summaries."""

    def __init__(self, config: CVConfig) -> None:
        self.config = config.ensure_directories()
        self.session_id = self.config.session_id

    def log_detection(
        self,
        *,
        timestamp: str,
        frame_id: int,
        track_id: int,
        predicted_student: str,
        recognition_score: float,
        recognition_margin: float,
        attendance_status: str,
        mood_label: str,
        mood_confidence: float,
        bbox: tuple[int, int, int, int],
    ) -> None:
        record = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "frame_id": frame_id,
            "track_id": track_id,
            "predicted_student": predicted_student,
            "recognition_score": round(float(recognition_score), 4),
            "recognition_margin": round(float(recognition_margin), 4),
            "attendance_status": attendance_status,
            "mood_label": mood_label,
            "mood_confidence": round(float(mood_confidence), 4),
            "bbox": list(bbox),
        }

        with self.config.session_log_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def write_attendance_summary(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return

        csv_path = self.config.attendance_log_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        existing_keys: set[tuple[str, str]] = set()
        fieldnames = [
            "session_id",
            "student_name",
            "attendance_status",
            "confirmed_at",
            "first_track_id",
        ]

        if csv_path.exists() and csv_path.stat().st_size > 0:
            with csv_path.open(encoding="utf-8") as file_handle:
                for row in csv.DictReader(file_handle):
                    existing_keys.add((row.get("session_id", ""), row.get("student_name", "")))

        with csv_path.open("a", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                key = (str(row.get("session_id", "")), str(row.get("student_name", "")))
                if key in existing_keys:
                    continue
                writer.writerow(row)
                existing_keys.add(key)

    def save_face_crop(
        self,
        *,
        face_crop,
        frame_id: int,
        track_id: int,
        predicted_student: str,
    ) -> Path | None:
        if not self.config.save_face_crops or face_crop is None:
            return None

        self.config.face_crops_dir.mkdir(parents=True, exist_ok=True)
        safe_name = predicted_student.replace(" ", "_")
        output_path = self.config.face_crops_dir / f"frame_{frame_id:06d}_track_{track_id}_{safe_name}.jpg"
        cv2.imwrite(str(output_path), face_crop)
        return output_path
