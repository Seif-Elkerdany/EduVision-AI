import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import api
from cv_module.recognizer import save_embedding_database


def configure_paths(monkeypatch, tmp_path):
    embeddings_path = tmp_path / "student_embeddings.pkl"
    attendance_path = tmp_path / "attendance_log.csv"
    session_path = tmp_path / "session_log.jsonl"
    questions_path = tmp_path / "questions_log.jsonl"
    vision_sessions_path = tmp_path / "vision_sessions.jsonl"

    monkeypatch.setattr(api, "EMBEDDINGS_PATH", embeddings_path)
    monkeypatch.setattr(api, "ATTENDANCE_CSV", attendance_path)
    monkeypatch.setattr(api, "SESSION_JSONL", session_path)
    monkeypatch.setattr(api, "QUESTIONS_JSONL", questions_path)
    monkeypatch.setattr(api, "VISION_SESSIONS_JSONL", vision_sessions_path)

    save_embedding_database(
        {
            "Alice": [np.array([1.0, 0.0], dtype=np.float32)],
            "Bob": [np.array([0.0, 1.0], dtype=np.float32)],
            "Charlie": [np.array([0.7, 0.7], dtype=np.float32)],
        },
        embeddings_path,
    )

    return attendance_path, session_path, vision_sessions_path


def test_sessions_include_absent_enrolled_students(monkeypatch, tmp_path):
    attendance_path, session_path, _ = configure_paths(monkeypatch, tmp_path)
    sid = "20260510_193000"

    with attendance_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "session_id",
                "student_name",
                "attendance_status",
                "confirmed_at",
                "first_track_id",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "session_id": sid,
            "student_name": "Alice",
            "attendance_status": "Present",
            "confirmed_at": "2026-05-10T19:30:10+03:00",
            "first_track_id": 1,
        })
        writer.writerow({
            "session_id": sid,
            "student_name": "Alice",
            "attendance_status": "Present",
            "confirmed_at": "2026-05-10T19:30:10+03:00",
            "first_track_id": 1,
        })

    session_path.write_text(
        json.dumps({
            "session_id": sid,
            "predicted_student": "Alice",
            "mood_label": "happy",
        }) + "\n",
        encoding="utf-8",
    )

    [session] = api.get_sessions_with_absences()
    by_name = {student["name"]: student for student in session["students"]}

    assert by_name["Alice"]["status"] == "Present"
    assert by_name["Bob"]["status"] == "Absent"
    assert by_name["Charlie"]["status"] == "Absent"
    assert len(session["students"]) == 3


def test_zero_attendance_session_marks_roster_absent(monkeypatch, tmp_path):
    _, _, vision_sessions_path = configure_paths(monkeypatch, tmp_path)
    sid = "20260510_194500"
    vision_sessions_path.write_text(
        json.dumps({
            "session_id": sid,
            "finalized_at": "2026-05-10T19:46:00+03:00",
            "frame_count": 20,
            "attendance_count": 0,
        }) + "\n",
        encoding="utf-8",
    )

    [session] = api.get_sessions_with_absences()

    assert {student["status"] for student in session["students"]} == {"Absent"}
    assert {student["name"] for student in session["students"]} == {
        "Alice",
        "Bob",
        "Charlie",
    }
