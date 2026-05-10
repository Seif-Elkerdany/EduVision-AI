import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient


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
    students_dir = tmp_path / "students"
    attendance_path = tmp_path / "attendance_log.csv"
    session_path = tmp_path / "session_log.jsonl"
    questions_path = tmp_path / "questions_log.jsonl"
    vision_sessions_path = tmp_path / "vision_sessions.jsonl"
    enrollment_report_path = tmp_path / "enrollment_report.json"

    monkeypatch.setattr(api, "EMBEDDINGS_PATH", embeddings_path)
    monkeypatch.setattr(api, "STUDENTS_DIR", students_dir)
    monkeypatch.setattr(api, "ATTENDANCE_CSV", attendance_path)
    monkeypatch.setattr(api, "SESSION_JSONL", session_path)
    monkeypatch.setattr(api, "QUESTIONS_JSONL", questions_path)
    monkeypatch.setattr(api, "VISION_SESSIONS_JSONL", vision_sessions_path)
    monkeypatch.setattr(api, "ENROLLMENT_REPORT_JSON", enrollment_report_path)
    monkeypatch.setattr(api, "_active_vision_sessions", set())

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


def test_delete_session_removes_attendance_and_vision_logs(monkeypatch, tmp_path):
    attendance_path, session_path, vision_sessions_path = configure_paths(monkeypatch, tmp_path)
    deleted_sid = "20260510_193000"
    kept_sid = "20260510_194500"

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
            "session_id": deleted_sid,
            "student_name": "Alice",
            "attendance_status": "Present",
            "confirmed_at": "2026-05-10T19:30:10+03:00",
            "first_track_id": 1,
        })
        writer.writerow({
            "session_id": kept_sid,
            "student_name": "Bob",
            "attendance_status": "Present",
            "confirmed_at": "2026-05-10T19:45:10+03:00",
            "first_track_id": 2,
        })

    session_path.write_text(
        "\n".join([
            json.dumps({"session_id": deleted_sid, "predicted_student": "Alice"}),
            json.dumps({"session_id": kept_sid, "predicted_student": "Bob"}),
        ]) + "\n",
        encoding="utf-8",
    )
    vision_sessions_path.write_text(
        "\n".join([
            json.dumps({"session_id": deleted_sid, "attendance_count": 1}),
            json.dumps({"session_id": kept_sid, "attendance_count": 1}),
        ]) + "\n",
        encoding="utf-8",
    )

    result = api.delete_session(deleted_sid)

    assert result["status"] == "deleted"
    assert result["removed"]["attendance_rows"] == 1
    assert result["removed"]["session_log_rows"] == 1
    assert result["removed"]["vision_session_rows"] == 1

    sessions = api.get_sessions_with_absences()
    assert [session["id"] for session in sessions] == [kept_sid]


def test_enroll_student_endpoint_saves_embedding_and_updates_roster(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    real_config = api.CVConfig

    def fake_config(**kwargs):
        config = real_config(**kwargs)
        config.students_dir = tmp_path / "students"
        config.embeddings_path = api.EMBEDDINGS_PATH
        return config

    class FakeDetection:
        detection_confidence = 0.95
        min_side = 96
        embedding = np.array([0.4, 0.6], dtype=np.float32)

    class FakeDetector:
        def __init__(self, config):
            self.config = config

        def detect(self, image, apply_filters=False):
            assert apply_filters is False
            return [FakeDetection()]

    monkeypatch.setattr(api, "CVConfig", fake_config)
    monkeypatch.setattr(api, "FaceDetector", FakeDetector)
    monkeypatch.setattr(api, "detect_device", lambda: "cpu")

    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    client = TestClient(api.app)
    response = client.post(
        "/api/enroll-student",
        data={
            "student_name": "Dana Sayed",
            "blur_threshold": "0",
            "min_detection_score": "0.55",
            "min_face_size": "50",
        },
        files=[("images", ("dana.jpg", encoded.tobytes(), "image/jpeg"))],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "enrolled"
    assert body["accepted"] == 1
    assert body["rejected"] == 0

    roster_names = [student["name"] for student in api.get_roster()]
    assert "Dana Sayed" in roster_names
    assert json.loads(api.ENROLLMENT_REPORT_JSON.read_text(encoding="utf-8"))[-1]["status"] == "accepted"


def test_delete_student_removes_embedding_and_image_folder(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)
    student_dir = api.STUDENTS_DIR / "Alice"
    student_dir.mkdir(parents=True)
    (student_dir / "alice.jpg").write_bytes(b"fake image")

    client = TestClient(api.app)
    response = client.delete("/api/students/Alice")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deleted"
    assert body["student"] == "Alice"
    assert body["removed"]["embedding_samples"] == 1
    assert body["removed"]["image_dirs"] == 1

    roster_names = [student["name"] for student in api.get_roster()]
    assert "Alice" not in roster_names
    assert not student_dir.exists()
