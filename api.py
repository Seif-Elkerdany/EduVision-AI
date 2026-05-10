"""
EduVision-AI — unified API + frontend server.

Start with:
    python api.py

Then open http://localhost:8000 in your browser.
"""
from __future__ import annotations

import asyncio
import csv
import json
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module import CVConfig, FaceDetector, SmartClassroomPipeline, save_embedding_database
from cv_module.config import detect_device
from cv_module.recognizer import load_embedding_database

# file paths needed
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "student_embeddings.pkl"
STUDENTS_DIR    = PROJECT_ROOT / "data" / "students"
ATTENDANCE_CSV  = PROJECT_ROOT / "data" / "logs" / "attendance_log.csv"
SESSION_JSONL   = PROJECT_ROOT / "data" / "logs" / "session_log.jsonl"
QUESTIONS_JSONL = PROJECT_ROOT / "data" / "logs" / "questions_log.jsonl"
VISION_SESSIONS_JSONL = PROJECT_ROOT / "data" / "logs" / "vision_sessions.jsonl"
ENROLLMENT_REPORT_JSON = PROJECT_ROOT / "data" / "logs" / "enrollment_report.json"
FRONTEND_DIR    = PROJECT_ROOT / "Frontend"


# helpers for consistent colors, initials, and mood emojis in the UI
_COLORS = ["blue", "green", "purple", "yellow"]

# Covers every normalized label that emotion.py can produce (happy/neutral/sad/
# angry/surprised/tired/unknown). "unknown" is emitted when the model fails
_MOOD_EMOJI = {
    "happy":     "😊 Happy",
    "neutral":   "😐 Neutral",
    "sad":       "😢 Sad",
    "angry":     "😠 Angry",
    "surprised": "😲 Surprised",
    "tired":     "😴 Tired",
    "unknown":   "❓ Unknown",
}
_TOPIC_CLASS = {
    "Computer Vision": "chip-cv", "Machine Learning": "chip-ml",
    "Natural Language Processing": "chip-ml", "Mathematics": "chip-math",
    "Data Science": "chip-ds", "Algorithms": "chip-algo",
    "Databases": "chip-ds", "Computer Networks": "chip-algo",
    "Operating Systems": "chip-algo",
}

# turns ex: "Nour Elbarawi" into "NE" for the avatar circle, and assigns a consistent color based on the name hash
def _initials(name: str) -> str:
    parts = name.split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else name[:2].upper()


def _color(name: str) -> str:
    return _COLORS[abs(hash(name)) % len(_COLORS)]


def _safe_student_dir_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {" ", "-", "_"} else "_" for ch in name)
    safe = " ".join(safe.split()).strip()
    return safe or "Student"


def _is_blurry(image: np.ndarray, blur_threshold: float) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < blur_threshold


def _enrolled_student_names() -> list[str]:
    if not EMBEDDINGS_PATH.exists():
        return []
    payload = load_embedding_database(EMBEDDINGS_PATH)
    return sorted(payload.get("students", {}).keys())


def _load_embedding_maps() -> tuple[dict[str, list[np.ndarray]], dict[str, list[str]]]:
    embeddings: dict[str, list[np.ndarray]] = {}
    sample_paths: dict[str, list[str]] = {}
    if not EMBEDDINGS_PATH.exists():
        return embeddings, sample_paths

    payload = load_embedding_database(EMBEDDINGS_PATH)
    for student_name, serialized in payload.get("students", {}).items():
        embeddings[student_name] = [
            np.asarray(embedding, dtype=np.float32)
            for embedding in np.asarray(serialized.get("raw_embeddings", []), dtype=np.float32)
        ]
        sample_paths[student_name] = list(serialized.get("sample_paths", []))
    return embeddings, sample_paths


def _append_enrollment_report(rows: list[dict[str, object]]) -> None:
    existing: list[dict[str, object]] = []
    if ENROLLMENT_REPORT_JSON.exists():
        try:
            existing = json.loads(ENROLLMENT_REPORT_JSON.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    ENROLLMENT_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ENROLLMENT_REPORT_JSON.write_text(
        json.dumps(existing + rows, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _read_vision_session_records() -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not VISION_SESSIONS_JSONL.exists():
        return records
    with VISION_SESSIONS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            sid = rec.get("session_id", "")
            if sid:
                records[sid] = rec
    return records


def _write_vision_session_record(record: dict) -> None:
    VISION_SESSIONS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with VISION_SESSIONS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _rewrite_jsonl_without_session(path: Path, session_id: str) -> int:
    if not path.exists():
        return 0

    removed = 0
    kept_lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                kept_lines.append(line)
                continue
            if rec.get("session_id") == session_id:
                removed += 1
                continue
            kept_lines.append(line)

    if removed:
        path.write_text("".join(kept_lines), encoding="utf-8")
    return removed


def _rewrite_attendance_without_session(session_id: str) -> int:
    if not ATTENDANCE_CSV.exists() or ATTENDANCE_CSV.stat().st_size == 0:
        return 0

    with ATTENDANCE_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or [
            "session_id",
            "student_name",
            "attendance_status",
            "confirmed_at",
            "first_track_id",
        ]
        kept_rows = []
        removed = 0
        for row in reader:
            if row.get("session_id") == session_id:
                removed += 1
            else:
                kept_rows.append(row)

    if removed:
        with ATTENDANCE_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
    return removed


# CV model/session state.
# A Vision Feed browser connection owns exactly one SmartClassroomPipeline.
# That keeps attendance/session state from leaking across reconnects.
_models_ready = False
_last_pipeline_runtime_device: str | None = None
_active_vision_sessions: set[str] = set()
_pipeline_lock = threading.Lock()


def create_vision_pipeline() -> SmartClassroomPipeline:
    cfg = CVConfig(device=detect_device())
    return SmartClassroomPipeline(cfg)


#  FastAPI app 
app = FastAPI(title="EduVision API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.on_event("startup")
async def _preload_pipeline() -> None:
    """Prepare the CV session factory without creating a user-facing session."""
    async def _warm():
        global _models_ready, _last_pipeline_runtime_device
        try:
            loop = asyncio.get_running_loop()
            pipeline = await loop.run_in_executor(None, create_vision_pipeline)
            with _pipeline_lock:
                _models_ready = True
                _last_pipeline_runtime_device = pipeline.detector.runtime_device
            print("[EduVision] CV session factory ready.")
        except Exception as exc:
            print(f"[EduVision] CV session factory warm-up skipped: {exc}")
    asyncio.create_task(_warm())


@app.get("/api/health")
def health() -> dict:
    detected_device = detect_device()
    torch_cuda_available = False
    ort_available_providers: list[str] = []

    try:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass

    try:
        import onnxruntime as ort

        ort_available_providers = list(ort.get_available_providers())
    except Exception:
        pass

    return {
        "status": "ok",
        "pipeline_ready": _models_ready,
        "detected_device": detected_device,
        "pipeline_runtime_device": _last_pipeline_runtime_device,
        "active_vision_sessions": sorted(_active_vision_sessions),
        "torch_cuda_available": torch_cuda_available,
        "ort_available_providers": ort_available_providers,
        "embeddings_exist": EMBEDDINGS_PATH.exists(),
        "sessions_exist": ATTENDANCE_CSV.exists(),
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "smart-classroom-assistant.html"))



# /api/roster reads the .pkl embeddings file and returns the list of enrolled students
@app.get("/api/roster")
def get_roster() -> list[dict]:
    return [
        {"name": name, "avatar": _initials(name), "ac": _color(name)}
        for name in _enrolled_student_names()
    ]


@app.delete("/api/students/{student_name}")
def delete_student(student_name: str) -> dict:
    normalized_name = " ".join(student_name.split()).strip()
    if not normalized_name:
        raise HTTPException(status_code=400, detail="Student name is required.")
    if _active_vision_sessions:
        raise HTTPException(
            status_code=409,
            detail="Stop the active vision session before deleting students.",
        )

    all_embeddings, all_sample_paths = _load_embedding_maps()
    matched_name = next(
        (name for name in all_embeddings if name == normalized_name),
        None,
    )
    if matched_name is None:
        lowered = normalized_name.casefold()
        matched_name = next(
            (name for name in all_embeddings if name.casefold() == lowered),
            None,
        )
    if matched_name is None:
        raise HTTPException(status_code=404, detail="Student not found.")

    removed_samples = len(all_embeddings.get(matched_name, []))
    all_embeddings.pop(matched_name, None)
    all_sample_paths.pop(matched_name, None)
    save_embedding_database(all_embeddings, EMBEDDINGS_PATH, all_sample_paths)

    removed_image_dirs = 0
    student_dir = STUDENTS_DIR / _safe_student_dir_name(matched_name)
    try:
        resolved = student_dir.resolve()
        allowed_root = STUDENTS_DIR.resolve()
        if student_dir.exists() and allowed_root in resolved.parents:
            shutil.rmtree(student_dir)
            removed_image_dirs = 1
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Student was removed from embeddings, but image cleanup failed: {exc}",
        ) from exc

    return {
        "status": "deleted",
        "student": matched_name,
        "removed": {
            "embedding_samples": removed_samples,
            "image_dirs": removed_image_dirs,
        },
    }


@app.post("/api/enroll-student")
async def enroll_student(
    student_name: str = Form(...),
    images: list[UploadFile] = File(...),
    blur_threshold: float = Form(35.0),
    min_detection_score: float = Form(0.55),
    min_face_size: int = Form(50),
) -> dict:
    normalized_name = " ".join(student_name.split()).strip()
    if not normalized_name:
        raise HTTPException(status_code=400, detail="Student name is required.")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    cfg = CVConfig(
        device=detect_device(),
        blur_variance_threshold=blur_threshold,
        enrollment_min_detection_score=min_detection_score,
        enrollment_min_face_size=min_face_size,
    ).ensure_directories()
    detector = FaceDetector(cfg)

    student_dir = cfg.students_dir / _safe_student_dir_name(normalized_name)
    student_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    accepted_embeddings: list[np.ndarray] = []
    accepted_paths: list[str] = []
    report_rows: list[dict[str, object]] = []

    for index, upload in enumerate(images, start=1):
        raw = await upload.read()
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            suffix = ".jpg"

        image_path = student_dir / f"{timestamp}_{index:02d}{suffix}"
        image_path.write_bytes(raw)

        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        reason = None
        if image is None:
            reason = "unreadable_image"
        elif _is_blurry(image, blur_threshold):
            reason = "blurry_image"
        else:
            detections = detector.detect(image, apply_filters=False)
            if len(detections) == 0:
                reason = "no_face"
            elif len(detections) > 1:
                reason = "multiple_faces"
            else:
                detection = detections[0]
                if detection.detection_confidence < min_detection_score:
                    reason = "low_detection_score"
                elif detection.min_side < min_face_size:
                    reason = "face_too_small"
                elif detection.embedding is None:
                    reason = "missing_embedding"
                else:
                    accepted_embeddings.append(detection.embedding)
                    accepted_paths.append(str(image_path))

        report_rows.append({
            "student_name": normalized_name,
            "image_path": str(image_path),
            "status": "accepted" if reason is None else "rejected",
            "reason": reason or "ok",
            "source": "web",
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        })

    _append_enrollment_report(report_rows)

    if not accepted_embeddings:
        return {
            "status": "failed",
            "student": normalized_name,
            "accepted": 0,
            "rejected": len(report_rows),
            "results": report_rows,
        }

    all_embeddings, all_sample_paths = _load_embedding_maps()
    all_embeddings.setdefault(normalized_name, [])
    all_sample_paths.setdefault(normalized_name, [])
    all_embeddings[normalized_name].extend(accepted_embeddings)
    all_sample_paths[normalized_name].extend(accepted_paths)
    payload = save_embedding_database(all_embeddings, EMBEDDINGS_PATH, all_sample_paths)
    enrolled = payload.get("students", {}).get(normalized_name, {})

    return {
        "status": "enrolled",
        "student": normalized_name,
        "accepted": len(accepted_embeddings),
        "rejected": len(report_rows) - len(accepted_embeddings),
        "total_samples": enrolled.get("num_samples", len(accepted_embeddings)),
        "results": report_rows,
    }


# /api/sessions returns attendance summaries for all sessions recorded in attendance_log.csv, enriched with mood data from session_log.jsonl
@app.get("/api/sessions-old", include_in_schema=False)
def get_sessions() -> list[dict]:
    if not ATTENDANCE_CSV.exists() or ATTENDANCE_CSV.stat().st_size == 0:
        return []

    # group rows by session_id
    session_map: dict[str, list[dict]] = {}
    with ATTENDANCE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            session_map.setdefault(row["session_id"], []).append(row)

    # last mood per student per session from session_log.jsonl
    mood_map: dict[str, dict[str, str]] = {}
    if SESSION_JSONL.exists():
        with SESSION_JSONL.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                except Exception:
                    continue
                sid, name = rec.get("session_id", ""), rec.get("predicted_student", "")
                if sid and name and name != "Unknown":
                    mood_map.setdefault(sid, {})[name] = rec.get("mood_label", "neutral")

    result = []
    for i, sid in enumerate(sorted(session_map)):
        rows = session_map[sid]
        moods = mood_map.get(sid, {})

        try:
            dt = datetime.strptime(sid, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%a, %d %b %Y")
            start_str = dt.strftime("%H:%M")
            end_times = []
            for row in rows:
                try:
                    end_times.append(datetime.fromisoformat(row["confirmed_at"]))
                except Exception:
                    pass
            end_str = max(end_times).strftime("%H:%M") if end_times else start_str
            time_str = start_str + " – " + end_str
        except Exception:
            date_str, time_str = sid, "—"

        students = []
        for row in rows:
            name = row["student_name"]
            mood_label = moods.get(name, "neutral")
            try:
                time_in = datetime.fromisoformat(row["confirmed_at"]).strftime("%H:%M")
            except Exception:
                time_in = "—"

            students.append({
                "name": name,
                "avatar": _initials(name), "ac": _color(name),
                "status": row.get("attendance_status", "Present"),
                "mood": _MOOD_EMOJI.get(mood_label, _MOOD_EMOJI["unknown"]),
                "time": time_in,
            })

        result.append({
            "label": f"Session {i + 1}",
            "date": date_str, "time": time_str,
            "students": students,
        })

    return result


# /api/sessions returns attendance summaries for all recorded sessions. Each
# session is expanded against the enrolled roster, so enrolled students who were
# not confirmed during that session appear as Absent.
@app.get("/api/sessions")
def get_sessions_with_absences() -> list[dict]:
    session_map: dict[str, dict[str, dict]] = {}
    if ATTENDANCE_CSV.exists() and ATTENDANCE_CSV.stat().st_size > 0:
        with ATTENDANCE_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("session_id", "")
                name = row.get("student_name", "")
                if not sid or not name:
                    continue

                student_rows = session_map.setdefault(sid, {})
                existing = student_rows.get(name)
                if existing is None:
                    student_rows[name] = row
                    continue

                current_dt = _parse_dt(row.get("confirmed_at"))
                existing_dt = _parse_dt(existing.get("confirmed_at"))
                if existing_dt is None or (current_dt is not None and current_dt < existing_dt):
                    student_rows[name] = row

    mood_map: dict[str, dict[str, str]] = {}
    if SESSION_JSONL.exists():
        with SESSION_JSONL.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                except Exception:
                    continue
                sid, name = rec.get("session_id", ""), rec.get("predicted_student", "")
                if sid and name and name != "Unknown":
                    mood_map.setdefault(sid, {})[name] = rec.get("mood_label", "neutral")

    vision_sessions = _read_vision_session_records()
    session_ids = sorted(set(session_map) | set(vision_sessions))
    if not session_ids:
        return []

    enrolled_names = _enrolled_student_names()
    result = []

    for i, sid in enumerate(session_ids):
        rows_by_name = session_map.get(sid, {})
        moods = mood_map.get(sid, {})
        session_record = vision_sessions.get(sid, {})

        try:
            dt = datetime.strptime(sid, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%a, %d %b %Y")
            start_str = dt.strftime("%H:%M")
            end_times = [
                parsed_dt
                for row in rows_by_name.values()
                if (parsed_dt := _parse_dt(row.get("confirmed_at"))) is not None
            ]
            finalized_at = _parse_dt(session_record.get("finalized_at"))
            if finalized_at is not None:
                end_times.append(finalized_at)
            end_str = max(end_times).strftime("%H:%M") if end_times else start_str
            time_str = start_str + " - " + end_str
        except Exception:
            date_str, time_str = sid, "-"

        students = []
        for name in sorted(set(enrolled_names) | set(rows_by_name)):
            row = rows_by_name.get(name)
            if row is None:
                students.append({
                    "name": name,
                    "avatar": _initials(name), "ac": _color(name),
                    "status": "Absent",
                    "mood": "-",
                    "time": "-",
                })
                continue

            mood_label = moods.get(name, "neutral")
            confirmed_at = _parse_dt(row.get("confirmed_at"))
            time_in = confirmed_at.strftime("%H:%M") if confirmed_at else "-"

            students.append({
                "name": name,
                "avatar": _initials(name), "ac": _color(name),
                "status": row.get("attendance_status", "Present"),
                "mood": _MOOD_EMOJI.get(mood_label, _MOOD_EMOJI["unknown"]),
                "time": time_in,
            })

        result.append({
            "id": sid,
            "label": f"Session {i + 1}",
            "date": date_str, "time": time_str,
            "students": students,
        })

    return result


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if session_id in set(_active_vision_sessions):
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a vision session while it is active.",
        )

    removed = {
        "attendance_rows": _rewrite_attendance_without_session(session_id),
        "session_log_rows": _rewrite_jsonl_without_session(SESSION_JSONL, session_id),
        "vision_session_rows": _rewrite_jsonl_without_session(VISION_SESSIONS_JSONL, session_id),
        "face_crop_dirs": 0,
    }

    face_crop_dir = PROJECT_ROOT / "data" / "logs" / "face_crops" / session_id
    try:
        resolved = face_crop_dir.resolve()
        allowed_root = (PROJECT_ROOT / "data" / "logs" / "face_crops").resolve()
        if face_crop_dir.exists() and allowed_root in resolved.parents:
            shutil.rmtree(face_crop_dir)
            removed["face_crop_dirs"] = 1
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete face crops: {exc}") from exc

    if not any(removed.values()):
        raise HTTPException(status_code=404, detail="Session not found.")

    return {"status": "deleted", "session_id": session_id, "removed": removed}


#  /api/questions 
@app.get("/api/questions")
def get_questions() -> list[dict]:
    if not QUESTIONS_JSONL.exists():
        return []
    out = []
    with QUESTIONS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line.strip()))
            except Exception:
                continue
    return list(reversed(out))


#  /api/classify 
def _run_classify(text: str) -> tuple[str, float]:
    """Sync wrapper — runs in thread pool so it never blocks the event loop."""
    from nlp_module.classifier import classify_question
    return classify_question(text)


@app.post("/api/classify")
async def classify_text(body: dict) -> dict:
    text = body.get("text", "").strip()
    if not text:
        return {"topic": "Unknown", "confidence": 0.0}

    loop = asyncio.get_running_loop()
    try:
        topic, score = await loop.run_in_executor(None, _run_classify, text)
    except Exception as exc:
        return {"topic": "Error", "confidence": 0.0, "error": str(exc)}

    student = body.get("student", "Student")
    record = {
        "student": student, "avatar": _initials(student), "ac": _color(student),
        "q": text, "topic": topic,
        "topicClass": _TOPIC_CLASS.get(topic, "chip-cv"),
        "time": datetime.now().strftime("%H:%M"),
        "conf": round(score * 100, 1),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    QUESTIONS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with QUESTIONS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"topic": topic, "confidence": round(score * 100, 1)}


# WebSocket /ws/vision 
def _process_frame(pipeline, frame):
    """Run face detection + recognition in a thread-pool worker."""
    _, results = pipeline.process_frame(frame)
    return [
        {
            "bbox": list(r.bbox),
            "name": r.display_name,
            "mood": r.mood.mood_label,
            "status": r.attendance.attendance_status,
            "score": round(r.recognition.similarity_score, 3),
        }
        for r in results
    ]


@app.websocket("/ws/vision")
async def vision_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    pipeline = create_vision_pipeline()
    session_id = pipeline.config.session_id
    with _pipeline_lock:
        _active_vision_sessions.add(session_id)
    print(f"[EduVision] Vision session started: {session_id}")
    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"session_id": session_id, "faces": []})
                continue
            # Run CPU-heavy work off the async event loop
            faces = await loop.run_in_executor(None, _process_frame, pipeline, frame)
            await websocket.send_json({"session_id": session_id, "faces": faces})
    except WebSocketDisconnect:
        print(f"[EduVision] Vision session disconnected: {session_id}")
    except Exception as exc:
        print(f"[EduVision] Vision session error ({session_id}): {exc}")
    finally:
        global _last_pipeline_runtime_device
        try:
            summary = pipeline.finalize()
            if pipeline.frame_index > 0 or summary["attendance_count"] > 0:
                _write_vision_session_record({
                    "session_id": session_id,
                    "finalized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "frame_count": pipeline.frame_index,
                    "attendance_count": summary["attendance_count"],
                })
            print(
                "[EduVision] Vision session finalized: "
                f"{session_id} ({summary['attendance_count']} attendance rows)"
            )
        except Exception as exc:
            print(f"[EduVision] Vision session finalize failed ({session_id}): {exc}")
        finally:
            with _pipeline_lock:
                _active_vision_sessions.discard(session_id)
                _last_pipeline_runtime_device = pipeline.detector.runtime_device


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
