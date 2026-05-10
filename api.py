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
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module import CVConfig, SmartClassroomPipeline
from cv_module.config import detect_device
from cv_module.recognizer import load_embedding_database

# file paths needed
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "student_embeddings.pkl"
ATTENDANCE_CSV  = PROJECT_ROOT / "data" / "logs" / "attendance_log.csv"
SESSION_JSONL   = PROJECT_ROOT / "data" / "logs" / "session_log.jsonl"
QUESTIONS_JSONL = PROJECT_ROOT / "data" / "logs" / "questions_log.jsonl"
VISION_SESSIONS_JSONL = PROJECT_ROOT / "data" / "logs" / "vision_sessions.jsonl"
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


def _enrolled_student_names() -> list[str]:
    if not EMBEDDINGS_PATH.exists():
        return []
    payload = load_embedding_database(EMBEDDINGS_PATH)
    return sorted(payload.get("students", {}).keys())


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
            "label": f"Session {i + 1}",
            "date": date_str, "time": time_str,
            "students": students,
        })

    return result


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
