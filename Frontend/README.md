# EduVision-AI — Frontend

This folder contains the web dashboard for EduVision-AI. It is a single HTML page that connects to the Python backend (`api.py`) running on `localhost:8000`.

---

## How to Run

**Step 1 — Start the backend server (from the project root):**
```bash
python api.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[EduVision] CV pipeline ready.
```

**Step 2 — Open the dashboard in your browser:**
```
http://localhost:8000
```

The page loads automatically. No separate frontend server is needed.

---

## How the Frontend Connects to the Backend

Every piece of data in the dashboard is read from real files through the `api.py` server.

| Frontend Page | API Endpoint | Data Source File |
|---|---|---|
| Dashboard stats | `GET /api/sessions` + `GET /api/questions` | `data/logs/attendance_log.csv`, `data/logs/session_log.jsonl`, `data/logs/questions_log.jsonl` |
| Roster | `GET /api/roster` | `data/embeddings/student_embeddings.pkl` |
| Attendance | `GET /api/sessions` | `data/logs/attendance_log.csv` + `data/logs/session_log.jsonl` |
| Vision Feed (camera) | `WebSocket /ws/vision` | Live camera → CV pipeline → JSON detections |
| Speech & NLP | `POST /api/classify` | Runs DeBERTa model → writes to `data/logs/questions_log.jsonl` |
| Question Log | `GET /api/questions` | `data/logs/questions_log.jsonl` |
| Settings | Static | Shows project config — no API call |

The **server status dot** in the top-right corner pings `GET /api/health` every few seconds:
- **Green** = server is running
- **Red** = server is offline

---

## Expected Output Per Page

### Dashboard
Shows summary cards computed from all recorded sessions:
- **Students Present** — count of unique students in the latest session
- **Avg Mood Score** — average mood across students (Happy=5, Neutral=3, Sad=1)
- **Questions Asked** — total questions logged
- **Top Topic** — most frequent NLP classification result

> If no sessions have been recorded yet, all cards show `—` and charts are empty. This is normal — run a Vision Feed session first.

### Roster
Lists every enrolled student pulled from the embeddings file.

> Empty if `data/embeddings/student_embeddings.pkl` does not exist. Run enrollment first:
> ```bash
> cd scripts
> python enroll_students.py --blur-threshold 35
> ```

### Attendance
Shows one card per recorded session with:
- Date and time
- Each student: name, mood emoji, check-in time, status (Present)

> Empty until at least one Vision Feed session has been completed and disconnected (disconnect saves to CSV).

### Vision Feed
Opens your webcam and streams frames to the Python CV pipeline via WebSocket.

Session lifecycle:
- Opening **Vision Feed** starts a new backend vision session.
- The session id is shown in the Vision Feed status text.
- Leaving the page, refreshing, closing the tab, or losing the camera/WebSocket connection finalizes that session.
- Attendance rows are written when the session finalizes.
- Duplicate rows for the same student in the same session are skipped.

Expected behavior:
1. Camera turns on when you click **Vision Feed** in the sidebar
2. Green bounding boxes appear around detected faces
3. Each box shows the student name and mood label
4. Attendance is saved to CSV automatically when you navigate away or close the browser

> If the camera does not start, check that your browser has camera permission granted.

### Speech & NLP
Click the microphone button to record a question.

Expected behavior:
1. Button turns red — browser asks for microphone permission (allow it)
2. Speak your question (e.g. *"What is backpropagation?"*)
3. Browser transcribes speech using Web Speech API
4. Transcript is sent to `/api/classify`
5. Result appears in the log: topic label + confidence percentage (e.g. `Machine Learning — 94.2%`)

> The NLP model (DeBERTa-v3-large) downloads automatically on first use (~1.5 GB). Classification takes 5–15 seconds on CPU.

> Supported topics: Computer Vision, Machine Learning, Natural Language Processing, Mathematics, Data Science, Algorithms, Databases, Computer Networks, Operating Systems.

### Analytics
Charts computed from all session and question data:
- Topic distribution (bar chart)
- Mood distribution (bar chart)
- Attendance over sessions (line chart)
- Top students by attendance

> All charts are empty until sessions and questions have been recorded.

---

## Files in This Folder

| File | Purpose |
|---|---|
| `smart-classroom-assistant.html` | Main dashboard — all pages, JavaScript, WebSocket client |
| `smart-classroom-assistant.css` | All styles |
| `AIU-logo.png` | University logo shown in the sidebar |

---

## Allow Camera and Microphone in Chrome (Required)

The Vision Feed and Speech & NLP pages will not work unless Chrome has permission to use your camera and microphone.


### How to confirm it is working

- **Camera** — go to **Vision Feed**. The camera preview should appear within 2 seconds. If you see a black box, permission is still blocked.
- **Microphone** — go to **Speech & NLP** and click the mic button. The button turns red and a "Listening…" indicator appears. If nothing happens or it immediately stops, microphone is blocked.


## Troubleshooting

| Problem | Fix |
|---|---|
| Page does not load | Make sure `python api.py` is running |
| Server status dot stays red | Backend is not running or crashed — check terminal output |
| Roster is empty | Run `python scripts/enroll_students.py --blur-threshold 35` first |
| Attendance page is empty | Complete a Vision Feed session (camera must detect faces), then close/navigate away |
| NLP says "backend offline" | `api.py` is not running, or the DeBERTa model is still loading (wait ~30 seconds after startup) |
| Camera does not start on Vision Feed | Grant camera permission in browser settings |
| Mic button does nothing | Grant microphone permission in browser settings (System Settings → Privacy → Microphone on macOS) |
