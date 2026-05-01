# Computer Vision Module

## Overview

This module provides a local, real-time computer vision pipeline for the EduVision Smart Classroom Assistant.

## Models

- Detection:
  `InsightFace buffalo_l` detector (`det_10g`, SCRFD-family)
- Alignment:
  5-point landmark-based alignment from `InsightFace FaceAnalysis` followed by `face_align.norm_crop`
- Recognition:
  `InsightFace buffalo_l` recognition model producing ArcFace-style normalized embeddings
- Matching:
  cosine similarity against enrolled student embeddings
- Emotion:
  `EmotiEffLib` ONNX backend using `enet_b0_8_best_vgaf`
- Tracking:
  lightweight IoU plus centroid matching, no separate tracking model
- Attendance confirmation:
  rule-based temporal smoothing, no separate attendance model

It is designed for:

- face detection and alignment
- student recognition for attendance
- lightweight facial-expression impression estimation
- frame-to-frame tracking
- temporal smoothing before attendance confirmation
- structured session logging

The implementation is local-first and intended to be easy to demo, test, and integrate into a FastAPI application later.

## Module Layout

```text
src/cv_module/
  __init__.py
  config.py
  detector.py
  recognizer.py
  emotion.py
  tracker.py
  attendance.py
  logger.py
  pipeline.py
scripts/
  enroll_students.py
  run_webcam.py
  run_video.py
  evaluate_thresholds.py
data/
  students/
  embeddings/
  logs/
tests/
  test_recognizer.py
  test_tracker.py
  test_attendance_logic.py
```

## Model Stack

- Face detection and alignment:
  `InsightFace FaceAnalysis` with the `buffalo_l` model pack, using its built-in detector and landmark outputs for alignment.
- Face recognition:
  ArcFace-style normalized embeddings from the `buffalo_l` recognition model, compared with cosine similarity, thresholding, and top-1 vs top-2 margin checks.
- Mood impression:
  `EmotiEffLib` with the ONNX backend by default through a lightweight wrapper, using the `enet_b0_8_best_vgaf` model.
- Tracking:
  Simple IoU plus centroid matching without heavy multi-object tracking dependencies.

## Why InsightFace Instead Of dlib or face_recognition

- It provides modern ArcFace-style embeddings, which are more aligned with current face-recognition practice than the older `dlib` stack.
- `FaceAnalysis` gives detection, landmarks, alignment, and recognition-friendly embeddings in one consistent pipeline.
- It supports ONNX Runtime execution, which keeps the system practical for local demos and constrained hardware.

## Why Mood Is Treated As An Impression

The emotion module is described as a facial-expression impression rather than a clinical or psychological fact. Facial-expression models are noisy, context-limited, and sensitive to pose, lighting, and occlusion. The pipeline therefore uses these outputs only as soft classroom signals.

## Setup

### 1. Activate the EduVision venv

From the repository root:

```powershell
..\Venvs\EduVision\Scripts\Activate.ps1
```

### 2. Install requirements

CPU-safe default:

```powershell
python -m pip install -r requirements.txt
```

Optional NVIDIA/CUDA path for faster InsightFace inference:

```powershell
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu==1.24.4
```

The project uses a single root `requirements.txt` for all modules.

If your machine shows an error about `cublasLt64_12.dll` when using `onnxruntime-gpu`, run the CV scripts with CPU explicitly:

```powershell
python scripts/run_webcam.py --device cpu
```

If OpenCV reports that `cvShowImage` or `cvDestroyAllWindows` is not implemented, your current OpenCV build has no GUI backend. In that case, use:

```powershell
python scripts/run_webcam.py --headless
```

## Enrollment

Place student images in:

```text
data/students/<student_name>/image1.jpg
data/students/<student_name>/image2.jpg
```

Run enrollment:

```powershell
python scripts/enroll_students.py
```

Example:

```powershell
python scripts/enroll_students.py --min-detection-score 0.75 --blur-threshold 120
```

The enrollment script rejects images with:

- no detected face
- multiple detected faces
- small faces
- low detection confidence
- blurry input

It stores raw embeddings and a mean embedding per student in `data/embeddings/student_embeddings.pkl`.

Student images, generated embeddings, logs, and downloaded local CV models are ignored by Git through the repository `.gitignore`.

## Run Webcam Demo

```powershell
python scripts/run_webcam.py
```

Example:

```powershell
python scripts/run_webcam.py --frame-skip 1 --recognition-threshold 0.48 --margin-threshold 0.06 --headless
```

## Run Video Demo

```powershell
python scripts/run_video.py --input path\to\classroom.mp4
```

Save annotated output:

```powershell
python scripts/run_video.py --input path\to\classroom.mp4 --output-video data\logs\annotated_classroom.mp4 --display
```

## Tune Thresholds

If you already have validation image pairs in CSV format:

```powershell
python scripts/evaluate_thresholds.py --pairs-csv path\to\pairs.csv
```

If you do not have a pairs CSV yet, the script can build genuine and impostor pairs from `data/students/` automatically:

```powershell
python scripts/evaluate_thresholds.py
```

The report is written to `data/logs/threshold_report.json`.

## Logging

The pipeline writes:

- `data/logs/session_log.jsonl`
  Per-frame structured events with timestamp, track id, predicted student, confidence, mood, attendance state, and bounding box.
- `data/logs/attendance_log.csv`
  Session-level attendance summary rows with session id and confirmation time.

By default, full face crops are not stored. Enable them only when needed with `--save-face-crops`.

## FastAPI Integration Note

The CV implementation is intentionally built as plain Python service classes rather than a hardcoded API. That makes it easy to reuse inside a FastAPI route or background worker later:

- `SmartClassroomPipeline` can be created once during app startup.
- `process_frame()` can be called from a streaming or upload endpoint.
- `finalize()` returns a session summary that can be exposed through your API layer.

## Hardware Notes For 4GB VRAM Targets

- The project uses only `buffalo_l` for face detection and recognition.
- Increase `frame_skip` to reduce detector load.
- The emotion module uses a lightweight ONNX backend and can be disabled with `--disable-emotion`.
- CPU fallback is supported if CUDA or ONNX GPU execution is unavailable.

## Limitations

- Small, dark, blurry, or strongly side-facing faces are harder to recognize reliably.
- Facial-expression estimation is noisy and should not be treated as a fact about a student.
- Attendance should rely on temporal confirmation, not single-frame recognition.
- Recognition thresholds should be calibrated on local classroom data rather than copied blindly.

## Tests

Run the focused CV tests with:

```powershell
..\Venvs\EduVision\Scripts\python.exe -m pytest tests/test_recognizer.py tests/test_tracker.py tests/test_attendance_logic.py
```
