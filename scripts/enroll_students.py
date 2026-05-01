from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module import CVConfig, FaceDetector, save_embedding_database  # noqa: E402
from cv_module.config import detect_device  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll students from local face images.")
    parser.add_argument("--students-dir", type=Path, default=PROJECT_ROOT / "data" / "students")
    parser.add_argument("--embeddings-path", type=Path, default=PROJECT_ROOT / "data" / "embeddings" / "student_embeddings.pkl")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "data" / "logs")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--min-detection-score", type=float, default=0.70)
    parser.add_argument("--min-face-size", type=int, default=80)
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--min-images-per-student", type=int, default=1)
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "data" / "logs" / "enrollment_report.json")
    return parser.parse_args()


def is_blurry(image, blur_threshold: float) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < blur_threshold


def iter_image_paths(student_dir: Path) -> list[Path]:
    return sorted(path for path in student_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> int:
    args = parse_args()
    config = CVConfig(
        device=args.device or detect_device(),
        students_dir=args.students_dir,
        embeddings_path=args.embeddings_path,
        logs_dir=args.logs_dir,
        enrollment_min_detection_score=args.min_detection_score,
        enrollment_min_face_size=args.min_face_size,
        blur_variance_threshold=args.blur_threshold,
    ).ensure_directories()

    detector = FaceDetector(config)
    accepted_embeddings: dict[str, list] = {}
    accepted_paths: dict[str, list[str]] = {}
    report_rows: list[dict[str, object]] = []

    if not args.students_dir.exists():
        raise FileNotFoundError(f"Students directory does not exist: {args.students_dir}")

    for student_dir in sorted(path for path in args.students_dir.iterdir() if path.is_dir()):
        student_name = student_dir.name
        for image_path in iter_image_paths(student_dir):
            image = cv2.imread(str(image_path))
            reason = None

            if image is None:
                reason = "unreadable_image"
            elif is_blurry(image, args.blur_threshold):
                reason = "blurry_image"
            else:
                detections = detector.detect(image, apply_filters=False)
                if len(detections) == 0:
                    reason = "no_face"
                elif len(detections) > 1:
                    reason = "multiple_faces"
                else:
                    detection = detections[0]
                    if detection.detection_confidence < args.min_detection_score:
                        reason = "low_detection_score"
                    elif detection.min_side < args.min_face_size:
                        reason = "face_too_small"
                    elif detection.embedding is None:
                        reason = "missing_embedding"
                    else:
                        accepted_embeddings.setdefault(student_name, []).append(detection.embedding)
                        accepted_paths.setdefault(student_name, []).append(str(image_path))

            report_rows.append(
                {
                    "student_name": student_name,
                    "image_path": str(image_path),
                    "status": "accepted" if reason is None else "rejected",
                    "reason": reason or "ok",
                }
            )

    filtered_embeddings = {
        student_name: embeddings
        for student_name, embeddings in accepted_embeddings.items()
        if len(embeddings) >= args.min_images_per_student
    }

    if not filtered_embeddings:
        print("No valid enrollment embeddings were created. Check the report JSON for rejection reasons.")
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")
        return 1

    payload = save_embedding_database(filtered_embeddings, args.embeddings_path, accepted_paths)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")

    print(f"Saved embeddings for {len(payload['students'])} students to {args.embeddings_path}")
    print(f"Enrollment report: {args.report_path}")
    for student_name, serialized in sorted(payload["students"].items()):
        print(f"  - {student_name}: {serialized['num_samples']} accepted images")

    skipped_students = sorted(set(accepted_embeddings) - set(filtered_embeddings))
    if skipped_students:
        print("Skipped students due to too few accepted images:")
        for student_name in skipped_students:
            print(f"  - {student_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
