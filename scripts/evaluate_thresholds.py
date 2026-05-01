from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module import CVConfig, FaceDetector  # noqa: E402
from cv_module.config import detect_device  # noqa: E402
from cv_module.recognizer import l2_normalize  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cosine-similarity thresholds for face recognition.")
    parser.add_argument("--students-dir", type=Path, default=PROJECT_ROOT / "data" / "students")
    parser.add_argument("--pairs-csv", type=Path, default=None, help="Optional CSV with columns: image_a,image_b,is_same")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "data" / "logs")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--min-threshold", type=float, default=0.20)
    parser.add_argument("--max-threshold", type=float, default=0.80)
    parser.add_argument("--steps", type=int, default=61)
    parser.add_argument("--max-impostor-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "data" / "logs" / "threshold_report.json")
    return parser.parse_args()


def image_paths_in(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def extract_single_face_embedding(image_path: Path, detector: FaceDetector) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    detections = detector.detect(image, apply_filters=False)
    if len(detections) != 1 or detections[0].embedding is None:
        return None
    return l2_normalize(detections[0].embedding)


def build_pairs_from_students(
    students_dir: Path,
    detector: FaceDetector,
    max_impostor_pairs: int,
    seed: int,
) -> tuple[list[tuple[float, int]], dict[str, int]]:
    student_embeddings: dict[str, list[np.ndarray]] = {}
    skipped_images = 0

    for student_dir in sorted(path for path in students_dir.iterdir() if path.is_dir()):
        embeddings = []
        for image_path in image_paths_in(student_dir):
            embedding = extract_single_face_embedding(image_path, detector)
            if embedding is None:
                skipped_images += 1
                continue
            embeddings.append(embedding)
        if embeddings:
            student_embeddings[student_dir.name] = embeddings

    pairs: list[tuple[float, int]] = []
    for embeddings in student_embeddings.values():
        for emb_a, emb_b in itertools.combinations(embeddings, 2):
            pairs.append((float(np.dot(emb_a, emb_b)), 1))

    impostor_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    students = sorted(student_embeddings)
    for idx, student_a in enumerate(students):
        for student_b in students[idx + 1 :]:
            for emb_a in student_embeddings[student_a]:
                for emb_b in student_embeddings[student_b]:
                    impostor_candidates.append((emb_a, emb_b))

    random.Random(seed).shuffle(impostor_candidates)
    for emb_a, emb_b in impostor_candidates[:max_impostor_pairs]:
        pairs.append((float(np.dot(emb_a, emb_b)), 0))

    stats = {
        "students_with_embeddings": len(student_embeddings),
        "skipped_images": skipped_images,
        "generated_pairs": len(pairs),
    }
    return pairs, stats


def build_pairs_from_csv(pairs_csv: Path, detector: FaceDetector) -> tuple[list[tuple[float, int]], dict[str, int]]:
    pairs: list[tuple[float, int]] = []
    skipped_pairs = 0

    with pairs_csv.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            image_a = Path(row["image_a"])
            image_b = Path(row["image_b"])
            label = int(row["is_same"])
            emb_a = extract_single_face_embedding(image_a, detector)
            emb_b = extract_single_face_embedding(image_b, detector)
            if emb_a is None or emb_b is None:
                skipped_pairs += 1
                continue
            pairs.append((float(np.dot(emb_a, emb_b)), label))

    stats = {
        "students_with_embeddings": 0,
        "skipped_images": skipped_pairs,
        "generated_pairs": len(pairs),
    }
    return pairs, stats


def compute_metrics(scores_and_labels: list[tuple[float, int]], threshold: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for score, label in scores_and_labels:
        predicted_same = score >= threshold
        if predicted_same and label == 1:
            tp += 1
        elif predicted_same and label == 0:
            fp += 1
        elif not predicted_same and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> int:
    args = parse_args()
    config = CVConfig(
        device=args.device or detect_device(),
        logs_dir=args.logs_dir,
        students_dir=args.students_dir,
    )
    detector = FaceDetector(config)

    if args.pairs_csv and args.pairs_csv.exists():
        scores_and_labels, stats = build_pairs_from_csv(args.pairs_csv, detector)
    elif args.students_dir.exists():
        scores_and_labels, stats = build_pairs_from_students(
            args.students_dir,
            detector,
            args.max_impostor_pairs,
            args.seed,
        )
    else:
        scores_and_labels, stats = [], {"students_with_embeddings": 0, "skipped_images": 0, "generated_pairs": 0}

    if len(scores_and_labels) < 2:
        print("Not enough validation pairs were created.")
        print("Provide either --pairs-csv with image pairs or a populated data/students/ directory.")
        return 0

    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.steps)
    metrics = [compute_metrics(scores_and_labels, float(threshold)) for threshold in thresholds]
    best_metric = max(metrics, key=lambda item: (item["f1"], item["accuracy"]))

    report = {
        "config": {
            "model_pack": config.insightface_model_pack,
            "device": config.device,
            "min_threshold": args.min_threshold,
            "max_threshold": args.max_threshold,
            "steps": args.steps,
        },
        "data_stats": stats,
        "best_threshold": best_metric,
        "all_metrics": metrics,
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(best_metric, indent=2))
    print(f"Saved threshold report to {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
