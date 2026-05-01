import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module.config import CVConfig
from cv_module.recognizer import FaceRecognizer, save_embedding_database


def test_recognizer_returns_student_name_when_similarity_is_strong(tmp_path):
    db_path = tmp_path / "student_embeddings.pkl"
    save_embedding_database(
        {
            "Alice": [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
            "Bob": [np.array([0.0, 1.0, 0.0], dtype=np.float32)],
        },
        db_path,
    )

    recognizer = FaceRecognizer(
        CVConfig(embeddings_path=db_path, recognition_threshold=0.40, top1_top2_margin_threshold=0.05),
        auto_load=True,
    )
    result = recognizer.recognize(np.array([0.98, 0.05, 0.0], dtype=np.float32))

    assert result.predicted_name == "Alice"
    assert result.is_unknown is False
    assert result.similarity_score > 0.9


def test_recognizer_returns_unknown_when_top_two_candidates_are_too_close(tmp_path):
    db_path = tmp_path / "student_embeddings.pkl"
    save_embedding_database(
        {
            "Alice": [np.array([1.0, 0.00], dtype=np.float32)],
            "Alicia": [np.array([0.99, 0.10], dtype=np.float32)],
        },
        db_path,
    )

    recognizer = FaceRecognizer(
        CVConfig(embeddings_path=db_path, recognition_threshold=0.40, top1_top2_margin_threshold=0.02),
        auto_load=True,
    )
    result = recognizer.recognize(np.array([1.0, 0.05], dtype=np.float32))

    assert result.predicted_name == "Unknown"
    assert result.is_unknown is True
    assert result.margin_between_top_1_and_top_2 < 0.02
