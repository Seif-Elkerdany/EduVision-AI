import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_module.config import CVConfig
from cv_module.logger import SessionLogger


def test_attendance_summary_skips_duplicate_session_student_rows(tmp_path):
    config = CVConfig(
        logs_dir=tmp_path,
        session_id="20260510_193000",
    )
    logger = SessionLogger(config)
    rows = [
        {
            "session_id": config.session_id,
            "student_name": "Alice",
            "attendance_status": "Present",
            "confirmed_at": "2026-05-10T19:30:00+03:00",
            "first_track_id": 1,
        }
    ]

    logger.write_attendance_summary(rows)
    logger.write_attendance_summary(rows)

    with config.attendance_log_path.open(encoding="utf-8") as file_handle:
        written_rows = list(csv.DictReader(file_handle))

    assert len(written_rows) == 1
    assert written_rows[0]["student_name"] == "Alice"
