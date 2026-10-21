# code/training/recorder.py
"""Small CSV writers used by the experiment loop."""
from pathlib import Path
import csv

def record_repetition(record: dict, path: str):
    """Append one repetition record to a CSV file, creating the header on first write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
