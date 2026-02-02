import json
from pathlib import Path


def load_valid_cutoff(processed_path: str) -> int:
    p = Path(processed_path) / "valid_cutoff.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return int(data["valid_cutoff"])

