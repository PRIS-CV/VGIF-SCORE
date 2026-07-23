from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_entries_file(path: Path) -> list[dict[str, Any]] | None:
    """Load benchmark entries from a JSON array or JSON Lines file."""
    if not path.is_file():
        return None

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return None
        return [item for item in payload if isinstance(item, dict)]

    if suffix == ".jsonl":
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}.")
                entries.append(item)
        return entries

    return None
