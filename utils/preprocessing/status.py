"""Append-friendly status files for resumable large preprocessing jobs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def write_status_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in row_list:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return output


def status_summary(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("status", "unknown")) for row in rows))
