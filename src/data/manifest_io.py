from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from src.utils.paths import ensure_dir

DEFAULT_COLUMNS = [
    "track_id",
    "track_num",
    "scenario",
    "domain",
    "track_dir",
    "annotation_path",
    "gt_text",
    "layout",
    "track_status",
    "num_lr_images",
    "num_hr_images",
    "lr_image_paths",
    "hr_image_paths",
    "annotation_notes",
]
LIST_COLUMNS = {"lr_image_paths", "hr_image_paths", "annotation_notes"}
INT_COLUMNS = {"track_num", "num_lr_images", "num_hr_images"}


def _ordered_fieldnames(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    if preferred is None:
        preferred = DEFAULT_COLUMNS
    seen: set[str] = set()
    fieldnames: list[str] = []
    for key in preferred:
        if key not in seen:
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def _encode_value(field: str, value: Any) -> str:
    if value is None:
        return ""
    if field in LIST_COLUMNS or isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_manifest_csv(
    rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    columns = _ordered_fieldnames(rows, fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _encode_value(field, row.get(field)) for field in columns})
    return path


def write_manifest_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path


def _decode_value(field: str, value: str) -> Any:
    if value == "":
        return [] if field in LIST_COLUMNS else ""
    if field in LIST_COLUMNS:
        return json.loads(value)
    if field in INT_COLUMNS:
        return int(value)
    return value


def read_manifest_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append({field: _decode_value(field, value) for field, value in row.items()})
    return rows


def read_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

