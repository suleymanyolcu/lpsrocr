from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.annotation_adapter import (
    extract_annotation_notes,
    extract_gt_text,
    extract_layout,
)
from src.data.manifest_io import write_manifest_csv, write_manifest_jsonl
from src.utils.paths import ensure_dir, relativize

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
FRAME_PATTERN = re.compile(r"^(?P<prefix>lr|hr)-(?P<index>\d+)$", re.IGNORECASE)
EXPECTED_FRAME_INDICES = tuple(range(1, 6))
DEFAULT_SCENARIO_ORDER = ("Scenario-A", "Scenario-B")


def _safe_int(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"(\d+)$", text)
    if not match:
        return None
    return int(match.group(1))


def _parse_track_record(dataset_root: Path, annotation_path: Path) -> tuple[str, str, str, int] | None:
    rel = annotation_path.relative_to(dataset_root)
    if len(rel.parts) != 4:
        return None
    scenario, domain, track_id, filename = rel.parts
    if filename != "annotations.json" or not track_id.startswith("track_"):
        return None
    track_num = _safe_int(track_id)
    if track_num is None:
        return None
    return scenario, domain, track_id, track_num


def _collect_prefixed_images(track_dir: Path, prefix: str) -> tuple[list[Path], list[str]]:
    indexed_paths: dict[int, Path] = {}
    notes: list[str] = []
    for path in sorted(track_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = FRAME_PATTERN.match(path.stem)
        if not match or match.group("prefix").lower() != prefix:
            continue
        frame_index = int(match.group("index"))
        if frame_index in indexed_paths:
            notes.append(f"{prefix}_duplicate_frame_index:{frame_index:03d}")
            continue
        indexed_paths[frame_index] = path

    ordered = [indexed_paths[index] for index in sorted(indexed_paths)]
    observed_indices = set(indexed_paths)
    missing = [index for index in EXPECTED_FRAME_INDICES if index not in observed_indices]
    extra = sorted(index for index in observed_indices if index not in EXPECTED_FRAME_INDICES)
    if missing:
        notes.append(f"{prefix}_missing_indices:{','.join(f'{index:03d}' for index in missing)}")
    if extra:
        notes.append(f"{prefix}_unexpected_indices:{','.join(f'{index:03d}' for index in extra)}")
    if len(ordered) != len(EXPECTED_FRAME_INDICES):
        notes.append(f"{prefix}_frame_count:{len(ordered)}")
    return ordered, notes


def _sorted_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["scenario"], row["domain"], row["track_num"], row["track_id"])


def scan_dataset(dataset_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    annotation_paths = sorted(
        dataset_root.rglob("annotations.json"),
        key=lambda path: relativize(path, dataset_root),
    )

    rows: list[dict[str, Any]] = []
    skipped_annotations = 0
    note_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    total_lr_images = 0
    total_hr_images = 0

    for annotation_path in annotation_paths:
        track_record = _parse_track_record(dataset_root, annotation_path)
        if track_record is None:
            skipped_annotations += 1
            continue

        scenario, domain, track_id, track_num = track_record
        track_dir = annotation_path.parent
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            annotation = {}
            parse_note = f"annotation_json_decode_error:{exc.__class__.__name__}"
        else:
            parse_note = ""

        lr_image_paths, lr_notes = _collect_prefixed_images(track_dir, "lr")
        hr_image_paths, hr_notes = _collect_prefixed_images(track_dir, "hr")

        notes: list[str] = []
        if parse_note:
            notes.append(parse_note)
        notes.extend(lr_notes)
        notes.extend(hr_notes)

        gt_text = extract_gt_text(annotation) if isinstance(annotation, dict) else None
        layout = extract_layout(annotation) if isinstance(annotation, dict) else None
        if not gt_text:
            notes.append("gt_text_missing_or_empty")
        if not layout:
            notes.append("layout_missing_or_empty")
        elif layout != domain:
            notes.append(f"layout_mismatch:{layout}")

        expected_image_names = [path.name for path in (*lr_image_paths, *hr_image_paths)]
        if isinstance(annotation, dict):
            notes.extend(extract_annotation_notes(annotation, expected_image_names=expected_image_names))
        else:
            notes.append("annotation_not_a_dict")

        track_status = "ok" if not notes else "warning"
        row = {
            "track_id": track_id,
            "track_num": track_num,
            "scenario": scenario,
            "domain": domain,
            "track_dir": relativize(track_dir, dataset_root),
            "annotation_path": relativize(annotation_path, dataset_root),
            "gt_text": gt_text or "",
            "layout": layout or "",
            "track_status": track_status,
            "num_lr_images": len(lr_image_paths),
            "num_hr_images": len(hr_image_paths),
            "lr_image_paths": [relativize(path, dataset_root) for path in lr_image_paths],
            "hr_image_paths": [relativize(path, dataset_root) for path in hr_image_paths],
            "annotation_notes": notes,
        }
        rows.append(row)

        scenario_counts[scenario] += 1
        domain_counts[f"{scenario}/{domain}"] += 1
        status_counts[track_status] += 1
        if layout:
            layout_counts[layout] += 1
        for note in notes:
            note_counts[note] += 1
        total_lr_images += len(lr_image_paths)
        total_hr_images += len(hr_image_paths)

    rows.sort(key=_sorted_row_key)

    summary = {
        "dataset_root": dataset_root.as_posix(),
        "track_count": len(rows),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "track_status_counts": dict(sorted(status_counts.items())),
        "layout_counts": dict(sorted(layout_counts.items())),
        "image_counts": {
            "lr": total_lr_images,
            "hr": total_hr_images,
            "total": total_lr_images + total_hr_images,
        },
        "annotation_note_counts": dict(sorted(note_counts.items())),
        "skipped_annotation_files": skipped_annotations,
    }
    return rows, summary


def write_scan_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    manifests_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    manifests_dir = ensure_dir(manifests_dir)
    output_dir = ensure_dir(output_dir)
    csv_path = write_manifest_csv(rows, manifests_dir / "full_manifest.csv")
    jsonl_path = write_manifest_jsonl(rows, manifests_dir / "full_manifest.jsonl")
    summary_path = Path(output_dir) / "scan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "full_manifest_csv": csv_path,
        "full_manifest_jsonl": jsonl_path,
        "scan_summary_json": summary_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan the competition dataset and build manifests.")
    parser.add_argument("--dataset-root", required=True, help="Path to the training dataset root.")
    parser.add_argument(
        "--manifests-dir",
        default="./manifests",
        help="Directory that receives full_manifest.csv/jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory that receives scan_summary.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rows, summary = scan_dataset(args.dataset_root)
    paths = write_scan_outputs(rows, summary, manifests_dir=args.manifests_dir, output_dir=args.output_dir)

    print(f"Scanned {summary['track_count']} tracks from {Path(args.dataset_root).expanduser().resolve()}")
    print(f"Wrote {paths['full_manifest_csv']}")
    print(f"Wrote {paths['full_manifest_jsonl']}")
    print(f"Wrote {paths['scan_summary_json']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
