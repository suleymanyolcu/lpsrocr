from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.data.manifest_io import read_manifest_csv
from src.utils.paths import ensure_dir


def _staged_image_path(output_dir: Path, relative_source_path: str) -> Path:
    return output_dir / "staged" / relative_source_path


def _materialize_file(source: Path, destination: Path, mode: str) -> None:
    ensure_dir(destination.parent)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        relative_source = os.path.relpath(source, start=destination.parent)
        os.symlink(relative_source, destination)
    else:
        raise ValueError(f"Unsupported staging mode: {mode}")


def _collect_rows(split_dir: str | Path, subset: str) -> list[dict[str, Any]]:
    split_dir = Path(split_dir)
    rows: list[dict[str, Any]] = []
    if subset in {"train", "both"}:
        rows.extend(read_manifest_csv(split_dir / "train_manifest.csv"))
    if subset in {"val", "both"}:
        rows.extend(read_manifest_csv(split_dir / "val_manifest.csv"))
    rows.sort(key=lambda row: (row["scenario"], row["domain"], row["track_num"], row["track_id"]))
    return rows


def export_gplpr_dataset(
    *,
    dataset_root: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    mode: str = "symlink",
    subset: str = "both",
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    split_dir = Path(split_dir).expanduser().resolve()
    output_dir = ensure_dir(output_dir)
    staged_root = ensure_dir(output_dir / "staged") if mode in {"copy", "symlink"} else None

    train_ids = {row["track_id"] for row in read_manifest_csv(split_dir / "train_manifest.csv")}
    rows = _collect_rows(split_dir, subset)
    lines: list[str] = []
    staged_count = 0

    for row in rows:
        split_label = "training" if row["track_id"] in train_ids else "validation"
        for relative_image_path in row["hr_image_paths"]:
            source = dataset_root / relative_image_path
            if mode in {"copy", "symlink"}:
                assert staged_root is not None
                staged_path = _staged_image_path(output_dir, relative_image_path)
                _materialize_file(source, staged_path, mode)
                label_path = staged_path.with_suffix(".txt")
                ensure_dir(label_path.parent)
                label_path.write_text(f"{row['gt_text']}\n", encoding="utf-8")
                output_path = staged_path.relative_to(output_dir).as_posix()
                staged_count += 1
            elif mode == "paths":
                output_path = relative_image_path
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            lines.append(f"{output_path};{split_label}")

    split_file = output_dir / "gplpr_split.txt"
    split_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "split_file": split_file,
        "line_count": len(lines),
        "staged_count": staged_count,
        "mode": mode,
        "subset": subset,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a GPLPR-friendly OCR staging tree.")
    parser.add_argument("--dataset-root", required=True, help="Path to the training dataset root.")
    parser.add_argument("--split-dir", required=True, help="Directory containing train/val split manifests.")
    parser.add_argument("--output-dir", required=True, help="Directory that receives the export.")
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "paths"),
        default="symlink",
        help="Staging mode for image files.",
    )
    parser.add_argument(
        "--subset",
        choices=("train", "val", "both"),
        default="both",
        help="Which split rows to export.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = export_gplpr_dataset(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        subset=args.subset,
    )
    print(f"Wrote {result['split_file']}")
    print(f"Lines: {result['line_count']}")
    print(f"Staged images: {result['staged_count']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
