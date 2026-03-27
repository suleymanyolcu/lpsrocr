from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.data.manifest_io import read_manifest_csv, read_manifest_jsonl, write_manifest_csv, write_manifest_jsonl
from src.stage_a.gplpr import StageAPaths, run_gplpr_eval
from src.utils.paths import ensure_dir, repo_root

DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_LR_STAGE_DIR = Path("external_data/stage_c_lr")
DEFAULT_SR_STAGE_DIR = Path("external_data/stage_c_sr")
DEFAULT_OUTPUT_DIR = Path("outputs/stage_c")
DEFAULT_SPLIT_DIR = Path("manifests/splits/scenario_b_dev_seed42_n400_v20")
DEFAULT_LR_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "lr"
DEFAULT_SR_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "sr"
DEFAULT_COMPARISON_DIR = DEFAULT_OUTPUT_DIR / "comparison"
FRAME_PATTERN = re.compile(r"^(?P<prefix>lr|hr|sr)-(?P<index>\d+)$", re.IGNORECASE)
STAGE_PATH_ANCHOR = "images"
EXPECTED_FRAMES_PER_TRACK = 5


def _resolve_path(path: str | Path | None, *, base: Path) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def _project_relpath(path: str | Path, base: str | Path) -> str:
    path = Path(path)
    base = Path(base)
    try:
        return os.path.relpath(path, start=base)
    except ValueError:
        return path.as_posix()


def _frame_index(path_like: str | Path) -> int:
    match = FRAME_PATTERN.match(Path(path_like).stem)
    if not match:
        raise ValueError(f"Could not parse frame index from {path_like}")
    return int(match.group("index"))


def _ordered_paths(paths: list[str]) -> list[str]:
    return sorted((str(path) for path in paths), key=lambda item: (_frame_index(item), item))


def _copy_or_symlink(source: Path, destination: Path, mode: str) -> None:
    ensure_dir(destination.parent)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        relative_source = os.path.relpath(source, start=destination.parent)
        os.symlink(relative_source, destination)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unsupported staging mode: {mode}")


def _read_text_gt(label_path: Path) -> str:
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("plate:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"Could not find plate text in {label_path}")


def _levenshtein_distance(lhs: str, rhs: str) -> int:
    if lhs == rhs:
        return 0
    if not lhs:
        return len(rhs)
    if not rhs:
        return len(lhs)

    previous = list(range(len(rhs) + 1))
    for i, left_char in enumerate(lhs, start=1):
        current = [i]
        for j, right_char in enumerate(rhs, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _normalized_edit_distance(gt_text: str, pred_text: str) -> float:
    denom = max(len(gt_text), len(pred_text), 1)
    return _levenshtein_distance(gt_text, pred_text) / float(denom)


def _track_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["scenario"], row["domain"], row["track_num"], row["track_id"])


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _coerce_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expand_lr_val_rows(val_rows: list[dict[str, Any]], dataset_root: Path) -> list[dict[str, Any]]:
    frame_rows: list[dict[str, Any]] = []
    for row in sorted(val_rows, key=_track_key):
        lr_paths = _ordered_paths(row["lr_image_paths"])
        hr_paths = _ordered_paths(row["hr_image_paths"])
        if len(lr_paths) != len(hr_paths):
            raise ValueError(
                f"Track {row['track_id']} has mismatched LR/HR counts: {len(lr_paths)} vs {len(hr_paths)}"
            )
        for lr_rel, hr_rel in zip(lr_paths, hr_paths):
            lr_idx = _frame_index(lr_rel)
            hr_idx = _frame_index(hr_rel)
            if lr_idx != hr_idx:
                raise ValueError(
                    f"Track {row['track_id']} has mismatched LR/HR frame indices: {lr_rel} vs {hr_rel}"
                )
            source_abs_path = (dataset_root / lr_rel).resolve()
            frame_rows.append(
                {
                    "track_id": row["track_id"],
                    "track_num": row["track_num"],
                    "scenario": row["scenario"],
                    "domain": row["domain"],
                    "track_dir": row["track_dir"],
                    "frame_idx": lr_idx,
                    "gt_text": row["gt_text"],
                    "source_mode": "lr",
                    "source_image_path": lr_rel,
                    "source_abs_path": source_abs_path.as_posix(),
                    "staged_relpath": lr_rel,
                }
            )
    return sorted(frame_rows, key=lambda row: (_track_key(row), row["frame_idx"], row["source_image_path"]))


def _expand_sr_restoration_rows(restored_rows: list[dict[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    frame_rows: list[dict[str, Any]] = []
    for row in sorted(restored_rows, key=lambda item: (item["scenario"], item["domain"], item["track_num"], item["frame_idx"])):
        frame_idx = int(row["frame_idx"])
        restored_rel = row["restored_image_path"]
        source_abs_path = _resolve_path(restored_rel, base=project_root)
        if source_abs_path is None:
            raise ValueError(f"Could not resolve restored image path: {restored_rel}")
        frame_rows.append(
            {
                "track_id": row["track_id"],
                "track_num": row["track_num"],
                "scenario": row["scenario"],
                "domain": row["domain"],
                "track_dir": row["track_dir"],
                "frame_idx": frame_idx,
                "gt_text": row["gt_text"],
                "source_mode": "sr",
                "source_image_path": restored_rel,
                "source_abs_path": source_abs_path.as_posix(),
                "staged_relpath": f"{row['track_dir']}/sr-{frame_idx:03d}.png",
            }
        )
    return sorted(frame_rows, key=lambda row: (_track_key(row), row["frame_idx"], row["source_image_path"]))


def _stage_frame_rows(
    *,
    frame_rows: list[dict[str, Any]],
    project_root: Path,
    stage_dir: Path,
    mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    stage_manifest_rows: list[dict[str, Any]] = []
    split_lines: list[str] = []
    for row in frame_rows:
        source_abs = Path(row["source_abs_path"]).expanduser().resolve()
        if not source_abs.exists():
            raise FileNotFoundError(f"Source image not found: {source_abs}")

        staged_relpath = Path(row["staged_relpath"])
        staged_path = stage_dir / "images" / "validation" / staged_relpath
        _copy_or_symlink(source_abs, staged_path, mode)

        label_path = staged_path.with_suffix(".txt")
        ensure_dir(label_path.parent)
        label_path.write_text(
            "\n".join(
                [
                    f"track_id: {row['track_id']}",
                    f"track_num: {row['track_num']}",
                    f"scenario: {row['scenario']}",
                    f"domain: {row['domain']}",
                    f"track_dir: {row['track_dir']}",
                    f"frame_idx: {row['frame_idx']}",
                    f"source_mode: {row['source_mode']}",
                    f"source_image_path: {row['source_image_path']}",
                    f"plate: {row['gt_text']}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        staged_rel = _project_relpath(staged_path, project_root)
        stage_manifest_rows.append(
            {
                "track_id": row["track_id"],
                "track_num": row["track_num"],
                "scenario": row["scenario"],
                "domain": row["domain"],
                "track_dir": row["track_dir"],
                "frame_idx": row["frame_idx"],
                "split_label": "validation",
                "source_mode": row["source_mode"],
                "source_image_path": row["source_image_path"],
                "source_abs_path": source_abs.as_posix(),
                "staged_image_path": staged_rel,
                "label_path": _project_relpath(label_path, project_root),
                "gt_text": row["gt_text"],
            }
        )
        split_lines.append(f"{staged_rel};validation")

    return stage_manifest_rows, split_lines


def _build_stage_a_paths(
    *,
    project_root: Path,
    dataset_root: Path | None,
    gplpr_repo: Path,
    stage_dir: Path,
    output_dir: Path,
    run_tag: str,
) -> StageAPaths:
    train_manifest = stage_dir / "train_manifest.csv"
    val_manifest = stage_dir / "validation_manifest.csv"
    return StageAPaths(
        project_root=project_root,
        dataset_root=dataset_root or project_root,
        gplpr_repo=gplpr_repo,
        split_dir=stage_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        stage_dir=stage_dir,
        output_dir=output_dir,
        split_file=stage_dir / "split_validation.txt",
        train_config=output_dir / "configs" / f"{run_tag}_train.yaml",
        eval_config=output_dir / "configs" / f"{run_tag}_eval.yaml",
        run_tag=run_tag,
    )


def _aggregate_track_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: (row["frame_idx"], row["image_path"]))
    predictions = [row["pred_text"] for row in ordered_rows]
    counts = Counter(predictions)
    has_frame_confidence = any(row.get("confidence") is not None for row in ordered_rows)

    def _frame_confidence(row: dict[str, Any]) -> float:
        confidence = row.get("confidence")
        return float(confidence) if confidence is not None else float("-inf")

    def _mean_confidence(text: str) -> float | None:
        values = [
            float(row["confidence"])
            for row in ordered_rows
            if row["pred_text"] == text and row.get("confidence") is not None
        ]
        return _mean_or_none(values)

    def _first_seen_index(text: str) -> int:
        for index, row in enumerate(ordered_rows):
            if row["pred_text"] == text:
                return index
        return len(ordered_rows)

    def _support_fraction(text: str) -> float:
        return counts[text] / float(len(ordered_rows))

    if has_frame_confidence:
        best_confidence_pred_text = min(
            counts.keys(),
            key=lambda text: (
                -(_mean_confidence(text) if _mean_confidence(text) is not None else float("-inf")),
                -counts[text],
                _first_seen_index(text),
                text,
            ),
        )
        majority_vote_pred_text = min(
            counts.keys(),
            key=lambda text: (
                -counts[text],
                -(_mean_confidence(text) if _mean_confidence(text) is not None else float("-inf")),
                text,
            ),
        )
    else:
        best_confidence_pred_text = min(
            counts.keys(),
            key=lambda text: (-counts[text], _first_seen_index(text), text),
        )
        majority_vote_pred_text = min(
            counts.keys(),
            key=lambda text: (-counts[text], text),
        )

    def _aggregated_confidence(text: str) -> float:
        mean_conf = _mean_confidence(text)
        if mean_conf is not None:
            return float(mean_conf)
        return _support_fraction(text)

    return {
        "best_confidence_pred_text": best_confidence_pred_text,
        "best_confidence": _aggregated_confidence(best_confidence_pred_text),
        "best_confidence_support_fraction": _support_fraction(best_confidence_pred_text),
        "majority_vote_pred_text": majority_vote_pred_text,
        "majority_vote": _aggregated_confidence(majority_vote_pred_text),
        "majority_vote_support_fraction": _support_fraction(majority_vote_pred_text),
        "majority_vote_count": counts[majority_vote_pred_text],
        "num_unique_predictions": len(counts),
        "has_frame_confidence": has_frame_confidence,
    }


def _load_stage_manifest_index(stage_manifest_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["staged_image_path"]: row for row in stage_manifest_rows}


def _augment_per_image_rows(
    *,
    raw_rows: list[dict[str, Any]],
    stage_manifest_rows: list[dict[str, Any]],
    source_mode: str,
) -> list[dict[str, Any]]:
    stage_index = _load_stage_manifest_index(stage_manifest_rows)
    augmented_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        image_path = row.get("image_path") or row.get("Image Name") or row.get("image_name") or row.get("img") or ""
        staged_meta = stage_index.get(image_path)
        if staged_meta is None:
            raise KeyError(f"Could not find staged metadata for image path: {image_path}")

        gt_text = row.get("gt_text") or staged_meta["gt_text"]
        pred_text = (row.get("pred_text") or row.get("Prediction") or row.get("prediction") or "").strip()
        confidence = _coerce_float(row.get("confidence") or row.get("Confidence") or row.get("score"))
        exact_match = pred_text == gt_text
        edit_distance = int(row.get("edit_distance") or _levenshtein_distance(gt_text, pred_text))
        normalized_edit_distance = float(
            row.get("normalized_edit_distance") or _normalized_edit_distance(gt_text, pred_text)
        )
        augmented_rows.append(
            {
                "track_id": staged_meta["track_id"],
                "track_num": staged_meta["track_num"],
                "scenario": staged_meta["scenario"],
                "domain": staged_meta["domain"],
                "track_dir": staged_meta["track_dir"],
                "frame_idx": int(staged_meta["frame_idx"]),
                "source_mode": source_mode,
                "image_path": image_path,
                "source_image_path": staged_meta["source_image_path"],
                "staged_image_path": staged_meta["staged_image_path"],
                "label_path": staged_meta["label_path"],
                "gt_text": gt_text,
                "pred_text": pred_text,
                "confidence": confidence,
                "exact_match": exact_match,
                "edit_distance": edit_distance,
                "normalized_edit_distance": normalized_edit_distance,
            }
        )
    return sorted(augmented_rows, key=lambda row: (_track_key(row), row["frame_idx"], row["image_path"]))


def _aggregate_per_track_rows(per_image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    track_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        track_groups[row["track_id"]].append(row)

    per_track_rows: list[dict[str, Any]] = []
    for track_id in sorted(track_groups):
        rows = sorted(track_groups[track_id], key=lambda row: (row["frame_idx"], row["image_path"]))
        if len(rows) != EXPECTED_FRAMES_PER_TRACK:
            raise ValueError(f"Track {track_id} has {len(rows)} predictions, expected {EXPECTED_FRAMES_PER_TRACK}")
        aggregate = _aggregate_track_rows(rows)
        gt_text = rows[0]["gt_text"]
        per_track_rows.append(
            {
                "track_id": track_id,
                "track_num": rows[0]["track_num"],
                "scenario": rows[0]["scenario"],
                "domain": rows[0]["domain"],
                "track_dir": rows[0]["track_dir"],
                "gt_text": gt_text,
                "num_frames": len(rows),
                "aggregated_text_best_conf": aggregate["best_confidence_pred_text"],
                "aggregated_conf_best_conf": aggregate["best_confidence"],
                "best_confidence_support_fraction": aggregate["best_confidence_support_fraction"],
                "exact_match_best_conf": aggregate["best_confidence_pred_text"] == gt_text,
                "aggregated_text_majority": aggregate["majority_vote_pred_text"],
                "aggregated_conf_majority": aggregate["majority_vote"],
                "majority_support_fraction": aggregate["majority_vote_support_fraction"],
                "majority_vote_count": aggregate["majority_vote_count"],
                "exact_match_majority": aggregate["majority_vote_pred_text"] == gt_text,
                "track_exact_match": aggregate["majority_vote_pred_text"] == gt_text,
                "mean_frame_confidence": _mean_or_none(
                    [float(row["confidence"]) for row in rows if row.get("confidence") is not None]
                ),
                "num_unique_predictions": aggregate["num_unique_predictions"],
            }
        )
    return sorted(per_track_rows, key=lambda row: (row["track_num"], row["track_id"]))


def _frame_summary_from_rows(per_image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_match_rate = (
        sum(1 for row in per_image_rows if row["exact_match"]) / len(per_image_rows) if per_image_rows else 0.0
    )
    mean_edit_distance = _mean_or_none([float(row["edit_distance"]) for row in per_image_rows])
    mean_normalized_edit_distance = _mean_or_none(
        [float(row["normalized_edit_distance"]) for row in per_image_rows]
    )
    scenario_counts = Counter(row["scenario"] for row in per_image_rows)
    domain_counts = Counter(f"{row['scenario']}/{row['domain']}" for row in per_image_rows)
    return {
        "num_images": len(per_image_rows),
        "image_exact_match_rate": exact_match_rate,
        "mean_edit_distance": mean_edit_distance,
        "mean_normalized_edit_distance": mean_normalized_edit_distance,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def _track_summary_from_rows(per_track_rows: list[dict[str, Any]]) -> dict[str, Any]:
    majority_rate = (
        sum(1 for row in per_track_rows if row["exact_match_majority"]) / len(per_track_rows)
        if per_track_rows
        else 0.0
    )
    best_rate = (
        sum(1 for row in per_track_rows if row["exact_match_best_conf"]) / len(per_track_rows)
        if per_track_rows
        else 0.0
    )
    mean_track_confidence = _mean_or_none([float(row["aggregated_conf_majority"]) for row in per_track_rows])
    mean_best_confidence = _mean_or_none([float(row["aggregated_conf_best_conf"]) for row in per_track_rows])
    mean_track_edit_distance_majority = _mean_or_none(
        [float(_levenshtein_distance(row["gt_text"], row["aggregated_text_majority"])) for row in per_track_rows]
    )
    mean_track_normalized_edit_distance_majority = _mean_or_none(
        [
            float(_normalized_edit_distance(row["gt_text"], row["aggregated_text_majority"]))
            for row in per_track_rows
        ]
    )
    scenario_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    domain_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_track_rows:
        scenario_groups[row["scenario"]].append(row)
        domain_groups[f"{row['scenario']}/{row['domain']}"] .append(row)
    by_scenario = {
        scenario: {
            "tracks": len(rows),
            "majority_exact_match_rate": sum(1 for row in rows if row["exact_match_majority"]) / len(rows),
            "best_conf_exact_match_rate": sum(1 for row in rows if row["exact_match_best_conf"]) / len(rows),
        }
        for scenario, rows in sorted(scenario_groups.items())
    }
    by_domain = {
        domain: {
            "tracks": len(rows),
            "majority_exact_match_rate": sum(1 for row in rows if row["exact_match_majority"]) / len(rows),
            "best_conf_exact_match_rate": sum(1 for row in rows if row["exact_match_best_conf"]) / len(rows),
        }
        for domain, rows in sorted(domain_groups.items())
    }
    return {
        "num_tracks": len(per_track_rows),
        "track_exact_match_majority_rate": majority_rate,
        "track_exact_match_best_conf_rate": best_rate,
        "mean_confidence_majority": mean_track_confidence,
        "mean_confidence_best_conf": mean_best_confidence,
        "mean_track_edit_distance_majority": mean_track_edit_distance_majority,
        "mean_track_normalized_edit_distance_majority": mean_track_normalized_edit_distance_majority,
        "by_scenario": by_scenario,
        "by_domain": by_domain,
    }


def _sanity_checks(
    *,
    per_image_rows: list[dict[str, Any]],
    per_track_rows: list[dict[str, Any]],
    expected_tracks: int,
    source_mode: str,
) -> dict[str, Any]:
    duplicate_rows = len(per_image_rows) - len(
        {(row["track_id"], row["frame_idx"]) for row in per_image_rows}
    )
    frames_per_track = Counter(row["track_id"] for row in per_image_rows)
    frame_count_distribution = dict(sorted(Counter(frames_per_track.values()).items()))
    tracks_with_bad_frame_count = sorted(
        track_id for track_id, count in frames_per_track.items() if count != EXPECTED_FRAMES_PER_TRACK
    )
    track_ids = sorted(row["track_id"] for row in per_track_rows)
    return {
        "source_mode": source_mode,
        "expected_tracks": expected_tracks,
        "observed_tracks": len(per_track_rows),
        "observed_images": len(per_image_rows),
        "duplicate_track_frame_rows": duplicate_rows,
        "frame_count_distribution": frame_count_distribution,
        "tracks_with_bad_frame_count": tracks_with_bad_frame_count,
        "track_ids_preview": track_ids[:5],
        "all_tracks_have_five_frames": not tracks_with_bad_frame_count,
    }


def _write_run_outputs(
    *,
    output_dir: Path,
    source_mode: str,
    checkpoint: Path,
    stage_manifest_rows: list[dict[str, Any]],
    per_image_rows: list[dict[str, Any]],
    per_track_rows: list[dict[str, Any]],
    expected_tracks: int,
    raw_eval_summary: dict[str, Any],
    run_tag: str,
    source_manifest_path: Path,
    stage_dir: Path,
    split_file: Path,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)

    per_image_csv = output_dir / "per_image.csv"
    per_track_csv = output_dir / "per_track.csv"
    per_image_fields = [
        "track_id",
        "track_num",
        "scenario",
        "domain",
        "track_dir",
        "frame_idx",
        "source_mode",
        "image_path",
        "source_image_path",
        "staged_image_path",
        "label_path",
        "gt_text",
        "pred_text",
        "confidence",
        "exact_match",
        "edit_distance",
        "normalized_edit_distance",
    ]
    per_track_fields = [
        "track_id",
        "track_num",
        "scenario",
        "domain",
        "track_dir",
        "gt_text",
        "num_frames",
        "aggregated_text_best_conf",
        "aggregated_conf_best_conf",
        "best_confidence_support_fraction",
        "exact_match_best_conf",
        "aggregated_text_majority",
        "aggregated_conf_majority",
        "majority_support_fraction",
        "majority_vote_count",
        "exact_match_majority",
        "track_exact_match",
        "mean_frame_confidence",
        "num_unique_predictions",
    ]
    write_manifest_csv(per_image_rows, per_image_csv, fieldnames=per_image_fields)
    write_manifest_csv(per_track_rows, per_track_csv, fieldnames=per_track_fields)

    frame_summary = _frame_summary_from_rows(per_image_rows)
    track_summary = _track_summary_from_rows(per_track_rows)
    summary = {
        "source_mode": source_mode,
        "checkpoint": checkpoint.as_posix(),
        "project_root": repo_root().as_posix(),
        "stage_dir": stage_dir.as_posix(),
        "split_file": split_file.as_posix(),
        "source_manifest": source_manifest_path.as_posix(),
        "mode": mode,
        "seed": seed,
        "run_tag": run_tag,
        "raw_eval_summary": raw_eval_summary,
        "per_image_csv": per_image_csv.as_posix(),
        "per_track_csv": per_track_csv.as_posix(),
        "frame_summary": frame_summary,
        "track_summary": track_summary,
        "sanity_checks": _sanity_checks(
            per_image_rows=per_image_rows,
            per_track_rows=per_track_rows,
            expected_tracks=expected_tracks,
            source_mode=source_mode,
        ),
        "confidence_note": (
            "GPLPR does not expose per-image confidence; track confidence is a deterministic vote-share proxy."
        ),
        "per_image_preview": per_image_rows[:3],
        "per_track_preview": per_track_rows[:3],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_path"] = summary_path.as_posix()

    stage_manifest_csv = stage_dir / "validation_manifest.csv"
    stage_manifest_jsonl = stage_dir / "validation_manifest.jsonl"
    write_manifest_csv(stage_manifest_rows, stage_manifest_csv)
    write_manifest_jsonl(stage_manifest_rows, stage_manifest_jsonl)
    split_text = "\n".join(
        f"{row['staged_image_path']};validation" for row in stage_manifest_rows
    )
    split_file.write_text(split_text + "\n", encoding="utf-8")

    setup_summary = {
        "source_mode": source_mode,
        "checkpoint": checkpoint.as_posix(),
        "project_root": repo_root().as_posix(),
        "stage_dir": stage_dir.as_posix(),
        "split_file": split_file.as_posix(),
        "stage_manifest_csv": stage_manifest_csv.as_posix(),
        "stage_manifest_jsonl": stage_manifest_jsonl.as_posix(),
        "output_dir": output_dir.as_posix(),
        "run_tag": run_tag,
        "mode": mode,
        "seed": seed,
        "num_stage_frames": len(stage_manifest_rows),
        "num_stage_tracks": len({row["track_id"] for row in stage_manifest_rows}),
    }
    setup_summary_path = output_dir / "stage_c_setup_summary.json"
    setup_summary_path.write_text(json.dumps(setup_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["setup_summary_path"] = setup_summary_path.as_posix()
    summary["stage_manifest_csv"] = stage_manifest_csv.as_posix()
    summary["stage_manifest_jsonl"] = stage_manifest_jsonl.as_posix()
    summary["raw_eval_per_image_jsonl"] = raw_eval_summary.get("per_image_jsonl", "")
    return summary


def _run_stage_c_ocr_pipeline(
    *,
    project_root: Path,
    gplpr_repo: Path,
    stage_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    frame_rows: list[dict[str, Any]],
    expected_tracks: int,
    source_manifest_path: Path,
    source_mode: str,
    mode: str,
    run_tag: str,
    seed: int = 42,
) -> dict[str, Any]:
    ensure_dir(stage_dir)
    ensure_dir(output_dir)
    stage_manifest_rows, _split_lines = _stage_frame_rows(
        frame_rows=frame_rows,
        project_root=project_root,
        stage_dir=stage_dir,
        mode=mode,
    )

    stage_manifest_csv = stage_dir / "validation_manifest.csv"
    stage_manifest_jsonl = stage_dir / "validation_manifest.jsonl"
    split_file = stage_dir / "split_validation.txt"
    write_manifest_csv(stage_manifest_rows, stage_manifest_csv)
    write_manifest_jsonl(stage_manifest_rows, stage_manifest_jsonl)
    split_file.write_text(
        "\n".join(f"{row['staged_image_path']};validation" for row in stage_manifest_rows) + "\n",
        encoding="utf-8",
    )

    stage_paths = _build_stage_a_paths(
        project_root=project_root,
        dataset_root=None,
        gplpr_repo=gplpr_repo,
        stage_dir=stage_dir,
        output_dir=output_dir,
        run_tag=run_tag,
    )

    raw_eval_summary = run_gplpr_eval(
        paths=stage_paths,
        checkpoint=checkpoint,
        seed=seed,
    )
    raw_per_image_rows = read_manifest_jsonl(raw_eval_summary["per_image_jsonl"])
    per_image_rows = _augment_per_image_rows(
        raw_rows=raw_per_image_rows,
        stage_manifest_rows=stage_manifest_rows,
        source_mode=source_mode,
    )
    per_track_rows = _aggregate_per_track_rows(per_image_rows)
    return _write_run_outputs(
        output_dir=output_dir,
        source_mode=source_mode,
        checkpoint=checkpoint,
        stage_manifest_rows=stage_manifest_rows,
        per_image_rows=per_image_rows,
        per_track_rows=per_track_rows,
        expected_tracks=expected_tracks,
        raw_eval_summary=raw_eval_summary,
        run_tag=run_tag,
        source_manifest_path=source_manifest_path,
        stage_dir=stage_dir,
        split_file=split_file,
        mode=mode,
        seed=seed,
    )


def run_stage_c_lr(
    *,
    project_root: str | Path = repo_root(),
    dataset_root: str | Path | None = None,
    gplpr_repo: str | Path,
    val_manifest: str | Path | None = None,
    split_dir: str | Path | None = None,
    stage_dir: str | Path = DEFAULT_LR_STAGE_DIR,
    output_dir: str | Path = DEFAULT_LR_OUTPUT_DIR,
    checkpoint: str | Path,
    mode: str = "symlink",
    run_tag: str | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    dataset_root_path = _resolve_path(dataset_root or "train", base=project_root)
    gplpr_repo = Path(gplpr_repo).expanduser().resolve()
    stage_dir = _resolve_path(stage_dir, base=project_root)
    output_dir = _resolve_path(output_dir, base=project_root)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GPLPR checkpoint not found: {checkpoint_path}")

    if val_manifest is None:
        if split_dir is None:
            split_dir = project_root / DEFAULT_SPLIT_DIR
        val_manifest_path = Path(split_dir).expanduser().resolve() / "val_manifest.csv"
    else:
        val_manifest_path = Path(val_manifest).expanduser().resolve()
    if not val_manifest_path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest_path}")
    val_rows = read_manifest_csv(val_manifest_path)
    frame_rows = _expand_lr_val_rows(val_rows, dataset_root_path)
    resolved_run_tag = run_tag or f"stage_c_lr_{val_manifest_path.parent.name}"

    if dry_run:
        return {
            "dry_run": True,
            "source_mode": "lr",
            "checkpoint": checkpoint_path.as_posix(),
            "stage_dir": stage_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "run_tag": resolved_run_tag,
            "num_frames": len(frame_rows),
            "num_tracks": len({row["track_id"] for row in frame_rows}),
        }

    return _run_stage_c_ocr_pipeline(
        project_root=project_root,
        gplpr_repo=gplpr_repo,
        stage_dir=stage_dir,
        output_dir=output_dir,
        checkpoint=checkpoint_path,
        frame_rows=frame_rows,
        expected_tracks=len(val_rows),
        source_manifest_path=val_manifest_path,
        source_mode="lr",
        mode=mode,
        run_tag=resolved_run_tag,
        seed=seed,
    )


def run_stage_c_sr(
    *,
    project_root: str | Path = repo_root(),
    gplpr_repo: str | Path,
    restored_manifest: str | Path | None = None,
    stage_b_output_dir: str | Path | None = None,
    stage_dir: str | Path = DEFAULT_SR_STAGE_DIR,
    output_dir: str | Path = DEFAULT_SR_OUTPUT_DIR,
    checkpoint: str | Path,
    mode: str = "symlink",
    run_tag: str | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    gplpr_repo = Path(gplpr_repo).expanduser().resolve()
    stage_dir = _resolve_path(stage_dir, base=project_root)
    output_dir = _resolve_path(output_dir, base=project_root)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GPLPR checkpoint not found: {checkpoint_path}")

    if restored_manifest is None:
        if stage_b_output_dir is None:
            stage_b_output_dir = project_root / "outputs" / "stage_b"
        restored_manifest_path = Path(stage_b_output_dir).expanduser().resolve() / "restored" / "restoration_manifest.jsonl"
    else:
        restored_manifest_path = Path(restored_manifest).expanduser().resolve()
    if not restored_manifest_path.exists():
        raise FileNotFoundError(f"Restoration manifest not found: {restored_manifest_path}")
    restored_rows = read_manifest_jsonl(restored_manifest_path)
    frame_rows = _expand_sr_restoration_rows(restored_rows, project_root)
    resolved_run_tag = run_tag or f"stage_c_sr_{restored_manifest_path.parent.parent.name}"

    if dry_run:
        return {
            "dry_run": True,
            "source_mode": "sr",
            "checkpoint": checkpoint_path.as_posix(),
            "stage_dir": stage_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "run_tag": resolved_run_tag,
            "num_frames": len(frame_rows),
            "num_tracks": len({row["track_id"] for row in frame_rows}),
        }

    return _run_stage_c_ocr_pipeline(
        project_root=project_root,
        gplpr_repo=gplpr_repo,
        stage_dir=stage_dir,
        output_dir=output_dir,
        checkpoint=checkpoint_path,
        frame_rows=frame_rows,
        expected_tracks=len({row["track_id"] for row in frame_rows}),
        source_manifest_path=restored_manifest_path,
        source_mode="sr",
        mode=mode,
        run_tag=resolved_run_tag,
        seed=seed,
    )


def _load_summary_and_tracks(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = output_dir / "summary.json"
    per_track_csv = output_dir / "per_track.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    if not per_track_csv.exists():
        raise FileNotFoundError(f"Per-track CSV not found: {per_track_csv}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_track_rows = read_manifest_csv(per_track_csv)
    return summary, per_track_rows


def build_stage_c_arg_parser_lr() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GPLPR on original LR validation images.")
    parser.add_argument("--project-root", default=str(repo_root()), help="Project repository root.")
    parser.add_argument("--dataset-root", default=None, help="Competition dataset root.")
    parser.add_argument("--gplpr-repo", required=True, help="Path to the GPLPR repository clone.")
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Scenario-B split directory.")
    parser.add_argument("--val-manifest", default=None, help="Optional explicit validation manifest path.")
    parser.add_argument(
        "--checkpoint",
        "--gplpr-checkpoint",
        dest="checkpoint",
        required=True,
        help="GPLPR OCR checkpoint from Stage A.",
    )
    parser.add_argument("--stage-dir", default=str(DEFAULT_LR_STAGE_DIR), help="Stage C LR dataset directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_LR_OUTPUT_DIR), help="Stage C LR output directory.")
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize staged images.",
    )
    parser.add_argument("--run-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded in summaries.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths without writing files.")
    return parser


def build_stage_c_arg_parser_sr() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GPLPR on SR-restored validation images.")
    parser.add_argument("--project-root", default=str(repo_root()), help="Project repository root.")
    parser.add_argument("--gplpr-repo", required=True, help="Path to the GPLPR repository clone.")
    parser.add_argument(
        "--restored-manifest",
        default=None,
        help="Optional explicit restoration_manifest.jsonl path from Stage B.",
    )
    parser.add_argument(
        "--stage-b-output-dir",
        default=None,
        help="Stage B output root. Used to infer restored/restoration_manifest.jsonl when --restored-manifest is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        "--gplpr-checkpoint",
        dest="checkpoint",
        required=True,
        help="GPLPR OCR checkpoint from Stage A.",
    )
    parser.add_argument("--stage-dir", default=str(DEFAULT_SR_STAGE_DIR), help="Stage C SR dataset directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_SR_OUTPUT_DIR), help="Stage C SR output directory.")
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize staged images.",
    )
    parser.add_argument("--run-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded in summaries.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths without writing files.")
    return parser


def main_run_lr(argv: list[str] | None = None) -> int:
    parser = build_stage_c_arg_parser_lr()
    args = parser.parse_args(argv)
    summary = run_stage_c_lr(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        gplpr_repo=args.gplpr_repo,
        val_manifest=args.val_manifest,
        split_dir=args.split_dir,
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        mode=args.mode,
        run_tag=args.run_tag,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main_run_sr(argv: list[str] | None = None) -> int:
    parser = build_stage_c_arg_parser_sr()
    args = parser.parse_args(argv)
    summary = run_stage_c_sr(
        project_root=args.project_root,
        gplpr_repo=args.gplpr_repo,
        restored_manifest=args.restored_manifest,
        stage_b_output_dir=args.stage_b_output_dir,
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        mode=args.mode,
        run_tag=args.run_tag,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0
