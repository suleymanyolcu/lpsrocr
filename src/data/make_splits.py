from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.manifest_io import write_manifest_csv, write_manifest_jsonl
from src.data.scan_dataset import scan_dataset
from src.utils.paths import ensure_dir


def _parse_multi_values(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    parsed: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.append(part)
    return parsed or None


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["scenario"], row["domain"], row["track_num"], row["track_id"])


def _group_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups[key].append(row)
    return groups


def _shuffle_rows(rows: list[dict[str, Any]], seed: int, salt: str) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(f"{seed}:{salt}").shuffle(shuffled)
    return shuffled


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _apportion_counts(sizes: dict[tuple[Any, ...], int], target_total: int) -> dict[tuple[Any, ...], int]:
    total = sum(sizes.values())
    if target_total >= total:
        return dict(sizes)
    if target_total <= 0:
        return {key: 0 for key in sizes}

    quotas: dict[tuple[Any, ...], float] = {
        key: (size * target_total) / total for key, size in sizes.items()
    }
    counts = {key: int(math.floor(quotas[key])) for key in sizes}
    remainder = target_total - sum(counts.values())
    ranked_keys = sorted(
        sizes.keys(),
        key=lambda key: (-(quotas[key] - counts[key]), key),
    )
    for key in ranked_keys[:remainder]:
        counts[key] += 1
    return counts


def _stratified_take(
    rows: list[dict[str, Any]],
    target_total: int,
    *,
    seed: int,
    salt: str,
    group_fields: tuple[str, ...] = ("scenario", "domain"),
) -> list[dict[str, Any]]:
    if target_total >= len(rows):
        return sorted(rows, key=_row_key)

    grouped = _group_rows(rows, group_fields)
    counts = _apportion_counts({key: len(group) for key, group in grouped.items()}, target_total)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = _shuffle_rows(grouped[key], seed, f"{salt}:{key}")
        selected.extend(group_rows[: counts[key]])
    return sorted(selected, key=_row_key)


def _stratified_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    group_fields: tuple[str, ...] = ("scenario", "domain"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []

    grouped = _group_rows(rows, group_fields)
    total = len(rows)
    val_total = _round_half_up(total * val_ratio)
    counts = _apportion_counts({key: len(group) for key, group in grouped.items()}, val_total)

    val_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = _shuffle_rows(grouped[key], seed, f"val:{key}")
        val_rows.extend(group_rows[: counts[key]])

    val_ids = {row["track_id"] for row in val_rows}
    train_rows = [row for row in rows if row["track_id"] not in val_ids]
    return sorted(train_rows, key=_row_key), sorted(val_rows, key=_row_key)


def _scenario_b_split(
    rows: list[dict[str, Any]], *, seed: int, val_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [row for row in rows if row["scenario"] == "Scenario-B"]
    if not eligible:
        raise ValueError("scenario_b_dev requires at least one Scenario-B track after filtering")
    grouped = _group_rows(eligible, ("scenario", "domain"))
    val_total = _round_half_up(len(eligible) * val_ratio)
    counts = _apportion_counts({key: len(group) for key, group in grouped.items()}, val_total)

    val_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = _shuffle_rows(grouped[key], seed, f"scenario_b_val:{key}")
        val_rows.extend(group_rows[: counts[key]])

    val_ids = {row["track_id"] for row in val_rows}
    train_rows = [row for row in rows if row["track_id"] not in val_ids]
    return sorted(train_rows, key=_row_key), sorted(val_rows, key=_row_key)


def _normalise_filters(values: list[str] | None) -> set[str] | None:
    parsed = _parse_multi_values(values)
    if parsed is None:
        return None
    return set(parsed)


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    scenarios: set[str] | None = None,
    domains: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        if scenarios is not None and row["scenario"] not in scenarios:
            continue
        if domains is not None and row["domain"] not in domains:
            continue
        filtered.append(row)
    return filtered


def _count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_counts = Counter(row["scenario"] for row in rows)
    domain_counts = Counter(f'{row["scenario"]}/{row["domain"]}' for row in rows)
    return {
        "tracks": len(rows),
        "lr_images": sum(row["num_lr_images"] for row in rows),
        "hr_images": sum(row["num_hr_images"] for row in rows),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def _build_split_name(split_mode: str, seed: int, val_ratio: float, max_tracks_total: int | None) -> str:
    ratio_tag = f"v{int(round(val_ratio * 100)):02d}"
    total_tag = f"_n{max_tracks_total}" if max_tracks_total is not None else ""
    if split_mode == "debug_small":
        return f"{split_mode}_seed{seed}{total_tag}_{ratio_tag}"
    return f"{split_mode}_seed{seed}{total_tag}_{ratio_tag}"


def make_split(
    rows: list[dict[str, Any]],
    *,
    split_mode: str,
    seed: int,
    val_ratio: float,
    max_tracks_total: int | None = None,
    max_train_tracks: int | None = None,
    max_val_tracks: int | None = None,
    scenarios: list[str] | None = None,
    domains: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scenarios_filter = _normalise_filters(scenarios)
    domains_filter = _normalise_filters(domains)
    filtered_rows = _filter_rows(rows, scenarios=scenarios_filter, domains=domains_filter)
    if not filtered_rows:
        raise ValueError("No tracks remain after applying the requested scenario/domain filters")

    if split_mode == "debug_small" and max_tracks_total is None:
        max_tracks_total = 32

    selected_rows = filtered_rows
    if max_tracks_total is not None:
        selected_rows = _stratified_take(
            selected_rows,
            max_tracks_total,
            seed=seed,
            salt="subset",
        )

    if split_mode == "balanced_dev" or split_mode == "debug_small":
        train_rows, val_rows = _stratified_split(selected_rows, seed=seed, val_ratio=val_ratio)
    elif split_mode == "scenario_b_dev":
        train_rows, val_rows = _scenario_b_split(selected_rows, seed=seed, val_ratio=val_ratio)
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    if max_train_tracks is not None:
        train_rows = train_rows[:max_train_tracks]
    if max_val_tracks is not None:
        val_rows = val_rows[:max_val_tracks]

    overlap = {row["track_id"] for row in train_rows} & {row["track_id"] for row in val_rows}
    if overlap:
        raise ValueError(f"Train/val leakage detected for track ids: {sorted(overlap)[:10]}")

    split_name = _build_split_name(split_mode, seed, val_ratio, max_tracks_total)
    metadata = {
        "split_name": split_name,
        "split_mode": split_mode,
        "seed": seed,
        "val_ratio": val_ratio,
        "filters": {
            "scenarios": sorted(scenarios_filter) if scenarios_filter else None,
            "domains": sorted(domains_filter) if domains_filter else None,
        },
        "caps": {
            "max_tracks_total": max_tracks_total,
            "max_train_tracks": max_train_tracks,
            "max_val_tracks": max_val_tracks,
        },
        "selected_track_count": len(selected_rows),
        "selected_counts": _count_rows(selected_rows),
        "train_counts": _count_rows(train_rows),
        "val_counts": _count_rows(val_rows),
        "leakage": {
            "train_val_overlap_tracks": 0,
        },
    }
    return split_name, train_rows, val_rows, metadata


def write_split_outputs(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    split_root = ensure_dir(Path(output_dir) / "splits" / metadata["split_name"])
    train_csv = write_manifest_csv(train_rows, split_root / "train_manifest.csv")
    val_csv = write_manifest_csv(val_rows, split_root / "val_manifest.csv")
    write_manifest_jsonl(train_rows, split_root / "train_manifest.jsonl")
    write_manifest_jsonl(val_rows, split_root / "val_manifest.jsonl")
    summary_path = split_root / "split_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "split_root": split_root,
        "train_manifest_csv": train_csv,
        "val_manifest_csv": val_csv,
        "split_summary_json": summary_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate track-level train/val splits.")
    parser.add_argument("--dataset-root", required=True, help="Path to the training dataset root.")
    parser.add_argument(
        "--output-dir",
        default="./manifests",
        help="Directory that receives the split manifests and summary files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic splits.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation ratio for balanced_dev and scenario_b_dev.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("balanced_dev", "scenario_b_dev", "debug_small"),
        required=True,
        help="Split strategy to use.",
    )
    parser.add_argument(
        "--max-tracks-total",
        type=int,
        default=None,
        help="Optional cap on the total number of tracks selected before splitting.",
    )
    parser.add_argument(
        "--max-train-tracks",
        type=int,
        default=None,
        help="Optional cap on the final train split size.",
    )
    parser.add_argument(
        "--max-val-tracks",
        type=int,
        default=None,
        help="Optional cap on the final val split size.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional scenario filter, e.g. Scenario-A Scenario-B.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Optional domain filter, e.g. Brazilian Mercosur.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rows, scan_summary = scan_dataset(args.dataset_root)
    split_name, train_rows, val_rows, metadata = make_split(
        rows,
        split_mode=args.split_mode,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_tracks_total=args.max_tracks_total,
        max_train_tracks=args.max_train_tracks,
        max_val_tracks=args.max_val_tracks,
        scenarios=args.scenarios,
        domains=args.domains,
    )

    outputs = write_split_outputs(train_rows, val_rows, metadata, output_dir=args.output_dir)
    write_manifest_csv(rows, Path(args.output_dir) / "full_manifest.csv")
    write_manifest_jsonl(rows, Path(args.output_dir) / "full_manifest.jsonl")

    print(f"Split name: {split_name}")
    print(f"Train tracks: {metadata['train_counts']['tracks']}")
    print(f"Val tracks: {metadata['val_counts']['tracks']}")
    print(f"Wrote {outputs['train_manifest_csv']}")
    print(f"Wrote {outputs['val_manifest_csv']}")
    print(f"Wrote {outputs['split_summary_json']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
