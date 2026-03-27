from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.manifest_io import read_manifest_csv
from src.stage_c.pipeline import (
    DEFAULT_COMPARISON_DIR,
    DEFAULT_LR_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SR_OUTPUT_DIR,
    _mean_or_none,
)
from src.utils.paths import ensure_dir, repo_root


def _load_stage_c_bundle(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = output_dir / "summary.json"
    per_track_csv = output_dir / "per_track.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Stage C summary not found: {summary_path}")
    if not per_track_csv.exists():
        raise FileNotFoundError(f"Stage C per-track CSV not found: {per_track_csv}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_track_rows = read_manifest_csv(per_track_csv)
    return summary, per_track_rows


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(rhs) - float(lhs)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _comparison_metric_rows(lr_summary: dict[str, Any], sr_summary: dict[str, Any]) -> list[dict[str, Any]]:
    lr_frame = lr_summary["frame_summary"]
    sr_frame = sr_summary["frame_summary"]
    lr_track = lr_summary["track_summary"]
    sr_track = sr_summary["track_summary"]
    return [
        {
            "metric": "image_exact_match_rate",
            "lr": lr_frame["image_exact_match_rate"],
            "sr": sr_frame["image_exact_match_rate"],
            "delta": _delta(lr_frame["image_exact_match_rate"], sr_frame["image_exact_match_rate"]),
        },
        {
            "metric": "track_exact_match_majority_rate",
            "lr": lr_track["track_exact_match_majority_rate"],
            "sr": sr_track["track_exact_match_majority_rate"],
            "delta": _delta(lr_track["track_exact_match_majority_rate"], sr_track["track_exact_match_majority_rate"]),
        },
        {
            "metric": "track_exact_match_best_conf_rate",
            "lr": lr_track["track_exact_match_best_conf_rate"],
            "sr": sr_track["track_exact_match_best_conf_rate"],
            "delta": _delta(lr_track["track_exact_match_best_conf_rate"], sr_track["track_exact_match_best_conf_rate"]),
        },
        {
            "metric": "mean_edit_distance",
            "lr": lr_frame["mean_edit_distance"],
            "sr": sr_frame["mean_edit_distance"],
            "delta": _delta(lr_frame["mean_edit_distance"], sr_frame["mean_edit_distance"]),
        },
        {
            "metric": "mean_normalized_edit_distance",
            "lr": lr_frame["mean_normalized_edit_distance"],
            "sr": sr_frame["mean_normalized_edit_distance"],
            "delta": _delta(
                lr_frame["mean_normalized_edit_distance"], sr_frame["mean_normalized_edit_distance"]
            ),
        },
        {
            "metric": "mean_confidence_majority",
            "lr": lr_track["mean_confidence_majority"],
            "sr": sr_track["mean_confidence_majority"],
            "delta": _delta(lr_track["mean_confidence_majority"], sr_track["mean_confidence_majority"]),
        },
    ]


def _format_comparison_markdown(
    *,
    lr_summary: dict[str, Any],
    sr_summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    shared_track_count: int,
    missing_tracks_lr: list[str],
    missing_tracks_sr: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Stage C LR vs SR Comparison")
    lines.append("")
    lines.append(f"- LR checkpoint: `{lr_summary['checkpoint']}`")
    lines.append(f"- SR checkpoint: `{sr_summary['checkpoint']}`")
    lines.append(f"- Shared tracks: {shared_track_count}")
    lines.append("")
    lines.append("| Metric | LR | SR | Delta (SR - LR) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in metrics:
        metric = row["metric"]
        if "rate" in metric:
            lr_text = _format_pct(row["lr"])
            sr_text = _format_pct(row["sr"])
            delta_text = _format_pct(row["delta"])
        else:
            lr_text = _format_float(row["lr"])
            sr_text = _format_float(row["sr"])
            delta_text = _format_float(row["delta"])
        lines.append(f"| `{metric}` | {lr_text} | {sr_text} | {delta_text} |")
    lines.append("")
    lines.append("## Scenario Breakdown")
    lines.append("")
    for scenario, lr_stats in lr_summary["track_summary"]["by_scenario"].items():
        sr_stats = sr_summary["track_summary"]["by_scenario"].get(scenario)
        if sr_stats is None:
            continue
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("| Mode | Tracks | Majority EM | Best-conf EM |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| LR | {lr_stats['tracks']} | {_format_pct(lr_stats['majority_exact_match_rate'])} | {_format_pct(lr_stats['best_conf_exact_match_rate'])} |"
        )
        lines.append(
            f"| SR | {sr_stats['tracks']} | {_format_pct(sr_stats['majority_exact_match_rate'])} | {_format_pct(sr_stats['best_conf_exact_match_rate'])} |"
        )
        lines.append("")
    lines.append("## Domain Breakdown")
    lines.append("")
    for domain, lr_stats in lr_summary["track_summary"]["by_domain"].items():
        sr_stats = sr_summary["track_summary"]["by_domain"].get(domain)
        if sr_stats is None:
            continue
        lines.append(f"### {domain}")
        lines.append("")
        lines.append("| Mode | Tracks | Majority EM | Best-conf EM |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| LR | {lr_stats['tracks']} | {_format_pct(lr_stats['majority_exact_match_rate'])} | {_format_pct(lr_stats['best_conf_exact_match_rate'])} |"
        )
        lines.append(
            f"| SR | {sr_stats['tracks']} | {_format_pct(sr_stats['majority_exact_match_rate'])} | {_format_pct(sr_stats['best_conf_exact_match_rate'])} |"
        )
        lines.append("")
    if missing_tracks_lr or missing_tracks_sr:
        lines.append("## Track Coverage")
        lines.append("")
        lines.append(f"- LR-only tracks: {len(missing_tracks_lr)}")
        lines.append(f"- SR-only tracks: {len(missing_tracks_sr)}")
    return "\n".join(lines).rstrip() + "\n"


def compare_stage_c_outputs(
    *,
    lr_output_dir: str | Path,
    sr_output_dir: str | Path,
    output_dir: str | Path = DEFAULT_COMPARISON_DIR,
) -> dict[str, Any]:
    lr_output_dir = Path(lr_output_dir).expanduser().resolve()
    sr_output_dir = Path(sr_output_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    lr_summary, lr_per_track_rows = _load_stage_c_bundle(lr_output_dir)
    sr_summary, sr_per_track_rows = _load_stage_c_bundle(sr_output_dir)

    lr_track_ids = {row["track_id"] for row in lr_per_track_rows}
    sr_track_ids = {row["track_id"] for row in sr_per_track_rows}
    shared_track_ids = sorted(lr_track_ids & sr_track_ids)
    missing_tracks_lr = sorted(lr_track_ids - sr_track_ids)
    missing_tracks_sr = sorted(sr_track_ids - lr_track_ids)

    metrics = _comparison_metric_rows(lr_summary, sr_summary)
    delta = {
        row["metric"]: row["delta"]
        for row in metrics
    }
    summary = {
        "lr": {
            "output_dir": lr_output_dir.as_posix(),
            "summary": lr_summary,
        },
        "sr": {
            "output_dir": sr_output_dir.as_posix(),
            "summary": sr_summary,
        },
        "shared_track_count": len(shared_track_ids),
        "missing_tracks_lr_only": missing_tracks_lr,
        "missing_tracks_sr_only": missing_tracks_sr,
        "metrics": metrics,
        "delta": delta,
    }
    summary_path = output_dir / "lr_vs_sr_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = output_dir / "lr_vs_sr_report.md"
    report_path.write_text(
        _format_comparison_markdown(
            lr_summary=lr_summary,
            sr_summary=sr_summary,
            metrics=metrics,
            shared_track_count=len(shared_track_ids),
            missing_tracks_lr=missing_tracks_lr,
            missing_tracks_sr=missing_tracks_sr,
        ),
        encoding="utf-8",
    )
    summary["summary_path"] = summary_path.as_posix()
    summary["report_path"] = report_path.as_posix()
    summary["shared_track_ids_preview"] = shared_track_ids[:5]
    return summary


def write_stage_c_submission_like_txt(
    *,
    per_track_csv: str | Path,
    output_file: str | Path,
    aggregation_mode: str = "majority",
    confidence_fallback: float = 0.0,
    sort_by: str = "track_num",
) -> dict[str, Any]:
    per_track_csv = Path(per_track_csv).expanduser().resolve()
    output_file = Path(output_file).expanduser().resolve()
    rows = read_manifest_csv(per_track_csv)
    if aggregation_mode not in {"best_conf", "majority"}:
        raise ValueError("aggregation_mode must be best_conf or majority")
    if sort_by not in {"track_num", "track_id"}:
        raise ValueError("sort_by must be track_num or track_id")

    text_column = "aggregated_text_best_conf" if aggregation_mode == "best_conf" else "aggregated_text_majority"
    confidence_column = (
        "aggregated_conf_best_conf" if aggregation_mode == "best_conf" else "aggregated_conf_majority"
    )

    def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        if sort_by == "track_num":
            try:
                return (int(row.get("track_num", 0)), row.get("track_id", ""))
            except (TypeError, ValueError):
                return (row.get("track_id", ""),)
        return (row.get("track_id", ""),)

    sorted_rows = sorted(rows, key=_sort_key)
    ensure_dir(output_file.parent)
    lines: list[str] = []
    for row in sorted_rows:
        text = str(row[text_column]).strip()
        confidence = row.get(confidence_column)
        try:
            confidence_value = float(confidence) if confidence not in {None, ""} else confidence_fallback
        except (TypeError, ValueError):
            confidence_value = confidence_fallback
        lines.append(f"{row['track_id']},{text};{confidence_value:.4f}")
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "per_track_csv": per_track_csv.as_posix(),
        "output_file": output_file.as_posix(),
        "aggregation_mode": aggregation_mode,
        "sort_by": sort_by,
        "num_tracks": len(sorted_rows),
        "confidence_fallback": confidence_fallback,
        "preview": lines[:3],
    }


def build_stage_c_arg_parser_compare() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare LR baseline against SR+OCR Stage C outputs.")
    parser.add_argument("--lr-output-dir", default=str(DEFAULT_LR_OUTPUT_DIR), help="Stage C LR output directory.")
    parser.add_argument("--sr-output-dir", default=str(DEFAULT_SR_OUTPUT_DIR), help="Stage C SR output directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_COMPARISON_DIR), help="Comparison output directory.")
    return parser


def build_stage_c_arg_parser_submission() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a competition-style prediction txt file from per-track CSV.")
    parser.add_argument("--per-track-csv", required=True, help="Stage C per-track CSV.")
    parser.add_argument("--output-file", required=True, help="Destination txt file.")
    parser.add_argument(
        "--aggregation-mode",
        choices=("best_conf", "majority"),
        default="majority",
        help="Which aggregated text/confidence column to export.",
    )
    parser.add_argument(
        "--confidence-fallback",
        type=float,
        default=0.0,
        help="Fallback confidence when the selected aggregation column is missing.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("track_num", "track_id"),
        default="track_num",
        help="Sort order for submission lines.",
    )
    return parser


def main_compare(argv: list[str] | None = None) -> int:
    parser = build_stage_c_arg_parser_compare()
    args = parser.parse_args(argv)
    summary = compare_stage_c_outputs(
        lr_output_dir=args.lr_output_dir,
        sr_output_dir=args.sr_output_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main_write_submission(argv: list[str] | None = None) -> int:
    parser = build_stage_c_arg_parser_submission()
    args = parser.parse_args(argv)
    summary = write_stage_c_submission_like_txt(
        per_track_csv=args.per_track_csv,
        output_file=args.output_file,
        aggregation_mode=args.aggregation_mode,
        confidence_fallback=args.confidence_fallback,
        sort_by=args.sort_by,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0
