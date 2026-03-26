from __future__ import annotations

import argparse

from src.stage_b.lpsrlacd import DEFAULT_STAGE_DIR, export_lpsrlacd_pairs
from src.utils.paths import repo_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export paired HR/LR manifests for lpsr-lacd style training.")
    parser.add_argument("--project-root", default=str(repo_root()), help="Project repository root.")
    parser.add_argument("--dataset-root", required=True, help="Path to the competition dataset root.")
    parser.add_argument("--split-dir", required=True, help="Directory containing train/val split manifests.")
    parser.add_argument(
        "--stage-dir",
        default=None,
        help="Directory that receives the staged paired images and split file. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_STAGE_DIR),
        help="Legacy alias for --stage-dir when --stage-dir is omitted.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "paths"),
        default="symlink",
        help="Materialization mode for staged files.",
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
    stage_dir = args.stage_dir or args.output_dir
    result = export_lpsrlacd_pairs(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        stage_dir=stage_dir,
        project_root=args.project_root,
        mode=args.mode,
        subset=args.subset,
    )
    print(f"Wrote {result['split_file']}")
    print(f"Lines: {result['total_pairs']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

