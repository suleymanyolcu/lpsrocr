from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import random
import shutil
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import yaml

from src.data.manifest_io import read_manifest_csv
from src.utils.paths import ensure_dir, repo_root

DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_STAGE_DIR = Path("external_data/gplpr_stage")
DEFAULT_OUTPUT_DIR = Path("outputs/stage_a")
DEFAULT_SPLIT_DIR = Path("manifests/splits/scenario_b_dev_seed42_v20")
TRAIN_CONFIG_NAME = "GP_LPR_HR_RODOSOL_train.yaml"
TEST_CONFIG_NAME = "GP_LPR_RODOSOL_test.yaml"
FRAME_PATTERN = re.compile(r"^(?P<prefix>hr|lr)-(?P<index>\d+)$", re.IGNORECASE)
STAGE_PATH_ANCHOR = "images"


@dataclass(slots=True)
class StageAPaths:
    project_root: Path
    dataset_root: Path
    gplpr_repo: Path
    split_dir: Path
    train_manifest: Path
    val_manifest: Path
    stage_dir: Path
    output_dir: Path
    split_file: Path
    train_config: Path
    eval_config: Path
    run_tag: str

    @property
    def train_run_dir(self) -> Path:
        return self.output_dir / "checkpoints" / self.run_tag

    @property
    def eval_run_dir(self) -> Path:
        return self.output_dir / "predictions" / self.run_tag

    @property
    def eval_summary_path(self) -> Path:
        return self.output_dir / "eval" / "summary.json"


class _NullSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        return None

    def flush(self) -> None:  # pragma: no cover - trivial
        return None

    def close(self) -> None:  # pragma: no cover - trivial
        return None


@contextmanager
def _temporary_sys_path(path: Path):
    path = path.resolve()
    inserted = False
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
        inserted = True
    try:
        yield
    finally:
        if inserted and str(path) in sys.path:
            sys.path.remove(str(path))


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - runtime environment specific
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _resolve_path(path: str | Path | None, *, base: Path) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def _read_split_summary(split_dir: Path) -> dict[str, Any]:
    summary_path = split_dir / "split_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"split_name": split_dir.name}


def _resolve_split_manifests(
    *,
    project_root: Path,
    split_dir: Path | None,
    train_manifest: Path | None,
    val_manifest: Path | None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    if train_manifest is None or val_manifest is None:
        if split_dir is None:
            split_dir = project_root / DEFAULT_SPLIT_DIR
        train_manifest = split_dir / "train_manifest.csv"
        val_manifest = split_dir / "val_manifest.csv"
    else:
        split_dir = split_dir or train_manifest.parent

    train_manifest = train_manifest.resolve()
    val_manifest = val_manifest.resolve()
    split_dir = split_dir.resolve()
    split_summary = _read_split_summary(split_dir)
    return train_manifest, val_manifest, split_dir, split_summary


def _build_run_tag(split_summary: dict[str, Any], *, prefix: str = "stage_a") -> str:
    split_name = split_summary.get("split_name") or "custom_split"
    return f"{prefix}_{split_name}"


def _frame_sort_key(relative_image_path: str) -> tuple[int, str]:
    path = Path(relative_image_path)
    match = FRAME_PATTERN.match(path.stem)
    if match:
        return int(match.group("index")), path.as_posix()
    return 999, path.as_posix()


def _ordered_frames(frame_paths: Iterable[str]) -> list[str]:
    return sorted((str(path) for path in frame_paths), key=_frame_sort_key)


def _parse_frame_index(image_path: Path) -> int | None:
    match = FRAME_PATTERN.match(image_path.stem)
    if not match:
        return None
    return int(match.group("index"))


def _relative_path_no_resolve(path: str | Path, base: str | Path) -> str:
    path = Path(path)
    base = Path(base)
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


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


def _read_plate_text(label_path: Path) -> str:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing GPLPR sidecar label: {label_path}")

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("plate:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    raise ValueError(f"Could not find a 'plate:' line in {label_path}")


def _validate_plate_text(gt_text: str, alphabet: str) -> None:
    allowed = set(alphabet)
    invalid = sorted(set(gt_text) - allowed)
    if invalid:
        raise ValueError(
            f"GT text {gt_text!r} contains unsupported characters: {''.join(invalid)}"
        )


def _stage_rows(
    *,
    dataset_root: Path,
    rows: list[dict[str, Any]],
    stage_dir: Path,
    project_root: Path,
    split_label: str,
    mode: str,
    alphabet: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    staged_rows: list[dict[str, Any]] = []
    split_lines: list[str] = []

    for row in rows:
        hr_image_paths = _ordered_frames(row["hr_image_paths"])
        if len(hr_image_paths) != 5:
            raise ValueError(
                f"Track {row['track_id']} expected 5 HR frames, found {len(hr_image_paths)}"
            )

        for frame_path in hr_image_paths:
            source_image = dataset_root / frame_path
            staged_image = stage_dir / "images" / split_label / frame_path
            _copy_or_symlink(source_image, staged_image, mode)

            frame_idx = _parse_frame_index(Path(frame_path))
            if frame_idx is None:
                raise ValueError(f"Could not parse frame index from {frame_path}")

            gt_text = str(row["gt_text"])
            _validate_plate_text(gt_text, alphabet)

            label_path = staged_image.with_suffix(".txt")
            ensure_dir(label_path.parent)
            label_path.write_text(
                "\n".join(
                    [
                        f"track_id: {row['track_id']}",
                        f"scenario: {row['scenario']}",
                        f"domain: {row['domain']}",
                        f"frame_idx: {frame_idx}",
                        f"source_image: {frame_path}",
                        f"plate: {gt_text}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            split_lines.append(_relative_path_no_resolve(staged_image, project_root) + f";{split_label}")
            staged_rows.append(
                {
                    "track_id": row["track_id"],
                    "scenario": row["scenario"],
                    "domain": row["domain"],
                    "track_dir": row["track_dir"],
                    "source_image_path": frame_path,
                    "staged_image_path": _relative_path_no_resolve(staged_image, project_root),
                    "label_path": _relative_path_no_resolve(label_path, project_root),
                    "frame_idx": frame_idx,
                    "gt_text": gt_text,
                    "split_label": split_label,
                }
            )

    return staged_rows, split_lines


def _load_rows(train_manifest: Path, val_manifest: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = read_manifest_csv(train_manifest)
    val_rows = read_manifest_csv(val_manifest)
    return train_rows, val_rows


def _count_split_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_counts = Counter(row["scenario"] for row in rows)
    domain_counts = Counter(f"{row['scenario']}/{row['domain']}" for row in rows)
    return {
        "tracks": len(rows),
        "frames": len(rows) * 5,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def _collect_charset(rows: list[dict[str, Any]]) -> str:
    chars = sorted({char for row in rows for char in str(row["gt_text"])})
    return "".join(chars)


def prepare_stage_a_paths(
    *,
    project_root: str | Path = repo_root(),
    dataset_root: str | Path | None = None,
    gplpr_repo: str | Path,
    split_dir: str | Path | None = None,
    train_manifest: str | Path | None = None,
    val_manifest: str | Path | None = None,
    stage_dir: str | Path = DEFAULT_STAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_tag: str | None = None,
) -> StageAPaths:
    project_root = Path(project_root).expanduser().resolve()
    dataset_root = _resolve_path(dataset_root or "train", base=project_root)
    gplpr_repo = Path(gplpr_repo).expanduser().resolve()
    stage_dir = _resolve_path(stage_dir, base=project_root)
    output_dir = _resolve_path(output_dir, base=project_root)
    train_manifest_path = _resolve_path(train_manifest, base=project_root)
    val_manifest_path = _resolve_path(val_manifest, base=project_root)
    split_dir_path = _resolve_path(split_dir, base=project_root)

    train_manifest_path, val_manifest_path, split_dir_path, split_summary = _resolve_split_manifests(
        project_root=project_root,
        split_dir=split_dir_path,
        train_manifest=train_manifest_path,
        val_manifest=val_manifest_path,
    )

    resolved_run_tag = run_tag or _build_run_tag(split_summary)
    return StageAPaths(
        project_root=project_root,
        dataset_root=dataset_root,
        gplpr_repo=gplpr_repo,
        split_dir=split_dir_path,
        train_manifest=train_manifest_path,
        val_manifest=val_manifest_path,
        stage_dir=stage_dir,
        output_dir=output_dir,
        split_file=stage_dir / "split_competition_hr.txt",
        train_config=output_dir / "configs" / f"{resolved_run_tag}_train.yaml",
        eval_config=output_dir / "configs" / f"{resolved_run_tag}_eval.yaml",
        run_tag=resolved_run_tag,
    )


def build_train_config(
    *,
    paths: StageAPaths,
    resume: str | Path | None = None,
    alphabet: str = DEFAULT_ALPHABET,
    base_config_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if base_config_path is None:
        base_config_path = paths.gplpr_repo / "config" / TRAIN_CONFIG_NAME
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"GPLPR train config not found: {base_config_path}")

    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid GPLPR config at {base_config_path}")

    config = deepcopy(config)
    split_file = paths.split_file.resolve()
    resume_path = Path(resume).expanduser().resolve() if resume is not None else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    config["alphabet"] = alphabet
    config.setdefault("model", {}).setdefault("args", {})
    config["model"]["args"]["alphabet"] = alphabet
    config["model"]["args"]["k"] = 7

    for dataset_key, phase in (("train_dataset", "training"), ("val_dataset", "validation")):
        dataset = config.setdefault(dataset_key, {})
        dataset.setdefault("dataset", {}).setdefault("args", {})
        dataset.setdefault("wrapper", {}).setdefault("args", {})
        dataset["dataset"]["args"]["path_split"] = split_file.as_posix()
        dataset["dataset"]["args"]["phase"] = phase
        dataset["wrapper"]["args"]["alphabet"] = alphabet
        dataset["wrapper"]["args"]["k"] = 7
        dataset["wrapper"]["args"]["with_lr"] = False

    config["resume"] = resume_path.as_posix() if resume_path is not None else None
    config["stage_a"] = {
        "project_root": paths.project_root.as_posix(),
        "dataset_root": paths.dataset_root.as_posix(),
        "gplpr_repo": paths.gplpr_repo.as_posix(),
        "stage_dir": paths.stage_dir.as_posix(),
        "split_file": split_file.as_posix(),
        "output_dir": paths.output_dir.as_posix(),
        "checkpoint_dir": paths.train_run_dir.as_posix(),
        "run_tag": paths.run_tag,
        "alphabet": alphabet,
        "resume": config["resume"],
    }
    if seed is not None:
        config["stage_a"]["seed"] = seed
    return config


def build_eval_config(
    *,
    paths: StageAPaths,
    checkpoint_path: str | Path,
    alphabet: str = DEFAULT_ALPHABET,
    base_config_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if base_config_path is None:
        base_config_path = paths.gplpr_repo / "config" / TEST_CONFIG_NAME
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"GPLPR test config not found: {base_config_path}")

    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid GPLPR config at {base_config_path}")

    config = deepcopy(config)
    split_file = paths.split_file.resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    config["alphabet"] = alphabet
    config.setdefault("model_ocr", {})
    config["model_ocr"]["load"] = checkpoint_path.as_posix()

    dataset = config.setdefault("test_dataset", {})
    dataset.setdefault("dataset", {}).setdefault("args", {})
    dataset.setdefault("wrapper", {}).setdefault("args", {})
    dataset["dataset"]["args"]["path_split"] = split_file.as_posix()
    dataset["dataset"]["args"]["phase"] = "validation"
    dataset["wrapper"]["args"]["alphabet"] = alphabet
    dataset["wrapper"]["args"]["k"] = 7
    dataset["wrapper"]["args"]["with_lr"] = False

    config["stage_a"] = {
        "project_root": paths.project_root.as_posix(),
        "dataset_root": paths.dataset_root.as_posix(),
        "gplpr_repo": paths.gplpr_repo.as_posix(),
        "stage_dir": paths.stage_dir.as_posix(),
        "split_file": split_file.as_posix(),
        "output_dir": paths.output_dir.as_posix(),
        "prediction_dir": paths.eval_run_dir.as_posix(),
        "eval_summary": paths.eval_summary_path.as_posix(),
        "run_tag": paths.run_tag,
        "alphabet": alphabet,
        "checkpoint": checkpoint_path.as_posix(),
    }
    if seed is not None:
        config["stage_a"]["seed"] = seed
    return config


def _write_yaml(data: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def export_stage_a_dataset(
    *,
    paths: StageAPaths,
    mode: str = "symlink",
    alphabet: str = DEFAULT_ALPHABET,
) -> dict[str, Any]:
    ensure_dir(paths.stage_dir)
    ensure_dir(paths.output_dir)
    train_rows, val_rows = _load_rows(paths.train_manifest, paths.val_manifest)
    train_rows = sorted(train_rows, key=lambda row: (row["scenario"], row["domain"], row["track_num"], row["track_id"]))
    val_rows = sorted(val_rows, key=lambda row: (row["scenario"], row["domain"], row["track_num"], row["track_id"]))

    staged_train_rows, train_split_lines = _stage_rows(
        dataset_root=paths.dataset_root,
        rows=train_rows,
        stage_dir=paths.stage_dir,
        project_root=paths.project_root,
        split_label="training",
        mode=mode,
        alphabet=alphabet,
    )
    staged_val_rows, val_split_lines = _stage_rows(
        dataset_root=paths.dataset_root,
        rows=val_rows,
        stage_dir=paths.stage_dir,
        project_root=paths.project_root,
        split_label="validation",
        mode=mode,
        alphabet=alphabet,
    )

    split_file = ensure_dir(paths.split_file.parent) / paths.split_file.name
    split_file.write_text("\n".join([*train_split_lines, *val_split_lines]) + "\n", encoding="utf-8")

    gt_charset = _collect_charset([*train_rows, *val_rows])
    alphabet_only = all(char in alphabet for char in gt_charset)

    summary = {
        "project_root": paths.project_root.as_posix(),
        "dataset_root": paths.dataset_root.as_posix(),
        "stage_dir": paths.stage_dir.as_posix(),
        "split_file": split_file.as_posix(),
        "split_name": paths.run_tag.removeprefix("stage_a_"),
        "mode": mode,
        "alphabet": alphabet,
        "observed_charset": gt_charset,
        "observed_charset_matches_alphabet": alphabet_only,
        "train": {
            **_count_split_rows(train_rows),
            "sample_count": len(staged_train_rows),
        },
        "validation": {
            **_count_split_rows(val_rows),
            "sample_count": len(staged_val_rows),
        },
        "total_samples": len(staged_train_rows) + len(staged_val_rows),
        "total_tracks": len(train_rows) + len(val_rows),
    }

    prepare_summary_path = paths.output_dir / "prepare_summary.json"
    prepare_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "summary": summary,
        "prepare_summary_path": prepare_summary_path,
        "split_file": split_file,
        "train_config": paths.train_config,
        "eval_config": paths.eval_config,
    }


def prepare_stage_a_assets(
    *,
    project_root: str | Path = repo_root(),
    dataset_root: str | Path | None = None,
    gplpr_repo: str | Path,
    split_dir: str | Path | None = None,
    train_manifest: str | Path | None = None,
    val_manifest: str | Path | None = None,
    stage_dir: str | Path = DEFAULT_STAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    mode: str = "symlink",
    alphabet: str = DEFAULT_ALPHABET,
    resume: str | Path | None = None,
    base_train_config: str | Path | None = None,
    base_test_config: str | Path | None = None,
    run_tag: str | None = None,
    seed: int = 42,
    export: bool = True,
) -> dict[str, Any]:
    _seed_everything(seed)
    paths = prepare_stage_a_paths(
        project_root=project_root,
        dataset_root=dataset_root,
        gplpr_repo=gplpr_repo,
        split_dir=split_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        stage_dir=stage_dir,
        output_dir=output_dir,
        run_tag=run_tag,
    )

    ensure_dir(paths.stage_dir)
    ensure_dir(paths.output_dir)
    export_result: dict[str, Any] | None = None
    if export:
        export_result = export_stage_a_dataset(paths=paths, mode=mode, alphabet=alphabet)

    train_config = build_train_config(
        paths=paths,
        resume=resume,
        alphabet=alphabet,
        base_config_path=base_train_config,
        seed=seed,
    )
    eval_config = build_eval_config(
        paths=paths,
        checkpoint_path=paths.train_run_dir / "best_model.pth",
        alphabet=alphabet,
        base_config_path=base_test_config,
        seed=seed,
    )

    ensure_dir(paths.output_dir)
    ensure_dir(paths.output_dir / "configs")
    train_config_path = _write_yaml(train_config, paths.train_config)
    eval_config_path = _write_yaml(eval_config, paths.eval_config)

    summary = {
        "paths": {
            "project_root": paths.project_root.as_posix(),
            "dataset_root": paths.dataset_root.as_posix(),
            "gplpr_repo": paths.gplpr_repo.as_posix(),
            "split_dir": paths.split_dir.as_posix(),
            "train_manifest": paths.train_manifest.as_posix(),
            "val_manifest": paths.val_manifest.as_posix(),
            "stage_dir": paths.stage_dir.as_posix(),
            "output_dir": paths.output_dir.as_posix(),
            "split_file": paths.split_file.as_posix(),
            "train_config": train_config_path.as_posix(),
            "eval_config": eval_config_path.as_posix(),
            "train_run_dir": paths.train_run_dir.as_posix(),
            "eval_run_dir": paths.eval_run_dir.as_posix(),
        },
        "run_tag": paths.run_tag,
        "mode": mode,
        "alphabet": alphabet,
        "seed": seed,
        "resume": Path(resume).expanduser().resolve().as_posix() if resume is not None else None,
    }
    if export_result is not None:
        summary["export"] = export_result["summary"]
        summary["prepare_summary_path"] = export_result["prepare_summary_path"].as_posix()

    ensure_dir(paths.output_dir)
    summary_path = paths.output_dir / "stage_a_setup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_path"] = summary_path.as_posix()
    return summary


def _load_gplpr_module(gplpr_repo: Path, module_name: str, relative_file: str) -> ModuleType:
    module_path = gplpr_repo / relative_file
    if not module_path.exists():
        raise FileNotFoundError(f"GPLPR module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GPLPR module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_gplpr_summary_writer(gplpr_module: ModuleType) -> None:
    utils_module = getattr(gplpr_module, "utils", None)
    if utils_module is None:
        return
    if not hasattr(utils_module, "SummaryWriter"):
        utils_module.SummaryWriter = _NullSummaryWriter  # type: ignore[attr-defined]


def _ensure_cuda_or_raise() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency error
        raise RuntimeError("GPLPR requires PyTorch in the execution environment") from exc

    if not torch.cuda.is_available():  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "GPLPR training/evaluation requires CUDA. Run this in Colab or another GPU environment."
        )


def _select_best_checkpoint(checkpoint_dir: Path) -> Path:
    best_named = sorted(checkpoint_dir.glob("best_model_Epoch_*.pth"))
    if best_named:
        def _epoch_value(path: Path) -> tuple[int, float, str]:
            match = re.search(r"Epoch_(\d+)", path.stem)
            epoch = int(match.group(1)) if match else -1
            return epoch, path.stat().st_mtime, path.name

        return max(best_named, key=_epoch_value)

    epoch_last = checkpoint_dir / "epoch-last.pth"
    if epoch_last.exists():
        return epoch_last
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def _materialize_checkpoint_alias(source: Path, alias_path: Path) -> Path:
    if alias_path.exists() or alias_path.is_symlink():
        alias_path.unlink()
    try:
        relative_source = os.path.relpath(source, start=alias_path.parent)
        os.symlink(relative_source, alias_path)
    except OSError:
        shutil.copy2(source, alias_path)
    return alias_path


def run_gplpr_train(
    *,
    paths: StageAPaths,
    alphabet: str = DEFAULT_ALPHABET,
    resume: str | Path | None = None,
    base_train_config: str | Path | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return {
            "dry_run": True,
            "config": paths.train_config.as_posix(),
            "save_path": paths.train_run_dir.as_posix(),
            "seed": seed,
        }

    _seed_everything(seed)
    _ensure_cuda_or_raise()
    if not paths.split_file.exists():
        raise FileNotFoundError(f"Stage A split file not found: {paths.split_file}")
    if not paths.stage_dir.exists():
        raise FileNotFoundError(f"Stage A stage directory not found: {paths.stage_dir}")
    config = build_train_config(
        paths=paths,
        resume=resume,
        alphabet=alphabet,
        base_config_path=base_train_config,
        seed=seed,
    )
    _write_yaml(config, paths.train_config)
    ensure_dir(paths.train_run_dir)

    with _temporary_sys_path(paths.gplpr_repo), _temporary_cwd(paths.project_root):
        gplpr_train = _load_gplpr_module(paths.gplpr_repo, "gplpr_train_stage_a", "train.py")
        _patch_gplpr_summary_writer(gplpr_train)
        gplpr_train.main(config, paths.train_run_dir)

    best_checkpoint = _select_best_checkpoint(paths.train_run_dir)
    alias = _materialize_checkpoint_alias(best_checkpoint, paths.train_run_dir / "best_model.pth")
    epoch_last = paths.train_run_dir / "epoch-last.pth"

    result = {
        "dry_run": False,
        "config": paths.train_config.as_posix(),
        "save_path": paths.train_run_dir.as_posix(),
        "best_checkpoint": best_checkpoint.as_posix(),
        "best_checkpoint_alias": alias.as_posix(),
        "epoch_last": epoch_last.as_posix() if epoch_last.exists() else None,
        "seed": seed,
    }
    train_summary_path = paths.output_dir / "train_summary.json"
    train_summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["summary_path"] = train_summary_path.as_posix()
    return result


def _parse_result_csv(results_csv: Path) -> list[dict[str, Any]]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _parse_stage_image_reference(image_path: str | Path, *, stage_dir: Path) -> dict[str, Any]:
    path = Path(image_path)
    staged_path = path if path.is_absolute() else (stage_dir.parent.parent / path)
    resolved_source = staged_path.resolve()
    parts = path.parts
    if STAGE_PATH_ANCHOR not in parts:
        raise ValueError(f"Could not parse staged GPLPR path: {image_path}")
    anchor_index = parts.index(STAGE_PATH_ANCHOR)
    try:
        split_label = parts[anchor_index + 1]
        scenario = parts[anchor_index + 2]
        domain = parts[anchor_index + 3]
        track_id = parts[anchor_index + 4]
        file_name = parts[anchor_index + 5]
    except IndexError as exc:
        raise ValueError(f"Could not parse staged GPLPR path: {image_path}") from exc

    frame_idx = _parse_frame_index(Path(file_name))
    if frame_idx is None:
        raise ValueError(f"Could not parse frame index from {image_path}")

    label_path = staged_path.with_suffix(".txt")
    gt_text = _read_plate_text(label_path)
    return {
        "image_path": path.as_posix(),
        "resolved_image_path": resolved_source.as_posix(),
        "label_path": label_path.as_posix(),
        "split_label": split_label,
        "scenario": scenario,
        "domain": domain,
        "track_id": track_id,
        "frame_idx": frame_idx,
        "gt_text": gt_text,
    }


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


def _aggregate_track_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: (row["frame_idx"], row["image_path"]))
    counts = Counter(row["pred_text"] for row in ordered_rows)

    def _confidence_value(row: dict[str, Any]) -> float:
        confidence = row.get("confidence")
        return float(confidence) if confidence is not None else float("-inf")

    def _mean_confidence(pred_text: str) -> float:
        values = [float(row["confidence"]) for row in ordered_rows if row["pred_text"] == pred_text and row.get("confidence") is not None]
        if not values:
            return float("-inf")
        return sum(values) / len(values)

    def _first_seen_index(pred_text: str) -> int:
        for index, row in enumerate(ordered_rows):
            if row["pred_text"] == pred_text:
                return index
        return len(ordered_rows)

    best_confidence_row = max(
        ordered_rows,
        key=lambda row: (_confidence_value(row), -row["frame_idx"], -len(row["pred_text"]), row["image_path"]),
    )
    majority_candidates = [pred for pred, count in counts.items() if count == max(counts.values())]
    majority_prediction = min(
        majority_candidates,
        key=lambda pred: (
            -counts[pred],
            -_mean_confidence(pred),
            _first_seen_index(pred),
            pred,
        ),
    )

    return {
        "best_confidence_pred_text": best_confidence_row["pred_text"],
        "best_confidence": best_confidence_row.get("confidence"),
        "majority_vote_pred_text": majority_prediction,
    }


def run_gplpr_eval(
    *,
    paths: StageAPaths,
    checkpoint: str | Path | None = None,
    alphabet: str = DEFAULT_ALPHABET,
    base_test_config: str | Path | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        checkpoint_path = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else (paths.train_run_dir / "best_model.pth")
        )
        return {
            "dry_run": True,
            "config": paths.eval_config.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "save_path": paths.eval_run_dir.as_posix(),
            "seed": seed,
        }

    _seed_everything(seed)
    _ensure_cuda_or_raise()
    if not paths.split_file.exists():
        raise FileNotFoundError(f"Stage A split file not found: {paths.split_file}")
    if not paths.stage_dir.exists():
        raise FileNotFoundError(f"Stage A stage directory not found: {paths.stage_dir}")
    if checkpoint is None:
        checkpoint_path = _select_best_checkpoint(paths.train_run_dir)
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = build_eval_config(
        paths=paths,
        checkpoint_path=checkpoint_path,
        alphabet=alphabet,
        base_config_path=base_test_config,
        seed=seed,
    )
    _write_yaml(config, paths.eval_config)
    ensure_dir(paths.eval_run_dir)

    with _temporary_sys_path(paths.gplpr_repo), _temporary_cwd(paths.project_root):
        gplpr_test = _load_gplpr_module(paths.gplpr_repo, "gplpr_test_stage_a", "test_ocr.py")
        gplpr_test.main(config, paths.eval_run_dir)

    results_csv = paths.eval_run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"GPLPR evaluation did not create {results_csv}")

    csv_rows = _parse_result_csv(results_csv)
    image_rows: list[dict[str, Any]] = []
    track_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in csv_rows:
        image_path = row.get("Image Name") or row.get("image_name") or row.get("img") or ""
        pred_text = (row.get("Prediction") or row.get("prediction") or "").strip()
        gt_metadata = _parse_stage_image_reference(image_path, stage_dir=paths.stage_dir)
        gt_text = gt_metadata["gt_text"]
        exact_match = pred_text == gt_text
        edit_distance = _levenshtein_distance(gt_text, pred_text)
        normalized_edit_distance = _normalized_edit_distance(gt_text, pred_text)

        image_row = {
            **gt_metadata,
            "pred_text": pred_text,
            "exact_match": exact_match,
            "edit_distance": edit_distance,
            "normalized_edit_distance": normalized_edit_distance,
            "confidence": None,
        }
        image_rows.append(image_row)
        track_groups[gt_metadata["track_id"]].append(image_row)

    per_track_rows: list[dict[str, Any]] = []
    for track_id in sorted(track_groups):
        rows = track_groups[track_id]
        rows = sorted(rows, key=lambda row: (row["frame_idx"], row["image_path"]))
        aggregate = _aggregate_track_predictions(rows)
        gt_text = rows[0]["gt_text"]
        best_confidence_pred_text = aggregate["best_confidence_pred_text"]
        majority_vote_pred_text = aggregate["majority_vote_pred_text"]
        per_track_rows.append(
            {
                "track_id": track_id,
                "scenario": rows[0]["scenario"],
                "domain": rows[0]["domain"],
                "gt_text": gt_text,
                "num_frames": len(rows),
                "best_confidence_pred_text": best_confidence_pred_text,
                "best_confidence": aggregate["best_confidence"],
                "best_confidence_exact_match": best_confidence_pred_text == gt_text,
                "majority_vote_pred_text": majority_vote_pred_text,
                "majority_vote_exact_match": majority_vote_pred_text == gt_text,
                "track_exact_match": majority_vote_pred_text == gt_text,
            }
        )

    per_image_path = paths.eval_run_dir / "per_image.jsonl"
    per_track_path = paths.eval_run_dir / "per_track.jsonl"
    per_image_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in image_rows) + "\n",
        encoding="utf-8",
    )
    per_track_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in per_track_rows) + "\n",
        encoding="utf-8",
    )

    image_exact_match_rate = (
        sum(1 for row in image_rows if row["exact_match"]) / len(image_rows)
        if image_rows
        else 0.0
    )
    best_confidence_track_rate = (
        sum(1 for row in per_track_rows if row["best_confidence_exact_match"]) / len(per_track_rows)
        if per_track_rows
        else 0.0
    )
    majority_vote_track_rate = (
        sum(1 for row in per_track_rows if row["majority_vote_exact_match"]) / len(per_track_rows)
        if per_track_rows
        else 0.0
    )
    summary = {
        "checkpoint": checkpoint_path.as_posix(),
        "results_csv": results_csv.as_posix(),
        "per_image_jsonl": per_image_path.as_posix(),
        "per_track_jsonl": per_track_path.as_posix(),
        "num_images": len(image_rows),
        "num_tracks": len(per_track_rows),
        "image_exact_match_rate": image_exact_match_rate,
        "track_exact_match_rate": majority_vote_track_rate,
        "best_confidence_track_exact_match_rate": best_confidence_track_rate,
        "confidence_available": any(row["best_confidence"] is not None for row in per_track_rows),
        "confidence_note": "GPLPR test output does not expose confidence, so the wrapper stores null.",
        "per_image_preview": image_rows[:3],
        "per_track_preview": per_track_rows[:3],
        "seed": seed,
    }

    ensure_dir(paths.eval_summary_path.parent)
    paths.eval_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser_prepare() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage competition HR OCR data for GPLPR.")
    parser.add_argument("--project-root", default=str(repo_root()), help="Project repository root.")
    parser.add_argument("--dataset-root", default=None, help="Competition dataset root.")
    parser.add_argument("--gplpr-repo", required=True, help="Path to the GPLPR repository clone.")
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Split directory from prompt 1.")
    parser.add_argument("--train-manifest", default=None, help="Optional explicit train manifest path.")
    parser.add_argument("--val-manifest", default=None, help="Optional explicit val manifest path.")
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_DIR), help="Staged GPLPR dataset root.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Stage A outputs root.")
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize staged images.",
    )
    parser.add_argument(
        "--symlink-mode",
        dest="mode",
        action="store_const",
        const="symlink",
        help="Alias for --mode symlink.",
    )
    parser.add_argument(
        "--copy-mode",
        dest="mode",
        action="store_const",
        const="copy",
        help="Alias for --mode copy.",
    )
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET, help="OCR alphabet.")
    parser.add_argument("--resume", default=None, help="Optional resume checkpoint for GPLPR training.")
    parser.add_argument("--base-train-config", default=None, help="Override GPLPR train config template.")
    parser.add_argument("--base-test-config", default=None, help="Override GPLPR test config template.")
    parser.add_argument("--run-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded in summaries.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths without writing files.")
    return parser


def build_arg_parser_train() -> argparse.ArgumentParser:
    parser = build_arg_parser_prepare()
    parser.description = "Fine-tune GPLPR on the staged HR OCR dataset."
    return parser


def build_arg_parser_eval() -> argparse.ArgumentParser:
    parser = build_arg_parser_prepare()
    parser.description = "Run GPLPR evaluation on HR validation images."
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate. Defaults to the best one.")
    return parser


def _print_resolved_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main_prepare(argv: list[str] | None = None) -> int:
    parser = build_arg_parser_prepare()
    args = parser.parse_args(argv)
    if args.dry_run:
        paths = prepare_stage_a_paths(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            gplpr_repo=args.gplpr_repo,
            split_dir=args.split_dir,
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            stage_dir=args.stage_dir,
            output_dir=args.output_dir,
            run_tag=args.run_tag,
        )
        _print_resolved_summary(
            {
                "project_root": paths.project_root.as_posix(),
                "dataset_root": paths.dataset_root.as_posix(),
                "gplpr_repo": paths.gplpr_repo.as_posix(),
                "split_dir": paths.split_dir.as_posix(),
                "train_manifest": paths.train_manifest.as_posix(),
                "val_manifest": paths.val_manifest.as_posix(),
                "stage_dir": paths.stage_dir.as_posix(),
                "output_dir": paths.output_dir.as_posix(),
                "split_file": paths.split_file.as_posix(),
                "train_config": paths.train_config.as_posix(),
                "eval_config": paths.eval_config.as_posix(),
                "run_tag": paths.run_tag,
                "mode": args.mode,
                "alphabet": args.alphabet,
                "seed": args.seed,
            }
        )
        return 0

    summary = prepare_stage_a_assets(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        gplpr_repo=args.gplpr_repo,
        split_dir=args.split_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        alphabet=args.alphabet,
        resume=args.resume,
        base_train_config=args.base_train_config,
        base_test_config=args.base_test_config,
        run_tag=args.run_tag,
        seed=args.seed,
        export=True,
    )
    _print_resolved_summary(summary)
    return 0


def main_train(argv: list[str] | None = None) -> int:
    parser = build_arg_parser_train()
    args = parser.parse_args(argv)
    paths = prepare_stage_a_paths(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        gplpr_repo=args.gplpr_repo,
        split_dir=args.split_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
    )
    result = run_gplpr_train(
        paths=paths,
        alphabet=args.alphabet,
        resume=args.resume,
        base_train_config=args.base_train_config,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        _print_resolved_summary(result)
    else:
        _print_resolved_summary(result)
    return 0


def main_eval(argv: list[str] | None = None) -> int:
    parser = build_arg_parser_eval()
    args = parser.parse_args(argv)
    paths = prepare_stage_a_paths(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        gplpr_repo=args.gplpr_repo,
        split_dir=args.split_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
    )
    result = run_gplpr_eval(
        paths=paths,
        checkpoint=args.checkpoint,
        alphabet=args.alphabet,
        base_test_config=args.base_test_config,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    _print_resolved_summary(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_prepare())
