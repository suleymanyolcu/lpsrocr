from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from src.data.manifest_io import read_manifest_csv, write_manifest_csv, write_manifest_jsonl
from src.stage_a.gplpr import _ensure_compatible_checkpoint, _seed_everything
from src.utils.paths import ensure_dir, repo_root

DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_STAGE_DIR = Path("external_data/lpsrlacd_stage")
DEFAULT_OUTPUT_DIR = Path("outputs/stage_b")
DEFAULT_SPLIT_DIR = Path("manifests/splits/scenario_b_dev_seed42_n400_v20")
DEFAULT_IMG_W = 48
DEFAULT_IMG_H = 16
DEFAULT_IMAGE_ASPECT_RATIO = 3
DEFAULT_BACKGROUND = "(127, 127, 127)"
TRAIN_CONFIG_NAME = "cgnetV2_deformable.yaml"
TEST_CONFIG_NAME = "cgnetV2_deformable_test.yaml"
FRAME_PATTERN = re.compile(r"^(?P<prefix>hr|lr)-(?P<index>\d+)$", re.IGNORECASE)


@dataclass(slots=True)
class StageBPaths:
    project_root: Path
    dataset_root: Path
    lpsrlacd_repo: Path
    split_dir: Path
    train_manifest: Path
    val_manifest: Path
    stage_dir: Path
    output_dir: Path
    split_file: Path
    pair_manifest: Path
    train_config: Path
    infer_config: Path
    run_tag: str

    @property
    def train_run_dir(self) -> Path:
        return self.output_dir / "checkpoints" / self.run_tag

    @property
    def restored_dir(self) -> Path:
        return self.output_dir / "restored"

    @property
    def eval_dir(self) -> Path:
        return self.output_dir / "eval"

    @property
    def setup_summary_path(self) -> Path:
        return self.output_dir / "stage_b_setup_summary.json"

    @property
    def export_summary_path(self) -> Path:
        return self.stage_dir / "stage_b_export_summary.json"

    @property
    def best_model_alias(self) -> Path:
        return self.train_run_dir / "best_model.pth"

    @property
    def best_sr_alias(self) -> Path:
        return self.train_run_dir / "best_model_sr.pth"

    @property
    def best_ocr_alias(self) -> Path:
        return self.train_run_dir / "best_model_ocr.pth"


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


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["scenario"], row["domain"], row["track_num"], row["track_id"])


def _frame_index(path_like: str | Path) -> int:
    match = FRAME_PATTERN.match(Path(path_like).stem)
    if not match:
        raise ValueError(f"Could not parse frame index from {path_like}")
    return int(match.group("index"))


def _ordered_frames(frame_paths: list[str]) -> list[str]:
    return sorted((str(path) for path in frame_paths), key=lambda item: (_frame_index(item), item))


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


def _track_pairs(row: dict[str, Any]) -> list[tuple[int, str, str]]:
    hr_paths = _ordered_frames(row["hr_image_paths"])
    lr_paths = _ordered_frames(row["lr_image_paths"])
    if len(hr_paths) != len(lr_paths):
        raise ValueError(
            f"Track {row['track_id']} has mismatched HR/LR counts: {len(hr_paths)} vs {len(lr_paths)}"
        )

    pairs: list[tuple[int, str, str]] = []
    for hr_rel, lr_rel in zip(hr_paths, lr_paths):
        hr_idx = _frame_index(hr_rel)
        lr_idx = _frame_index(lr_rel)
        if hr_idx != lr_idx:
            raise ValueError(
                f"Track {row['track_id']} has mismatched frame indices: {hr_rel} vs {lr_rel}"
            )
        pairs.append((hr_idx, hr_rel, lr_rel))
    return pairs


def _count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_counts = Counter(row["scenario"] for row in rows)
    domain_counts = Counter(f'{row["scenario"]}/{row["domain"]}' for row in rows)
    return {
        "tracks": len(rows),
        "frames": sum(row["num_hr_images"] for row in rows),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def _count_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_counts = Counter(row["scenario"] for row in rows)
    domain_counts = Counter(f'{row["scenario"]}/{row["domain"]}' for row in rows)
    track_ids = {row["track_id"] for row in rows}
    return {
        "pairs": len(rows),
        "tracks": len(track_ids),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def _build_run_tag(split_summary: dict[str, Any], prefix: str = "stage_b") -> str:
    split_name = split_summary.get("split_name") or "custom_split"
    return f"{prefix}_{split_name}"


def prepare_stage_b_paths(
    *,
    project_root: str | Path = repo_root(),
    dataset_root: str | Path | None = None,
    lpsrlacd_repo: str | Path,
    split_dir: str | Path | None = None,
    train_manifest: str | Path | None = None,
    val_manifest: str | Path | None = None,
    stage_dir: str | Path = DEFAULT_STAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_tag: str | None = None,
) -> StageBPaths:
    project_root = Path(project_root).expanduser().resolve()
    dataset_root = _resolve_path(dataset_root or "train", base=project_root)
    lpsrlacd_repo = Path(lpsrlacd_repo).expanduser().resolve()
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
    return StageBPaths(
        project_root=project_root,
        dataset_root=dataset_root,
        lpsrlacd_repo=lpsrlacd_repo,
        split_dir=split_dir_path,
        train_manifest=train_manifest_path,
        val_manifest=val_manifest_path,
        stage_dir=stage_dir,
        output_dir=output_dir,
        split_file=stage_dir / "split_pairs.txt",
        pair_manifest=stage_dir / "pairs_manifest.jsonl",
        train_config=output_dir / "configs" / f"{resolved_run_tag}_train.yaml",
        infer_config=output_dir / "configs" / f"{resolved_run_tag}_infer.yaml",
        run_tag=resolved_run_tag,
    )


def _staged_image_path(paths: StageBPaths, split_label: str, relative_image_path: str) -> Path:
    return paths.stage_dir / "images" / split_label / relative_image_path


def _stage_track_pairs(
    *,
    paths: StageBPaths,
    row: dict[str, Any],
    split_label: str,
    mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_rows: list[dict[str, Any]] = []
    split_lines: list[str] = []

    for frame_idx, hr_rel, lr_rel in _track_pairs(row):
        hr_source = paths.dataset_root / hr_rel
        lr_source = paths.dataset_root / lr_rel

        if mode in {"copy", "symlink"}:
            hr_dest = _staged_image_path(paths, split_label, hr_rel)
            lr_dest = _staged_image_path(paths, split_label, lr_rel)
            _copy_or_symlink(hr_source, hr_dest, mode)
            _copy_or_symlink(lr_source, lr_dest, mode)

            label_path = hr_dest.with_suffix(".txt")
            ensure_dir(label_path.parent)
            label_path.write_text(
                "\n".join(
                    [
                        f"track_id: {row['track_id']}",
                        f"scenario: {row['scenario']}",
                        f"domain: {row['domain']}",
                        f"frame_idx: {frame_idx}",
                        f"source_hr_image: {hr_rel}",
                        f"source_lr_image: {lr_rel}",
                        f"plate: {row['gt_text']}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            hr_output = _project_relpath(hr_dest, paths.project_root)
            lr_output = _project_relpath(lr_dest, paths.project_root)
            label_output = _project_relpath(label_path, paths.project_root)
        elif mode == "paths":
            hr_output = _project_relpath(hr_source, paths.project_root)
            lr_output = _project_relpath(lr_source, paths.project_root)
            label_output = None
        else:  # pragma: no cover - guarded by argparse
            raise ValueError(f"Unsupported staging mode: {mode}")

        split_lines.append(f"{hr_output};{lr_output};{split_label}")
        manifest_rows.append(
            {
                "track_id": row["track_id"],
                "track_num": row["track_num"],
                "scenario": row["scenario"],
                "domain": row["domain"],
                "track_dir": row["track_dir"],
                "frame_idx": frame_idx,
                "split_label": split_label,
                "gt_text": row["gt_text"],
                "hr_image_path": hr_rel,
                "lr_image_path": lr_rel,
                "staged_hr_path": hr_output if mode != "paths" else None,
                "staged_lr_path": lr_output if mode != "paths" else None,
                "label_path": label_output,
                "source_hr_exists": hr_source.exists(),
                "source_lr_exists": lr_source.exists(),
            }
        )

    return manifest_rows, split_lines


def export_lpsrlacd_pairs(
    *,
    dataset_root: str | Path,
    split_dir: str | Path,
    stage_dir: str | Path,
    project_root: str | Path = repo_root(),
    mode: str = "symlink",
    subset: str = "both",
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    split_dir = Path(split_dir).expanduser().resolve()
    stage_dir = Path(stage_dir).expanduser().resolve()
    ensure_dir(stage_dir)

    paths = prepare_stage_b_paths(
        project_root=project_root,
        dataset_root=dataset_root,
        lpsrlacd_repo=project_root,
        split_dir=split_dir,
        stage_dir=stage_dir,
        output_dir=stage_dir,
    )

    train_rows = read_manifest_csv(split_dir / "train_manifest.csv") if subset in {"train", "both"} else []
    val_rows = read_manifest_csv(split_dir / "val_manifest.csv") if subset in {"val", "both"} else []
    train_rows = sorted(train_rows, key=_row_key)
    val_rows = sorted(val_rows, key=_row_key)

    train_manifest_rows: list[dict[str, Any]] = []
    val_manifest_rows: list[dict[str, Any]] = []
    split_lines: list[str] = []

    for row in train_rows:
        staged_rows, lines = _stage_track_pairs(paths=paths, row=row, split_label="training", mode=mode)
        train_manifest_rows.extend(staged_rows)
        split_lines.extend(lines)

    for row in val_rows:
        staged_rows, lines = _stage_track_pairs(paths=paths, row=row, split_label="validation", mode=mode)
        val_manifest_rows.extend(staged_rows)
        split_lines.extend(lines)

    split_file = stage_dir / "split_pairs.txt"
    split_alias = stage_dir / "lpsrlacd_pairs.txt"
    split_text = "\n".join(split_lines) + "\n"
    split_file.write_text(split_text, encoding="utf-8")
    split_alias.write_text(split_text, encoding="utf-8")

    manifest_rows = [*train_manifest_rows, *val_manifest_rows]
    write_manifest_jsonl(manifest_rows, stage_dir / "pairs_manifest.jsonl")

    summary = {
        "project_root": project_root.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "split_dir": split_dir.as_posix(),
        "stage_dir": stage_dir.as_posix(),
        "split_file": split_file.as_posix(),
        "split_file_alias": split_alias.as_posix(),
        "mode": mode,
        "subset": subset,
        "train": _count_pairs(train_manifest_rows),
        "validation": _count_pairs(val_manifest_rows),
        "total_pairs": len(manifest_rows),
        "total_tracks": len({row["track_id"] for row in manifest_rows}),
        "pair_format": "HR_path;LR_path;training|validation",
        "sidecar_label_format": "plate: <GT_TEXT>",
        "notes": [
            "HR sidecar .txt files are written next to staged HR images.",
            "The pair file is track-safe because each SR output is saved under its track directory.",
        ],
    }
    (stage_dir / "stage_b_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_yaml(data: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _normalize_ocr_checkpoint(path: str | Path | None, *, output_dir: Path) -> Path | None:
    if path is None:
        return None
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return _ensure_compatible_checkpoint(checkpoint, output_dir=output_dir)


def build_train_config(
    *,
    paths: StageBPaths,
    resume_sr: str | Path | None = None,
    resume_ocr: str | Path | None = None,
    base_config_path: str | Path | None = None,
    alphabet: str = DEFAULT_ALPHABET,
    seed: int | None = None,
) -> dict[str, Any]:
    if base_config_path is None:
        base_config_path = paths.lpsrlacd_repo / "configs" / TRAIN_CONFIG_NAME
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"lpsr-lacd train config not found: {base_config_path}")

    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid lpsr-lacd config at {base_config_path}")

    config = deepcopy(config)
    split_file = _project_relpath(paths.split_file, paths.project_root)

    resume_sr_path = Path(resume_sr).expanduser().resolve() if resume_sr is not None else None
    if resume_sr_path is not None and not resume_sr_path.exists():
        raise FileNotFoundError(f"SR resume checkpoint not found: {resume_sr_path}")
    resume_ocr_path = _normalize_ocr_checkpoint(resume_ocr, output_dir=paths.output_dir)

    config["alphabet"] = alphabet
    config.setdefault("MODEL_OCR", {}).setdefault("args", {})
    config["MODEL_OCR"]["args"]["alphabet"] = alphabet
    config["MODEL_OCR"]["args"]["K"] = 7
    config["MODEL_OCR"]["args"].pop("k", None)

    for dataset_key, phase in (("train_dataset", "training"), ("val_dataset", "validation")):
        dataset = config.setdefault(dataset_key, {})
        dataset.setdefault("dataset", {}).setdefault("args", {})
        dataset.setdefault("wrapper", {}).setdefault("args", {})
        dataset["dataset"]["args"]["path_split"] = split_file
        dataset["dataset"]["args"]["phase"] = phase
        dataset["wrapper"]["args"]["imgW"] = DEFAULT_IMG_W
        dataset["wrapper"]["args"]["imgH"] = DEFAULT_IMG_H
        dataset["wrapper"]["args"]["aug"] = dataset_key == "train_dataset"
        dataset["wrapper"]["args"]["image_aspect_ratio"] = DEFAULT_IMAGE_ASPECT_RATIO
        dataset["wrapper"]["args"]["background"] = DEFAULT_BACKGROUND

    config.setdefault("loss_sr", {}).setdefault("args", {})
    config["loss_sr"]["args"]["load"] = None

    if resume_sr_path is not None:
        config["resume"] = [
            _project_relpath(resume_sr_path, paths.project_root),
            _project_relpath(resume_ocr_path, paths.project_root) if resume_ocr_path is not None else None,
        ]
        config["LOAD_PRE_TRAINED_OCR"] = None
    else:
        config["resume"] = None
        config["LOAD_PRE_TRAINED_OCR"] = (
            _project_relpath(resume_ocr_path, paths.project_root) if resume_ocr_path is not None else None
        )

    config["stage_b"] = {
        "project_root": paths.project_root.as_posix(),
        "dataset_root": paths.dataset_root.as_posix(),
        "lpsrlacd_repo": paths.lpsrlacd_repo.as_posix(),
        "stage_dir": paths.stage_dir.as_posix(),
        "split_file": paths.split_file.as_posix(),
        "output_dir": paths.output_dir.as_posix(),
        "train_run_dir": paths.train_run_dir.as_posix(),
        "restored_dir": paths.restored_dir.as_posix(),
        "run_tag": paths.run_tag,
        "alphabet": alphabet,
        "resume_sr": resume_sr_path.as_posix() if resume_sr_path is not None else None,
        "resume_ocr": resume_ocr_path.as_posix() if resume_ocr_path is not None else None,
        "load_pretrained_ocr": config["LOAD_PRE_TRAINED_OCR"],
        "loss_sr_load": None,
    }
    if seed is not None:
        config["stage_b"]["seed"] = seed
    return config


def build_infer_config(
    *,
    paths: StageBPaths,
    checkpoint: str | Path,
    base_config_path: str | Path | None = None,
    alphabet: str = DEFAULT_ALPHABET,
    seed: int | None = None,
    allow_missing_checkpoint: bool = False,
) -> dict[str, Any]:
    if base_config_path is None:
        base_config_path = paths.lpsrlacd_repo / "configs" / TEST_CONFIG_NAME
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"lpsr-lacd test config not found: {base_config_path}")

    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid lpsr-lacd test config at {base_config_path}")

    config = deepcopy(config)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists() and not allow_missing_checkpoint:
        raise FileNotFoundError(f"SR checkpoint not found: {checkpoint_path}")

    config.setdefault("model", {})
    config["model"]["load"] = _project_relpath(checkpoint_path, paths.project_root)
    config["alphabet"] = alphabet

    test_dataset = config.setdefault("test_dataset", {})
    test_dataset.setdefault("dataset", {}).setdefault("args", {})
    test_dataset.setdefault("wrapper", {}).setdefault("args", {})
    test_dataset["dataset"]["args"]["path_split"] = _project_relpath(paths.split_file, paths.project_root)
    test_dataset["dataset"]["args"]["phase"] = "validation"
    test_dataset["wrapper"]["args"]["imgW"] = DEFAULT_IMG_W
    test_dataset["wrapper"]["args"]["imgH"] = DEFAULT_IMG_H
    test_dataset["wrapper"]["args"]["aug"] = False
    test_dataset["wrapper"]["args"]["image_aspect_ratio"] = DEFAULT_IMAGE_ASPECT_RATIO
    test_dataset["wrapper"]["args"]["background"] = DEFAULT_BACKGROUND
    test_dataset["wrapper"]["args"]["test"] = True

    config.setdefault("model_ocr", {})
    config["model_ocr"].setdefault("args", {})
    config["model_ocr"]["args"]["load"] = None

    config["stage_b"] = {
        "project_root": paths.project_root.as_posix(),
        "dataset_root": paths.dataset_root.as_posix(),
        "lpsrlacd_repo": paths.lpsrlacd_repo.as_posix(),
        "stage_dir": paths.stage_dir.as_posix(),
        "split_file": paths.split_file.as_posix(),
        "output_dir": paths.output_dir.as_posix(),
        "restored_dir": paths.restored_dir.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "alphabet": alphabet,
        "note": "Custom inference wrapper is used instead of the upstream test.py because the upstream script saves by basename.",
    }
    if seed is not None:
        config["stage_b"]["seed"] = seed
    return config


def _load_lpsrlacd_module(repo_root_path: Path, module_name: str, relative_file: str) -> ModuleType:
    module_path = repo_root_path / relative_file
    if not module_path.exists():
        raise FileNotFoundError(f"lpsr-lacd module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load lpsr-lacd module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_step_lr_verbose_compat(module: ModuleType | None = None) -> None:
    try:
        from torch.optim import lr_scheduler as torch_lr_scheduler
    except Exception:
        return

    original_step_lr = torch_lr_scheduler.StepLR
    if getattr(original_step_lr, "_lpsrocr_compat", False):
        compat_step_lr = original_step_lr
    else:
        class CompatStepLR(original_step_lr):  # type: ignore[misc, valid-type]
            def __init__(self, optimizer, *args, **kwargs):
                kwargs.pop("verbose", None)
                super().__init__(optimizer, *args, **kwargs)

        CompatStepLR._lpsrocr_compat = True  # type: ignore[attr-defined]
        CompatStepLR.__name__ = "StepLR"
        CompatStepLR.__qualname__ = "StepLR"
        compat_step_lr = CompatStepLR
        torch_lr_scheduler.StepLR = compat_step_lr

    if module is not None:
        setattr(module, "StepLR", compat_step_lr)


def _seed_torch(seed: int) -> None:
    _seed_everything(seed)


def _select_best_checkpoint(checkpoint_dir: Path, *, model_name: str | None = None) -> Path:
    if model_name is None:
        candidates = sorted(checkpoint_dir.glob("best_model_*.pth"))
    else:
        candidates = sorted(checkpoint_dir.glob(f"best_model_{model_name}_Epoch_*.pth"))
    if candidates:
        def _epoch_value(path: Path) -> tuple[int, float, str]:
            match = re.search(r"Epoch_(\d+)", path.stem)
            epoch = int(match.group(1)) if match else -1
            return epoch, path.stat().st_mtime, path.name

        return max(candidates, key=_epoch_value)

    if model_name is not None:
        fallback = checkpoint_dir / f"epoch-last-{model_name}.pth"
        if fallback.exists():
            return fallback
    epoch_last = checkpoint_dir / "epoch-last.pth"
    if epoch_last.exists():
        return epoch_last
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def _materialize_checkpoint_alias(source: Path, alias_path: Path) -> Path:
    ensure_dir(alias_path.parent)
    if alias_path.exists() or alias_path.is_symlink():
        alias_path.unlink()
    try:
        relative_source = os.path.relpath(source, start=alias_path.parent)
        os.symlink(relative_source, alias_path)
    except OSError:
        shutil.copy2(source, alias_path)
    return alias_path


def _load_image_rgb(path: Path) -> Any:
    from PIL import Image
    import numpy as np

    return np.array(Image.open(path).convert("RGB"))


def _pad_to_ratio(image: Any, min_ratio: float, max_ratio: float, color: tuple[int, int, int]) -> Any:
    import numpy as np

    height, width = image.shape[:2]
    ratio = float(width) / float(height)
    border_w = 0
    border_h = 0

    if min_ratio <= ratio <= max_ratio:
        return image

    if ratio < min_ratio:
        while ratio < min_ratio:
            border_w += 1
            ratio = float(width + border_w) / float(height + border_h)
    else:
        while ratio > max_ratio:
            border_h += 1
            ratio = float(width) / float(height + border_h)

    border_w //= 2
    border_h //= 2
    padded = np.empty((height + 2 * border_h, width + 2 * border_w, 3), dtype=image.dtype)
    padded[:] = color
    padded[border_h : border_h + height, border_w : border_w + width] = image
    return padded


def _resize_rgb(image: Any, size: tuple[int, int]) -> Any:
    from PIL import Image
    import numpy as np

    resampling = getattr(Image, "Resampling", Image)
    return np.array(Image.fromarray(image).resize(size, resample=resampling.BICUBIC))


def _rgb_to_tensor(image: Any) -> Any:
    import numpy as np
    import torch

    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)


def _tensor_to_rgb_image(tensor: Any) -> Any:
    import numpy as np

    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def _compute_psnr_ssim(sr_rgb: Any, hr_rgb: Any) -> tuple[float | None, float | None]:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except Exception:
        return None, None

    import numpy as np

    sr_arr = np.asarray(sr_rgb, dtype=np.uint8)
    hr_arr = np.asarray(hr_rgb, dtype=np.uint8)
    if sr_arr.shape != hr_arr.shape:
        return None, None

    psnr = float(peak_signal_noise_ratio(hr_arr, sr_arr, data_range=255))
    try:
        ssim = float(structural_similarity(hr_arr, sr_arr, channel_axis=-1, data_range=255))
    except TypeError:  # older skimage
        ssim = float(structural_similarity(hr_arr, sr_arr, multichannel=True, data_range=255))
    return psnr, ssim


def _run_sr_model(
    *,
    model: Any,
    lr_path: Path,
    hr_path: Path,
    img_h: int = DEFAULT_IMG_H,
    img_w: int = DEFAULT_IMG_W,
    image_aspect_ratio: float = DEFAULT_IMAGE_ASPECT_RATIO,
) -> tuple[Any, Any, Any, tuple[float | None, float | None]]:
    import torch

    lr_rgb = _load_image_rgb(lr_path)
    hr_rgb = _load_image_rgb(hr_path)

    lr_min_ratio = image_aspect_ratio - 0.15
    lr_max_ratio = image_aspect_ratio + 0.15
    hr_min_ratio = image_aspect_ratio - 0.15
    hr_max_ratio = image_aspect_ratio + 0.15

    lr_rgb = _pad_to_ratio(lr_rgb, lr_min_ratio, lr_max_ratio, (127, 127, 127))
    hr_rgb = _pad_to_ratio(hr_rgb, hr_min_ratio, hr_max_ratio, (127, 127, 127))

    lr_rgb = _resize_rgb(lr_rgb, (img_w, img_h))
    hr_rgb = _resize_rgb(hr_rgb, (img_w * 2, img_h * 2))

    lr_tensor = _rgb_to_tensor(lr_rgb).unsqueeze(0)
    with torch.no_grad():
        sr = model(lr_tensor)
        if isinstance(sr, tuple):
            sr = sr[0]
    sr_rgb = _tensor_to_rgb_image(sr[0])
    metrics = _compute_psnr_ssim(sr_rgb, hr_rgb)
    return sr_rgb, lr_rgb, hr_rgb, metrics


def run_lpsrlacd_train(
    *,
    paths: StageBPaths,
    resume_sr: str | Path | None = None,
    resume_ocr: str | Path | None = None,
    alphabet: str = DEFAULT_ALPHABET,
    base_config_path: str | Path | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    import torch

    if dry_run:
        resume_sr_path = Path(resume_sr).expanduser().resolve() if resume_sr is not None else None
        resume_ocr_path = Path(resume_ocr).expanduser().resolve() if resume_ocr is not None else None
        return {
            "dry_run": True,
            "config": paths.train_config.as_posix(),
            "save_path": paths.train_run_dir.as_posix(),
            "resume_sr": resume_sr_path.as_posix() if resume_sr_path is not None else None,
            "resume_ocr": resume_ocr_path.as_posix() if resume_ocr_path is not None else None,
            "seed": seed,
            "preview": {
                "resume_strategy": "resume" if resume_sr_path is not None else "load_pretrained_ocr",
                "train_config_path": paths.train_config.as_posix(),
            },
        }

    _seed_torch(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("lpsr-lacd training requires CUDA. Run this in Colab or another GPU environment.")
    if not paths.split_file.exists():
        raise FileNotFoundError(f"Stage B split file not found: {paths.split_file}")
    if not paths.stage_dir.exists():
        raise FileNotFoundError(f"Stage B stage directory not found: {paths.stage_dir}")

    resume_sr_path = Path(resume_sr).expanduser().resolve() if resume_sr is not None else None
    if resume_sr_path is not None and not resume_sr_path.exists():
        raise FileNotFoundError(f"SR resume checkpoint not found: {resume_sr_path}")
    resume_ocr_path = _normalize_ocr_checkpoint(resume_ocr, output_dir=paths.output_dir)

    config = build_train_config(
        paths=paths,
        resume_sr=resume_sr_path,
        resume_ocr=resume_ocr_path,
        base_config_path=base_config_path,
        alphabet=alphabet,
        seed=seed,
    )
    _write_yaml(config, paths.train_config)
    ensure_dir(paths.train_run_dir)

    _patch_step_lr_verbose_compat()
    with _temporary_sys_path(paths.lpsrlacd_repo), _temporary_cwd(paths.project_root):
        lpsr_train = _load_lpsrlacd_module(paths.lpsrlacd_repo, "lpsrlacd_train_stage_b", "ParallelNetTrain.py")
        _patch_step_lr_verbose_compat(lpsr_train)
        lpsr_train.main(config, paths.train_run_dir)

    sr_checkpoint = _select_best_checkpoint(paths.train_run_dir, model_name=config["MODEL_SR"]["name"])
    ocr_checkpoint = _select_best_checkpoint(paths.train_run_dir, model_name=config["MODEL_OCR"]["name"])
    sr_alias = _materialize_checkpoint_alias(sr_checkpoint, paths.best_sr_alias)
    _materialize_checkpoint_alias(sr_checkpoint, paths.best_model_alias)
    ocr_alias = _materialize_checkpoint_alias(ocr_checkpoint, paths.best_ocr_alias)

    result = {
        "dry_run": False,
        "config": paths.train_config.as_posix(),
        "save_path": paths.train_run_dir.as_posix(),
        "best_sr_checkpoint": sr_checkpoint.as_posix(),
        "best_sr_alias": sr_alias.as_posix(),
        "best_checkpoint": sr_alias.as_posix(),
        "best_ocr_checkpoint": ocr_checkpoint.as_posix(),
        "best_ocr_alias": ocr_alias.as_posix(),
        "seed": seed,
    }
    summary_path = paths.output_dir / "train_summary.json"
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["summary_path"] = summary_path.as_posix()
    return result


def run_lpsrlacd_infer(
    *,
    paths: StageBPaths,
    checkpoint: str | Path | None = None,
    alphabet: str = DEFAULT_ALPHABET,
    base_config_path: str | Path | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    import torch

    if dry_run:
        if checkpoint is None:
            checkpoint_path = paths.best_model_alias if paths.best_model_alias.exists() else (
                paths.train_run_dir / "best_model.pth"
            )
        else:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
        return {
            "dry_run": True,
            "config": paths.infer_config.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "restored_dir": paths.restored_dir.as_posix(),
            "seed": seed,
        }

    if checkpoint is None:
        checkpoint = paths.best_model_alias if paths.best_model_alias.exists() else _select_best_checkpoint(
            paths.train_run_dir, model_name="cgnetV2_deformable"
        )

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SR checkpoint not found: {checkpoint_path}")

    config = build_infer_config(
        paths=paths,
        checkpoint=checkpoint_path,
        base_config_path=base_config_path,
        alphabet=alphabet,
        seed=seed,
    )
    _write_yaml(config, paths.infer_config)

    _seed_torch(seed)
    if not paths.val_manifest.exists():
        raise FileNotFoundError(f"Validation manifest not found: {paths.val_manifest}")
    if not paths.stage_dir.exists():
        raise FileNotFoundError(f"Stage directory not found: {paths.stage_dir}")

    with _temporary_sys_path(paths.lpsrlacd_repo), _temporary_cwd(paths.project_root):
        import models as lpsr_models

        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
        sr_model = lpsr_models.make(checkpoint_obj["model"], load_model=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr_model = sr_model.to(device)
    sr_model.eval()

    val_rows = read_manifest_csv(paths.val_manifest)
    val_rows = sorted(val_rows, key=_row_key)

    restored_rows: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []

    with torch.no_grad():
        for row in val_rows:
            for frame_idx, hr_rel, lr_rel in _track_pairs(row):
                lr_path = paths.dataset_root / lr_rel
                hr_path = paths.dataset_root / hr_rel
                sr_rgb, lr_rgb, hr_rgb, metrics = _run_sr_model(
                    model=sr_model,
                    lr_path=lr_path,
                    hr_path=hr_path,
                    img_h=DEFAULT_IMG_H,
                    img_w=DEFAULT_IMG_W,
                    image_aspect_ratio=DEFAULT_IMAGE_ASPECT_RATIO,
                )

                restored_path = paths.restored_dir / row["track_dir"] / f"sr-{frame_idx:03d}.png"
                ensure_dir(restored_path.parent)
                from PIL import Image

                Image.fromarray(sr_rgb).save(restored_path)

                psnr, ssim = metrics
                if psnr is not None:
                    psnr_values.append(psnr)
                if ssim is not None:
                    ssim_values.append(ssim)

                restored_rows.append(
                    {
                        "track_id": row["track_id"],
                        "track_num": row["track_num"],
                        "scenario": row["scenario"],
                        "domain": row["domain"],
                        "track_dir": row["track_dir"],
                        "frame_idx": frame_idx,
                        "split_label": "validation",
                        "lr_image_path": lr_rel,
                        "hr_image_path": hr_rel,
                        "restored_image_path": _project_relpath(restored_path, paths.project_root),
                        "gt_text": row["gt_text"],
                        "psnr": psnr,
                        "ssim": ssim,
                    }
                )

    restored_manifest = paths.restored_dir / "restoration_manifest.jsonl"
    ensure_dir(restored_manifest.parent)
    write_manifest_jsonl(restored_rows, restored_manifest)

    metrics_rows = [
        {
            "image_path": item["restored_image_path"],
            "track_id": item["track_id"],
            "frame_idx": item["frame_idx"],
            "scenario": item["scenario"],
            "domain": item["domain"],
            "split_label": item["split_label"],
            "lr_image_path": item["lr_image_path"],
            "hr_image_path": item["hr_image_path"],
            "psnr": item["psnr"],
            "ssim": item["ssim"],
        }
        for item in restored_rows
    ]
    sr_per_image_csv = paths.eval_dir / "sr_per_image.csv"
    write_manifest_csv(metrics_rows, sr_per_image_csv)

    summary = {
        "checkpoint": checkpoint_path.as_posix(),
        "num_images": len(restored_rows),
        "num_tracks": len({row["track_id"] for row in restored_rows}),
        "restored_dir": paths.restored_dir.as_posix(),
        "restoration_manifest": restored_manifest.as_posix(),
        "sr_per_image_csv": sr_per_image_csv.as_posix(),
        "mean_psnr": float(sum(psnr_values) / len(psnr_values)) if psnr_values else None,
        "mean_ssim": float(sum(ssim_values) / len(ssim_values)) if ssim_values else None,
        "psnr_available": bool(psnr_values),
        "ssim_available": bool(ssim_values),
        "seed": seed,
        "preview": restored_rows[:3],
    }
    sr_summary = paths.eval_dir / "sr_summary.json"
    ensure_dir(sr_summary.parent)
    sr_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def prepare_stage_b_assets(
    *,
    project_root: str | Path = repo_root(),
    dataset_root: str | Path | None = None,
    lpsrlacd_repo: str | Path,
    split_dir: str | Path | None = None,
    train_manifest: str | Path | None = None,
    val_manifest: str | Path | None = None,
    stage_dir: str | Path = DEFAULT_STAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    mode: str = "symlink",
    resume_sr: str | Path | None = None,
    resume_ocr: str | Path | None = None,
    base_train_config: str | Path | None = None,
    base_test_config: str | Path | None = None,
    run_tag: str | None = None,
    alphabet: str = DEFAULT_ALPHABET,
    seed: int = 42,
    export: bool = True,
) -> dict[str, Any]:
    _seed_torch(seed)
    paths = prepare_stage_b_paths(
        project_root=project_root,
        dataset_root=dataset_root,
        lpsrlacd_repo=lpsrlacd_repo,
        split_dir=split_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        stage_dir=stage_dir,
        output_dir=output_dir,
        run_tag=run_tag,
    )

    ensure_dir(paths.stage_dir)
    ensure_dir(paths.output_dir)

    export_summary: dict[str, Any] | None = None
    if export:
        export_summary = export_lpsrlacd_pairs(
            dataset_root=paths.dataset_root,
            split_dir=paths.split_dir,
            stage_dir=paths.stage_dir,
            project_root=paths.project_root,
            mode=mode,
        )

    train_config = build_train_config(
        paths=paths,
        resume_sr=resume_sr,
        resume_ocr=resume_ocr,
        base_config_path=base_train_config,
        alphabet=alphabet,
        seed=seed,
    )
    infer_config = build_infer_config(
        paths=paths,
        checkpoint=paths.best_model_alias if paths.best_model_alias.exists() else paths.train_run_dir / "best_model.pth",
        base_config_path=base_test_config,
        alphabet=alphabet,
        seed=seed,
        allow_missing_checkpoint=True,
    )

    train_config_path = _write_yaml(train_config, paths.train_config)
    infer_config_path = _write_yaml(infer_config, paths.infer_config)

    summary = {
        "paths": {
            "project_root": paths.project_root.as_posix(),
            "dataset_root": paths.dataset_root.as_posix(),
            "lpsrlacd_repo": paths.lpsrlacd_repo.as_posix(),
            "split_dir": paths.split_dir.as_posix(),
            "train_manifest": paths.train_manifest.as_posix(),
            "val_manifest": paths.val_manifest.as_posix(),
            "stage_dir": paths.stage_dir.as_posix(),
            "output_dir": paths.output_dir.as_posix(),
            "split_file": paths.split_file.as_posix(),
            "pair_manifest": paths.pair_manifest.as_posix(),
            "train_config": train_config_path.as_posix(),
            "infer_config": infer_config_path.as_posix(),
            "train_run_dir": paths.train_run_dir.as_posix(),
            "restored_dir": paths.restored_dir.as_posix(),
        },
        "run_tag": paths.run_tag,
        "mode": mode,
        "seed": seed,
        "resume_sr": Path(resume_sr).expanduser().resolve().as_posix() if resume_sr is not None else None,
        "resume_ocr": Path(resume_ocr).expanduser().resolve().as_posix() if resume_ocr is not None else None,
    }
    if export_summary is not None:
        summary["export"] = export_summary

    summary_path = paths.setup_summary_path
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_path"] = summary_path.as_posix()
    return summary


def _print_resolved_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_arg_parser_prepare() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage competition LR/HR pairs for lpsr-lacd.")
    parser.add_argument("--project-root", default=str(repo_root()), help="Project repository root.")
    parser.add_argument("--dataset-root", default=None, help="Competition dataset root.")
    parser.add_argument("--lpsrlacd-repo", required=True, help="Path to the lpsr-lacd repository clone.")
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Split directory from prompt 1.")
    parser.add_argument("--train-manifest", default=None, help="Optional explicit train manifest path.")
    parser.add_argument("--val-manifest", default=None, help="Optional explicit val manifest path.")
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_DIR), help="Staging directory for paired images.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Stage B outputs root.")
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy", "paths"),
        default="symlink",
        help="How to materialize image files.",
    )
    parser.add_argument("--run-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded in summaries.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths without writing files.")
    return parser


def build_arg_parser_train() -> argparse.ArgumentParser:
    parser = build_arg_parser_prepare()
    parser.description = "Fine-tune lpsr-lacd on the paired competition dataset."
    parser.add_argument("--resume-sr", default=None, help="Optional SR checkpoint to resume from.")
    parser.add_argument(
        "--resume-ocr",
        default=None,
        help="Optional GPLPR OCR checkpoint to initialize the OCR branch. Use the Stage A best_model.pth here.",
    )
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET, help="OCR alphabet.")
    parser.add_argument("--base-train-config", default=None, help="Override the upstream train config template.")
    return parser


def build_arg_parser_infer() -> argparse.ArgumentParser:
    parser = build_arg_parser_prepare()
    parser.description = "Run lpsr-lacd validation inference and export restored SR images."
    parser.add_argument("--checkpoint", default=None, help="SR checkpoint to evaluate. Defaults to the best one.")
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET, help="OCR alphabet.")
    parser.add_argument("--base-test-config", default=None, help="Override the upstream test config template.")
    return parser


def main_prepare(argv: list[str] | None = None) -> int:
    parser = build_arg_parser_prepare()
    args = parser.parse_args(argv)
    if args.dry_run:
        paths = prepare_stage_b_paths(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            lpsrlacd_repo=args.lpsrlacd_repo,
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
                "lpsrlacd_repo": paths.lpsrlacd_repo.as_posix(),
                "split_dir": paths.split_dir.as_posix(),
                "train_manifest": paths.train_manifest.as_posix(),
                "val_manifest": paths.val_manifest.as_posix(),
                "stage_dir": paths.stage_dir.as_posix(),
                "output_dir": paths.output_dir.as_posix(),
                "split_file": paths.split_file.as_posix(),
                "pair_manifest": paths.pair_manifest.as_posix(),
                "train_config": paths.train_config.as_posix(),
                "infer_config": paths.infer_config.as_posix(),
                "run_tag": paths.run_tag,
                "mode": args.mode,
                "seed": args.seed,
            }
        )
        return 0

    summary = prepare_stage_b_assets(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        lpsrlacd_repo=args.lpsrlacd_repo,
        split_dir=args.split_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        run_tag=args.run_tag,
        seed=args.seed,
        export=True,
    )
    _print_resolved_summary(summary)
    return 0


def main_train(argv: list[str] | None = None) -> int:
    parser = build_arg_parser_train()
    args = parser.parse_args(argv)
    paths = prepare_stage_b_paths(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        lpsrlacd_repo=args.lpsrlacd_repo,
        split_dir=args.split_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
    )
    result = run_lpsrlacd_train(
        paths=paths,
        resume_sr=args.resume_sr,
        resume_ocr=args.resume_ocr,
        alphabet=args.alphabet,
        base_config_path=args.base_train_config,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    _print_resolved_summary(result)
    return 0


def main_infer(argv: list[str] | None = None) -> int:
    parser = build_arg_parser_infer()
    args = parser.parse_args(argv)
    paths = prepare_stage_b_paths(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        lpsrlacd_repo=args.lpsrlacd_repo,
        split_dir=args.split_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
    )
    result = run_lpsrlacd_infer(
        paths=paths,
        checkpoint=args.checkpoint,
        alphabet=args.alphabet,
        base_config_path=args.base_test_config,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    _print_resolved_summary(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_prepare())
