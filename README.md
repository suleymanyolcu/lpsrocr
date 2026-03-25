# Low-Resolution License Plate Recognition Bootstrap

This repository is a small, boring Python workspace for the ICPR 2026 low-resolution license plate recognition benchmark.

It is set up for:
- local editing in VSCode
- GPU-heavy work in Google Colab
- dataset and checkpoints stored on Drive or mounted storage
- track-level manifests, splits, and export staging only

## Stages

- Stage A: fine-tune OCR on competition data
- Stage B: fine-tune SR on competition data
- Stage C: run LR -> SR -> OCR -> track aggregation

This repo only prepares the dataset and export scaffolding. Training code is intentionally out of scope for now.

## Stage A

Stage A fine-tunes GPLPR on the competition HR frames only.

What it does:
- stages HR images into `external_data/gplpr_stage/images/training` and `.../validation`
- writes same-stem `.txt` labels next to each staged image
- uses the current Scenario-B dev split as the default Stage A split
- patches a GPLPR train config and eval config into `outputs/stage_a/configs`
- runs GPLPR through its Python entrypoints, not the upstream CLI, so the upstream email tail is bypassed

Label format:
- the staged `.txt` files contain `plate: <GT_TEXT>` because GPLPR’s OCR wrapper reads that exact pattern
- the competition labels observed in the scan are 7 characters long and use only `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`

Prepare data:

```bash
python scripts/stage_a_prepare_gplpr.py \
  --project-root . \
  --dataset-root ./train \
  --gplpr-repo /path/to/gplpr \
  --split-dir ./manifests/splits/scenario_b_dev_seed42_n400_v20 \
  --stage-dir ./external_data/gplpr_stage \
  --output-dir ./outputs/stage_a \
  --mode symlink
```

Train GPLPR in Colab:

```bash
python scripts/stage_a_train_gplpr.py \
  --project-root . \
  --dataset-root ./train \
  --gplpr-repo /path/to/gplpr \
  --split-dir ./manifests/splits/scenario_b_dev_seed42_n400_v20 \
  --stage-dir ./external_data/gplpr_stage \
  --output-dir ./outputs/stage_a \
  --resume /path/to/optional_resume.pth
```

Evaluate on HR validation images:

```bash
python scripts/stage_a_eval_gplpr.py \
  --project-root . \
  --dataset-root ./train \
  --gplpr-repo /path/to/gplpr \
  --split-dir ./manifests/splits/scenario_b_dev_seed42_n400_v20 \
  --stage-dir ./external_data/gplpr_stage \
  --output-dir ./outputs/stage_a \
  --checkpoint ./outputs/stage_a/checkpoints/stage_a_scenario_b_dev_seed42_n400_v20/best_model.pth
```

Stage A outputs:
- `outputs/stage_a/configs/`
- `outputs/stage_a/checkpoints/`
- `outputs/stage_a/predictions/`
- `outputs/stage_a/eval/summary.json`

Note:
- GPLPR’s current test output does not expose confidence, so the evaluation wrapper records `confidence: null`.
- If Colab says `No module named 'kornia'`, rerun `pip install -r requirements.txt` or `pip install kornia` before training.
- If you use a pretrained GPLPR checkpoint with `--resume`, the wrapper adds a small `ReduceLROnPlateau` block because GPLPR’s resume path expects one.
- If Colab says `No module named 'Levenshtein'`, rerun `pip install -r requirements.txt` or `pip install python-Levenshtein` before evaluation.

## Workflow

1. Edit code locally in VSCode.
2. Keep the dataset under `./train` or a mounted Drive path.
3. Run scanning and split generation locally or in Colab.
4. Point Colab notebooks at the generated manifests and export folders.
5. Save checkpoints and outputs to Drive, not to the notebook runtime.

## Project Layout

- `src/`: scanner, split generator, manifest helpers, export adapters
- `scripts/`: thin CLI wrappers
- `manifests/`: full manifest and split manifests
- `outputs/`: scan summary and export outputs
- `external_data/`: staged data for downstream repos
- `notebooks/`: Colab handoff templates

## First Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Scan the dataset:

```bash
python scripts/scan_dataset.py --dataset-root ./train --manifests-dir ./manifests --output-dir ./outputs
```

Make the Scenario-B-focused dev split:

```bash
python scripts/make_splits.py --dataset-root ./train --output-dir ./manifests --split-mode scenario_b_dev --seed 42 --val-ratio 0.2
```

## Notes

- Split logic is track-level only.
- LR/HR pairing is index-based.
- `plate_text` is treated as the GT source.
- `plate_layout` is treated as the domain label.
