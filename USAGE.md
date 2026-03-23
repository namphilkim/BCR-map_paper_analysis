# Usage

MIL on ViT patch embeddings in `.h5`. Checkpoints: Zenodo, not `git clone`.

## Install

```bash
git clone https://github.com/namphilkim/BCR-map_paper_analysis.git
cd BCR-map_paper_analysis
pip install -r requirements.txt
pip install -e .
```

Unpack Zenodo weights under `checkpoints/` before loading `.ckpt`.

## CLI

| Command | |
|---------|---|
| `bcr-map train …` | LightningCLI — `bcr-map train -- --help` |
| `bcr-map extract …` | ViT → `.h5` — `bcr-map extract -- --help` |

## Files

| File | |
|------|---|
| `bcr_map/cli.py` | CLI |
| `bcr_map/train.py` | Training |
| `main.py` | `bcr-map train` |
| `configs/mil_training.yaml` | Defaults |
| `extract_vit_patch_embeddings.py` | Extraction driver |
| `utils/mil_model.py` | Model |
| `utils/mil_data.py` | Data |
| `utils/loss.py` | Loss |
| `scripts/train_mil_single_run.sh` | Env-based train |
| `scripts/extract_embeddings_user_images.sh` | Env-based extract |

## Embeddings

1. BCR-map images under one root.  
2. `bcr-map extract --data_dir /path/to/images`  
3. Fold CSV: columns `fold`, `class`, `image_path` (relative to dataset root). See `utils/mil_data.py`.

## Training (env)

```bash
export MIL_DATAPATH=/path/to/root_used_in_csv
export MIL_FOLDS_CSV=/path/to/folds.csv
export MIL_EXPERIMENT_ID=my_project/exp01/mil_hipt_vit_b16_6class
export MIL_NUM_CLASSES=6
export MIL_K_FOLDS=5
bash scripts/train_mil_single_run.sh
```

## Environment variables (`train_mil_single_run.sh`)

| Variable | Required | Default / note |
|----------|----------|----------------|
| `MIL_DATAPATH` | yes | Dataset root |
| `MIL_FOLDS_CSV` | yes | Folds CSV |
| `MIL_EXPERIMENT_ID` | yes | Subdir under `checkpoints/` and `logs` |
| `MIL_NUM_CLASSES` | no | `6` |
| `MIL_K_FOLDS` | no | `3` |
| `MIL_MAX_EPOCHS` | no | `100` |
| `MIL_BATCH_SIZE` | no | `2` |
| `MIL_MAX_PATCHES` | no | `10000` |
| `MIL_GPU` | no | `0` |
| `MIL_AGGREGATION` | no | `hipt` |
| `MIL_BACKBONE` | no | `vit-b16-224-in21k` |
| `MIL_CONFIG` | no | `$REPO_ROOT/configs/mil_training.yaml` |
| `BCR_MAP_ROOT` | no | Repo root |
| `BCR_MAP_STABLE_VERSION` | no | `1` |
| `BCR_MAP_CSV_LOGGER_ONLY` | no | `1` (CSV; `0` for W&B) |

## Zenodo weights

1. Download archive from Zenodo (DOI in main README when set).  
2. Extract so `checkpoints/In-house/...`, `checkpoints/Mal-ID/...` match training paths.  
`checkpoints/README.md`.

## Example env (paper-style paths)

In-house:

```bash
export MIL_DATAPATH="${BCRMAP_DATA_ROOT}/In-house_maps_naive_vj_fixed/sampling_100000/none"
export MIL_FOLDS_CSV="${BCRMAP_DATA_ROOT}/In-house_maps_naive_vj_fixed/data_folds_6class_no_RA.csv"
export MIL_EXPERIMENT_ID="In-house/sampling_100000/none/mil_hipt_vit_b16_6class"
export MIL_NUM_CLASSES=6
export MIL_K_FOLDS=3
bash scripts/train_mil_single_run.sh
```

Mal-ID:

```bash
export MIL_DATAPATH="${BCRMAP_DATA_ROOT}/Mal-ID_maps_naive_vj_fixed/sampling_5000/none"
export MIL_FOLDS_CSV="${BCRMAP_DATA_ROOT}/Mal-ID_maps_naive_vj_fixed/data_folds_260108.csv"
export MIL_EXPERIMENT_ID="Mal-ID/naive_vj_fixed/sampling_5000/none/mil_hipt_vit_b16_6class"
export MIL_NUM_CLASSES=6
export MIL_K_FOLDS=5
bash scripts/train_mil_single_run.sh
```

## Notebooks

`BCR-map_figure_*.ipynb`, `MyBasics.py`, `BCR-map_intermediate_files/` at repo root.

## Reproducibility

Pin deps: `pip freeze > requirements-lock.txt` after a validated run.
