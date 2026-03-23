# Usage protocol

This document supplements the main **[README.md](README.md)** in the **[BCR-map_paper_analysis](https://github.com/namphilkim/BCR-map_paper_analysis)** repository. It focuses on the **image-based MIL** training and embedding pipeline (*BCR-map: Integrative Visualization of B-cell Receptor Repertoires*). **Pretrained `.ckpt` files are obtained from Zenodo**, not from `git clone`. Figure notebooks live in the same repository (see README table).

**Scope:** Inputs are **BCR-map images** (the paper’s 2D repertoire visualizations). MIL training operates on **precomputed** ViT patch embeddings (`.h5`) derived from those images. Offline embedding generation uses **`bcr-map extract`**. Training uses **`bcr-map train`** (or `python main.py`) with **`configs/mil_training.yaml`** by default (HiPT aggregator).

---

## 1. Installation

```bash
git clone https://github.com/namphilkim/BCR-map_paper_analysis.git
cd BCR-map_paper_analysis
pip install -r requirements.txt
pip install -e .   # provides the `bcr-map` console script
```

Then download model weights from **Zenodo** (see §6) into `checkpoints/` before running evaluations that load saved weights.

---

## 2. Command-line interface

| Command | Purpose |
|---------|---------|
| `bcr-map train …` | MIL training via LightningCLI. Full help: `bcr-map train -- --help` |
| `bcr-map extract …` | ViT patch features from each BCR-map image → `.h5` beside the source file. Help: `bcr-map extract -- --help` |

---

## 3. Source layout (software)

| File | Role |
|------|------|
| `bcr_map/cli.py` | Typer CLI (`train` / `extract`). |
| `bcr_map/train.py` | MIL training implementation. |
| `main.py` | Delegates to `bcr_map.train` (equivalent to `bcr-map train`). |
| `configs/mil_training.yaml` | Default hyperparameters; override via CLI. |
| `extract_vit_patch_embeddings.py` | Argparse driver for extraction (invoked by `bcr-map extract`). |
| `utils/mil_model.py` | Classifier and aggregation heads. |
| `utils/mil_data.py` | Precomputed `.h5` embeddings and fold CSV loading. |
| `utils/loss.py` | Loss for the MIL model. |
| `scripts/train_mil_single_run.sh` | One k-fold job via environment variables; runs `python -m bcr_map.cli train` from the repository root. |
| `scripts/extract_embeddings_user_images.sh` | Embeddings for an arbitrary image tree; wraps `python -m bcr_map.cli extract`. |

---

## 4. End-to-end pipeline (user data)

### 4.1 Embedding generation

1. Place **BCR-map images** (PNG/JPEG/TIFF as exported from your BCR-map pipeline) under a single directory tree (recursive walk).
2. Run (from repository root, or after `pip install -e .`):

```bash
export GPU=0   # optional; passed to CUDA_VISIBLE_DEVICES in shell wrappers
bcr-map extract --data_dir /path/to/your/images
# or: bash scripts/extract_embeddings_user_images.sh /path/to/your/images
```

3. Build a **fold CSV** with columns `fold`, `class`, `image_path` (paths relative to the dataset root). See `utils/mil_data.py` for `.h5` resolution next to each image.

### 4.2 Training

```bash
export MIL_DATAPATH=/path/to/root_used_in_csv
export MIL_FOLDS_CSV=/path/to/folds.csv
export MIL_EXPERIMENT_ID=my_project/exp01/mil_hipt_vit_b16_6class
export MIL_NUM_CLASSES=6
export MIL_K_FOLDS=5
bash scripts/train_mil_single_run.sh
```

---

## 5. Environment variables (`train_mil_single_run.sh`)

| Variable | Required | Description |
|----------|----------|-------------|
| `MIL_DATAPATH` | yes | Dataset root (paths in CSV are relative to this). |
| `MIL_FOLDS_CSV` | yes | Fold assignment CSV. |
| `MIL_EXPERIMENT_ID` | yes | Subdirectory under `checkpoints/` and `logs`. |
| `MIL_NUM_CLASSES` | no | Default `6`. |
| `MIL_K_FOLDS` | no | Default `3`. |
| `MIL_MAX_EPOCHS` | no | Default `100`. |
| `MIL_BATCH_SIZE` | no | Default `2`. |
| `MIL_MAX_PATCHES` | no | Default `10000`. |
| `MIL_GPU` | no | Default `0`. |
| `MIL_AGGREGATION` | no | Default `hipt`. |
| `MIL_BACKBONE` | no | Default `vit-b16-224-in21k`. |
| `MIL_CONFIG` | no | Default `$REPO_ROOT/configs/mil_training.yaml`. |
| `BCR_MAP_ROOT` | no | Repository root (default: parent of `scripts/`). |
| `BCR_MAP_STABLE_VERSION` | no | Default `1` (stable `--version` paths). |
| `BCR_MAP_CSV_LOGGER_ONLY` | no | Default `1` (CSV only; `0` for Weights & Biases). |

---

## 6. Zenodo model weights and local layout

**Checkpoints (`*.ckpt`)** are **not** in Git. After you publish a Zenodo deposit, add its **DOI** to the main README and cite it in publications.

1. Open the Zenodo record (placeholder DOI: `10.5281/zenodo.XXXXXXXX`).
2. Download the archive (`.zip` / `.tar` as you uploaded).
3. Extract into the repository root so relative paths match training, e.g.:

| Expected path (after extract) |
|-------------------------------|
| `checkpoints/In-house/...` |
| `checkpoints/Mal-ID/...` |

See [`checkpoints/README.md`](checkpoints/README.md).

**Logs** (CSVs for downstream figure reproduction — may be tracked in git for paper figures):

| Directory |
|-----------|
| `logs/lightning_logs/In-house/` |
| `logs/lightning_logs/Mal-ID/` |

---

## 7. Example commands (paper-style naming)

**In-house**

```bash
export MIL_DATAPATH="${BCRMAP_DATA_ROOT}/In-house_maps_naive_vj_fixed/sampling_100000/none"
export MIL_FOLDS_CSV="${BCRMAP_DATA_ROOT}/In-house_maps_naive_vj_fixed/data_folds_6class_no_RA.csv"
export MIL_EXPERIMENT_ID="In-house/sampling_100000/none/mil_hipt_vit_b16_6class"
export MIL_NUM_CLASSES=6
export MIL_K_FOLDS=3
bash scripts/train_mil_single_run.sh
```

**Mal-ID**

```bash
export MIL_DATAPATH="${BCRMAP_DATA_ROOT}/Mal-ID_maps_naive_vj_fixed/sampling_5000/none"
export MIL_FOLDS_CSV="${BCRMAP_DATA_ROOT}/Mal-ID_maps_naive_vj_fixed/data_folds_260108.csv"
export MIL_EXPERIMENT_ID="Mal-ID/naive_vj_fixed/sampling_5000/none/mil_hipt_vit_b16_6class"
export MIL_NUM_CLASSES=6
export MIL_K_FOLDS=5
bash scripts/train_mil_single_run.sh
```

---

## 8. Figure notebooks (same repository)

Figure reproduction notebooks (`BCR-map_figure_*.ipynb`), `MyBasics.py`, and `BCR-map_intermediate_files/` are part of **[BCR-map_paper_analysis](https://github.com/namphilkim/BCR-map_paper_analysis)** alongside the MIL code in this tree; **USAGE.md** documents only the training and embedding workflow.

---

## 9. Dependency note

`requirements.txt` and `pyproject.toml` specify compatible ranges. For reproducibility across machines, export a **fully pinned** environment after validation (`pip freeze > requirements-lock.txt`) and archive it with the paper supplement.
