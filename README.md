# BCR-map_paper_analysis

Codes utilized for analysis performed in the manuscript *BCR-map: Integrative Visualization of B-cell Receptor Repertoires*.

This [GitHub repository](https://github.com/namphilkim/BCR-map_paper_analysis) contains **both** (1) figure-level analysis notebooks and intermediate files already on `main`, and (2) **model training and ViT patch embedding extraction** for image-based BCR-map classification (`bcr_map/`, `utils/`, `configs/`, and related scripts).

**Release:** **v1.0.0** — Python package version in [`pyproject.toml`](pyproject.toml), [`bcr_map/__init__.py`](bcr_map/__init__.py), and [`CITATION.cff`](CITATION.cff).

## Publication

**Title:** *BCR-map: Integrative Visualization of B-cell Receptor Repertoires*

**Abstract:** The B-cell receptor (BCR) repertoire encodes cumulative immune history shaped by antigen-driven responses. However, most BCR repertoire analyses rely on aggregated summary metrics that obscure how individual clonotypes collectively organize immune responses. Here, we present BCR-map, an integrative visualization framework that spatially organizes clonotypes to preserve global repertoire structure while retaining clonotype-level resolution and enabling simultaneous interpretation of multiple repertoire features. Using BCR-map, we identify disease-associated repertoire signatures across COVID-19, autoimmune diseases, and Alzheimer’s disease in 111 individuals, track longitudinal immune responses following vaccination, and enable robust multi-disease classification using image-based deep learning. These results establish BCR-map as an interpretable framework for system-level immune-repertoire analysis and facilitate discovery and monitoring of immune states across diseases and over time.

> **To cite:** use [`CITATION.cff`](CITATION.cff) and add the final DOI, journal, volume, and author list in that file when available.

---

## Repository layout

| Path | Description |
|------|-------------|
| `BCR-map_figure_*.ipynb` | Figure reproduction notebooks (analysis). |
| `BCR-map_intermediate_files/` | Intermediate tables and inputs for analyses. |
| `MyBasics.py` | Plotting and parsing utilities for notebooks. |
| `bcr_map/` | Installable package: MIL training (`train.py`) and CLI (`cli.py`). |
| `utils/` | `MILDataModule`, `MILClassificationModel`, losses. |
| `configs/mil_training.yaml` | Default training configuration (HiPT, ViT-B/16). |
| `extract_vit_patch_embeddings.py` | ViT patch features from BCR-map images, written as `.h5` next to source images. |
| `main.py` | Legacy-compatible entry point (`python main.py` ≡ `bcr-map train`). |
| `scripts/` | Optional shell wrappers for portable paths and environment variables. |
| `USAGE.md` | Full protocol: environment variables, checkpoint layouts, examples. |
| `checkpoints/` | **Local only** — place Zenodo-downloaded `.ckpt` files here (see below). |

**Large files:** model weights (≈10+ GB) are **not** in Git. Use **Zenodo** (below). Selected **metrics logs** under `logs/lightning_logs/` may still be tracked for figure reproduction per [`.gitignore`](.gitignore).

---

## Image-based MIL (concise)

Inputs are **BCR-map images** (2D repertoire layouts defined in the paper). Each image is treated as a bag of ViT patch embeddings; a multiple-instance learning (MIL) head aggregates patch features for **image-level** (repertoire-level) prediction. Patch embeddings are produced offline (`bcr-map extract`); training uses `bcr-map train` with **PyTorch Lightning** and YAML configuration.

---

## System requirements

- **Python** ≥ 3.9 (3.10+ recommended).
- **CUDA**-capable GPU strongly recommended for extraction and training; CPU runs are possible but slow.
- **Disk:** sufficient space for image trees, per-image `.h5` embedding files, and checkpoints.

---

## Installation

```bash
git clone https://github.com/namphilkim/BCR-map_paper_analysis.git
cd BCR-map_paper_analysis
python -m venv .venv && source .venv/bin/activate   # optional
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

**Conda (optional):** see [`environment.yml`](environment.yml).

Verify:

```bash
bcr-map --help
```

---

## Minimal workflow (model training)

**1. Patch embeddings** from each BCR-map image (writes `<image_stem>.h5` beside the file):

```bash
bcr-map extract --data_dir /path/to/images
```

**2. Training** (requires a fold CSV and dataset root; defaults in `configs/mil_training.yaml`):

```bash
bcr-map train --config configs/mil_training.yaml
```

For a single scripted run with environment variables (recommended for shared clusters), see **`USAGE.md`** and `scripts/train_mil_single_run.sh`.

**Lightning CLI help:** `bcr-map train -- --help`  
**Extraction help:** `bcr-map extract -- --help`

---

## Method overview (MIL)

1. **Patch embeddings:** sliding window over each **BCR-map** RGB image; ViT forward on patches; stored as compressed arrays in HDF5.
2. **MIL training:** bags of patch vectors per BCR-map image; aggregation methods include **HiPT** (default), attention-based MIL, gated attention, transformer MIL, and simple pooling (see `utils/mil_model.py`).

Full parameter descriptions and aggregation choices are documented in the code and in `configs/mil_training.yaml`.

---

## Data and ethics

This repository **does not** contain repertoire data, BCR-map images, identifiers, or model checkpoint binaries (those are on **Zenodo**). Users must supply their own training data where applicable and ensure compliance with institutional review, consent, and data-use agreements. Paths in the paper supplement (e.g. `In-house`, `Mal-ID`) refer to **local** checkpoint and log layouts after you download weights; **USAGE.md** describes directory naming.

---

## Citation

Cite the article ***BCR-map: Integrative Visualization of B-cell Receptor Repertoires*** and this repository. Machine-readable metadata: [`CITATION.cff`](CITATION.cff). Example BibTeX entries (fill in DOI, journal, and authors when published):

```bibtex
@article{bcr_map_2025,
  title   = {BCR-map: Integrative Visualization of B-cell Receptor Repertoires},
  author  = {…},
  journal = {…},
  year    = {2025},
  doi     = {…}
}

@software{bcr_map_paper_analysis,
  title        = {BCR-map_paper_analysis: figure analysis and MIL training code},
  version      = {1.0.0},
  year         = {2026},
  url          = {https://github.com/namphilkim/BCR-map_paper_analysis},
  note         = {Includes BCR-map_figure notebooks and image-based MIL pipeline; model checkpoints on Zenodo}
}

% Weights archive (replace with your Zenodo DOI):
% @dataset{bcr_map_weights_2025,
%   title   = {BCR-map MIL model checkpoints},
%   year    = {2025},
%   doi     = {10.5281/zenodo.XXXXXXXX},
%   publisher = {Zenodo}
% }
```

---

## License

See [`LICENSE`](LICENSE) (MIT).

---

## Further documentation

- **`USAGE.md`** — end-to-end MIL protocol, environment variables, paper-style run commands.
