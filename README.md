# BCR-map_paper_analysis

https://github.com/namphilkim/BCR-map_paper_analysis

Code for *BCR-map: Integrative Visualization of B-cell Receptor Repertoires*: figure notebooks (`BCR-map_figure_*.ipynb`), `BCR-map_intermediate_files/`, `MyBasics.py`, and MIL training / ViT patch embedding (`bcr_map/`, `utils/`, `configs/`, `extract_vit_patch_embeddings.py`, `scripts/`).

Package version **1.0.0** (`pyproject.toml`, `bcr_map/__init__.py`, `CITATION.cff`).

## Contents

| Path | Role |
|------|------|
| `BCR-map_figure_*.ipynb` | Figure notebooks |
| `BCR-map_intermediate_files/` | Tables for analyses |
| `MyBasics.py` | Helpers for notebooks |
| `bcr_map/` | `bcr-map` CLI, training |
| `utils/` | Data module, model, loss |
| `configs/mil_training.yaml` | Default MIL config |
| `extract_vit_patch_embeddings.py` | ViT → `.h5` per image |
| `main.py` | Same as `bcr-map train` |
| `scripts/` | Shell wrappers |
| `USAGE.md` | Env vars, examples |
| `checkpoints/` | Put Zenodo weights here (not in git) |

Model weights (`.ckpt`) are **not** in this repo; use Zenodo (DOI in README when published). See `checkpoints/README.md`. Some `logs/lightning_logs/` CSVs may be tracked per `.gitignore`.

## Requirements

- Python ≥ 3.9  
- GPU recommended for `extract` / `train`

## Install

```bash
git clone https://github.com/namphilkim/BCR-map_paper_analysis.git
cd BCR-map_paper_analysis
python -m venv .venv && source .venv/bin/activate   # optional
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Optional: `environment.yml`.

## MIL commands

```bash
bcr-map extract --data_dir /path/to/images
bcr-map train --config configs/mil_training.yaml
```

`bcr-map train -- --help` — LightningCLI. `bcr-map extract -- --help` — extraction args.

More: `USAGE.md`, `scripts/train_mil_single_run.sh`.

## Data

No study data or checkpoints in git. Training data and ethics are the user’s responsibility. Path names like `In-house` / `Mal-ID` match local `checkpoints/` / `logs/` after you add Zenodo weights.

## Citation

`CITATION.cff`. BibTeX example:

```bibtex
@software{bcr_map_paper_analysis,
  title   = {BCR-map_paper_analysis},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/namphilkim/BCR-map_paper_analysis},
}
```

Article citation: add when published. Zenodo dataset DOI for weights: add when published.

## License

MIT — see `LICENSE`.
