#!/usr/bin/env bash
#
# Train one MIL model (ViT embeddings + HiPT aggregation), k-fold cross-validation.
# Edit the variables in the "User settings" block below, or export them before running.
#
# Data layout (MIL_DATAPATH):
#   CSV column image_path is relative to this root. Each BCR-map image should have a matching
#   .h5 file of ViT patch embeddings beside it (see extract_embeddings_user_images.sh).
#
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${BCR_MAP_ROOT:-$REPO_ROOT}"

# =============================================================================
# User settings (override with environment variables)
# =============================================================================

# Root folder for this run: class subdirectories or paths as expected by your folds CSV.
: "${MIL_DATAPATH:?Set MIL_DATAPATH to the dataset root (images + .h5 embeddings).}"

# CSV with columns including: fold, class, image_path (path relative to MIL_DATAPATH).
: "${MIL_FOLDS_CSV:?Set MIL_FOLDS_CSV to the absolute or relative path of your fold definition CSV.}"

# Subdirectory name under checkpoints/ and logs (use letters, numbers, slashes; no spaces).
: "${MIL_EXPERIMENT_ID:?Set MIL_EXPERIMENT_ID, e.g. my_cohort/run_01/mil_hipt_vit_b16_6class}"

MIL_NUM_CLASSES="${MIL_NUM_CLASSES:-6}"
MIL_K_FOLDS="${MIL_K_FOLDS:-3}"
MIL_MAX_EPOCHS="${MIL_MAX_EPOCHS:-100}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-2}"
MIL_MAX_PATCHES="${MIL_MAX_PATCHES:-10000}"
MIL_GPU="${MIL_GPU:-0}"
MIL_AGGREGATION="${MIL_AGGREGATION:-hipt}"
MIL_BACKBONE="${MIL_BACKBONE:-vit-b16-224-in21k}"

CONFIG_YAML="${MIL_CONFIG:-${REPO_ROOT}/configs/mil_training.yaml}"

# Reproducible checkpoint paths; CSV-only logs (no W&B account).
export BCR_MAP_STABLE_VERSION="${BCR_MAP_STABLE_VERSION:-1}"
export BCR_MAP_CSV_LOGGER_ONLY="${BCR_MAP_CSV_LOGGER_ONLY:-1}"

# =============================================================================
# Run
# =============================================================================

echo "Repository:  ${REPO_ROOT}"
echo "Datapath:    ${MIL_DATAPATH}"
echo "Folds CSV:   ${MIL_FOLDS_CSV}"
echo "Experiment:  ${MIL_EXPERIMENT_ID}"
echo "GPU:         ${MIL_GPU}"
echo "K-folds:     ${MIL_K_FOLDS} | classes: ${MIL_NUM_CLASSES} | epochs: ${MIL_MAX_EPOCHS}"
echo ""

(
  cd "${REPO_ROOT}" || exit 1
  CUDA_VISIBLE_DEVICES="${MIL_GPU}" OMP_NUM_THREADS=1 python -m bcr_map.cli train \
    --config "${CONFIG_YAML}" \
    --model.aggregation_method "${MIL_AGGREGATION}" \
    --model.model_name "${MIL_BACKBONE}" \
    --data.num_classes "${MIL_NUM_CLASSES}" \
    --data.datapath "${MIL_DATAPATH}" \
    --data.data_folds "${MIL_FOLDS_CSV}" \
    --data.max_patches "${MIL_MAX_PATCHES}" \
    --data.batch_size "${MIL_BATCH_SIZE}" \
    --trainer.max_epochs "${MIL_MAX_EPOCHS}" \
    --version "${MIL_EXPERIMENT_ID}" \
    --k_folds "${MIL_K_FOLDS}"
)

echo ""
echo "Done. Checkpoints: ${REPO_ROOT}/checkpoints/${MIL_EXPERIMENT_ID}/fold_*"
