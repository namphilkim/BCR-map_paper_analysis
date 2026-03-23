#!/usr/bin/env bash
# ViT patch .h5 per image. Args: IMAGE_ROOT or $1. See USAGE.md for fold CSV.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${BCR_MAP_ROOT:-$REPO_ROOT}"

IMAGE_ROOT="${IMAGE_ROOT:-${1:-}}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Usage: $0 /path/to/images" >&2
  echo "   or:  IMAGE_ROOT=/path/to/images $0" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "ERROR: Not a directory: ${IMAGE_ROOT}" >&2
  exit 1
fi

EMBED_MODEL="${EMBED_MODEL:-vit-b16-224-in21k}"
PATCH_SIZE="${PATCH_SIZE:-224}"
STRIDE="${STRIDE:-224}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU="${GPU:-0}"

echo "Repository:    ${REPO_ROOT}"
echo "Image root:    ${IMAGE_ROOT}"
echo "ViT model:     ${EMBED_MODEL}"
echo "Patch/stride:  ${PATCH_SIZE} / ${STRIDE}"
echo "Batch size:    ${BATCH_SIZE}"
echo "GPU:           ${GPU}"
echo ""

(
  cd "${REPO_ROOT}" || exit 1
  CUDA_VISIBLE_DEVICES="${GPU}" python -m bcr_map.cli extract \
    --data_dir "${IMAGE_ROOT}" \
    --model_name "${EMBED_MODEL}" \
    --patch_size "${PATCH_SIZE}" \
    --stride "${STRIDE}" \
    --batch_size "${BATCH_SIZE}"
)

echo ""
echo "Done. .h5 files are next to each source image under ${IMAGE_ROOT}"
