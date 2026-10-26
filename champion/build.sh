#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-team1547}"
TAG="${TAG:-final}"

AMD64_IMAGE="${IMAGE_NAME}:${TAG}"
ARM64_IMAGE="${IMAGE_NAME}-arm64:${TAG}"

OUTPUT_DIR="${OUTPUT_DIR:-./dist}"
mkdir -p "${OUTPUT_DIR}"

echo "==> Building arm64 image (local use): ${ARM64_IMAGE}"
docker build \
  --platform linux/arm64 \
  -t "${ARM64_IMAGE}" \
  --load \
  .

echo "==> Building amd64 image (for distribution): ${AMD64_IMAGE}"
docker build \
  --platform linux/amd64 \
  -t "${AMD64_IMAGE}" \
  --load \
  .

TAR_FILE="${OUTPUT_DIR}/${IMAGE_NAME}_${TAG}.tar.gz"
echo "==> Saving amd64 image to ${TAR_FILE}"
docker save "${AMD64_IMAGE}" | gzip > "${TAR_FILE}"

echo "==> Done."
echo "    arm64 image (local): ${ARM64_IMAGE}"
echo "    amd64 image:         ${AMD64_IMAGE}"
echo "    amd64 archive:       ${TAR_FILE}"
