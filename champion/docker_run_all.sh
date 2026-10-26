#!/bin/bash

# 用法: bash docker_run_all.sh [image-tag] [run-id]
# image-tag: Docker 镜像 tag，例如 v4、v5（可选，优先级：命令行参数 > 脚本内默认值）
# run-id: 运行目录名，默认为时间戳

# ↓ 在此修改默认 tag
DEFAULT_IMAGE_TAG="final"

IMAGE_TAG=${1:-${DEFAULT_IMAGE_TAG}}
RUN_ID=${2:-$(date +%Y%m%d-%H%M%S)}
BASE_DIR="/Users/zhe/Documents/work/20260400-data-agent/workdir/docker_runs/${RUN_ID}"

mkdir -p "${BASE_DIR}/output"
mkdir -p "${BASE_DIR}/temp"
mkdir -p "${BASE_DIR}/logs"

echo "镜像 tag:      team1547-arm64:${IMAGE_TAG}"
echo "本次运行目录: ${BASE_DIR}"

docker run --rm \
    -v /Users/zhe/Documents/work/20260400-data-agent/phase2/input:/input:ro \
    -v "${BASE_DIR}/output:/output:rw" \
    -v "${BASE_DIR}/temp:/tmp/temp:rw" \
    -v "${BASE_DIR}/logs:/logs:rw" \
    -e MODEL_API_URL=http://10.166.163.101:8417/v1 \
    -e MODEL_API_KEY=empty \
    -e MODEL_NAME=qwen3.5-35b-a3b \
    team1547-arm64:${IMAGE_TAG}


echo "运行完成，结果保存在: ${BASE_DIR}"


