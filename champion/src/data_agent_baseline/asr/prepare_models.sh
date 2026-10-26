#!/usr/bin/env bash
# 把本机 HF 缓存里的 faster-whisper 模型 flatten 拷到 repo 内 asr_models/<name>/，
# 作为打镜像的构建上下文（镜像无外网，需把权重烤进去离线加载）。
#
# 每个模型只拷 4 个必需文件：model.bin / config.json / tokenizer.json / vocabulary.txt。
# 用 cp -L 解符号链接——HF 缓存里 snapshots/*/model.bin 是指向 blobs/ 的软链，直接拷会断链。
#
# 用法（打镜像前在宿主机跑一次）：
#   bash src/data_agent_baseline/asr/prepare_models.sh            # 拷 medium + tiny
#   bash src/data_agent_baseline/asr/prepare_models.sh --force    # 已存在也重建
#   MODELS="medium tiny base" bash src/.../prepare_models.sh      # 自定义模型集
set -euo pipefail

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# 仓库根 = 本脚本所在目录的上溯三级（asr/ -> data_agent_baseline/ -> src/ -> repo）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
DST_ROOT="${REPO_ROOT}/asr_models"
MODELS="${MODELS:-medium tiny}"
FILES="model.bin config.json tokenizer.json vocabulary.txt"

echo "==> HF 缓存: ${HUB}"
echo "==> 输出到:  ${DST_ROOT}"
echo "==> 模型:    ${MODELS}"

for name in ${MODELS}; do
  dst="${DST_ROOT}/${name}"
  if [ -f "${dst}/model.bin" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [跳过] ${name} 已存在 (--force 可重建)"
    continue
  fi

  # 找该模型的 snapshot 目录（HF 标准布局）
  snap=$(ls -d "${HUB}/models--Systran--faster-whisper-${name}/snapshots/"*/ 2>/dev/null | head -1 || true)
  if [ -z "${snap}" ]; then
    echo "  [错误] 未在 HF 缓存找到 ${name}：${HUB}/models--Systran--faster-whisper-${name}" >&2
    echo "         先在有网机器上让 faster-whisper 下载一次该模型，或手动放入缓存。" >&2
    exit 1
  fi

  rm -rf "${dst}"
  mkdir -p "${dst}"
  for f in ${FILES}; do
    if [ ! -e "${snap}/${f}" ]; then
      echo "  [错误] ${name} 缺少文件 ${f}（在 ${snap}）" >&2
      exit 1
    fi
    cp -L "${snap}/${f}" "${dst}/${f}"   # -L 解软链，拷真实字节
  done
  sz=$(du -sh "${dst}" | cut -f1)
  echo "  [完成] ${name} -> ${dst} (${sz})"
done

echo "==> 全部完成。总体积："
du -sh "${DST_ROOT}"
echo "    现在可以 build.sh 打镜像（Dockerfile 会 COPY asr_models）。"
