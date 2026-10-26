#!/bin/bash
# 创建一份"代码快照",用于跑批量 run 时冻结当前代码:
#   在快照目录里跑 run, 主目录可继续改代码, 互不影响.
#
# 用法:
#   ./make_snapshot.sh                 # 生成/覆盖 ../kdd_run_snapshot
#   ./make_snapshot.sh /path/to/dst    # 指定快照目录
#   NO_SYNC=1 ./make_snapshot.sh       # 跳过 uv sync (仅同步源码)
#
# 设计要点:
#   - rsync 排除大目录(.venv/dist/run_logs/.git/experiments/asr_models)与数据产物,
#     快照只含源码, 体积小.
#   - 快照里 `uv sync` 重建独立 .venv: 第三方依赖走 uv 全局缓存(硬链接,不重复占盘),
#     仅项目自身可编辑包 data-agent-baseline 各指各的源码 => 与主目录真正隔离.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-$(dirname "$SRC")/kdd_run_snapshot}"

# 不需要进快照的目录/产物. 按需增删.
EXCLUDES=(
    ".venv"          # 重建独立 venv
    ".git"           # 不需要版本历史
    "dist"           # 构建产物, 跑 run 用不到
    "run_logs"       # 历史运行日志
    "experiments"    # 历史实验结果(大)
    "asr_models"     # 本地 ASR 模型(大); 若跑 run 需本地 ASR 请删掉此行
    "__pycache__"
    "*.pyc"
)

echo "源目录:   $SRC"
echo "快照目录: $DST"

if [ -e "$DST" ]; then
    echo "目标已存在, 将增量同步并删除快照中多余文件 (--delete)."
fi

RSYNC_ARGS=(-a --delete)
for e in "${EXCLUDES[@]}"; do
    RSYNC_ARGS+=(--exclude="$e")
done

# 注意 SRC 结尾的 /, 表示同步目录内容而非目录本身.
rsync "${RSYNC_ARGS[@]}" "$SRC/" "$DST/"

echo "--- 源码同步完成, 快照大小 ---"
du -sh "$DST"

if [ "${NO_SYNC:-0}" = "1" ]; then
    echo "NO_SYNC=1, 跳过 uv sync. 注意: 快照内无 .venv, 直接 uv run 会自行解析."
    exit 0
fi

echo "--- 在快照内重建独立 venv (uv sync) ---"
( cd "$DST" && unset VIRTUAL_ENV && uv sync )

echo "--- 拷贝 static-ffmpeg 二进制 (uv sync 不含, 否则首次调用要联网下载, 很慢) ---"
# static_ffmpeg 的 ffmpeg/ffprobe 不随 pip 包安装, 首次调用 get_or_fetch 才联网下载到
# 各自 .venv 的 site-packages/static_ffmpeg/bin/ 下. 主目录通常已下载过, 直接拷贝即可,
# 省去快照内重新下载(很慢/可能无网). 失败不阻断, 跑 run 时会回退到联网下载.
_SFF_SUBPATH="lib/python3.12/site-packages/static_ffmpeg/bin"
_SFF_SRC="$SRC/.venv/$_SFF_SUBPATH"
_SFF_DST="$DST/.venv/$_SFF_SUBPATH"
if [ -d "$_SFF_SRC" ] && [ -n "$(ls -A "$_SFF_SRC" 2>/dev/null)" ]; then
    mkdir -p "$_SFF_DST"
    rsync -a "$_SFF_SRC/" "$_SFF_DST/"
    echo "✅ static-ffmpeg 二进制已从主目录拷贝到快照."
else
    echo "⚠️  主目录无 static-ffmpeg 二进制 ($_SFF_SRC), 跳过; 跑 run 时将联网下载."
fi

echo ""
echo "✅ 快照就绪: $DST"
echo "   在快照内跑批量 run (主目录可继续改代码, 互不影响):"
echo "     cd \"$DST\" && ./run_agent_n_times.sh -n 10"
