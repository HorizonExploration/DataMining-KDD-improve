# ============ Builder 阶段:只装依赖,不碰 src ============
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 先只复制依赖文件,利用 Docker 层缓存——改代码时无需重装依赖
COPY pyproject.toml uv.lock uv.toml README.md ./

# 占位 src,让 uv 能解析 project 元数据
RUN mkdir -p src/data_agent_baseline && touch src/data_agent_baseline/__init__.py

# 只装第三方依赖,不装项目本身(src 变化不会使这层缓存失效)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# static-ffmpeg 的 wheel 不含二进制,首次运行才从 GitHub 下载。
# 在构建时(联网)触发下载,把对应平台(由 --platform 决定)的 ffmpeg/ffprobe
# 烤进 /app/.venv,随后 COPY 到 runtime,保证运行时离线可用。
RUN /app/.venv/bin/python -c "from static_ffmpeg import run; print(run.get_or_fetch_platform_executables_else_raise())"

# 校验:确认下载的 ffmpeg/ffprobe 可执行(任一失败则构建中断)
RUN /app/.venv/bin/python - <<'PY'
import subprocess
from static_ffmpeg import run
ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()
for name, exe in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
    out = subprocess.run([exe, "-version"], capture_output=True, text=True, check=True)
    print(f"[OK] {name}: {out.stdout.splitlines()[0]}")
PY

# ============ Runtime 阶段(用同一个基础镜像,避免拉 docker.io)============
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app/src/data_agent_baseline" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_AGENT_SHORT_TB=1 \
    MAX_PARALLEL=8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# 从 builder 复制纯依赖的 .venv(src 怎么变都不影响这层)
COPY --from=builder /app/.venv ./.venv

# 烤入 ASR 模型权重(离线加载;打镜像前先在宿主机跑 src/data_agent_baseline/asr/prepare_models.sh)。
# 比 src 稳定,放在 src 之前,改代码不会让这层缓存失效。
ENV ASR_MODEL_ROOT=/app/asr_models
COPY asr_models /app/asr_models

# 直接从本地复制项目源码和脚本(最常变化的层放最后)
COPY src ./src
#COPY codetest ./codetest

# 复制 entrypoint 脚本
COPY docker-entrypoint.sh ./

# 数据输入和输出目录(运行时挂载)
RUN mkdir -p /app/data /app/artifacts && \
    chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--help"]
