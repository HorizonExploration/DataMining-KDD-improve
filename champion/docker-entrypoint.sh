#!/bin/sh
mkdir -p /logs && chmod 777 /logs
{
  # 启动自检:PyAV / static-ffmpeg 二进制 / 本地 CPU whisper 模型,结果写入日志(非致命)。
  python src/data_agent_baseline/preflight.py || echo "[preflight] 自检发现问题(见上),继续运行任务"
  exec python src/data_agent_baseline/zz_agent_v2.py "$@"
} 2>&1 | tee /logs/runtime.log
