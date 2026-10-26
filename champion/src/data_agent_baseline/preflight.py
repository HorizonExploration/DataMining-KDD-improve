#!/usr/bin/env python
"""容器启动自检 —— 确认运行时依赖在镜像内真实可用,结果打到 stdout 供日志留痕。

检查三件事(与运行时实际用法一致):
  1. PyAV          : 视频抽帧(video/frames.py)走 PyAV 自带的 libav 库,确认可导入并能报出版本。
  2. static-ffmpeg : ffmpeg/ffprobe CLI 二进制已烤进镜像(非首次运行联网下载),且能执行。
  3. whisper       : ASR 在容器内走【本地 CPU】,按相对仓库根的 asr_models/<name> 离线加载
                     (与 video/audio_transcribe.py 同口径,无需任何环境变量),逐个真实 load 验证。

设计为「非致命」:任何一项失败只打印 [FAIL] 并最终返回非零退出码,由调用方决定是否中断;
entrypoint 中我们打印后仍继续跑,避免自检 bug 误杀真实任务,但 FAIL 在日志里一眼可见。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# whisper 在容器内的部署口径(与 asr/server.py 一致):本地 CPU + int8。
_CPU_DEVICE = "cpu"
_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE") or "int8"
# 需要离线加载验证的模型名(与 asr/prepare_models.sh 烤进镜像的子目录对应)。
_WHISPER_MODELS = ("tiny", "medium")
# 离线模型根目录:相对仓库根的 asr_models/(本文件 = src/<pkg>/preflight.py,仓库根 =
# parents[2])。容器内 /app/src/.../preflight.py -> /app/asr_models(Dockerfile COPY 到此),
# 本机则指向 prepare_models.sh 的输出目录。与 video/audio_transcribe.py 的 _DEFAULT_MODEL_ROOT
# 口径一致,容器与本机同一相对路径,无需任何环境变量。
_DEFAULT_MODEL_ROOT = str(Path(__file__).resolve().parents[2] / "asr_models")


def _ok(msg: str) -> None:
    print(f"[preflight][OK]   {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[preflight][FAIL] {msg}", flush=True)


def check_pyav() -> bool:
    """PyAV 可导入,并报出绑定的 ffmpeg/libav 版本(视频解码自包含的证据)。"""
    try:
        import av  # noqa: PLC0415

        libs = getattr(av, "library_versions", {}) or {}
        lav = libs.get("libavcodec") or libs.get("avcodec")
        _ok(f"PyAV {av.__version__} (libavcodec={lav})")
        return True
    except Exception as e:  # noqa: BLE001
        _fail(f"PyAV import/probe 失败: {e!r}")
        return False


def check_static_ffmpeg() -> bool:
    """static-ffmpeg 的 ffmpeg/ffprobe 已就位且可执行(离线,不触发下载)。"""
    try:
        from static_ffmpeg import run  # noqa: PLC0415

        ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()
    except Exception as e:  # noqa: BLE001
        _fail(f"static-ffmpeg 解析二进制失败: {e!r}")
        return False

    all_ok = True
    for name, exe in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        try:
            out = subprocess.run(
                [exe, "-version"], capture_output=True, text=True, check=True
            )
            _ok(f"{name}: {out.stdout.splitlines()[0]}  ({exe})")
        except Exception as e:  # noqa: BLE001
            _fail(f"{name} 执行失败 ({exe}): {e!r}")
            all_ok = False
    return all_ok


def check_whisper() -> bool:
    """按相对仓库根的 asr_models/<name> 在 CPU 上离线真实 load,验证烤进镜像的权重可用。"""
    # 与 video/audio_transcribe.py 同口径:固定相对仓库根的目录,不读任何环境变量。
    model_root = _DEFAULT_MODEL_ROOT
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        _fail(f"faster_whisper 导入失败: {e!r}")
        return False

    all_ok = True
    for name in _WHISPER_MODELS:
        local = os.path.join(model_root, name)
        if not os.path.isdir(local):
            _fail(f"whisper 模型目录缺失: {local}")
            all_ok = False
            continue
        try:
            # 与 server.get_model 的 offline 分支一致:按本地目录路径加载,纯 CPU。
            WhisperModel(local, device=_CPU_DEVICE, compute_type=_COMPUTE_TYPE)
            _ok(f"whisper '{name}' 本地 CPU 加载成功 (compute_type={_COMPUTE_TYPE})  ({local})")
        except Exception as e:  # noqa: BLE001
            _fail(f"whisper '{name}' 加载失败 ({local}): {e!r}")
            all_ok = False
    return all_ok


def main() -> int:
    """跑全部自检并打印汇总。

    绝不抛异常 / 绝不阻断:返回码仅供日志参考(0=全过,1=有失败),
    entrypoint 无论返回什么都继续跑评测——缺模块时部分任务仍可运行。
    """
    print("=" * 60, flush=True)
    print("[preflight] 运行时依赖自检开始(非致命,失败仅告警)", flush=True)
    print("=" * 60, flush=True)

    checks = {
        "PyAV": check_pyav,
        "static-ffmpeg": check_static_ffmpeg,
        "whisper(local CPU)": check_whisper,
    }
    results: dict[str, bool] = {}
    for name, fn in checks.items():
        try:
            results[name] = bool(fn())
        except Exception as e:  # noqa: BLE001  自检自身异常也不得外泄
            _fail(f"{name} 自检过程异常: {e!r}")
            results[name] = False

    print("-" * 60, flush=True)
    for name, ok in results.items():
        print(f"[preflight] {name:<22} {'PASS' if ok else 'FAIL'}", flush=True)
    all_pass = all(results.values())
    print(f"[preflight] 总体: {'ALL PASS' if all_pass else 'HAS FAILURE(继续运行)'}", flush=True)
    print("=" * 60, flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001  兜底:任何意外都不得让自检中断容器
        print(f"[preflight][FAIL] 自检主流程崩溃,已忽略以保证评测继续: {e!r}", flush=True)
        sys.exit(1)

