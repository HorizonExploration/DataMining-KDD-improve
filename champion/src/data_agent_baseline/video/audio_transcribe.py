"""Video → 带时间戳的 Whisper 转写（供分帧/旁白对齐使用）。

ASR 后端有两种，**默认一律走 ``local``**（不区分容器/本机）：

  * ``local``（默认）：直接用 ``faster_whisper.WhisperModel`` 跑 CPU 推理。
    这也是提交给比赛评估系统的路径（评估容器无外网，必须本地推理）。**以仓库内烤入的
    模型路径为权威基准**：按 ``asr_models/<name>`` 离线目录加载（该目录相对仓库根计算，
    容器内即 ``/app/asr_models``，本机即 ``<repo>/asr_models``，无需任何环境变量）。
    目录缺失直接报错，不静默联网下载。设备固定 ``cpu``，量化 ``int8``。

  * ``remote``（仅在特别指定时）：把推理转发到远程 GPU faster-whisper 服务
    （``asr.RemoteWhisperModel``，见 ``asr/server.py``）。**不会自动启用**，需显式传
    ``backend='remote'`` 或设环境变量 ``export ASR_BACKEND=remote`` 才走它。

两种后端都直接从 mp4 解码音频并请求 word 级时间戳 —— 对齐需要 word 粒度，
一句话经常横跨多张幻灯片。``RemoteWhisperModel`` 是原生 ``WhisperModel`` 的
drop-in 替身，``.transcribe(audio, **kwargs) -> (segments, info)`` 签名一致，
故下游 segment 处理逻辑两条后端共用、零分叉。

模型选型口径（与 ASR eval 结论一致）：
  * 用 ``medium``（``tiny`` 仅做语言检测）。
  * 默认不强制 ``language``：自动检测，避免英文音频被误判成中文；需要锚定再显式传。
  * ``large-v3`` 末尾易幻觉；``medium`` 更稳。
  * 输出做繁→简归一（opencc t2s）以对齐 eval 约定；opencc 不可用时回退原样。

返回 dict 的 ``segments[].{start,end,text}`` 可直接喂给
``video.message.build_multimodal_message``（whisper 风格转写）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from faster_whisper.audio import decode_audio

DEFAULT_ASR_ENDPOINT = "http://10.166.158.46:8416"

# 本地 CPU 后端的部署口径（与 asr/server.py 一致）：CPU + int8。
_CPU_DEVICE = "cpu"
_COMPUTE_TYPE = "int8"
# 离线模型根目录：相对仓库根的 ``asr_models/``（本文件 = src/<pkg>/video/audio_transcribe.py，
# 仓库根 = parents[3]）。容器内 /app/src/.../audio_transcribe.py → /app/asr_models（Dockerfile
# COPY 到此），本机则指向 prepare_models.sh 的输出目录——容器与本机同一相对路径，无需任何
# 环境变量。local 模式以此为权威基准：按 <root>/<name> 离线目录加载，不走 HF 缓存/外网。
_DEFAULT_MODEL_ROOT = str(Path(__file__).resolve().parents[3] / "asr_models")

__all__ = ["transcribe_video", "transcribe_to_file", "detect_audio_language",
           "DEFAULT_ASR_ENDPOINT"]


def _to_simplified(text: str) -> str:
    """繁→简归一；opencc 不可用时回退原样。"""
    try:
        from opencc import OpenCC

        return OpenCC("t2s").convert(text)
    except Exception:
        return text


def _local_model_dir(name: str) -> str | None:
    """``_DEFAULT_MODEL_ROOT/<name>`` 存在则返回其路径，否则 None（不报错，供探测用）。"""
    local = os.path.join(_DEFAULT_MODEL_ROOT, name)
    return local if os.path.isdir(local) else None


def _resolve_local_model(name: str) -> str:
    """把模型名解析成本地离线目录 target（local 模式以仓库内模型路径为权威基准）。

    要求 ``_DEFAULT_MODEL_ROOT/<name>`` 目录存在，命中即按本地目录离线加载（完全离线、
    零网络）；缺失则直接报错——容器内无外网，不允许静默回退去联网下载。
    """
    local = _local_model_dir(name)
    if local is None:
        raise FileNotFoundError(
            f"本地 ASR 模型目录缺失: {os.path.join(_DEFAULT_MODEL_ROOT, name)}\n"
            f"（model={name!r}）。打镜像前请在宿主机跑 asr/prepare_models.sh 烤入权重；"
            f"本机调试请确保 {_DEFAULT_MODEL_ROOT} 下已放好对应模型目录。"
        )
    return local


def _build_model(model: str, *, backend: str, endpoint: str,
                 token: str | None, timeout: float):
    """按 backend 构造 whisper 模型；两条后端都暴露同一 ``.transcribe`` 接口。"""
    if backend == "local":
        from faster_whisper import WhisperModel

        # 以 asr_models/<name> 离线目录为准，CPU + int8（与容器部署口径一致）。
        local_dir = _resolve_local_model(model)
        print(
            f"[ASR] backend=local（本地 CPU）model={model!r} "
            f"endpoint={local_dir} device={_CPU_DEVICE} compute={_COMPUTE_TYPE}",
            flush=True,
        )
        return WhisperModel(
            local_dir,
            device=_CPU_DEVICE,
            compute_type=_COMPUTE_TYPE,
        )
    if backend == "remote":
        from asr.remote_whisper import RemoteWhisperModel

        m = RemoteWhisperModel(model, endpoint=endpoint, token=token, timeout=timeout)
        print(
            f"[ASR] backend=remote（远程 GPU）model={model!r} endpoint={m.endpoint}",
            flush=True,
        )
        return m
    raise ValueError(f"未知 ASR backend: {backend!r}（应为 'local' 或 'remote'）")


def _resolve_backend(backend: str | None) -> str:
    """选择 ASR 后端。

    优先级：显式 ``backend`` 参数 > 环境变量 ``ASR_BACKEND`` > 默认。
    默认一律走 ``local``（CPU 离线推理）；远程 GPU 只在**特别指定**时才用
    （显式传 ``backend='remote'`` 或 ``export ASR_BACKEND=remote``）。
    """
    if backend is not None:
        return backend
    env_backend = os.environ.get("ASR_BACKEND", "").strip().lower()
    if env_backend in ("local", "remote"):
        return env_backend
    return "local"


def detect_audio_language(
    audio,
    *,
    backend: str | None = None,         # None -> 默认 local；ASR_BACKEND=remote 才走远程
    endpoint: str = DEFAULT_ASR_ENDPOINT,  # 仅 backend='remote' 时使用
    token: str | None = None,           # 仅 remote：服务端开了 ASR_TOKEN 鉴权时填
    timeout: float = 600.0,             # 仅 remote：单次 HTTP 超时（秒）
    model: str = "tiny",
    sampling_rate: int = 16000,
) -> dict:
    """现场检测一段音频的语言（限定 zh/en，取概率大者）。

    用 ``tiny`` 模型对【音频本身】检测中/英——不读题目语言、不看 gold：评测已证实有
    task 音频中文但题目英文，下游 ASR ``initial_prompt`` 必须按【音频语言】写。后端选择
    与 :func:`transcribe_video` 同口径（默认 local CPU；ASR_BACKEND=remote 才走远程 GPU）。

    Args:
        audio: ``decode_audio`` 得到的 float32 一维数组，或可被解码的视频/音频路径。
        backend: ``None`` 时按 ``general_tools.is_docker()`` 自动选择（显式传值可覆盖）。
        endpoint / token / timeout: 仅 ``backend='remote'`` 生效（同 transcribe_video）。
        model: 检测用模型名（``tiny`` 即可，省算力）。local 后端若该模型未烤入则自动
            回退到 ``tiny``——语言检测对模型大小不敏感，``tiny`` 足够且必然在镜像里。
        sampling_rate: 传入路径时 ``decode_audio`` 的目标采样率（默认 16000）。

    Returns:
        ``{"lang": "zh"|"en", "p_zh": float, "p_en": float, "infer_s": float}``。
    """
    if isinstance(audio, (str, Path)):
        audio = decode_audio(str(audio), sampling_rate=sampling_rate)

    backend = _resolve_backend(backend)
    # local 后端：所选模型未烤入时回退到 tiny（检测不挑模型，tiny 必然在镜像内）。
    if backend == "local" and model != "tiny" and _local_model_dir(model) is None:
        model = "tiny"
    m = _build_model(model, backend=backend,
                     endpoint=endpoint, token=token, timeout=timeout)
    t0 = time.time()
    _, _, all_probs = m.detect_language(audio=audio)
    dt = time.time() - t0
    d = dict(all_probs)
    p_zh, p_en = d.get("zh", 0.0), d.get("en", 0.0)
    lang = "zh" if p_zh >= p_en else "en"
    return {"lang": lang, "p_zh": round(p_zh, 4), "p_en": round(p_en, 4),
            "infer_s": round(dt, 2)}


def transcribe_video(
    video_path: str | Path,
    *,
    backend: str | None = None,         # None -> 默认 local；ASR_BACKEND=remote 才走远程
    endpoint: str = DEFAULT_ASR_ENDPOINT,  # 仅 backend='remote' 时使用
    token: str | None = None,           # 仅 remote：服务端开了 ASR_TOKEN 鉴权时填
    timeout: float = 600.0,             # 仅 remote：单次 HTTP 超时（秒）
    model: str = "medium",
    language: str | None = None,        # None = 自动检测（不要默认成 zh）
    initial_prompt: str | None = None,  # ASR 软引导（如 asr.gen_init_prompt 产物）
    beam_size: int = 5,
    vad_filter: bool = True,
    sampling_rate: int = 16000,         # decode_audio 目标采样率；也用于由采样点数反算时长
) -> dict:
    """转写视频，返回带 word 时间戳的 Whisper 结果 dict。

    Args:
        video_path: 源视频；音频直接解码（不显式 demux）。
        backend: ``'local'``（CPU 离线推理，提交路径，**默认**）或 ``'remote'``（转发远程
            GPU）。``None`` 时默认走 ``local``；仅当显式传 ``'remote'`` 或设环境变量
            ``ASR_BACKEND=remote`` 才走远程 GPU。
        endpoint: 远程 faster-whisper 服务地址（仅 ``backend='remote'`` 生效）。
        token / timeout: 远程服务鉴权 token 与 HTTP 超时（仅 ``backend='remote'`` 生效）。
        model: whisper 模型名（推荐 ``medium``）。
        language: 锚定语言码，``None`` 则自动检测。
        initial_prompt: 可选软引导 prompt，纠正同音字（如 ``asr.gen_init_prompt`` 的产物）。
        sampling_rate: 音频解码目标采样率（默认 16000，whisper 固定输入）。同一值既传给
            ``decode_audio`` 也用于由采样点数反算 ``duration_s``，二者同源不会错位。

    Returns:
        dict, 形如::

            {
              "video": "briefing.mp4", "model": "medium", "language": "zh",
              "backend": "local", "duration_s": 49.98, "text": "...",
              "segments": [
                {"start": 0.16, "end": 7.8, "text": "...",
                 "words": [{"start": .., "end": .., "word": ".."}, ...]},
                ...
              ]
            }
    """
    backend = _resolve_backend(backend)

    video_path = str(video_path)
    audio = decode_audio(video_path, sampling_rate=sampling_rate)
    duration_s = round(len(audio) / sampling_rate, 2)

    m = _build_model(model, backend=backend, endpoint=endpoint,
                     token=token, timeout=timeout)
    segs, info = m.transcribe(
        audio,
        language=language,             # None -> faster-whisper 自动检测
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=True,          # 关键：word 级时间戳用于对齐
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
    )

    segments = []
    for s in segs:
        words = None
        if getattr(s, "words", None):
            words = [
                {"start": round(w.start, 3), "end": round(w.end, 3),
                 "word": _to_simplified(w.word)}
                for w in s.words
            ]
        segments.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": _to_simplified(s.text),
            "words": words,
        })

    return {
        "video": Path(video_path).name,
        "model": model,
        "backend": backend,
        "language": getattr(info, "language", None),
        "duration_s": duration_s,
        "text": "".join(seg["text"] for seg in segments),
        "segments": segments,
    }


def transcribe_to_file(
    video_path: str | Path,
    out_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """转写并写出结果 JSON；返回输出路径。

    默认输出 ``<video_dir>/briefing.asr.json``（与帧放一起，消费方一处读两份）。
    后端及其参数（``backend`` / ``endpoint`` / ``model`` ...）通过 ``**kwargs`` 透传给
    ``transcribe_video``。
    """
    video_path = Path(video_path)
    out = Path(out_path) if out_path else video_path.with_name("briefing.asr.json")
    result = transcribe_video(video_path, **kwargs)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
