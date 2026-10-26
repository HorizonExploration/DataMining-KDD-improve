#!/usr/bin/env python
"""
远程 GPU faster-whisper 服务（极薄无状态 HTTP 层）。

职责：只做纯计算 —— 收到「音频 + 一堆 faster-whisper 参数」，在 GPU 上跑
`WhisperModel.transcribe` / `detect_language`，把结果原样返回。
语言检测 / prompt 选择 / 两遍策略 / CER 计算等业务逻辑全部留在 client。

两个端点：
  POST /transcribe       multipart: audio(.npy 二进制) + model + params(JSON) -> {segments, info}
  POST /detect_language  multipart: audio(.npy 二进制) + model + params(JSON) -> {language, ...}
  GET  /health           {status, device, compute_type, loaded}

模型按名缓存常驻显存（首次用时懒加载），tiny / medium 可同时驻留。

环境变量（均可不设，默认自动探测）：
  ASR_DEVICE         设备；不设时自动探测：有 CUDA 用 "cuda"，否则回退 "cpu"
                     （故本机无 GPU 也能直接 uvicorn 启动测试）
  ASR_COMPUTE_TYPE   量化；默认 "int8"（CPU/GPU 一致——最终部署在 CPU）
  ASR_TOKEN          若设置，则所有请求需带 ?token=<它> 才放行（简单鉴权）
  ASR_MODEL_DIR      HF 模型缓存根目录（作为 download_root）；默认 ~/.cache/huggingface/hub。
                     把 models--Systran--faster-whisper-* 放此即可加载（离线配合 HF_HUB_OFFLINE=1）
  ASR_MODEL_ROOT     烤进镜像的离线模型根目录；若设置且 <root>/<name> 存在，则直接按本地目录
                     路径加载（完全离线、不走 HF 名字解析）。见 asr/prepare_models.sh。

启动：ASR_MODEL_DIR=~/.cache/huggingface/hub ASR_COMPUTE_TYPE=int8 uvicorn asr.server:app --host 0.0.0.0 --port 8416
"""
import io
import json
import os

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from faster_whisper import WhisperModel


def _auto_device() -> str:
    """未显式指定 ASR_DEVICE 时：有可用 CUDA 设备用 cuda，否则回退 cpu。"""
    dev = os.environ.get("ASR_DEVICE")
    if dev:
        return dev
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = _auto_device()
# 量化：默认 int8（CPU/GPU 一致——最终在 CPU 部署，GPU 上也用 int8 保持结果一致）。
COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE") or "int8"
TOKEN = os.environ.get("ASR_TOKEN")  # None => 不鉴权
# HF 模型缓存根目录（作为 WhisperModel 的 download_root）；默认 HF 默认缓存。
# 把 models--Systran--faster-whisper-* 放到这里即可（离线时配合 HF_HUB_OFFLINE=1）。
MODEL_DIR = os.environ.get("ASR_MODEL_DIR") or os.path.expanduser("~/.cache/huggingface/hub")
# 烤进镜像的离线模型根目录（每个模型一个扁平子目录 <name>/{model.bin,config.json,...}）。
# 若设置且 <ASR_MODEL_ROOT>/<name> 存在，则直接按本地目录路径加载——完全离线、零网络、
# 不走 HF 名字解析。见 asr/prepare_models.sh。
MODEL_ROOT = os.environ.get("ASR_MODEL_ROOT")

try:
    import ctranslate2 as _ct2
    _CUDA_COUNT = _ct2.get_cuda_device_count()
except Exception:
    _CUDA_COUNT = 0
print(f"[asr] startup device={DEVICE} compute_type={COMPUTE_TYPE} "
      f"cuda_device_count={_CUDA_COUNT} model_dir={MODEL_DIR} model_root={MODEL_ROOT}", flush=True)

app = FastAPI(title="faster-whisper remote service")

_MODELS: dict[str, WhisperModel] = {}  # name -> WhisperModel


def get_model(name: str) -> WhisperModel:
    """按模型名缓存已加载模型，首次用时懒加载。

    解析顺序：
      1) 若设了 ASR_MODEL_ROOT 且 <root>/<name> 是个目录 -> 直接按本地目录路径加载
         （完全离线，烤进镜像的场景走这条）。
      2) 否则按模型名从 MODEL_DIR（HF 缓存）解析（远程 GPU / 本机带缓存的场景）。
    """
    if name not in _MODELS:
        target, source = name, f"download_root={MODEL_DIR}"
        if MODEL_ROOT:
            local = os.path.join(MODEL_ROOT, name)
            if os.path.isdir(local):
                target, source = local, "local dir (offline)"
        print(f"[asr] loading model={name} from {source} on device={DEVICE} "
              f"compute_type={COMPUTE_TYPE} ...", flush=True)
        _MODELS[name] = WhisperModel(
            target, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_DIR,
        )
        print(f"[asr] loaded model={name} (device={DEVICE})", flush=True)
    return _MODELS[name]


def _check_token(token: str | None):
    if TOKEN and token != TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing token")


def _load_audio(raw: bytes) -> np.ndarray:
    """client 用 np.save 序列化的 float32 一维音频（sample_rate=16000）。"""
    return np.load(io.BytesIO(raw), allow_pickle=False)


def _seg_to_dict(s) -> dict:
    words = getattr(s, "words", None)
    if words:
        words = [
            {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
            for w in words
        ]
    else:
        words = None
    return {
        "id": getattr(s, "id", None),
        "seek": getattr(s, "seek", None),
        "start": s.start,
        "end": s.end,
        "text": s.text,
        "tokens": list(s.tokens) if getattr(s, "tokens", None) is not None else None,
        "avg_logprob": getattr(s, "avg_logprob", None),
        "compression_ratio": getattr(s, "compression_ratio", None),
        "no_speech_prob": getattr(s, "no_speech_prob", None),
        "temperature": getattr(s, "temperature", None),
        "words": words,
    }


def _info_to_dict(info) -> dict:
    return {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
        "all_language_probs": getattr(info, "all_language_probs", None),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "model_dir": MODEL_DIR,
        "model_root": MODEL_ROOT,
        "loaded": sorted(_MODELS.keys()),
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Form(...),
    params: str = Form("{}"),
    token: str | None = Query(default=None),
):
    _check_token(token)
    arr = _load_audio(await audio.read())
    kwargs = json.loads(params)
    m = get_model(model)
    segments, info = m.transcribe(arr, **kwargs)
    # segments 是惰性生成器，这里在 GPU 端物化后再回传。
    seg_list = [_seg_to_dict(s) for s in segments]
    return {"segments": seg_list, "info": _info_to_dict(info)}


@app.post("/detect_language")
async def detect_language(
    audio: UploadFile = File(...),
    model: str = Form(...),
    params: str = Form("{}"),
    token: str | None = Query(default=None),
):
    _check_token(token)
    arr = _load_audio(await audio.read())
    kwargs = json.loads(params)
    m = get_model(model)
    language, language_probability, all_language_probs = m.detect_language(
        audio=arr, **kwargs
    )
    return {
        "language": language,
        "language_probability": language_probability,
        # list of (lang, prob)；JSON 中为 [[lang, prob], ...]，client dict(...) 可还原
        "all_language_probs": all_language_probs,
    }
