#!/usr/bin/env python
"""
RemoteWhisperModel —— faster_whisper.WhisperModel 的 drop-in 远程替身。

把 transcribe / detect_language 的重计算转发到远程 GPU 上的 FastAPI 服务
（见同目录 server.py），方法签名与返回值跟原生 WhisperModel 完全一致，
下游业务逻辑（tiny→medium 两遍、prompt 选择、CER 计算等）一行都不用改。

用法：
    # 改前
    from faster_whisper import WhisperModel
    model = WhisperModel("medium", device="cpu", compute_type="int8")

    # 改后（只改 import + 构造这一行）
    from asr import RemoteWhisperModel as WhisperModel
    model = WhisperModel("medium", endpoint="http://<gpu-host>:8900")

下游不变：
    segs, info = model.transcribe(audio, language=det, beam_size=5, vad_filter=True)
    text = "".join(s.text for s in segs)
    lang, prob, all_probs = model.detect_language(audio=audio)

返回对象用 types.SimpleNamespace 承载：server 返回什么字段就有什么属性
（s.text / s.start / s.end / s.words[i].word / info.language ...），透传无需手工维护。

约定：audio 为 decode_audio 得到的 float32 numpy 一维数组（sample_rate=16000），
以 .npy 二进制 POST，省去服务端重复解码、也省传整个 mp4。也兼容直接传文件路径
（此时本地 lazy 调用 faster_whisper.audio.decode_audio 解码后再发）。
"""
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests

DEFAULT_ENDPOINT = os.environ.get("REMOTE_WHISPER_ENDPOINT", "http://localhost:8900")


def _segment_ns(d: dict) -> SimpleNamespace:
    """把 server 返回的 segment dict 直接透传成属性对象；嵌套 words 也转一层。

    server 发什么字段就有什么属性（s.text / s.start / s.end / s.words ...），
    client 无需手工维护字段列表，faster_whisper 加字段也不用改这里。
    """
    d = dict(d)
    words = d.get("words")
    d["words"] = [SimpleNamespace(**w) for w in words] if words else None
    return SimpleNamespace(**d)


def _audio_to_npy_bytes(audio) -> bytes:
    """把音频统一序列化成 .npy 二进制（float32 一维）。

    - numpy 数组：直接物化转 float32。
    - 文件路径（str/Path）：lazy 调用 faster_whisper.audio.decode_audio 解码。
    """
    if isinstance(audio, (str, Path)):
        from faster_whisper.audio import decode_audio  # 仅传路径时才需要
        audio = decode_audio(str(audio))
    arr = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


class RemoteWhisperModel:
    """faster_whisper.WhisperModel 的远程替身。

    构造参数：
        model_size_or_path : 模型名（如 "tiny" / "medium"），透传给远程服务的 get_model。
        endpoint           : 远程服务地址，默认取环境变量 REMOTE_WHISPER_ENDPOINT
                             或 http://localhost:8900。
        token              : 若服务端开了 ASR_TOKEN 鉴权，这里填同样的值。
        timeout            : 单次 HTTP 超时（秒）。

    其余 device / compute_type 等原生参数会被接收并忽略（量化由服务端决定）。
    """

    def __init__(
        self,
        model_size_or_path: str,
        endpoint: str | None = None,
        *,
        token: str | None = None,
        timeout: float = 600.0,
        **_ignored,
    ):
        self.model = model_size_or_path
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self.token = token or os.environ.get("ASR_TOKEN")
        self.timeout = timeout

    def _post(self, path: str, audio, params: dict) -> dict:
        files = {"audio": ("audio.npy", _audio_to_npy_bytes(audio), "application/octet-stream")}
        data = {"model": self.model, "params": json.dumps(params)}
        query = {"token": self.token} if self.token else None
        r = requests.post(
            self.endpoint + path, files=files, data=data,
            params=query, timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def transcribe(self, audio, **kwargs):
        """与原生一致：返回 (segments, info)。

        segments 为已物化的 list（同样可迭代），保证 s.text / s.start / s.end 可用。
        所有 faster-whisper 参数（language / beam_size / vad_filter /
        initial_prompt / condition_on_previous_text ...）原样透传给服务端。
        """
        resp = self._post("/transcribe", audio, kwargs)
        segments = [_segment_ns(d) for d in resp["segments"]]
        info = SimpleNamespace(**(resp.get("info") or {}))
        return segments, info

    def detect_language(self, audio=None, **kwargs):
        """与原生一致：返回 (language, language_probability, all_language_probs)。

        all_language_probs 为 [(lang, prob), ...]，可直接 dict(...) 还原。
        """
        if audio is None:
            raise ValueError("RemoteWhisperModel.detect_language 需要 audio 参数")
        resp = self._post("/detect_language", audio, kwargs)
        all_probs = [tuple(x) for x in (resp.get("all_language_probs") or [])]
        return resp["language"], resp["language_probability"], all_probs
