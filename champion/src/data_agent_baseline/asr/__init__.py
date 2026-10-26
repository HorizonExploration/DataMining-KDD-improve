"""远程 GPU faster-whisper ASR 子包。

- ``server``        : 部署在 GPU host 的 FastAPI 纯计算服务。
- ``remote_whisper``: 本机用的 drop-in 替身 ``RemoteWhisperModel``。
- ``gen_init_prompt``: 两阶段（tiny 检测语言 + qwen3.5 写）生成 ASR ``initial_prompt``。

便捷导入：
    from asr import RemoteWhisperModel
    from asr import gen_init_prompt        # 单条端到端（video 调用）
    from asr import detect_audio_language, write_init_prompt
"""
# from .gen_init_prompt import (
#     detect_audio_language,
#     gen_init_prompt,
#     write_init_prompt,
# )
# from .remote_whisper import RemoteWhisperModel
#
# __all__ = [
#     "RemoteWhisperModel",
#     "gen_init_prompt",
#     "detect_audio_language",
#     "write_init_prompt",
# ]
