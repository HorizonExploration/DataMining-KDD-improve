"""把视频整合成「幻灯片帧 + 旁白」按时间轴交错的多模态输入(喂给 solver agent)。

这是 ``video`` 子包对外的【组装入口】,把已有的两块能力拼到一起:

  1. :func:`slide_coarse_det.extract_slides_coarse` —— 从 H.264 码率突变检测幻灯片
     切换(无需全解码),为每张稳定下来的幻灯片导出一张全分辨率 PNG;
  2. :func:`audio_transcribe.transcribe_video` —— 把视频音频转写成【带 word 级
     时间戳】的 whisper 结果(word 级是对齐的前提:一句旁白经常横跨多张幻灯片)。

然后按时间轴把两条信息源【交错】:**幻灯片帧在前,紧随其后是该帧出现之后、
下一帧出现之前的那段旁白文本**。交错算法沿用
``experiments/video/extract_slides.py:build_multimodal_message`` 的口径——逐词遍历
转写,凡是时间戳 ≤ 当前词起点的帧都先吐出来,于是模型看到的永远是
"先看到这张幻灯片,再听到讲它的话"。

产物是一个 pydantic-ai 用户消息可直接消费的 parts 列表::

    ["[幻灯片 #0 @6.0s]", BinaryContent(png0), "[旁白 6.4-17.5s] 行业分类字典同步完成…",
     "[幻灯片 #1 @18.3s]", BinaryContent(png1), "[旁白 …] …", ...]

库用法::

    from video.build_video_input import build_video_input
    result = build_video_input(
        video_path="context/video/briefing.mp4",
        frames_dir="workdir/_video_frames",   # 帧 PNG 落盘处(勿写 input)
        question=task_question,                # 可选:用于生成 ASR initial_prompt
        knowledge=knowledge_md,                # 可选:同上
    )
    parts = result["parts"]                    # 直接拼到 agent 的 user message

CLI(本机调试,默认远程 GPU 转写;产物写到 experiments/video/)::

    uv run python -m video.build_video_input --task-dir <task_dir>
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# remote ASR 后端(本机调试)走 ``from asr.remote_whisper import ...`` 裸包名,
# 需要 ``src/data_agent_baseline`` 在 path 上(与 zz_agent_v2.py 同款兜底)。
# docker 提交走 local 后端不触发此 import,但加上无副作用。
_PKG_ROOT = Path(__file__).resolve().parents[1]  # .../src/data_agent_baseline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from video.audio_transcribe import transcribe_video
from video.slide_coarse_det import SlideFrame, extract_slides_coarse

__all__ = [
    "build_video_input",
    "video_parts_for_task",
    "rebuild_from_saved",
    "interleave",
    "events_to_parts",
    "events_to_text",
    "save_video_input_logs",
    "TimelineEvent",
]


# ----------------------------------------------------------------------------
# 交错:把帧序列与转写按时间轴穿插
# ----------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """交错时间轴上的一个事件:一张幻灯片帧,或一段旁白文本。"""

    kind: str               # "frame" | "speech"
    t_start: float          # 事件起点(s)。frame 即帧时间戳;speech 即该段首词起点
    t_end: float            # 事件终点(s)。frame 与 t_start 相同;speech 即末词终点
    text: str = ""          # speech:旁白文本;frame:空
    slide_index: int = -1   # frame:幻灯片序号;speech:-1
    path: str = ""          # frame:PNG 绝对路径;speech:空
    hiccup: str = ""        # frame:该帧的 hiccup 版面树(可空);speech:空

    def to_dict(self) -> dict:
        d = {"type": self.kind, "t_start": round(self.t_start, 2),
             "t_end": round(self.t_end, 2)}
        if self.kind == "frame":
            d["slide"] = self.slide_index
            d["path"] = self.path
            if self.hiccup:
                d["hiccup"] = self.hiccup
        else:
            d["text"] = self.text
        return d


def _transcript_events(transcript: dict | None,
                       word_level: bool) -> list[tuple[float, float, str]]:
    """把 whisper 转写 dict 摊平成 ``[(t_start, t_end, text)]`` 旁白事件,按时间排序。

    ``word_level=True`` 时优先用 word 级时间戳(对齐更细);某 segment 缺 words 时
    退化用整段 ``{start,end,text}``。这是交错对齐的最小颗粒。
    """
    if not transcript:
        return []
    events: list[tuple[float, float, str]] = []
    for seg in transcript.get("segments", []):
        words = seg.get("words") if word_level else None
        if words:
            for w in words:
                events.append((float(w["start"]), float(w["end"]), w["word"]))
        else:
            events.append((float(seg["start"]),
                           float(seg.get("end", seg["start"])), seg["text"]))
    events.sort(key=lambda e: e[0])
    return events


def interleave(
    frames: list[SlideFrame],
    transcript: dict | None,
    *,
    word_level: bool = True,
    hiccup_map: dict[str, str] | None = None,
) -> list[TimelineEvent]:
    """把帧与转写按时间轴交错成一串事件:帧在前,紧随其后是对应旁白。

    **对齐时刻用 ``burst_start``(幻灯片开始切换的时刻),不是导出帧的 ``t``。**
    每张帧导出的 ``t`` 是幻灯片【沉降后】的时间戳(``burst_end + settle_margin``,
    比真实切换晚 2~3s,为的是截到稳定画面)。但讲新页的旁白往往在【转场动画进行中】
    就已经开口——若按沉降 ``t`` 对齐,这段"已在讲新页"的旁白会被错归到上一帧后面,
    且常从句子中间被切断。而 ``burst_start`` 通常恰好落在讲解者点击换页造成的【停顿
    处】,以它为切分点,帧会落进句子间隙、讲新页的旁白也正确跟在新帧之后。帧的
    ``t_start``/label 也显示 ``burst_start``(与对齐口径一致,即帧在旁白轴上的位置);
    图片内容仍是沉降帧(``t`` 那一刻的稳定画面),只是对外不再单独暴露 ``t``。

    逐词遍历转写:每遇到一个词,先把所有"对齐时刻 ≤ 该词中点"的帧吐出(帧在前),
    再把连续的词攒成一段 speech 文本(到下一张帧出现前合并成一块,减少碎片)。
    用【词中点】而非起点判断,避免"刚开口就换页"的词被劈在上一帧尾部(见下方循环注释)。
    最后把末帧之后的尾部旁白、以及没有任何旁白时的剩余帧补齐。

    Args:
        hiccup_map: 可选 ``{帧PNG路径: hiccup文本}``(由 ``frame_html_tool.
            frames_to_hiccup`` 生成的内存映射)。命中则挂到对应 frame 事件的
            ``hiccup`` 字段,供后续以"图片 + 版面树"模式拼装。

    Returns:
        ``[TimelineEvent]``,严格按出现顺序;可经 :func:`events_to_parts` 转成
        pydantic-ai parts,或 ``[e.to_dict() for e in events]`` 落盘成可读时间轴。
    """
    hiccup_map = hiccup_map or {}

    def _align_t(f: SlideFrame) -> float:
        # 用 burst_start(开始切换时刻)作为对齐点。开场帧 burst_start==0,语义即
        # "从视频最开头就显示着",理应早于一切旁白排在最前——所以无条件取 burst_start,
        # 不回退到沉降 t(回退会把开场帧错误地推到 0.0s 那段旁白之后)。
        return f.burst_start

    frames = sorted(frames, key=_align_t)
    events = _transcript_events(transcript, word_level)
    out: list[TimelineEvent] = []

    fi = 0
    buf: list[str] = []
    buf_start: float | None = None
    buf_end: float | None = None

    def flush_speech():
        nonlocal buf, buf_start, buf_end
        if buf:
            out.append(TimelineEvent("speech", buf_start, buf_end,
                                     text="".join(buf).strip()))
            buf, buf_start, buf_end = [], None, None

    def emit_frame(fm: SlideFrame):
        # t_start/label 用 burst_start(开始切换时刻),与对齐口径一致:label 标的
        # "第几秒"就是帧在旁白时间轴上的位置。图片内容仍是沉降帧(fm.t 那一刻),
        # 但对外只暴露 burst_start 这一个时间,避免两套时间让模型困惑。
        out.append(TimelineEvent("frame", fm.burst_start, fm.burst_start,
                                 slide_index=fm.index, path=fm.path,
                                 hiccup=hiccup_map.get(fm.path, "")))

    for (s, e, text) in events:
        # 帧插在哪个词之前,看【词的中点】而非起点:讲解者常"边说出新句第一个词、
        # 边点下一页"(This|换页|pop-up),此时换页时刻 burst_start 落在该词发音过程
        # 中间——词起点早于 burst、但词主体在 burst 之后。用中点比较,这类"刚开口就
        # 换页"的词(其讲的已是新页内容)会正确归到新帧之后,不再被劈在上一帧尾部。
        mid = (s + e) / 2
        while fi < len(frames) and _align_t(frames[fi]) <= mid:
            flush_speech()
            emit_frame(frames[fi])
            fi += 1
        if buf_start is None:
            buf_start = s
        buf_end = e
        buf.append(text)
    flush_speech()

    # 末帧之后再无旁白的剩余帧
    while fi < len(frames):
        emit_frame(frames[fi])
        fi += 1
    return out


# ----------------------------------------------------------------------------
# 事件 -> pydantic-ai 多模态 parts / 纯文本预览
# ----------------------------------------------------------------------------

# Qwen3.5 系列 ViT 图像 token 成本(供做帧下采样取舍参考)
# -----------------------------------------------------------
# 模型侧 Qwen2VLImageProcessorFast 的 smart_resize 把 H/W 各自对齐到 32 的整数倍
# (factor = patch_size × spatial_merge_size = 16 × 2),且总像素夹到
# [65536(=256²), 16777216(=4096²)]。每个 image token 覆盖 32×32 = 1024 像素,
# 故:
#       image_tokens = H' × W' / 1024
#
# briefing.mp4 通常是 720p,extract_slides_coarse 导出的帧也是全分辨率(1280×720)。
# 以 16:9 为例,几档常见下采样的 token 成本:
#
#   原始尺寸          对齐后 H'×W'        总像素      image tokens
#   1280× 720  → 1280×704   (720→704)    901,120     **880**   ← 当前默认
#   1024× 576  → 1024×576                589,824       576
#    960× 544  →  960×544   (540→544)    522,240       510
#    864× 480  →  864×480   (854→864)    414,720       405
#    768× 432  →  768×448   (432→448)    344,064       336
#    640× 352  →  640×352   (360→352)    225,280       220
#    512× 288  →  512×288                147,456       144
#    480× 256  →  480×256   (270→256)    122,880       120
#    416× 256  →  416×256   (426→416)    106,496       104
#    256× 256  →  256×256              ≥  65,536        64   ← min 档(再小会被升采样)
#    320× 180  →  352×192   (升采样)     67,584      ~66    ← 已触底,与 256² 接近
#
# 一段 10 分钟视频常出 30–60 张幻灯片帧:
#   - 默认 1280×720 → 26k–53k vision token(27B / 256k context 仍很轻);
#   - 砍到 640×352  → 6.6k–13k,大幅省 token,牺牲细节(小字读不出来时再用 hiccup 补)。
# 想省 token 也可走 frames_mode="hiccup":vision token 归零,只留版面树文本。
# 注:目前 slide_coarse_det 导出 PNG 不做 resize;真要省 vision 成本需要在导出后
# 或 _read_png 里加一次缩放(本文件未实现)。

def _read_png(path: str):
    from pydantic_ai import BinaryContent

    return BinaryContent(data=Path(path).read_bytes(), media_type="image/png")


def events_to_parts(
    events: list[TimelineEvent],
    *,
    label: bool = True,
    frames_mode: str = "image+hiccup",
) -> list:
    """把交错事件转成 pydantic-ai 用户消息 parts(text 与 BinaryContent 交替)。

    ``label=True`` 时给每个事件加一行时间标注([幻灯片 #k @t] / [旁白 a-b]),
    让模型能把帧、旁白和时间对上号。

    ``frames_mode`` 控制帧如何呈现:
      * ``"image+hiccup"``(默认):图片 + 其 hiccup 版面树并列。图片信息量最大,
        hiccup 把小字表格/数值转成模型易精确读取的文本,二者互补。无 hiccup 的帧
        自动退化为纯图片。
      * ``"image"``:仅图片。
      * ``"hiccup"``:仅 hiccup 文本(无 vision / 省 token 时用;缺 hiccup 的帧回退图片)。
    """
    parts: list = []
    for ev in events:
        if ev.kind == "frame":
            if label:
                parts.append(f"[幻灯片 #{ev.slide_index} @{ev.t_start:.1f}s]")
            want_img = frames_mode in ("image", "image+hiccup") or not ev.hiccup
            want_hic = frames_mode in ("hiccup", "image+hiccup") and ev.hiccup
            if want_img:
                parts.append(_read_png(ev.path))
            if want_hic:
                parts.append(f"[版面 #{ev.slide_index}]\n{ev.hiccup}")
        else:
            prefix = f"[旁白 {ev.t_start:.1f}-{ev.t_end:.1f}s] " if label else ""
            parts.append(prefix + ev.text)
    return parts


def events_to_text(events: list[TimelineEvent], *, with_hiccup: bool = False) -> str:
    """把交错事件渲染成【纯文本】预览(图片用占位行),供调试/日志/落盘。

    ``with_hiccup=True`` 时把帧的 hiccup 版面树也打印出来,便于核对对齐+版面。
    """
    lines = []
    for ev in events:
        if ev.kind == "frame":
            lines.append(f"[幻灯片 #{ev.slide_index} @{ev.t_start:.1f}s] "
                         f"<{Path(ev.path).name}>")
            if with_hiccup and ev.hiccup:
                lines.append(ev.hiccup)
        else:
            lines.append(f"[旁白 {ev.t_start:.1f}-{ev.t_end:.1f}s] {ev.text}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 端到端编排:抽帧 + 转写 + 交错
# ----------------------------------------------------------------------------

def build_video_input(
    video_path: str | Path,
    frames_dir: str | Path,
    *,
    question: str | None = None,
    knowledge: str | None = None,
    initial_prompt: str | None = None,
    language: str | None = None,
    word_level: bool = True,
    label: bool = True,
    with_hiccup: bool = True,
    hiccup_workers: int = 8,
    frames_mode: str = "image+hiccup",
    backend: str | None = None,
    transcribe_kwargs: dict | None = None,
) -> dict:
    """端到端:对一个视频抽帧 + 转写(+ 可选 hiccup 版面),交错成多模态 parts。

    处理【单个 task】,全程在内存完成(抽帧→转写→生成 hiccup→对齐→parts),不依赖
    任何盘上中间产物。批量请另写脚本循环调用本函数。

    Args:
        video_path: 源视频(briefing.mp4)。只读,绝不写入。
        frames_dir: 帧 PNG 的落盘目录(请放在 workdir/ 或 experiments/ 下,**勿写
            input**)。``extract_slides_coarse`` 会在此创建。
        question / knowledge: 可选。当未显式给 ``initial_prompt`` 时,用它们调
            ``asr_init_prompt.gen_init_prompt`` 现场生成 ASR 软引导(纠正同音字)。
            不传则转写裸跑。
        initial_prompt: 直接指定 ASR initial_prompt(优先于 question/knowledge 生成)。
        language: 锚定转写语言(``None`` 自动检测;不要默认强制 zh)。
        word_level: 交错对齐颗粒,``True`` 用 word 级时间戳(默认,更细)。
        label: 是否给 parts 加时间标注行。
        with_hiccup: 是否为每帧生成 hiccup 版面树(默认开)。所有帧抽完后【统一】并发
            生成(``frame_html_tool.frames_to_hiccup``),结果留在内存并挂到帧事件;
            端点不可用 / 关闭时优雅降级——hiccup 为空,``image+hiccup`` 模式自动只给图片。
        hiccup_workers: hiccup 并发线程数(网络绑定)。
        frames_mode: 帧呈现模式,见 :func:`events_to_parts`(默认 ``image+hiccup``)。
        backend: ASR 后端(``None`` 默认 local；ASR_BACKEND=remote 才走远程 GPU);透传转写。
        transcribe_kwargs: 透传给 ``transcribe_video`` 的其它参数(model 等)。

    Returns:
        dict::

            {
              "video": "briefing.mp4",
              "language": "zh",
              "initial_prompt": "…" | None,
              "frames": [SlideFrame.to_dict(), ...],
              "transcript": {whisper 结果 dict},
              "hiccup": {帧路径: hiccup文本},               # 内存映射(with_hiccup 时)
              "timeline": [TimelineEvent.to_dict(), ...],   # 中间层:帧+hiccup+旁白对齐
              "parts": [str | BinaryContent, ...],          # 直接喂 agent 的多模态消息
            }
    """
    video_path = Path(video_path)
    tkw = dict(transcribe_kwargs or {})

    # 1) 抽帧(码率突变法,无需全解码)
    _t0 = time.perf_counter()
    frames = extract_slides_coarse(video_path, frames_dir)
    logger.info('[video] extract_frames: {:.2f}s ({} frames)', time.perf_counter() - _t0, len(frames))

    # 2) 生成 ASR initial_prompt(可选):优先用显式传入,否则按 question/knowledge 现场写
    if initial_prompt is None and question:
        from video.asr_init_prompt import gen_init_prompt

        initial_prompt = gen_init_prompt(
            video_path, question=question, knowledge=knowledge, lang=language,
        )

    # 3) 转写(带 word 级时间戳)
    _t1 = time.perf_counter()
    transcript = transcribe_video(
        video_path,
        backend=backend,
        language=language,
        initial_prompt=initial_prompt,
        **tkw,
    )
    logger.info('[video] ASR transcribe: {:.2f}s', time.perf_counter() - _t1)

    # 4) 所有帧抽完后【统一】并发生成 hiccup 版面树(留在内存)。端点不可用即降级为空。
    hiccup_map: dict[str, str] = {}
    if with_hiccup and frames:
        from video.frame_html_tool import frames_to_hiccup

        _t2 = time.perf_counter()
        try:
            hiccup_map = frames_to_hiccup(
                [f.path for f in frames], workers=hiccup_workers)
        except Exception:
            hiccup_map = {}  # 端点不可达等:优雅降级为纯图片
        logger.info('[video] hiccup generation: {:.2f}s ({} frames)', time.perf_counter() - _t2, len(frames))

    # 5) 按时间轴交错(把 hiccup 挂到帧事件)
    events = interleave(frames, transcript, word_level=word_level,
                        hiccup_map=hiccup_map)
    parts = events_to_parts(events, label=label, frames_mode=frames_mode)

    return {
        "video": video_path.name,
        "language": transcript.get("language"),
        "initial_prompt": initial_prompt,
        "frames": [f.to_dict() for f in frames],
        "transcript": transcript,
        "hiccup": hiccup_map,
        "timeline": [e.to_dict() for e in events],
        "parts": parts,
    }


def rebuild_from_saved(
    saved: str | Path | dict,
    *,
    word_level: bool = True,
    label: bool = True,
    frames_mode: str = "image+hiccup",
) -> dict:
    """从已保存的 ``video_input.json`` 复用 frames + transcript + hiccup,只重新交错。

    抽帧、ASR 转写、hiccup 生成是最贵的三步(后两步都要打端点)。``build_video_input``
    落盘的 ``video_input.json`` 已含 ``frames``(带 burst 窗)、``transcript`` 和
    ``hiccup`` 映射,故调交错口径 / 排版 / frames_mode 时无需重跑——读回这几块,重新
    :func:`interleave` 即可,秒级完成。

    Args:
        saved: ``video_input.json`` 的路径,或已 ``json.load`` 的 dict。
        word_level / label / frames_mode: 同 :func:`build_video_input`。

    Returns:
        与 :func:`build_video_input` 同构的 dict(含新的 ``timeline`` 与 ``parts``)。
    """
    if isinstance(saved, (str, Path)):
        saved = json.loads(Path(saved).read_text())
    frames = [
        SlideFrame(index=f["index"], t=f["t"], path=f["path"],
                   burst_start=f["burst_start"], burst_end=f["burst_end"])
        for f in saved["frames"]
    ]
    transcript = saved["transcript"]
    hiccup_map = saved.get("hiccup") or {}
    events = interleave(frames, transcript, word_level=word_level,
                        hiccup_map=hiccup_map)
    return {
        "video": saved.get("video"),
        "language": saved.get("language"),
        "initial_prompt": saved.get("initial_prompt"),
        "frames": saved["frames"],
        "transcript": transcript,
        "hiccup": hiccup_map,
        "timeline": [e.to_dict() for e in events],
        "parts": events_to_parts(events, label=label, frames_mode=frames_mode),
    }


# ----------------------------------------------------------------------------
# task 级便捷入口:供 solver agent 在 task 工作目录里直接取用视频多模态 parts
# ----------------------------------------------------------------------------

VIDEO_REL = "context/video/briefing.mp4"  # task 内视频的固定相对路径


def video_parts_for_task(
    task_dir: str | Path,
    *,
    question: str | None = None,
    knowledge: str | None = None,
    frames_subdir: str = "workdir/_video_frames",
    save_meta: bool = True,
    log_dir: str | Path | None = None,
    **kwargs,
) -> list:
    """为一个 task 生成视频多模态 parts(无视频则返回空列表)。

    solver agent 的 task 工作目录约定:视频在 ``<task_dir>/context/video/briefing.mp4``,
    仅 ``workdir/`` 可写。本函数按此约定:检测视频→抽帧到 ``workdir/_video_frames``→
    转写→生成 hiccup→交错,返回可直接拼进 ``agent.run`` user message 的
    ``[str | BinaryContent]``。无视频时返回 ``[]``,调用方据此决定是否拼接。

    Args:
        task_dir: task 根目录(含 ``context/`` 与可写的 ``workdir/``)。
        question / knowledge: 透传给 :func:`build_video_input`,用于 ASR initial_prompt。
            通常即 task.json 的 question 与 knowledge.md 文本。
        frames_subdir: 帧 PNG 落盘子目录(相对 task_dir,须在 ``workdir/`` 下,勿写
            context/input)。
        save_meta: 是否把可读交错时间轴落盘到帧目录(``video_input.json`` /
            ``timeline.txt``),便于排查;不影响返回的 parts。
        log_dir: 若提供,额外把 ``video_input.json`` / ``hiccup.json`` /
            ``timeline_hiccup.txt`` 落到该日志目录。
        **kwargs: 透传给 :func:`build_video_input`(frames_mode / with_hiccup / backend …)。

    Returns:
        ``[str | BinaryContent]``(图片+版面+旁白交错);无视频则 ``[]``。
    """
    task_dir = Path(task_dir)
    video = task_dir / VIDEO_REL
    if not video.exists():
        return []

    frames_dir = task_dir / frames_subdir
    result = build_video_input(
        video, frames_dir,
        question=question, knowledge=knowledge, **kwargs,
    )

    if save_meta or log_dir is not None:
        try:
            meta = {k: v for k, v in result.items() if k != "parts"}
            events = [_ev_from_dict(d) for d in result["timeline"]]
            if log_dir is not None:
                save_video_input_logs(meta, log_dir)
            if save_meta:
                (frames_dir / "video_input.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                (frames_dir / "timeline.txt").write_text(
                    events_to_text(events),
                    encoding="utf-8")
        except Exception as e:
            logger.warning("video input meta/log save failed: {}", e)

    return result["parts"]


def save_video_input_logs(meta: dict, log_dir: str | Path) -> None:
    """把视频输入中间产物落到 runner 的 task 日志目录.

    ``meta`` 是 ``build_video_input`` 返回值去掉 ``parts`` 后的 JSON 可序列化对象。
    除完整元信息外,额外拆出 hiccup 映射和带 hiccup 的可读时间轴,方便直接核对版面
    预测质量。
    """
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        (log_dir / "video_input.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        hiccup_map = meta.get("hiccup") or {}
        if not isinstance(hiccup_map, dict):
            hiccup_map = {}
        (log_dir / "hiccup.json").write_text(
            json.dumps(hiccup_map, ensure_ascii=False, indent=2), encoding="utf-8")

        events = [_ev_from_dict(d) for d in meta.get("timeline") or []]
        (log_dir / "timeline_hiccup.txt").write_text(
            events_to_text(events, with_hiccup=True), encoding="utf-8")
    except Exception as e:
        logger.warning("video input log save failed to {}: {}", log_dir, e)


# ----------------------------------------------------------------------------
# CLI: 针对单个 task 跑一遍,把交错时间轴写到 experiments/video/(便于核对)
# ----------------------------------------------------------------------------

def _resolve_video(task_dir: str | None, video: str | None) -> Path:
    if video:
        return Path(video)
    if task_dir:
        return Path(task_dir) / "context/video/briefing.mp4"
    raise SystemExit("需要 --video 或 --task-dir")


def _ev_from_dict(d: dict) -> TimelineEvent:
    """timeline dict 还原成 TimelineEvent(仅 CLI 渲染纯文本用)。"""
    if d["type"] == "frame":
        return TimelineEvent("frame", d["t_start"], d["t_end"],
                             slide_index=d["slide"], path=d["path"],
                             hiccup=d.get("hiccup", ""))
    return TimelineEvent("speech", d["t_start"], d["t_end"], text=d["text"])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="抽帧 + 转写 + 按时间轴交错,生成视频多模态输入(调试)。")
    ap.add_argument("--task-dir", default=None, help="task 目录(取其 context/video/briefing.mp4)")
    ap.add_argument("--video", default=None, help="直接指定视频路径")
    ap.add_argument("--out-dir", default="experiments/video/build_video_test",
                    help="产物输出目录(帧 + timeline,勿指向 input)")
    ap.add_argument("--question", default=None, help="任务问题(用于生成 ASR initial_prompt)")
    ap.add_argument("--language", default=None, help="锚定转写语言(默认自动检测)")
    ap.add_argument("--backend", default=None, choices=["local", "remote"],
                    help="ASR 后端(默认按是否容器自动选)")
    ap.add_argument("--no-hiccup", action="store_true", help="跳过 hiccup 版面生成")
    args = ap.parse_args()

    video = _resolve_video(args.task_dir, args.video)
    if not video.exists():
        raise SystemExit(f"视频不存在: {video}")

    # 题目 / knowledge:优先命令行,否则尝试从 task_dir 读取
    question = args.question
    knowledge = None
    if args.task_dir:
        td = Path(args.task_dir)
        tj = td / "task.json"
        if question is None and tj.exists():
            question = json.loads(tj.read_text()).get("question")
        km = td / "context/knowledge.md"
        if km.exists():
            knowledge = km.read_text()

    out_dir = Path(args.out_dir)
    if args.task_dir:
        out_dir = out_dir / Path(args.task_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    print(f"[build_video_input] video={video}\n  frames -> {frames_dir}", flush=True)
    result = build_video_input(
        video, frames_dir,
        question=question, knowledge=knowledge,
        language=args.language, backend=args.backend,
        with_hiccup=not args.no_hiccup,
    )

    # 落盘:可读时间轴 + 元信息(不含 BinaryContent,图片只留路径)
    meta = {k: v for k, v in result.items() if k != "parts"}
    (out_dir / "video_input.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    events = [_ev_from_dict(d) for d in result["timeline"]]
    (out_dir / "timeline.txt").write_text(
        events_to_text(events), encoding="utf-8")
    # 带 hiccup 版面的详版,便于核对对齐 + 版面识别质量
    (out_dir / "timeline_hiccup.txt").write_text(
        events_to_text(events, with_hiccup=True), encoding="utf-8")

    n_frame = sum(1 for e in result["timeline"] if e["type"] == "frame")
    n_speech = sum(1 for e in result["timeline"] if e["type"] == "speech")
    n_hic = sum(1 for v in (result.get("hiccup") or {}).values() if v)
    print(f"  language={result['language']}  帧={n_frame}  旁白段={n_speech}  "
          f"hiccup={n_hic}/{n_frame}  parts={len(result['parts'])}", flush=True)
    print(f"  -> {out_dir / 'video_input.json'}\n  -> {out_dir / 'timeline.txt'}\n"
          f"  -> {out_dir / 'timeline_hiccup.txt'}", flush=True)


if __name__ == "__main__":
    main()
