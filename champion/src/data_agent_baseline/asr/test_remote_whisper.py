#!/usr/bin/env python
"""
校验 RemoteWhisperModel（远程替身）与原生 faster_whisper.WhisperModel 的
**输入/输出字段一致性**。具体识别文本可能因模型/量化波动略有差异，故本测试
只断言「字段名集合 + 字段类型」对齐，不断言文本逐字相等。

覆盖：
  - detect_language: 返回 (language, language_probability, all_language_probs) 三元组结构/类型
  - transcribe:      返回 (segments, info)；逐段对象字段、info 对象字段
  - Word:            word_timestamps=True 时逐词对象字段

前置：先启动服务（本机自动 CPU/int8，可直接启动）：
  uvicorn asr.server:app --host 127.0.0.1 --port 8900

用法：uv run python -m asr.test_remote_whisper [task_id] [endpoint]
"""
import dataclasses as dc
import numbers
import sys
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from asr import RemoteWhisperModel

BASE = Path("/Users/zhe/Documents/work/20260400-data-agent/demo_samples_phase2")
task = sys.argv[1] if len(sys.argv) > 1 else "task_12"
endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8900"
video = BASE / f"input/{task}/context/video/briefing.mp4"

# info 中 client 刻意不透传的复杂字段（namedtuple, 下游用不到）。
INFO_OMITTED = {"transcription_options", "vad_options"}


def field_names(obj) -> list[str]:
    """统一拿一个对象的字段名（dataclass / namedtuple / __slots__ / __dict__）。"""
    if dc.is_dataclass(obj):
        return [f.name for f in dc.fields(obj)]
    t = type(obj)
    if hasattr(t, "_fields"):
        return list(t._fields)
    if getattr(t, "__slots__", None):
        return list(t.__slots__)
    return list(vars(obj).keys())


def assert_fields_superset(native, remote, label, omitted=frozenset()):
    """断言 remote 对象暴露了 native 的全部（减去 omitted）字段，并打印对比。"""
    nf, rf = set(field_names(native)), set(field_names(remote))
    required = nf - set(omitted)
    missing = required - rf
    print(f"  [{label}] native={sorted(nf)}")
    print(f"  [{label}] remote={sorted(rf)}")
    if omitted:
        print(f"  [{label}] 刻意省略(下游不用)={sorted(omitted)}")
    assert not missing, f"{label}: remote 缺少字段 {sorted(missing)}"
    print(f"  [{label}] 字段齐备 ✓")


def _kind(x) -> str:
    """字段类型按「类别」比较：JSON 往返会把 numpy.float64 变成 python float，
    二者数值与下游用法完全等价，故归为同一类，不算字段不一致。"""
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, numbers.Integral):
        return "int"
    if isinstance(x, numbers.Real):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, (list, tuple)):
        return "seq"
    return type(x).__name__


def assert_types_match(native, remote, fields, label):
    """对共有字段逐一比对类型类别（None 容忍：内容波动可能使某些字段一侧为 None）。"""
    bad = []
    for f in fields:
        nv, rv = getattr(native, f, None), getattr(remote, f, None)
        if nv is None or rv is None:
            continue
        if _kind(nv) != _kind(rv):
            bad.append(f"{f}: native={_kind(nv)}({type(nv).__name__}) "
                       f"remote={_kind(rv)}({type(rv).__name__})")
    assert not bad, f"{label}: 字段类型不一致 -> {bad}"
    print(f"  [{label}] 共有字段类型一致 ✓")


print(f"task={task}  endpoint={endpoint}\n{video}")
audio = decode_audio(str(video))
print(f"audio: shape={audio.shape} dtype={audio.dtype}\n")

loc = WhisperModel("tiny", device="cpu", compute_type="int8")
rem = RemoteWhisperModel("tiny", endpoint=endpoint)

# ---------- 1) detect_language ----------
print("=== detect_language: 返回三元组结构/类型 ===")
l_out = loc.detect_language(audio=audio)
r_out = rem.detect_language(audio=audio)
assert isinstance(l_out, tuple) and isinstance(r_out, tuple), "返回应为 tuple"
assert len(l_out) == len(r_out) == 3, "应为 3 元组 (lang, prob, all_probs)"
(l_lang, l_prob, l_all), (r_lang, r_prob, r_all) = l_out, r_out
assert type(l_lang) is type(r_lang) is str
assert isinstance(l_prob, float) and isinstance(r_prob, float)
# all_language_probs: 两侧都应是 [(str, float), ...]，可 dict() 还原
for nm, allp in (("native", l_all), ("remote", r_all)):
    assert isinstance(allp, list) and allp, f"{nm} all_language_probs 应为非空 list"
    a, b = allp[0]
    assert isinstance(a, str) and isinstance(b, float), f"{nm} 元素应为 (str, float)"
print(f"  native: lang={l_lang!r} prob={l_prob:.4f} all[0]={l_all[0]}")
print(f"  remote: lang={r_lang!r} prob={r_prob:.4f} all[0]={r_all[0]}")
print("  三元组结构/类型一致 ✓（内容可能因模型波动略有差异，不强制相等）\n")

# ---------- 2) transcribe: segments + info ----------
print("=== transcribe(language=det, beam_size=5, vad_filter=True) ===")
det = "zh" if dict(r_all).get("zh", 0) >= dict(r_all).get("en", 0) else "en"
kw = dict(language=det, beam_size=5, vad_filter=True)
l_segs, l_info = loc.transcribe(audio, **kw)
r_segs, r_info = rem.transcribe(audio, **kw)
l_segs = list(l_segs)  # 原生是惰性生成器, 物化
assert l_segs and r_segs, "两侧都应有 segment"
assert isinstance(r_segs, list), "remote segments 应为已物化 list"

print("- Segment 字段:")
assert_fields_superset(l_segs[0], r_segs[0], "Segment")
assert_types_match(l_segs[0], r_segs[0],
                   ["id", "seek", "start", "end", "text", "avg_logprob",
                    "compression_ratio", "no_speech_prob"], "Segment")

print("- TranscriptionInfo 字段:")
assert_fields_superset(l_info, r_info, "Info", omitted=INFO_OMITTED)
assert_types_match(l_info, r_info,
                   ["language", "language_probability", "duration"], "Info")

l_text = "".join(s.text for s in l_segs).strip()
r_text = "".join(s.text for s in r_segs).strip()
print(f"  native text({len(l_text)}): {l_text[:70]}...")
print(f"  remote text({len(r_text)}): {r_text[:70]}...")
print("  (文本内容仅供肉眼比对, 不做相等断言)\n")

# ---------- 3) Word 字段 (word_timestamps=True) ----------
print("=== transcribe(word_timestamps=True): Word 字段 ===")
lw_segs, _ = loc.transcribe(audio, word_timestamps=True, **kw)
rw_segs, _ = rem.transcribe(audio, word_timestamps=True, **kw)
lw = next((w for s in lw_segs for w in (s.words or [])), None)
rw = next((w for s in rw_segs if s.words for w in s.words), None)
assert lw is not None and rw is not None, "word_timestamps 应产出逐词对象"
assert_fields_superset(lw, rw, "Word")
assert_types_match(lw, rw, ["start", "end", "word", "probability"], "Word")
print()

print("ALL FIELD-PARITY CHECKS PASSED ✓")
