"""Python traceback 折叠 —— 方案 A / B 共用的单一事实源.

solver agent 用 ``execute`` 工具跑临时脚本时, pandas/sqlalchemy 这类深层库
抛出的异常会带 20+ 行的完整调用栈, 真正有用的只有"我哪行触发"(用户帧) 和
"最终在哪报错"(末帧) + 异常消息. 中间一长串库内部帧纯属噪声, 既刷屏又白白
吃掉 LLM 上下文.

本模块把"保留首帧 + 末若干帧 + 异常消息, 折叠中间帧"的策略集中在一处, 供:
  - 方案 A: ``sitecustomize.py`` 的自定义 ``sys.excepthook`` 调 :func:`fold_frames`
            (源头拦截, 拿到结构化 frame 对象, 最精确; 但只管未捕获异常)
  - 方案 B: ``LocalBackend`` 子类的 ``execute`` 对合并输出调 :func:`fold_text`
            (汇聚点兜底, 处理已渲染文本; 管一切经 execute 返回的栈)

两者共用同一组阈值常量, 保证不变式 ``HEAD + TAIL <= MAX``:
折叠后帧数 = HEAD + TAIL, 必须 <= MAX, 这样 B 再处理 A 已折过的文本时
``len(frames) > MAX`` 不成立 -> 原样透传, 折叠幂等, 不会二次删信息.
"""

from __future__ import annotations

import re
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

# ---- 折叠策略 (A / B 必须共用, 勿各自硬编码) ----
HEAD = 1  # 保留最外层(用户代码)帧数
TAIL = 2  # 保留最内层(报错点)帧数
MAX = 4   # 总帧数 <= MAX 时不折叠; 超过才折
assert HEAD + TAIL <= MAX, "折叠后帧数(HEAD+TAIL)必须<=MAX, 否则方案B会二次折叠A的输出"


def _fold_marker(hidden: int) -> str:
    return f"  ... [{hidden} 中间帧已折叠] ...\n"


def fold_frames(
    etype: type[BaseException],
    exc: BaseException,
    tb: "TracebackType | None",
) -> str:
    """把一个异常渲染成"折叠中间帧"的 traceback 文本 (方案 A / excepthook 用).

    传入结构化的 (类型, 异常, traceback), 用 :mod:`traceback` 提取 frame 后
    只保留首 ``HEAD`` 帧 + 末 ``TAIL`` 帧, 中间折叠成一行标记. 异常类型与消息
    (``format_exception_only``) 始终完整保留.

    不处理链式异常的 ``__cause__`` / ``__context__`` (excepthook 场景下交由
    解释器默认行为已足够; 方案 B 的 :func:`fold_text` 才需处理多块文本).
    """
    frames = traceback.extract_tb(tb)
    parts = ["Traceback (most recent call last):\n"]
    if len(frames) > MAX:
        hidden = len(frames) - HEAD - TAIL
        parts += traceback.format_list(frames[:HEAD])
        parts.append(_fold_marker(hidden))
        parts += traceback.format_list(frames[-TAIL:])
    else:
        parts += traceback.format_list(frames)
    parts += traceback.format_exception_only(etype, exc)
    return "".join(parts)


_TB_START = re.compile(r"^Traceback \(most recent call last\):\s*$")
_FRAME_HDR = re.compile(r'^  File ".*", line \d+')


def fold_text(text: str) -> str:
    """折叠任意文本里的 Python traceback 块 (方案 B / 工具层用).

    扫描每一行: 遇到 ``Traceback (most recent call last):`` 视为一个栈块开始,
    收集其后连续的 ``  File "...", line N`` 帧 (含各自的源码/``^^^`` 缩进行),
    超过 ``MAX`` 帧则保留首 ``HEAD`` + 末 ``TAIL``, 中间折叠. 非栈文本(stdout、
    异常消息行、链式异常的 "The above exception ..." 等) 原样透传.

    幂等: 对已被 :func:`fold_frames` 折叠过的文本 (仅 HEAD+TAIL 帧, <= MAX)
    不会再折, 整块透传 —— A->B 串联不丢信息.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if _TB_START.match(lines[i].rstrip("\n")):
            out.append(lines[i])
            i += 1
            # 收集帧: 每帧 = '  File "...' 行 + 其后更深缩进的源码/标记行
            frames: list[list[str]] = []
            while i < n and _FRAME_HDR.match(lines[i]):
                frame = [lines[i]]
                i += 1
                while i < n and lines[i].startswith("    ") and not _FRAME_HDR.match(lines[i]):
                    frame.append(lines[i])
                    i += 1
                frames.append(frame)
            if len(frames) > MAX:
                hidden = len(frames) - HEAD - TAIL
                for fr in frames[:HEAD]:
                    out.extend(fr)
                out.append(_fold_marker(hidden))
                for fr in frames[-TAIL:]:
                    out.extend(fr)
            else:
                for fr in frames:
                    out.extend(fr)
            # 帧块之后是异常消息行 (可能多行), 透传直到下一个 Traceback 块或空行
            while i < n and not _TB_START.match(lines[i].rstrip("\n")):
                out.append(lines[i])
                i += 1
                if i < n and lines[i].strip() == "":
                    break
        else:
            out.append(lines[i])
            i += 1
    return "".join(out)
