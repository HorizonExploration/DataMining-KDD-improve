"""站点级钩子: 折叠 execute 子进程里未捕获异常的 traceback (方案 A).

Python 解释器启动时, 若本文件所在目录在 ``sys.path`` / ``PYTHONPATH`` 上,
会被自动 ``import`` (site 机制). solver agent 的 ``execute`` 工具每次都 fork 一个
全新 python 子进程跑临时脚本 (heredoc / -c / solver.py), 这些子进程都会继承
父进程的 ``PYTHONPATH`` (本仓库入口脚本与 Dockerfile 都把
``src/data_agent_baseline`` 放进了 PYTHONPATH), 于是本钩子在每个子进程开机即生效,
无需 agent 写任何代码 —— 这是比"靠 prompt 让 agent 自己写 sys.tracebacklimit"
可靠得多的源头拦截.

折叠逻辑复用 :mod:`agents_v2.tb_fold` (与方案 B 工具层兜底同一事实源).

开关 ``DATA_AGENT_SHORT_TB`` (默认生效):
  - 显式关闭: ``0`` / ``false`` / ``no`` / ``off`` (大小写不限) -> 不装钩子, 看完整栈
  - 其它任何值, 或未设置 -> 装钩子, 折叠中间帧

本地调试想看完整栈时, ``export DATA_AGENT_SHORT_TB=0`` 即可.
"""

import os
import sys

_OFF = {"0", "false", "no", "off", ""}


def _enabled() -> bool:
    # 默认生效: 只有显式关闭值才返回 False
    return os.environ.get("DATA_AGENT_SHORT_TB", "1").strip().lower() not in _OFF


def _install() -> None:
    try:
        from agents_v2 import tb_fold
    except Exception:
        # tb_fold 不可用 (路径异常等) 时静默放弃, 绝不能让 sitecustomize 报错
        # 影响子进程正常启动.
        return

    _default_hook = sys.excepthook

    def _short_tb_excepthook(etype, exc, tb):
        try:
            sys.stderr.write(tb_fold.fold_frames(etype, exc, tb))
        except Exception:
            # 折叠本身出错时回退到默认全栈, 保证报错信息绝不丢失.
            _default_hook(etype, exc, tb)

    sys.excepthook = _short_tb_excepthook


if _enabled():
    _install()
