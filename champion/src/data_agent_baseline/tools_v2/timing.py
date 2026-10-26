"""轻量级任务计时模块 — 统计各环节耗时并落盘 JSON.

用法::

    from tools_v2.timing import TaskTimer

    timer = TaskTimer("task_042")
    with timer.stage("doc_prepare"):
        await prepare_doc_tables_async(...)
    with timer.stage("agent_run.attempt1"):
        result = await agent.run(...)

    timer.dump(task_log_dir / "timing.json")

特性:
- 同名 stage 多次调用时**累加**(适用于 agent_run 多 attempt)
- 每段退出时 logger.info 打印(本地可见;docker 无 sink 静默)
- dump() 落盘 JSON, docker 也能事后分析
- 支持嵌套(外层包含内层时间,各自独立统计)
- 线程/协程安全:每个 TaskTimer 实例绑定一个 task_id,不共享状态
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger


class TaskTimer:
    """单个 task 的计时器,跟踪各 stage 耗时."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._stages: dict[str, float] = {}  # stage_name -> 累计秒数
        self._order: list[str] = []  # 保持首次出现顺序
        self._t0 = time.perf_counter()  # task 整体起点

    @contextmanager
    def stage(self, name: str):
        """上下文管理器:记录 name 对应环节的耗时(同名累加)."""
        t_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t_start
            if name not in self._stages:
                self._order.append(name)
                self._stages[name] = 0.0
            self._stages[name] += elapsed
            logger.info(
                "[{}] ⏱ {} {:.2f}s (cumulative {:.2f}s)",
                self.task_id,
                name,
                elapsed,
                self._stages[name],
            )

    @property
    def total_elapsed(self) -> float:
        """从 TaskTimer 创建到现在的墙钟时间."""
        return time.perf_counter() - self._t0

    def summary(self) -> dict[str, Any]:
        """生成计时摘要 dict."""
        return {
            "task_id": self.task_id,
            "total_wall_s": round(self.total_elapsed, 3),
            "stages": {name: round(self._stages[name], 3) for name in self._order},
        }

    def dump(self, path: str | Path) -> Path:
        """把计时摘要写入 JSON 文件(自动创建父目录)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[{}] ⏱ timing dumped → {} (total {:.1f}s)", self.task_id, path, data["total_wall_s"])
        return path


def format_timing_summary(all_timings: list[dict[str, Any]]) -> str:
    """将多个 task 的 timing 汇总为可读表格字符串(用于终端打印).

    Args:
        all_timings: 每项为 TaskTimer.summary() 的返回值.

    Returns:
        多行字符串,可直接 logger.info 输出.
    """
    if not all_timings:
        return "(no timing data)"

    # 收集所有出现过的 stage 名(保持首次出现顺序)
    all_stages: list[str] = []
    seen: set[str] = set()
    for t in all_timings:
        for s in t.get("stages", {}):
            if s not in seen:
                seen.add(s)
                all_stages.append(s)

    lines: list[str] = []
    sep = "-" * 80
    lines.append("=" * 80)
    lines.append(f"TIMING SUMMARY ({len(all_timings)} tasks)")
    lines.append(sep)

    # header
    header = f"  {'task_id':<14} | {'total':>7}"
    for s in all_stages:
        # 截短 stage 名到 12 字符
        short = s[:12]
        header += f" | {short:>8}"
    lines.append(header)
    lines.append(sep)

    # per task
    totals_per_stage: dict[str, float] = {s: 0.0 for s in all_stages}
    for t in sorted(all_timings, key=lambda x: x.get("task_id", "")):
        row = f"  {t['task_id']:<14} | {t['total_wall_s']:>6.1f}s"
        stages = t.get("stages", {})
        for s in all_stages:
            v = stages.get(s, 0.0)
            totals_per_stage[s] += v
            row += f" | {v:>7.1f}s" if v > 0 else f" | {'—':>8}"
        lines.append(row)

    # sum / avg
    lines.append(sep)
    n = len(all_timings)
    total_wall = sum(t["total_wall_s"] for t in all_timings)
    row_sum = f"  {'SUM':<14} | {total_wall:>6.1f}s"
    row_avg = f"  {'AVG':<14} | {total_wall / n:>6.1f}s"
    for s in all_stages:
        row_sum += f" | {totals_per_stage[s]:>7.1f}s"
        row_avg += f" | {totals_per_stage[s] / n:>7.1f}s"
    lines.append(row_sum)
    lines.append(row_avg)

    # 最慢 task
    slowest = max(all_timings, key=lambda x: x["total_wall_s"])
    lines.append(sep)
    lines.append(f"  Slowest: {slowest['task_id']} ({slowest['total_wall_s']:.1f}s)")
    lines.append("=" * 80)

    return "\n".join(lines)
