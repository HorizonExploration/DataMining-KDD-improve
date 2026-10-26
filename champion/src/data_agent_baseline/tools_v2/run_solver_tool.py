"""run_solver_tool.py — 运行 ``workdir/solver.py`` 并自检产出的单一职责工具.

替代旧的通用 ``execute`` 工具在 solver agent 里的角色. 设计动机见
``analysis`` 下相关报告: 通用 shell 给 agent 过多自由度, 诱导它绕开
``explore_data`` 在 heredoc 里手写脏路径(cd / sqlite3 / read_csv). 本工具
把 agent 唯一需要的"执行"动作收窄为: 跑 ``python3 ./workdir/solver.py``, 跑完
顺带回读 ``workdir/prediction.csv`` 做保存前自检.

设计要点
--------
- 固定执行 ``python3 ./workdir/solver.py`` (cwd=task_dir), 不暴露 command/path,
  杜绝当通用 shell 用. 与旧 execute 一样继承父进程 env(含 PYTHONPATH), 故
  sitecustomize 的 traceback 折叠(方案 A) 对子进程同样生效.
- 输出再过一遍 :func:`agents_v2.tb_fold.fold_text` (方案 B 兜底), 与旧
  ``_wrap_backend_block_cd`` 行为一致.
- solver.py 首次执行会逐表打印 schema (``_schema_shown.flag`` 控制), 对 agent
  是噪音, 折叠成一行摘要.
- 三态返回, 显式告诉 agent 发生了什么:
    FAILED   —— 进程非零退出 / 有 traceback;
    NO OUTPUT —— 跑通但 ``result is None`` (scaffold 未写 prediction.csv);
    OK + 自检 —— 跑通且产出, 回读 prediction.csv 给 shape/列名/非空数/前 N 行.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from agents_v2 import tb_fold

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 120
DEFAULT_HEAD = 5
MAX_OUTPUT_CHARS = 100_000  # 与旧 LocalBackend.execute 的截断阈值保持一致

# scaffold 生成的 solver.py 在 result 仍为 None 时打印的稳定标记
# (见 tools_v2/scaffold_tool.py 末尾段). 用它判定"跑通但没产出".
_RESULT_NONE_MARKER = "查询尚未实现 (result is None)"

# solver.py 首次执行打印的 schema 行, 历史上有两种格式:
#   `df_xxx (csv, 12cols) [别名 df_xxx]: 列A:VARCHAR, 列B:BIGINT, ...`  (当前 scaffold)
#   `df_xxx (rows=474, 7cols): id:Int64, CompanyCode:Int64, ...`        (旧 scaffold)
# 共同特征: 行首非空白, 括号内以 `<数字>cols)` 结尾, 其后 `: ` 接 `列名:类型` 列表.
# 用来识别并折叠这段噪音.
_SCHEMA_LINE_RE = re.compile(r"^\S.*\d+cols\)(?:\s*\[别名[^\]]*\])?:\s")

_TB_FOLD_OFF = {"0", "false", "no", "off", ""}


def _tb_fold_enabled() -> bool:
    """traceback 折叠开关, 与 sitecustomize / agent_util 共用 ``DATA_AGENT_SHORT_TB``.

    默认生效; 仅显式关闭值(0/false/no/off, 大小写不限) 时不折叠.
    """
    return os.environ.get("DATA_AGENT_SHORT_TB", "1").strip().lower() not in _TB_FOLD_OFF


# ---------------------------------------------------------------------------
# schema 噪音折叠
# ---------------------------------------------------------------------------


def _fold_schema_dump(text: str) -> str:
    """把 solver.py 首跑打印的连续 schema 行折叠成一行摘要.

    只折叠**连续**的 schema 行块, 块内行数 >= 2 才折(单行不值得折). 其余文本
    原样透传. 折叠后留一行 ``[run_solver] 已折叠 N 张表的 schema 打印(需要时用
    explore_data 的 \\schema <表名> 查)``.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if _SCHEMA_LINE_RE.match(lines[i]):
            j = i
            while j < n and _SCHEMA_LINE_RE.match(lines[j]):
                j += 1
            block = j - i
            if block >= 2:
                out.append(
                    f"[run_solver] 已折叠 {block} 张表的 schema 打印"
                    f"(需要某表字段/样本时用 explore_data 的 \\schema <表名> 查)\n"
                )
            else:
                out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "".join(out)


def _post_process_output(text: str) -> str:
    """对子进程合并输出做后处理: 先折 traceback(若启用), 再折 schema 噪音."""
    if not isinstance(text, str) or not text:
        return text
    if _tb_fold_enabled():
        try:
            text = tb_fold.fold_text(text)
        except Exception:  # noqa: BLE001 — 折叠失败绝不吞报错, 保留原文
            pass
    try:
        text = _fold_schema_dump(text)
    except Exception:  # noqa: BLE001
        pass
    return text


# ---------------------------------------------------------------------------
# prediction.csv 自检
# ---------------------------------------------------------------------------


def _inspect_prediction(pred_path: Path, head: int) -> str:
    """回读 prediction.csv, 输出 shape / 列名 / 各列非空数 / 前 head 行.

    读取失败显式报出, 不静默(自检失败本身就是要让 agent 知道的信号).
    """
    if not pred_path.exists():
        return (
            "[run_solver] 警告: solver.py 跑通但未找到 workdir/prediction.csv. "
            "请确认 pred_df 已正确赋值并走到 to_csv 分支."
        )
    try:
        df = pd.read_csv(pred_path)
    except Exception as e:  # noqa: BLE001
        return f"[run_solver] 自检失败: 无法读取 prediction.csv: {e}"

    n_rows, n_cols = df.shape
    lines = [
        f"[run_solver] OK  prediction.csv: rows={n_rows}, cols={n_cols}",
        f"  columns ({n_cols}): {', '.join(map(str, df.columns))}",
        "  各列非空数 / 总行数:",
    ]
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        lines.append(f"    - {col}: {non_null}/{n_rows}")
    h = max(0, head)
    if h:
        lines.append(f"  前 {min(h, n_rows)} 行:")
        for idx, row in df.head(h).iterrows():
            cells = {c: (None if pd.isna(v) else v) for c, v in row.items()}
            lines.append(f"    row {idx}: {cells}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def _run_solver(
    task_dir: Path,
    inspect: bool = True,
    head: int = DEFAULT_HEAD,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    task_dir = Path(task_dir).resolve()
    solver_path = task_dir / "workdir" / "solver.py"
    pred_path = task_dir / "workdir" / "prediction.csv"

    if not solver_path.exists():
        return (
            f"[run_solver] 错误: 未找到 {solver_path.relative_to(task_dir)}. "
            "正常情况下 scaffold 已生成该文件; 可用 read_solver 确认其内容."
        )

    if timeout <= 0:
        timeout = DEFAULT_TIMEOUT

    # 记录跑前 prediction.csv 的 mtime, 用于辅助判定本次是否真正刷新了产出.
    pred_mtime_before = pred_path.stat().st_mtime if pred_path.exists() else None

    t0 = time.time()
    # solver.py 会 import pandas/duckdb/tools_v2 等: 必须用当前解释器 sys.executable
    # (主进程跑在项目 venv 里, 依赖都装在这个 venv; 写死 'python3' 会落到系统 python
    # 导致 ModuleNotFoundError), 并把项目源码目录加到 PYTHONPATH 让子进程能 import
    # tools_v2.datasource_runtime. 与 zz_agent_v2.py 的兜底路径保持一致.
    src_root = str(Path(__file__).resolve().parents[1])  # .../src/data_agent_baseline
    env = dict(os.environ)
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, "./workdir/solver.py"],
            cwd=task_dir,
            env=env,  # 当前 venv 解释器 + 注入 PYTHONPATH, 保证 pandas/tools_v2 可 import
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            f"[run_solver] TIMEOUT: solver.py 执行超过 {timeout}s 被终止. "
            "请检查是否有死循环 / 笛卡尔积 join / 超大数据全量物化; "
            "先用 explore_data 验证查询规模再运行."
        )
    except Exception as e:  # noqa: BLE001
        return f"[run_solver] 错误: 无法启动 solver.py: {e}"
    elapsed_ms = int((time.time() - t0) * 1000)

    raw_output = (proc.stdout or "") + (proc.stderr or "")
    output = _post_process_output(raw_output)
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... <输出截断, 原长 {len(output)}>"

    header = f"# run_solver  exit_code={proc.returncode}  elapsed_ms={elapsed_ms}"
    out_block = output.strip()
    out_section = f"\n--- solver.py 输出 ---\n{out_block}" if out_block else ""

    # 三态判定
    if proc.returncode != 0:
        return (
            f"{header}\n[run_solver] FAILED: solver.py 非零退出, 见下方报错."
            f"{out_section}"
        )

    if _RESULT_NONE_MARKER in raw_output:
        return (
            f"{header}\n[run_solver] NO OUTPUT: result 仍为 None, 未写出 prediction.csv. "
            f'请在 solver.py 的 "## 进行查询" 段把 result 赋为真实查询结果后再运行.'
            f"{out_section}"
        )

    # 跑通且(应当)有产出
    pred_mtime_after = pred_path.stat().st_mtime if pred_path.exists() else None
    if pred_mtime_before is not None and pred_mtime_after == pred_mtime_before:
        logger.debug("run_solver: prediction.csv mtime 未变化, 可能未真正重写")

    if not inspect:
        note = "" if pred_path.exists() else "\n[run_solver] 警告: 未找到 prediction.csv."
        return f"{header}\n[run_solver] OK (inspect=False){note}{out_section}"

    inspect_block = _inspect_prediction(pred_path, head)
    return f"{header}\n{inspect_block}{out_section}"


# ---------------------------------------------------------------------------
# 暴露给 pydantic_ai 的 tool factory
# ---------------------------------------------------------------------------


def build_run_solver_tool(task_dir: Path):
    """构造一个绑定到 ``task_dir`` 的 ``run_solver`` 函数, 可直接传给 ``Tool(...)``.

    用法 (在 ``solver_agent.py``)::

        from pydantic_ai import Tool
        from tools_v2.run_solver_tool import build_run_solver_tool
        run_solver = build_run_solver_tool(task_dir)
        agent = create_deep_agent(..., tools=[Tool(run_solver, takes_ctx=False)])
    """
    task_dir = Path(task_dir).resolve()

    def run_solver(
        inspect: bool = True,
        head: int = DEFAULT_HEAD,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        """运行 ``workdir/solver.py`` 并自检产出的 ``prediction.csv``.

        写完 / 改完 solver.py 后用本工具执行它. 工具固定在任务根目录下跑
        ``python3 ./workdir/solver.py``, 跑完会自动回读 ``workdir/prediction.csv``
        给出 shape / 列名 / 各列非空数 / 前几行, 替代手动 cat 自检.

        三态返回:
        - FAILED: solver.py 报错(非零退出), 返回折叠后的 traceback;
        - NO OUTPUT: 跑通但 result 仍为 None(未填查询), 未写 prediction.csv;
        - OK: 跑通且有产出, 附 prediction.csv 自检.

        注意: 结构化数据(csv/json/sqlite)的探查请用 ``explore_data`` 工具, 不要
        靠反复跑 solver.py 试错; 本工具只负责"执行最终脚本 + 校验产出".

        Args:
            inspect: 跑完是否自动回读 prediction.csv 做自检, 默认 True.
            head: 自检时展示前几行, 默认 5.
            timeout: 执行超时秒数, 默认 120.

        Returns:
            人类可读字符串, 含 exit_code / 耗时 / 三态结论 / 自检 / solver 输出.
        """
        return _run_solver(task_dir, inspect=inspect, head=head, timeout=timeout)

    return run_solver


# ---------------------------------------------------------------------------
# CLI 调试
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_solver_tool.py <task_dir> [inspect] [head] [timeout]")
        sys.exit(1)
    td = Path(sys.argv[1])
    fn = build_run_solver_tool(td)
    print(fn())
