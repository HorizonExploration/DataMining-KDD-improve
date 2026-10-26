"""solver_file_tool.py — 只读写 ``workdir/solver.py`` 的专用 read/edit 工具.

设计动机
--------
旧方案给 solver agent 挂的是 deep agent 自带的整组 console 工具
(``ls/glob/grep/read_file/write_file/edit_file``). 这带来两个问题:

1. **绕开 explore_data 探查数据**: agent 会用 read_file/grep 直接去翻
   ``context/`` 下的 csv/json/doc 原文 (尤其 doc 抽取表不理想时, 临时手写
   正则解析 ``doc/*.md``), 既慢又脆, 还破坏了 explore_data(DuckDB) 的
   表名/列名归一保证. 典型失败见 ``analysis`` 下 task_32 排查记录:
   60 次请求耗尽预算, 大半花在 read_file/hashline_edit 反复空转.
2. **文件系统自由度过大**: ls/glob 在"所有表已注册给 agent"的前提下纯属噪音.

因此把 agent 对文件的能力**收窄到唯一需要的动作**: 读 / 改
``./workdir/solver.py``. 数据探查一律走 ``explore_data``; 执行走 ``run_solver``.

实现要点
--------
- 两个工具**不接受 path 参数**, 路径写死为 ``<task_dir>/workdir/solver.py``,
  agent 无法借此读写任何其它文件.
- 完全复用底层 ``pydantic_ai_backends.hashline`` 的 ``format_hashline_output``
  (读) 与 ``apply_hashline_edit_with_summary`` (写), 因此行首
  ``行号:hash|`` 标记、hash 校验、insert/delete 语义与旧 hashline_edit 完全一致.
- ``hashline.line_hash`` 已被 ``agents_v2.hashline_patch`` monkey-patch 成全字母
  hash; 本模块走同一函数, patch 自动生效 (调用方 solver_agent 已先 import 该 patch).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from pydantic_ai_backends import hashline


# solver.py 相对 task_dir 的固定位置. 与 scaffold_tool / run_solver_tool 一致.
_SOLVER_REL = "workdir/solver.py"

# edit 成功后回传"新 hash 快照"时, 在受影响区域上下各带几行上下文.
# 让 agent 直接从 edit 返回值拿到最新 行号:hash, 无需再 read_solver 即可继续编辑,
# 既消除"凭旧 hash 连续编辑 -> hash mismatch"死循环, 又省掉一次 re-read 请求.
_EDIT_SNAPSHOT_CONTEXT = 3

# 底层 hashline 的 hash mismatch 错误形如:
#   "Hash mismatch at line 66: expected 'XP', got 'NZ'. File may have changed — re-read it first."
# 实测 (60-task run 中 39 个 task 首次范围编辑即失配) 绝大多数是 agent 把 end_hash
# 取成了同一次 read 里**邻近行**的 hash (行号→hash 取串), 文件并未改动: 此时 'got'
# 后面就是该行的真实 hash. 底层那句 "re-read it first" 反而误导 agent 白白多读一轮.
# 这里把它改写成"直接用真实 hash 重发"的可执行提示, 仍然拒绝落盘(保持精准校验).
_HASH_MISMATCH_RE = re.compile(
    r"Hash mismatch at line (\d+): expected '([^']*)', got '([^']*)'"
)


def _rewrite_hash_mismatch(error: str) -> str:
    """把底层 hash mismatch 错误改写成可直接重试的提示; 非该类错误原样返回."""
    m = _HASH_MISMATCH_RE.search(error)
    if not m:
        return f"Error: {error}"
    line_no, given, actual = m.group(1), m.group(2), m.group(3)
    return (
        f"Error: hash 校验失败 at line {line_no}: 你给的 hash='{given}', "
        f"但该行当前真实 hash='{actual}'.\n"
        f"  常见原因: 范围编辑时 end_hash 取成了邻近行的 hash (行号→hash 取串), "
        f"文件本身未被改动.\n"
        f"  → 直接把该处 hash 改成 '{actual}' 重发本次编辑即可, 无需 read_solver 重读.\n"
        f"  (仅当你怀疑行号 {line_no} 本身取错、或文件确被改动时, 才需 read_solver 重读.)"
    )


def _solver_path(task_dir: Path) -> Path:
    return Path(task_dir).resolve() / _SOLVER_REL


def build_read_solver_tool(task_dir: Path):
    """构造绑定到 ``task_dir`` 的 ``read_solver`` 函数, 传给 ``Tool(...)``."""
    solver_path = _solver_path(task_dir)

    def read_solver(offset: int = 0, limit: int = 2000) -> str:
        """读取 ``./workdir/solver.py``, 行首带 ``行号:hash|`` 标记.

        这是本 agent 唯一的文件读取通道, 路径固定为 solver.py, 不接受其它路径.
        结构化数据(csv/json/sqlite)的探查请用 ``explore_data``, 不要试图读原始数据文件.

        每行格式 ``{行号}:{hash}|{内容}``; 行号为 1-based int, hash 为两位大写字母.
        修改前必须先用本工具读取, 拿到 ``行号:hash`` 后才能调用 ``edit_solver``.

        Args:
            offset: 0-based 起始行 (分页用), 默认从头读.
            limit: 最多返回行数, 默认 2000 (solver.py 通常可一次读完).

        Returns:
            带 hashline 标记的文件内容; 文件不存在时返回错误说明.
        """
        if not solver_path.exists():
            return (
                "Error: 未找到 ./workdir/solver.py. 正常情况下 scaffold 已生成该文件, "
                "请确认任务初始化是否成功."
            )
        content = solver_path.read_text()
        return hashline.format_hashline_output(content, offset=offset, limit=limit)

    return read_solver


def _render_syntax_error_window(new_text: str, err_line: int, ctx: int = 3) -> str:
    """把编辑后(未落盘)的 new_text 在出错行上下各 ``ctx`` 行带行号渲染出来.

    SyntaxError 报的 ``err_line`` 是**编辑后坐标系**的行号, 而 agent 手上有效的
    行号:hash 与可 read 到的文件都还是**编辑前**的 (回滚了), 两个坐标系无锚点可对,
    agent 因此定位不到自己 new_content 里哪行错. 这里直接展示编辑后的出错窗口,
    让 agent 眼见为实——无论错因是全角标点/漏 `#`/结构断裂都通用.

    刻意**不带 hash**: 这份内容并未落盘, 给 hash 会诱导 agent 拿假 hash 去编辑.
    """
    lines = new_text.split("\n")
    lo = max(1, err_line - ctx)
    hi = min(len(lines), err_line + ctx)
    width = len(str(hi))
    out = []
    for n in range(lo, hi + 1):
        mark = ">>" if n == err_line else "  "
        out.append(f"  {mark} {str(n).rjust(width)}| {lines[n - 1]}")
    return "\n".join(out)


def build_edit_solver_tool(task_dir: Path):
    """构造绑定到 ``task_dir`` 的 ``edit_solver`` 函数, 传给 ``Tool(...)``."""
    solver_path = _solver_path(task_dir)

    def edit_solver(
        start_line: int,
        start_hash: str,
        new_content: str,
        end_line: int | None = None,
        end_hash: str | None = None,
        insert_after: bool = False,
    ) -> str:
        """编辑 ``./workdir/solver.py`` (唯一可写文件), 通过 ``行号:hash`` 定位.

        必须先用 ``read_solver`` 读取, 拿到 ``行号:hash`` 标记后再调用本工具.
        ``start_line``/``start_hash`` 必须与读取结果严格对应. 范围编辑时 ``end_hash``
        最易取错 (常误填成邻近行的 hash); 若失配, 工具会直接告知该行真实 hash,
        据此改正重发即可, 通常无需 re-read.

        **编辑成功后, 本工具会直接回传编辑区域附近的最新 ``行号:hash`` 快照**, 你可
        据此继续编辑该区域, 无需再次 read_solver. 仅当要编辑的目标行不在返回的快照
        范围内时, 才需重新 read_solver 取该处的最新 hash. 多处编辑建议**从下往上**改
        (先改大行号, 小行号的 hash 不受影响).

        操作语义:
        - 替换单行: start_line + start_hash + new_content
        - 替换范围: 再加 end_line + end_hash (含两端)
        - 在某行后插入: start_line + start_hash + insert_after=True + new_content
        - 删除: new_content="" (置空)

        注意: 向 py 中加注释时行内不要包含换行符 ``\\n``, 否则被注释内容会进入下一行
        导致语法错误.

        **重要**: 编辑的是 Python 源码, **不是 Markdown**. 文件里 ``## xxx`` 是 Python
        注释行, 不要把它当 Markdown标题, 进而把后续注释行写成无 ``#`` 前缀的纯文本——那会变成裸语句, 导致
        SyntaxError. 在 ``new_content`` 中写多行注释时, **每一行都必须以 ``#`` 开头**.
        例如 ``# 说明1\\n# 说明2``

        Args:
            start_line: 1-based 起始行号 (来自 read_solver).
            start_hash: 起始行的两位字母 hash (来自 read_solver).
            new_content: 替换文本; 置空字符串表示删除.
            end_line: 范围替换时的结束行号 (含), 单行编辑留空.
            end_hash: 范围替换时结束行的 hash.
            insert_after: True 表示在 start_line 之后插入, 而非替换.

        Returns:
            成功时返回编辑摘要; 失败 (hash mismatch / 越界) 时返回错误说明.
        """
        if not solver_path.exists():
            return "Error: 未找到 ./workdir/solver.py, 无法编辑."
        content = solver_path.read_text()
        new_text, error, summary = hashline.apply_hashline_edit_with_summary(
            content,
            start_line,
            start_hash,
            new_content,
            end_line=end_line,
            end_hash=end_hash,
            insert_after=insert_after,
        )
        if error is not None:
            return _rewrite_hash_mismatch(error)

        # 写入前先做 Python 语法体检; 失败回滚, 把错误位置 + 提示回给 agent.
        # ast.parse 只会报第一处 SyntaxError (parser 一遇错即停), 但常见的
        # "## 注释行下方漏 #" 通常是同类错, agent 改一处即可继续, 多轮改完.
        try:
            ast.parse(new_text, filename=str(solver_path))
        except SyntaxError as e:
            err_line = e.lineno or 0
            window = _render_syntax_error_window(new_text, err_line) if err_line else ""
            return (
                f"Error: 编辑被回滚 — solver.py 出现 SyntaxError (line {e.lineno}, "
                f"col {e.offset}): {e.msg}\n"
                f"以下是【你这次编辑若应用后】出错行附近的内容 (行号为编辑后坐标, "
                f"仅供定位; 文件并未真正改动):\n"
                f"{window}\n"
                f"提示:\n"
                f"  - 上面 `>>` 标出的就是报错行, 对照你提交的 new_content 修正后重发;\n"
                f"  - solver.py 是 Python 不是 Markdown, 多行注释每行都要以 `#` 开头;\n"
                f"  - 常见错因: 注释/字符串里混入全角标点(，、：等, 应改半角), 或注释行漏 `#`, "
                f"或范围替换时漏写了后续代码(如 result= / to_csv) 导致结构断裂;\n"
                f"  - 仅报告第一处错误, 你的编辑里可能有多处同类错, 一并修复;\n"
                f"  - 文件已回滚, 磁盘内容未变, 旧 行号:hash 仍然有效, 无需 re-read, "
                f"修正 new_content 后用原 hash 重发即可."
            )

        solver_path.write_text(new_text)

        # 回传编辑区域附近的最新 行号:hash 快照, 让 agent 无需 re-read 即可继续编辑,
        # 从根本上消除"凭旧 hash 连续编辑 -> hash mismatch"死循环.
        # 新内容行数: 删除为 0, 否则按 new_content 实际行数(与底层 split('\n') 一致).
        n_new = 0 if not new_content else len(new_content.split("\n"))
        if insert_after:
            # 插入在 start_line 之后, 新行占据 start_line+1 .. start_line+n_new.
            region_start = start_line + 1
            region_end = start_line + n_new
        else:
            # 替换 start_line..effective_end, 新行占据 start_line .. start_line+n_new-1.
            region_start = start_line
            region_end = start_line + max(n_new, 1) - 1
        offset = max(0, region_start - 1 - _EDIT_SNAPSHOT_CONTEXT)
        limit = (region_end - region_start + 1) + 2 * _EDIT_SNAPSHOT_CONTEXT
        snapshot = hashline.format_hashline_output(new_text, offset=offset, limit=limit)
        return (
            f"Edited ./workdir/solver.py: {summary}\n"
            f"--- 编辑后该区域最新 行号:hash (可据此继续编辑, 无需 re-read) ---\n"
            f"{snapshot}"
        )

    return edit_solver
