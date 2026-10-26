"""scaffold_tool.py — Generate a thin solver.py scaffold for a task directory.

Given a task directory (containing ``context/`` and ``task.json``), this writes
``workdir/solver.py`` with a **minimal** skeleton:

- Imports the shared data-source runtime (``tools_v2.datasource_runtime``).
- Builds a unified DuckDB state over all csv/json/sqlite sources (same code path
  as the ``explore_data`` tool → "探查通过即执行通过").
- Exposes ``run_sql(query)`` with built-in SQL dialect + column-space normalization.
- Prints a compact schema overview on first run (table + ``col:type``).
- A ``## 进行查询`` stub for the agent to fill in, and the final
  ``pred_df.to_csv(...)`` save line.

历史上 scaffold 会逐文件生成 ``pd.read_csv`` / ``json.loads`` / ``duckdb ATTACH``
读数据 block, 并内联手抄一份 SQL 方言归一 helper (与 ``tools_v2.naming`` 重复).
现已全部收口到 ``datasource_runtime``: solver.py 子进程能 import 它
(``src/data_agent_baseline`` 在 PYTHONPATH 上, 本地与 Docker 均已验证), 故 scaffold
只需生成几行胶水代码, 不再重复读数据/方言归一逻辑.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools_v2.datasource_runtime import build_datasource_state


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------


def _collect_table_names(task_dir: Path) -> list[str]:
    """构建一次统一 state, 收集 canonical 表名 + ``df_`` 别名, 供注入 query 段注释.

    与 solver.py 运行时用的是同一函数 (build_datasource_state), 故这里列出的表名
    与 solver 实际可查的表名完全一致. 失败时返回空列表 (不阻塞 scaffold 生成).
    """
    try:
        state = build_datasource_state(task_dir)
    except Exception:
        return []
    names: list[str] = []
    for t in state.tables:
        names.append(t.canonical)
        # 只额外列出 df_ 风格别名 (agent 习惯), 其余原名/大小写别名不刷屏.
        for a in t.aliases:
            if a == f"df_{t.canonical}":
                names.append(a)
    try:
        state.conn.close()
    except Exception:
        pass
    return names


def _render_scaffold(task_dir: Path, question: str) -> str:
    table_names = _collect_table_names(task_dir)
    tables_comment = ", ".join(table_names) if table_names else "(运行 solver.py 首次执行查看 schema 速览)"

    lines = [
        "# this file path is `./workdir/solver.py`",
        "from pathlib import Path",
        "import pandas as pd",
        "",
        "# 数据源统一读取 + SQL 方言归一由共享运行时提供, 与 explore_data 工具同一引擎",
        "# 同一列名归一逻辑 —— 在 explore_data 里验证通过的 SQL, 写进 run_sql 即可执行通过.",
        "from tools_v2.datasource_runtime import build_datasource_state, run_sql as _run_sql",
        "",
        "",
        "## 输入输出路径, 不要修改",
        "input_base = Path('./context')   # use this, don't change",
        "output_base = Path('./workdir')  # use this, don't change",
        "",
        "",
        "## 读取数据源 (csv/json/sqlite 全部注册为 DuckDB 视图, 表名见下方 schema 速览)",
        "## 不要修改这两行, 也不要再手写 pd.read_csv / sqlite3 直读原始文件.",
        "_state = build_datasource_state(Path('.').resolve())  # task_dir = 当前工作目录",
        "",
        "",
        "def run_sql(query):",
        '    """执行 SQL 返回完整 DataFrame; 自动处理引号方言(反引号/方括号->双引号)',
        '    与列名多余/缺失空格(归一到真实列名). 直接写真实表名/列名即可.',
        '    表名用 scaffold 注册的名字 (canonical 或 df_<名>), 见下方 schema 速览."""',
        "    return _run_sql(_state, query)",
        "",
        "",
        "## 数据 schema 速览（仅首次执行打印: 表名 + 列名:类型）",
        "## 需要某表的样本/空值分布时, 用 explore_data 工具的 `\\schema <表名>` 按需查",
        "## 不需要修改本段, 也不要删除 workdir/_schema_shown.flag",
        "_COL_MAX = 80  # 单列 '列名:类型' 最大字符; 仅截断过长的个别列, 不影响同行其他列",
        "_schema_flag = Path(__file__).resolve().parent / '_schema_shown.flag'  # 与 solver.py 同目录",
        "if not _schema_flag.exists():",
        "    for _t in _state.tables:",
        "        try:",
        "            _info = _state.conn.execute(f'DESCRIBE \"{_t.canonical}\"').fetchall()",
        "        except Exception as _e:",
        "            print(f'{_t.canonical}: <describe failed: {_e}>'); continue",
        "        _parts = []",
        "        for _row in _info:",
        "            _c, _ty = str(_row[0]), str(_row[1])",
        "            _entry = f'{_c}:{_ty}'",
        "            if len(_entry) > _COL_MAX:",
        "                _entry = _entry[:_COL_MAX] + f'…(此列名共{len(_entry)}字符,已截断{len(_entry)-_COL_MAX})'",
        "            _parts.append(_entry)",
        "        _alias = f\" [别名 df_{_t.canonical}]\" if f'df_{_t.canonical}' in _t.aliases else ''",
        "        print(f'{_t.canonical} ({_t.source_type}, {len(_parts)}cols){_alias}: ' + ', '.join(_parts))",
        "    _schema_flag.write_text('1')  # 标记已展示, 后续执行不再重复打印",
        "",
        "",
        "## 进行查询",
        f"### question: {question}",
        f"### 可用表 (run_sql 内直接用这些名字): {tables_comment}",
        "### TODO: 根据 question 和 knowledge.md 理解任务, 结合数据源, 完成查询逻辑.",
        "### 在此处用注释写下分析过程: 每个字段的含义/来源/join 逻辑/过滤与聚合粒度/空值处理.",
        '### 写法示例: result = run_sql("""SELECT ... FROM df_xxx WHERE ...""")',
        "result = None  # TODO: 替换为上面示例形式的真实查询",
        "",
        "",
        "## 数据字段转换(如有必要) 根据 question 要求对结果进行转化，输出所需字段",
        "",
        "",
        "## 保存预测结果",
        "pred_df = result  # TODO: 替换为最终结果 DataFrame",
        "if pred_df is None:",
        "    # 查询尚未实现: 首次执行/未填 SQL 时走这里, 不写 prediction.csv, 也不报错.",
        "    print('[solver] 查询尚未实现 (result is None): 请在 \"## 进行查询\" 段补全 SQL 后再运行; 本次未写出 prediction.csv')",
        "else:",
        "    pred_df.to_csv(output_base / 'prediction.csv', index=False)  # don't change",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_solver_scaffold(task_dir: str | Path) -> Path:
    """Inspect *task_dir* and write a thin ``workdir/solver.py``.

    Args:
        task_dir: Absolute (or relative) path to the task folder, which must
                  contain ``context/`` and ``task.json``.

    Returns:
        The ``Path`` of the written ``solver.py`` file.

    Raises:
        FileNotFoundError: If *task_dir* or ``context/`` does not exist.
    """
    task_dir = Path(task_dir).resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task_dir not found: {task_dir}")

    ctx = task_dir / "context"
    if not ctx.is_dir():
        raise FileNotFoundError(f"context/ not found under {task_dir}")

    # Read question from task.json (best-effort)
    task_json = task_dir / "task.json"
    question = "(see task.json)"
    if task_json.is_file():
        try:
            td = json.loads(task_json.read_text(encoding="utf-8"))
            question = td.get("question", question)
        except Exception:
            pass

    # Ensure workdir exists
    workdir = task_dir / "workdir"
    workdir.mkdir(exist_ok=True)

    code = _render_scaffold(task_dir, question)

    solver_path = workdir / "solver.py"
    solver_path.write_text(code, encoding="utf-8")
    return solver_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scaffold_tool.py <task_dir> [task_dir ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        out = generate_solver_scaffold(arg)
        print(f"Written: {out}")
        print(out.read_text())
        print("-" * 60)
