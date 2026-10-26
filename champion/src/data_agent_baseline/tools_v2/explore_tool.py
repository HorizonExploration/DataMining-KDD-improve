"""explore_tool.py — 只读 SQL 数据探查工具.

给 solver agent 一个轻量的"试错沙盒":在写 / 调 ``solver.py`` 之前(或卡住时)
直接对 ``context/`` 下的 csv / json / sqlite 用 SQL 做联合查询、看 schema、看样本、
验证 join 关系, 而不必每次都改 ``solver.py`` 跑全流程.

设计要点
--------
- 后端: DuckDB (进程内, 只读, 原生支持 csv / json / sqlite).
- 表注册: 自动扫描 ``task_dir/context/`` 下 ``*.csv / *.json / *.db / *.sqlite``;
  每张表同时注册多个 alias (裸名 + scaffold 风格 ``df_<stem>`` + 原大小写),
  与 ``solver.py`` / ``knowledge.md`` 的命名都对得上.
- SQLite 多 db 同名表: 主名加前缀 ``<dbstem>__<table>``, 仍然额外提供
  ``df_<table>`` alias (会在与其它源裸名冲突时跳过, 避免歧义).
- JSON: 已确认所有 JSON 都是 ``{"table": ..., "records": [...]}`` 格式, 直接
  ``json.loads + DataFrame(records)`` register 进 DuckDB.
- 安全: 只允许 SELECT / WITH / PRAGMA / DESCRIBE / SHOW 起头的 SQL;
  另支持两个伪命令: ``\\tables`` 列出所有表, ``\\schema <table>`` 看字段 + 5 行样本.
- 行数限制: 不分页, 只用一个 ``limit`` 参数, 默认 50, 最大 500.
- 单元格截断: 超过 ``max_cell_chars`` 的字符串截断, binary 转 ``<binary len=N>``.
- 缓存: 同一 ``task_dir`` 复用一个 DuckDB connection, 避免每次重新 register.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# 数据源读取 + SQL 方言归一已收口到共享运行时 datasource_runtime, 与 solver.py
# 共用同一份代码 (探查通过即执行通过). 本模块只保留"给 LLM 的工具展示层":
# 分页/LIMIT 包裹、count(*) total、schema/tables 伪命令、结果截断与格式化.
# 下列 re-export (_get_state / _quote 等) 供 table_profile.py 等下游沿用.
from tools_v2.datasource_runtime import (
    DatasourceState as _ExploreState,  # 历史名沿用, 减少本文件改动面
    RegisteredTable as _RegisteredTable,
    build_datasource_state,
    normalize_query,
    _quote,
    _sql_str,
    _stem_to_ident,
)


# ---------------------------------------------------------------------------
# 限制 / 默认值
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 50
MAX_LIMIT = 500
DEFAULT_MAX_CELL_CHARS = 200
SCHEMA_SAMPLE_ROWS = 5
QUERY_TIMEOUT_SECONDS = 30
# DuckDB 错误消息里常带一长串 "Candidate bindings"/"LINE 1: ..." echo,
# 对 LLM 是噪音, 截到这个长度.
MAX_ERROR_CHARS = 600


# 仅允许这些关键字开头的 SQL.
_ALLOWED_SQL_PREFIXES = ("select", "with", "pragma", "describe", "desc", "show", "explain")


def _format_db_error(e: BaseException, max_chars: int = MAX_ERROR_CHARS) -> str:
    """把 DuckDB / pandas 的长错误消息压缩到对 LLM 友好的长度.

    DuckDB 喜欢在错误里 echo 整段 SQL + 一大串 "Candidate bindings:" 列表
    (列名很多时几百行), 对 agent 是纯噪音. 这里:
    1. 只留第一段 (到第一个空行/双换行/`LINE` 字样为止).
    2. 把 "Candidate bindings: ..." 整段去掉, 只保留 "Did you mean X" 这一行.
    3. 整体硬截断.
    """
    msg = str(e).strip()
    # 去掉 LINE N: ... ^^^ 这样的 SQL 回显
    msg = re.sub(r"\nLINE \d+:.*?(?:\n\s*\^+)?", "", msg, flags=re.DOTALL)
    # 去掉 Candidate bindings / Candidate functions 长列表 (保留 "Did you mean ..." 行)
    msg = re.sub(
        r"\n\s*Candidate (?:bindings|functions):.*?(?=\n\n|\Z)",
        "",
        msg,
        flags=re.DOTALL,
    )
    # 多重空行收成一个
    msg = re.sub(r"\n\s*\n+", "\n", msg).strip()
    if len(msg) > max_chars:
        msg = msg[:max_chars] + f"... <truncated, full_len={len(str(e))}>"
    # strftime 方言陷阱: qwen3.5 常按 SQLite 写 strftime('%Y', 文本列), 但 DuckDB 的
    # strftime 第一参必须是 DATE/TIMESTAMP(非文本), 报 "Could not choose a best
    # candidate function for the function call strftime". 仅改参数顺序救不了(根因是
    # 文本列未转 DATE), 故在报错里直接给出 DuckDB 正确写法引导. (实测 prompt 对此弱效.)
    # 在截断之后追加, 保证提示不被切掉.
    low = msg.lower()
    if "strftime" in low and ("could not choose" in low or "candidate function" in low):
        msg += (
            "\nHINT: DuckDB 的 strftime 第一个参数必须是 DATE/TIMESTAMP 类型(不是文本), "
            "且参数顺序与 SQLite 相反. 取年份直接用 YEAR(TRY_CAST(\"日期列\" AS DATE)); "
            "需格式化用 strftime(TRY_CAST(\"日期列\" AS DATE), '%Y'); "
            "若日期是非 ISO 文本(如 20230501)先 try_strptime(\"日期列\", '%Y%m%d')."
        )
    return msg


# ---------------------------------------------------------------------------
# 工具内部状态 (按 task_dir 缓存 DuckDB 连接)
# ---------------------------------------------------------------------------
#
# 数据类 _RegisteredTable / _ExploreState 已移到 datasource_runtime
# (RegisteredTable / DatasourceState), 此处通过顶部 import 别名沿用旧名.


_state_cache: dict[str, _ExploreState] = {}
_state_lock = threading.Lock()



def _get_state(task_dir: Path) -> _ExploreState:
    key = str(task_dir.resolve())
    # 快速路径: 无锁读 (Python dict 读是原子的)
    st = _state_cache.get(key)
    if st is not None:
        return st
    # 慢路径: 先在锁外构建, 再持锁写入 (避免持锁期间做耗时的 INSTALL/ATTACH 操作)
    new_st = build_datasource_state(task_dir.resolve())
    with _state_lock:
        # 双重检查: 并发时可能已被其他线程写入
        st = _state_cache.get(key)
        if st is None:
            _state_cache[key] = new_st
            return new_st
        return st


# ---------------------------------------------------------------------------
# SQL 安全校验
# ---------------------------------------------------------------------------


def _validate_sql(sql: str) -> tuple[bool, str]:
    """只允许只读 SQL. 返回 (ok, reason)."""
    s = sql.strip()
    if not s:
        return False, "empty sql"
    # 简单 strip 注释
    s_no_comment = re.sub(r"--[^\n]*", "", s)
    s_no_comment = re.sub(r"/\*.*?\*/", "", s_no_comment, flags=re.DOTALL)
    head = s_no_comment.lstrip().split(None, 1)
    if not head:
        return False, "empty sql"
    first = head[0].lower()
    if first not in _ALLOWED_SQL_PREFIXES:
        return False, (
            f"only read-only SQL is allowed (got '{first}'); "
            f"allowed prefixes: {', '.join(_ALLOWED_SQL_PREFIXES)}"
        )
    # 黑名单关键词 (覆盖所有可能的写操作)
    banned = re.compile(
        r"\b(insert|update|delete|drop|create|alter|truncate|attach|detach|copy|"
        r"vacuum|reindex|grant|revoke|call|export|import|load|install)\b",
        re.IGNORECASE,
    )
    # PRAGMA / DESCRIBE / SHOW / SELECT 内部不应出现写操作
    if banned.search(s_no_comment):
        m = banned.search(s_no_comment)
        return False, f"forbidden keyword '{m.group(0)}' in SQL"
    return True, ""


# ---------------------------------------------------------------------------
# 单元格 / 行截断
# ---------------------------------------------------------------------------


def _truncate_value(v: Any, max_chars: int) -> Any:
    if isinstance(v, (bytes, bytearray, memoryview)):
        n = len(bytes(v))
        return f"<binary len={n}>"
    if isinstance(v, str) and len(v) > max_chars:
        return v[:max_chars] + f"...<truncated, full_len={len(v)}>"
    return v


def _is_na_scalar(val: Any) -> bool:
    """安全的标量 NA 判断: 对 list/dict/ndarray 等非标量(pd.isna 会返回数组或报错)
    一律视为非 NA, 避免 "truth value of an array is ambiguous"."""
    if isinstance(val, (list, dict, tuple, set, bytes, bytearray)):
        return False
    try:
        res = pd.isna(val)
    except (ValueError, TypeError):
        return False
    return bool(res) if not hasattr(res, "__len__") else False


def _df_to_records(df: pd.DataFrame, max_chars: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        rec = {}
        for col, val in zip(df.columns, row):
            if _is_na_scalar(val):
                rec[col] = None
            else:
                rec[col] = _truncate_value(val, max_chars)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# 伪命令 (\tables / \schema)
# ---------------------------------------------------------------------------


def _format_tables_listing(state: _ExploreState) -> str:
    if not state.tables:
        return "(no tables registered; check that context/ contains csv/json/db files)"
    lines = [f"# Registered tables ({len(state.tables)})", ""]
    for t in state.tables:
        try:
            n = state.conn.execute(
                f"SELECT count(*) FROM {_quote(t.canonical)}"
            ).fetchone()[0]
            n_str = str(n)
        except Exception as e:
            n_str = f"?? ({e})"
        cols_info = state.conn.execute(
            f"DESCRIBE {_quote(t.canonical)}"
        ).fetchall()
        col_names = [c[0] for c in cols_info]
        lines.append(
            f"- {t.canonical}  ({t.source_type}, rows={n_str}, cols={len(col_names)})  "
            f"<- {t.source_rel}"
            + (f"  [sqlite_table={t.sqlite_table}]" if t.sqlite_table else "")
        )
        if t.aliases:
            lines.append(f"    aliases: {', '.join(t.aliases)}")
        lines.append(f"    columns: {', '.join(col_names)}")
    return "\n".join(lines)


def _resolve_table_name(state: _ExploreState, name: str) -> str | None:
    """把用户给的 name (canonical 或 alias, case-insensitive) 解析回 canonical 名."""
    cf = name.casefold()
    for t in state.tables:
        if t.canonical.casefold() == cf:
            return t.canonical
        for a in t.aliases:
            if a.casefold() == cf:
                return t.canonical
    return None


def _format_schema(state: _ExploreState, table_name: str) -> str:
    # 先尝试解析成 canonical (允许传 alias / 大小写不一致)
    resolved = _resolve_table_name(state, table_name) or table_name
    try:
        cols_info = state.conn.execute(f"DESCRIBE {_quote(resolved)}").fetchall()
        n = state.conn.execute(f"SELECT count(*) FROM {_quote(resolved)}").fetchone()[0]
        sample_df = state.conn.execute(
            f"SELECT * FROM {_quote(resolved)} LIMIT {SCHEMA_SAMPLE_ROWS}"
        ).df()
    except Exception as e:  # noqa: BLE001 — 任何 DuckDB / pandas / pyarrow 异常都吞掉, 只回 LLM 看
        known = ", ".join(sorted(t.canonical for t in state.tables)) or "(none)"
        return (
            f"[error] failed to inspect table {table_name!r} (resolved to {resolved!r}): "
            f"{_format_db_error(e)}\n"
            f"  known tables: {known}\n"
            f"  hint: use \\tables to see all aliases."
        )
    out = [f"# Schema: {resolved}"]
    if resolved != table_name:
        out.append(f"  (resolved from alias {table_name!r})")
    out += [f"  rows: {n}", f"  columns ({len(cols_info)}):"]
    for c in cols_info:
        # DESCRIBE 返回: column_name, column_type, null, key, default, extra
        col_name, col_type = c[0], c[1]
        out.append(f"    - {col_name}  {col_type}")
    out.append(f"  sample (first {SCHEMA_SAMPLE_ROWS} rows):")
    sample_records = _df_to_records(sample_df, DEFAULT_MAX_CELL_CHARS)
    for i, r in enumerate(sample_records, 1):
        out.append(f"    row {i}: {json.dumps(r, ensure_ascii=False, default=str)}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def _run_explore(
    task_dir: Path,
    sql: str,
    limit: int = DEFAULT_LIMIT,
    max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
) -> str:
    state = _get_state(task_dir)

    # 限值
    if limit <= 0:
        limit = DEFAULT_LIMIT
    if limit > MAX_LIMIT:
        limit = MAX_LIMIT
    if max_cell_chars <= 0:
        max_cell_chars = DEFAULT_MAX_CELL_CHARS

    # 伪命令
    s = sql.strip()
    if s in (r"\tables", r"\dt", "\\tables", "\\dt"):
        with state._conn_lock:
            return _format_tables_listing(state)
    m = re.match(r"^\\schema\s+(\S+)\s*$", s)
    if m:
        with state._conn_lock:
            return _format_schema(state, m.group(1))

    ok, reason = _validate_sql(sql)
    if not ok:
        return f"[error] {reason}"

    # 取首关键词: DESCRIBE/DESC/PRAGMA/SHOW/EXPLAIN 是元数据语句, 不能作为
    # 子查询/CTE 的内容(放进 `WITH (...)` 或 `FROM (...)` 会报 "A CTE needs a
    # SELECT"). 这类语句行数本就少, 直接原样执行, 跳过分页+count 包装.
    # 只有 SELECT/WITH 才走 CTE 包一层加 LIMIT 的逻辑.
    s_no_comment = re.sub(r"/\*.*?\*/", "", re.sub(r"--[^\n]*", "", sql), flags=re.DOTALL)
    first_kw = s_no_comment.lstrip().split(None, 1)[0].lower() if s_no_comment.strip() else ""
    is_meta_stmt = first_kw in ("describe", "desc", "pragma", "show", "explain")

    # SQL 方言归一化 (引号方言 + 列名空格) 收口到共享 normalize_query, 与 solver.py
    # 的 run_sql 完全一致 —— 探查通过即执行通过. quote_rewrites / col_rewrites 用于
    # 下方 header 提示 agent "已自动改写", 让它在 solver.py 里直接用归一后的写法.
    sql, quote_rewrites, col_rewrites = normalize_query(state, sql)

    t0 = time.time()
    try:
        with state._conn_lock:
            if is_meta_stmt:
                # 元数据语句: 直接执行, 不包 CTE, 不算 total.
                df = state.conn.execute(sql).df()
                total = len(df)
            else:
                # SELECT / WITH: 用 CTE 包一层加 LIMIT (不论用户有没有写 LIMIT 都 work),
                # 并单独算 total 以便提示是否被截断.
                wrapped = f"WITH _user_q AS (\n{sql}\n) SELECT * FROM _user_q LIMIT {limit}"
                total = state.conn.execute(
                    f"SELECT count(*) FROM (\n{sql}\n) AS _cnt"
                ).fetchone()[0]
                df = state.conn.execute(wrapped).df()
    except Exception as e:  # noqa: BLE001 — duckdb / pandas / pyarrow 等都可能抛, 一律转给 LLM
        return f"[sql error] {_format_db_error(e)}"
    elapsed_ms = int((time.time() - t0) * 1000)

    records = _df_to_records(df, max_cell_chars)
    truncated_note = ""
    if total > len(records):
        truncated_note = (
            f"  WARNING: result has {total} rows, showing first {len(records)} (limit={limit}); "
            f"add WHERE/aggregation in SQL or raise limit (max {MAX_LIMIT})."
        )

    header = [
        f"# explore_data  rows_returned={len(records)}/{total}  elapsed_ms={elapsed_ms}",
        f"  columns ({len(df.columns)}): {', '.join(map(str, df.columns))}",
    ]
    if quote_rewrites:
        # 告诉 agent 反引号/方括号被自动转成双引号, 让它在 solver.py 里直接用双引号.
        fixes = "; ".join(f'{a}->{b}' for a, b in quote_rewrites)
        header.append(f"  NOTE: 标识符引号已归一(DuckDB 只用双引号, 勿用反引号/方括号): {fixes}")
    if col_rewrites:
        # 告诉 agent 列名被自动纠正, 让它在 solver.py 里直接用真实列名(逐字, 勿加空格).
        fixes = "; ".join(f'"{a}"->"{b}"' for a, b in col_rewrites)
        header.append(f"  NOTE: 列名空格已自动归一(请用真实列名): {fixes}")
    if truncated_note:
        header.append(truncated_note)
    body = [
        f"  row {i}: {json.dumps(r, ensure_ascii=False, default=str)}"
        for i, r in enumerate(records, 1)
    ]
    if not body:
        body = ["  (no rows)"]
    return "\n".join(header + body)


# ---------------------------------------------------------------------------
# 暴露给 pydantic_ai 的 tool factory
# ---------------------------------------------------------------------------


def build_explore_tool(task_dir: Path):
    """构造一个绑定到 ``task_dir`` 的 ``explore_data`` 函数, 可直接传给 ``Tool(...)``.

    用法 (在 ``solver_agent.py``)::

        from pydantic_ai import Tool
        from tools_v2.explore_tool import build_explore_tool
        explore_data = build_explore_tool(task_dir)
        agent = create_deep_agent(..., tools=[Tool(explore_data, takes_ctx=False)])
    """
    task_dir = Path(task_dir).resolve()

    def explore_data(
        sql: str,
        limit: int = DEFAULT_LIMIT,
        max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
    ) -> str:
        """对当前任务的 csv / json / sqlite 数据源执行只读 SQL, 用于探查 schema / 样本 / 联合查询.

        所有 ``context/`` 下的 csv / json / db 都已经按 scaffold 命名注册为 DuckDB 视图.
        每张表同时支持多个 alias: 裸名 (members)、scaffold 风格 (df_members)、原大小写 (Members).

        Args:
            sql: 只读 SQL, 必须以 SELECT/WITH/PRAGMA/DESCRIBE/SHOW/EXPLAIN 开头.
                 也支持两个伪命令:
                 - ``\\tables`` —— 列出所有已注册表 (含行数/列名/出处).
                 - ``\\schema <table>`` —— 看某张表的字段 + 5 行样本.
            limit: 最多返回多少行, 默认 50, 最大 500. 总行数会一并返回, 便于判断结果是否被截断.
            max_cell_chars: 单元格字符串截断阈值, 默认 200.

        Returns:
            人类可读的字符串, 含列名 / 截断提示 / 各行 JSON.
            出错时以 ``[error] ...`` / ``[sql error] ...`` 开头.
        """
        return _run_explore(task_dir, sql, limit=limit, max_cell_chars=max_cell_chars)

    return explore_data


# ---------------------------------------------------------------------------
# CLI 调试
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python explore_tool.py <task_dir> <sql>")
        sys.exit(1)
    td = Path(sys.argv[1])
    q = sys.argv[2]
    fn = build_explore_tool(td)
    print(fn(q))
