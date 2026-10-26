"""datasource_runtime.py — 数据源统一读取 + SQL 执行的**共享运行时**.

把 csv / json / sqlite 三种数据源统一注册成同一个 DuckDB 连接里的 view
(带 ``df_<stem>`` / 裸名 / 原大小写 多 alias), 并提供经过 SQL 方言归一化的
``run_sql`` 执行入口.

为什么独立成模块
----------------
``explore_tool`` (给 agent 的只读探查工具) 与 scaffold 生成的 ``solver.py``
(最终产出 prediction.csv 的脚本) **必须走同一段读数据 + 方言归一逻辑**, 否则
"explore 验证通过但 solver 执行报错" 的漂移会反复出现. 历史上 solver.py 内联
手抄了一份方言归一 helper (反引号/方括号→双引号、列名空格归一), 与
``tools_v2.naming`` 的实现逐行重复, 极易漂移.

实测确认 solver.py 子进程 (workdir 下 ``python workdir/solver.py``) 能 import
本模块: ``src/data_agent_baseline`` 在 PYTHONPATH 上 (run_agent_n_times.sh /
Dockerfile 都设置了). 因此两边都 ``from tools_v2.datasource_runtime import ...``,
共用同一份代码, parity 由"同一段代码"保证, 而非靠注释提醒.

线程安全
--------
DuckDB in-process connection **不是线程安全的**. 同一 task 一轮里 LLM 可能并发
发多个 explore_data 工具调用 (各跑在一个 anyio worker thread 上) 命中同一
connection. 所有 ``conn.execute`` 必须持 ``DatasourceState._conn_lock``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from tools_v2.naming import (
    stem_to_ident,
    rewrite_quoted_columns,
    normalize_quote_dialect,
)


# ---------------------------------------------------------------------------
# 标识符 / 字符串字面量 quote
# ---------------------------------------------------------------------------


def _stem_to_ident(stem: str) -> str:
    """把文件名 stem 转成合法 SQL/Python 标识符 (lower).

    统一命名规则收口在 ``tools_v2.naming.stem_to_ident`` (与 scaffold_tool 共用),
    保证 explore 注册的表名与 solver.py 的 DataFrame 变量名一致, 避免错配.
    """
    return stem_to_ident(stem)


def _quote(ident: str) -> str:
    """DuckDB / SQL 标识符 quote, 用双引号 + 双写 escape."""
    return '"' + ident.replace('"', '""') + '"'


def _sql_str(s: str) -> str:
    """把 Python str 转成 SQL 字符串字面量."""
    return "'" + s.replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class RegisteredTable:
    """注册到 DuckDB 的一张逻辑表."""

    canonical: str          # 主名, 例如 ``members`` 或 ``postHistory__users``
    aliases: list[str]      # 额外 alias, 例如 [``df_members``, ``Members``]
    source_type: str        # ``csv`` / ``json`` / ``sqlite``
    source_rel: str         # 相对 task_dir 的路径
    sqlite_table: str | None = None  # SQLite 内的原表名 (仅 sqlite 类型)


@dataclass
class DatasourceState:
    conn: duckdb.DuckDBPyConnection
    tables: list[RegisteredTable] = field(default_factory=list)
    task_dir: Path = field(default_factory=Path)
    # 持有 pandas DF 引用, 防止被 GC 后 DuckDB 拿到野指针.
    _df_refs: dict[str, pd.DataFrame] = field(default_factory=dict)
    # 保护 conn 的并发访问 (DuckDB in-process connection 不是线程安全的).
    _conn_lock: threading.Lock = field(default_factory=threading.Lock)
    # 所有真实列名(跨表合并去重), 懒加载缓存, 用于列名空格归一化改写.
    _all_columns: set[str] | None = None

    def all_columns(self) -> set[str]:
        """收集所有已注册表的真实列名(去重), 供 rewrite_quoted_columns 使用.

        必须持 ``_conn_lock``: DuckDB in-process connection 不是线程安全的, 而同一
        task 一轮里 LLM 可能发多个 explore_data 并行工具调用(各跑在一个 anyio worker
        thread 上)并发命中同一 connection. 历史上这里裸调 conn.execute 不加锁, 两个
        worker 同时进来会把 DuckDB 内部状态搞坏 → 卡死在 native 里(py-spy 实测).
        懒加载缓存 ``_all_columns`` 也一并在锁内填, 避免并发重复构建/竞态.
        """
        with self._conn_lock:
            if self._all_columns is None:
                cols: set[str] = set()
                for t in self.tables:
                    try:
                        info = self.conn.execute(f"DESCRIBE {_quote(t.canonical)}").fetchall()
                        cols.update(r[0] for r in info)
                    except Exception:  # noqa: BLE001 — 单表失败不影响其它
                        pass
                self._all_columns = cols
            return self._all_columns


# ---------------------------------------------------------------------------
# 注册逻辑
# ---------------------------------------------------------------------------


def _register_csv(state: DatasourceState, p: Path) -> None:
    rel = str(p.relative_to(state.task_dir))
    canonical = _stem_to_ident(p.stem)
    aliases = _build_aliases(canonical, p.stem, used=_used_names(state))
    state.conn.execute(
        f"CREATE OR REPLACE VIEW {_quote(canonical)} AS "
        f"SELECT * FROM read_csv_auto({_sql_str(str(p))})"
    )
    for a in aliases:
        state.conn.execute(
            f"CREATE OR REPLACE VIEW {_quote(a)} AS SELECT * FROM {_quote(canonical)}"
        )
    state.tables.append(
        RegisteredTable(
            canonical=canonical, aliases=aliases, source_type="csv", source_rel=rel
        )
    )


def _register_json(state: DatasourceState, p: Path) -> None:
    """JSON 已统一为 ``{"table": ..., "records": [...]}`` 格式."""
    rel = str(p.relative_to(state.task_dir))
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("datasource_runtime: skip json {} ({})", rel, e)
        return
    if not (isinstance(data, dict) and isinstance(data.get("records"), list)):
        logger.warning("datasource_runtime: skip json {} (not records-shaped)", rel)
        return
    table_name = data.get("table") or p.stem
    canonical = _stem_to_ident(table_name)
    aliases = _build_aliases(canonical, table_name, p.stem, used=_used_names(state))

    df = pd.DataFrame(data["records"])
    # DuckDB register 名字必须唯一, 用 canonical
    register_name = f"_explore_json_{canonical}"
    state._df_refs[register_name] = df
    state.conn.register(register_name, df)
    state.conn.execute(
        f"CREATE OR REPLACE VIEW {_quote(canonical)} AS SELECT * FROM {_quote(register_name)}"
    )
    for a in aliases:
        state.conn.execute(
            f"CREATE OR REPLACE VIEW {_quote(a)} AS SELECT * FROM {_quote(canonical)}"
        )
    state.tables.append(
        RegisteredTable(
            canonical=canonical, aliases=aliases, source_type="json", source_rel=rel
        )
    )


def _register_sqlite(state: DatasourceState, p: Path) -> None:
    rel = str(p.relative_to(state.task_dir))
    db_stem = _stem_to_ident(p.stem)
    # 列出所有表
    try:
        conn_ro = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        rows = conn_ro.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        sqlite_tables = [r[0] for r in rows]
        conn_ro.close()
    except Exception as e:
        logger.warning("datasource_runtime: skip sqlite {} ({})", rel, e)
        return

    # ATTACH 这个 db (DuckDB sqlite scanner)
    attach_alias = f"sqlite_{db_stem}"
    try:
        state.conn.execute("INSTALL sqlite")
        state.conn.execute("LOAD sqlite")
        state.conn.execute(
            f"ATTACH {_sql_str(str(p))} AS {_quote(attach_alias)} (TYPE SQLITE, READ_ONLY)"
        )
    except duckdb.Error as e:
        # 已经 attach 过的话忽略
        if "already" not in str(e).lower():
            logger.warning("datasource_runtime: ATTACH sqlite {} failed: {}", rel, e)
            return

    used = _used_names(state)
    used_cf = {u.casefold() for u in used}
    for t in sqlite_tables:
        # 同名时 (case-insensitive) 主名加 db 前缀, 但仍尝试给一个裸 alias
        bare = _stem_to_ident(t)
        if bare.casefold() in used_cf:
            canonical = f"{db_stem}__{bare}"
            extra_aliases = []  # 裸名已被占, 不再 alias
        else:
            canonical = bare
            extra_aliases = []
        aliases = _build_aliases(canonical, t, used=used) + extra_aliases
        used_cf = {u.casefold() for u in used}  # 同步

        qualified = f"{_quote(attach_alias)}.{_quote(t)}"
        state.conn.execute(
            f"CREATE OR REPLACE VIEW {_quote(canonical)} AS SELECT * FROM {qualified}"
        )
        for a in aliases:
            state.conn.execute(
                f"CREATE OR REPLACE VIEW {_quote(a)} AS SELECT * FROM {_quote(canonical)}"
            )
        state.tables.append(
            RegisteredTable(
                canonical=canonical,
                aliases=aliases,
                source_type="sqlite",
                source_rel=rel,
                sqlite_table=t,
            )
        )


def _build_aliases(canonical: str, *original_names: str, used: set[str]) -> list[str]:
    """生成 alias 列表 (df_<canonical> + 各原名 + 原名小写), 跳过冲突.

    重要: DuckDB 对未加引号的标识符 case-insensitive 解析, 所以即便
    ``CREATE VIEW "Examination"`` 和 ``CREATE VIEW "examination"`` 在 catalog
    中是两个独立条目, resolve ``"Examination"`` 时也会同时匹配到 ``examination``,
    再加上 alias 本身指向 canonical, 会触发"recursive bind view"异常.
    所以这里用 ``casefold`` 做去重判断.
    """
    canonical_cf = canonical.casefold()
    used_cf = {u.casefold() for u in used}

    candidates: list[str] = []
    candidates.append(f"df_{canonical}")
    for name in original_names:
        if not name:
            continue
        if name.casefold() == canonical_cf:
            continue
        candidates.append(name)
        low = name.lower()
        if low != name:
            candidates.append(low)

    out: list[str] = []
    seen_cf: set[str] = set()
    for c in candidates:
        if not c:
            continue
        c_cf = c.casefold()
        if c_cf == canonical_cf or c_cf in used_cf or c_cf in seen_cf:
            continue
        seen_cf.add(c_cf)
        out.append(c)
    used.update(out)
    used.add(canonical)
    return out


def _used_names(state: DatasourceState) -> set[str]:
    """所有已注册名字 (canonical + aliases). 调用方做 case-insensitive 判断时
    应自行 ``.casefold()`` 一遍 —— DuckDB 的标识符默认大小写不敏感, 必须按
    casefold 去重, 否则 ``Examination`` / ``examination`` 会自我递归引用."""
    used: set[str] = set()
    for t in state.tables:
        used.add(t.canonical)
        used.update(t.aliases)
    return used


# ---------------------------------------------------------------------------
# 状态构造
# ---------------------------------------------------------------------------


def build_datasource_state(task_dir: Path) -> DatasourceState:
    """扫描 ``task_dir/context/`` 下 csv/json/db/sqlite, 全部注册到一个新的
    DuckDB in-memory 连接, 返回 :class:`DatasourceState`.

    注意: 本函数不做进程级缓存 (explore_tool 侧有自己的 ``_get_state`` 缓存;
    solver.py 每次执行只建一次). 重复调用会建多个独立连接.
    """
    conn = duckdb.connect(":memory:")
    state = DatasourceState(conn=conn, task_dir=task_dir)

    ctx = task_dir / "context"
    if not ctx.is_dir():
        return state

    csvs: list[Path] = []
    if (ctx / "csv").is_dir():
        csvs += sorted((ctx / "csv").glob("*.csv"))
    csvs += sorted(ctx.glob("*.csv"))

    jsons: list[Path] = []
    if (ctx / "json").is_dir():
        jsons += sorted((ctx / "json").glob("*.json"))
    jsons += sorted(ctx.glob("*.json"))

    dbs: list[Path] = []
    if (ctx / "db").is_dir():
        dbs += sorted((ctx / "db").glob("*.db")) + sorted((ctx / "db").glob("*.sqlite"))
    dbs += sorted(ctx.glob("*.db")) + sorted(ctx.glob("*.sqlite"))

    for p in csvs:
        try:
            _register_csv(state, p)
        except Exception as e:
            logger.warning("datasource_runtime: register csv {} failed: {}", p, e)
    for p in jsons:
        try:
            _register_json(state, p)
        except Exception as e:
            logger.warning("datasource_runtime: register json {} failed: {}", p, e)
    for p in dbs:
        try:
            _register_sqlite(state, p)
        except Exception as e:
            logger.warning("datasource_runtime: register sqlite {} failed: {}", p, e)

    return state


# ---------------------------------------------------------------------------
# SQL 方言归一化 + 执行
# ---------------------------------------------------------------------------

# 元数据语句首关键词: 这些语句不涉及列引用, 跳过列名空格归一化.
_META_STMT_KEYWORDS = ("describe", "desc", "pragma", "show", "explain")


def _is_meta_stmt(sql: str) -> bool:
    """判断 SQL 是否为元数据语句 (DESCRIBE/PRAGMA/SHOW/EXPLAIN).

    这类语句放进 ``WITH (...)`` / ``FROM (...)`` 会报错, 且不涉及列引用,
    故列名空格归一化要跳过它们.
    """
    s_no_comment = re.sub(r"/\*.*?\*/", "", re.sub(r"--[^\n]*", "", sql), flags=re.DOTALL)
    first_kw = s_no_comment.lstrip().split(None, 1)[0].lower() if s_no_comment.strip() else ""
    return first_kw in _META_STMT_KEYWORDS


def normalize_query(
    state: DatasourceState, sql: str
) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]]]:
    """对用户 SQL 做两步方言归一, 返回 ``(归一后 sql, quote_rewrites, col_rewrites)``.

    1. 引号方言归一: qwen3.5 常写 MySQL 反引号或方括号 ``[列]``
       (DuckDB 不支持, 反引号直接 ParserError). 统一改写成双引号标识符.
       必须在列名空格归一化**之前**做 (后者只认双引号标识符).
    2. 列名空格归一: LLM 常给中文列名多加/漏掉括号前空格 (``在任基金数 (只)``
       vs 真实 ``在任基金数(只)``), DuckDB 严格报 column not found. 把双引号标识符
       "去空格后唯一匹配真实列"的改写成真实列名. 元数据语句跳过.

    explore_tool 与 solver.py 共用本函数, 保证"探查通过即执行通过".
    """
    quote_rewrites: list[tuple[str, str]] = []
    sql, quote_rewrites = normalize_quote_dialect(sql)

    col_rewrites: list[tuple[str, str]] = []
    if not _is_meta_stmt(sql):
        sql, col_rewrites = rewrite_quoted_columns(sql, state.all_columns())

    return sql, quote_rewrites, col_rewrites


def run_sql(state: DatasourceState, query: str) -> pd.DataFrame:
    """执行 SQL 并返回完整 DataFrame (不分页/不加 LIMIT).

    供 solver.py 使用: 自动做引号方言归一 + 列名空格归一 (与 explore_data 完全
    一致). 出错时直接抛出底层异常 (solver agent 的 execute 工具会折叠 traceback).
    """
    sql, _, _ = normalize_query(state, query)
    with state._conn_lock:
        return state.conn.execute(sql).df()
