"""table_profile.py — 结构化表的"列画像"预计算.

给 plan agent 的**静态版**(无 explore_data 工具)用: 把 agent 原本要靠多轮 explore_data
才能查到的"决策性统计"离线算好, 一次性注入 context, 让 plan 单轮产出, 省掉 ReAct 往返.

覆盖 explore_data 的三大高价值用途 (见 plan-agent-eval 记忆):
- **去重粒度**: 每列 distinct 数 vs 总行数 → 决定 count(*) 还是 count(distinct).
- **单位/量纲**: 数值列 min/max → 看数量级是否自洽, 是否需 ×100/×1e6 换算.
- **空值/孤儿行**: 每列非空计数 → doc 抽取表常有"只含 ID 的叙述片段行".

复用 ``explore_tool._get_state``: csv/json/sqlite + doc 抽取库 ``_doc_extracted.db`` 都已
注册成 DuckDB 视图, 这里只补一层"每表一条聚合 SQL"算画像, 不重复注册.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from tools_v2.explore_tool import _get_state, _quote


# 这些 DuckDB 类型视为数值列, 额外算 MIN/MAX (用于量纲判断).
_NUMERIC_TYPE_TOKENS = (
    "INT", "DECIMAL", "NUMERIC", "DOUBLE", "FLOAT", "REAL", "HUGEINT", "BIGINT",
)

# 单列样例值预览的最大字符数.
_MAX_EXAMPLE_CHARS = 40

# distinct 数过大时只标注 "≈唯一" 的阈值比例 (distinct/total >= 此值视为近似唯一键).
_NEAR_UNIQUE_RATIO = 0.99


def _is_numeric(duckdb_type: str) -> bool:
    t = (duckdb_type or "").upper()
    return any(tok in t for tok in _NUMERIC_TYPE_TOKENS)


def _fmt_num(v: object) -> str:
    """把 min/max 数值渲染得紧凑可读 (整数不带小数, 浮点保留有效信息)."""
    if v is None:
        return "?"
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


def _profile_one_table(state, canonical: str) -> str | None:
    """对一张已注册表算列画像, 返回 markdown 片段; 失败返回 None.

    一表一条聚合 SQL: 总行数 + 每列 (distinct 数, 非空数) + 数值列 (min, max).
    用列序号做别名 (c0__nd 等), 避免中文/特殊列名做别名时的转义问题.
    """
    conn = state.conn
    try:
        with state._conn_lock:
            cols_info = conn.execute(f"DESCRIBE {_quote(canonical)}").fetchall()
    except Exception as e:  # noqa: BLE001
        logger.debug("[table_profile] describe {} failed: {}", canonical, e)
        return None

    # cols_info: (column_name, column_type, null, key, default, extra)
    columns = [(c[0], c[1]) for c in cols_info]
    if not columns:
        return None

    exprs = ["COUNT(*) AS _total"]
    meta: list[tuple[int, str, str, bool]] = []  # (idx, name, type, is_numeric)
    for i, (name, typ) in enumerate(columns):
        q = _quote(name)
        numeric = _is_numeric(typ)
        meta.append((i, name, typ, numeric))
        exprs.append(f'COUNT(DISTINCT {q}) AS "c{i}__nd"')
        exprs.append(f'COUNT({q}) AS "c{i}__nn"')
        if numeric:
            exprs.append(f'MIN({q}) AS "c{i}__min"')
            exprs.append(f'MAX({q}) AS "c{i}__max"')

    sql = f'SELECT {", ".join(exprs)} FROM {_quote(canonical)}'
    try:
        with state._conn_lock:
            row = conn.execute(sql).fetchdf().iloc[0].to_dict()
            # 一行样例值 (供 agent 看真实格式, 如日期/编码形态)
            ex_row = conn.execute(
                f"SELECT * FROM {_quote(canonical)} LIMIT 1"
            ).fetchdf()
    except Exception as e:  # noqa: BLE001
        logger.debug("[table_profile] profile {} failed: {}", canonical, e)
        return None

    total = int(row.get("_total", 0) or 0)
    examples = ex_row.iloc[0].to_dict() if not ex_row.empty else {}

    lines = [f"### {canonical}  (rows={total}, cols={len(columns)})"]
    for i, name, typ, numeric in meta:
        nd = row.get(f"c{i}__nd")
        nn = row.get(f"c{i}__nn")
        nd = int(nd) if nd is not None else 0
        nn = int(nn) if nn is not None else 0
        nulls = total - nn

        parts = [f"distinct={nd}"]
        # 去重提示: 近似唯一键 / 大量重复
        if total > 0 and nd >= total * _NEAR_UNIQUE_RATIO:
            parts.append("≈唯一键")
        if nulls > 0:
            parts.append(f"nulls={nulls}/{total}")
        if numeric:
            mn = _fmt_num(row.get(f"c{i}__min"))
            mx = _fmt_num(row.get(f"c{i}__max"))
            parts.append(f"min={mn} max={mx}")

        ex = examples.get(name)
        ex_str = ""
        if ex is not None:
            s = str(ex)
            if len(s) > _MAX_EXAMPLE_CHARS:
                s = s[:_MAX_EXAMPLE_CHARS] + "…"
            ex_str = f"  e.g. {s!r}"

        lines.append(
            f"  - {name}  type={typ}  {' '.join(parts)}{ex_str}"
        )
    return "\n".join(lines)


def build_table_profiles(task_dir: Path) -> str:
    """对 task 的所有结构化表算列画像, 返回 markdown 文本.

    无结构化表时返回空串. 单表失败不影响其余表.
    """
    state = _get_state(Path(task_dir).resolve())
    if not state.tables:
        return ""

    chunks: list[str] = []
    for t in state.tables:
        frag = _profile_one_table(state, t.canonical)
        if frag:
            # 标注出处 (csv/json/sqlite/doc抽取), 便于 agent 区分 sub_db vs _doc_extracted.
            frag_lines = frag.split("\n", 1)
            frag_lines[0] += f"   <- {t.source_rel}"
            chunks.append("\n".join(frag_lines))

    if not chunks:
        return ""
    return "\n\n".join(chunks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python table_profile.py <task_dir>")
        sys.exit(1)
    print(build_table_profiles(Path(sys.argv[1])))
