"""Describe tools for context files in input_demo task directories.

Path model
----------
All public functions take **two** path parts:

- ``base_path``: the absolute prefix used to actually open the file. Outside
  the sandbox this is the host path; inside the sandbox this is the sandbox
  mount root. The describe output never shows ``base_path``.
- ``rel_path``: path relative to ``base_path``. This is the only path
  reported in the human-readable output, so the same description is valid
  inside or outside the sandbox.

Effective filesystem path = ``os.path.join(base_path, rel_path)``.

Supports the file types observed under ``<task>/context/`` in
``public/input_demo``:

- JSON  (``context/json/*.json``)  — dicts with ``{"table", "records": [...]}``
- CSV   (``context/csv/*.csv``)
- SQLite DB (``context/db/*.db``)
- Markdown (``context/knowledge.md``, ``context/doc/*.md``)
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3

from loguru import logger
from collections import OrderedDict
from typing import Any, Iterable



# ---------------------------------------------------------------------------
# Tunable limits
# ---------------------------------------------------------------------------
# All "max units" are counted in words (ASCII whitespace-separated tokens) /
# CJK characters. See ``_truncate_line`` for the exact rule.

# How many sample data rows to show for each structured file.
DEFAULT_SAMPLE_ROWS = 2

# Truncation applied to each value inside a sample row (CSV/JSON/SQLite).
MAX_VALUE_UNITS = 30

# Truncation applied to the example value rendered in a column summary line.
MAX_COLUMN_EXAMPLE_UNITS = 30

# Truncation applied to top-level dict value previews in describe_json.
MAX_JSON_DICT_PREVIEW_UNITS = 30

# Truncation applied to scalar JSON values.
MAX_JSON_SCALAR_UNITS = 30

# Truncation applied to each line of the markdown / text preview.
MAX_MARKDOWN_LINE_UNITS = 150

# How many lines of the markdown / text preview to show.
DEFAULT_MARKDOWN_HEAD_LINES = 20

# How many headings to list before collapsing the rest with "+N more".
MAX_MARKDOWN_HEADINGS = 15

# How many top-level keys to list for a plain-dict JSON file.
MAX_JSON_TOPLEVEL_KEYS = 20

# CSV: how many rows to scan when inferring column types.
CSV_SCAN_ROWS = 1000


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve(base_path: str, rel_path: str) -> str:
    """Join base_path + rel_path into an effective filesystem path.

    ``rel_path`` is treated as relative even if it starts with ``/`` so that
    callers can pass either ``"context/foo.json"`` or ``"/context/foo.json"``
    without surprises.
    """
    rel = rel_path.lstrip(os.sep).lstrip("/")
    return os.path.join(base_path, rel)


# ---------------------------------------------------------------------------
# Truncation helpers (defined before column/sample formatters that use them)
# ---------------------------------------------------------------------------

def _truncate_line(line: str, max_units: int = MAX_MARKDOWN_LINE_UNITS) -> str:
    """Truncate a line if it exceeds ``max_units`` words/CJK chars.

    A "unit" is either an ASCII word (whitespace-separated) or a single CJK
    character. If the line is too long, keep the first ``max_units`` units
    and append ``... (omitted N words)``.
    """
    units: list[str] = []
    buf: list[str] = []

    def _flush():
        if buf:
            units.append("".join(buf))
            buf.clear()

    for ch in line:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            _flush()
            units.append(ch)
        elif ch.isspace():
            _flush()
            units.append(ch)
        else:
            buf.append(ch)
    _flush()

    kept: list[str] = []
    count = 0
    for u in units:
        if count >= max_units and not u.isspace():
            break
        kept.append(u)
        if not u.isspace():
            count += 1

    total = sum(1 for u in units if not u.isspace())
    if total <= max_units:
        return line
    omitted = total - count
    return "".join(kept).rstrip() + f" ... (omitted {omitted} words)"


def _truncate_value(v: Any, max_units: int = MAX_VALUE_UNITS) -> Any:
    if isinstance(v, str):
        return _truncate_line(v, max_units=max_units)
    return v


def _truncate_row(row: Any, max_units: int = MAX_VALUE_UNITS) -> Any:
    if isinstance(row, dict):
        return {k: _truncate_value(v, max_units) for k, v in row.items()}
    if isinstance(row, list):
        return [_truncate_value(v, max_units) for v in row]
    return _truncate_value(row, max_units)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _columns_from_records(records: Iterable[dict]) -> "OrderedDict[str, dict]":
    cols: "OrderedDict[str, dict]" = OrderedDict()
    total = 0
    for rec in records:
        total += 1
        if not isinstance(rec, dict):
            continue
        for k, v in rec.items():
            info = cols.setdefault(
                k, {"types": set(), "nulls": 0, "examples": []}
            )
            t = _infer_type(v)
            info["types"].add(t)
            if v is None or v == "":
                info["nulls"] += 1
            elif len(info["examples"]) < 3:
                info["examples"].append(v)
    for info in cols.values():
        info["total"] = total
    return cols


def _format_columns_table(
    cols: "OrderedDict[str, dict]",
    max_units: int = MAX_COLUMN_EXAMPLE_UNITS,
) -> str:
    if not cols:
        return "  (no columns)"
    lines = []
    name_w = max(len(c) for c in cols) if cols else 4
    for name, info in cols.items():
        types = "/".join(sorted(t for t in info["types"] if t != "null"))
        if not types:
            types = "null"
        nullable = "null" in info["types"] or info["nulls"] > 0
        ex = info["examples"][0] if info["examples"] else None
        if isinstance(ex, str):
            ex_repr = repr(_truncate_line(ex, max_units=max_units))
        else:
            ex_repr = _truncate_line(repr(ex), max_units=max_units)
        lines.append(
            f"  - {name.ljust(name_w)}  type={types:<12} "
            f"nullable={'Y' if nullable else 'N'}  e.g. {ex_repr}"
        )
    return "\n".join(lines)


def _format_sample_rows(
    rows: list,
    n: int = DEFAULT_SAMPLE_ROWS,
    max_units: int = MAX_VALUE_UNITS,
) -> str:
    out = []
    for i, row in enumerate(rows[:n], 1):
        truncated = _truncate_row(row, max_units)
        out.append(
            f"  row {i}: {json.dumps(truncated, ensure_ascii=False, default=str)}"
        )
    return "\n".join(out) if out else "  (no rows)"


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def describe_json(base_path: str, rel_path: str, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> str:
    """Describe a JSON file. Output shows only ``rel_path``."""
    full = _resolve(base_path, rel_path)
    with open(full, "r", encoding="utf-8") as f:
        data = json.load(f)

    size = os.path.getsize(full)
    header = [f"[JSON] {rel_path}", f"  size: {size:,} bytes"]

    if isinstance(data, dict) and isinstance(data.get("records"), list):
        table = data.get("table", os.path.splitext(os.path.basename(rel_path))[0])
        records = data["records"]
        cols = _columns_from_records(records)
        header.append(f"  shape: object with 'records' (table='{table}')")
        header.append(f"  rows: {len(records)}    columns: {len(cols)}")
        header.append("  columns:")
        header.append(_format_columns_table(cols))
        header.append(f"  sample rows (first {sample_rows}):")
        header.append(_format_sample_rows(records, sample_rows))
        return "\n".join(header)

    if isinstance(data, list) and data and isinstance(data[0], dict):
        cols = _columns_from_records(data)
        header.append("  shape: array of objects")
        header.append(f"  rows: {len(data)}    columns: {len(cols)}")
        header.append("  columns:")
        header.append(_format_columns_table(cols))
        header.append(f"  sample rows (first {sample_rows}):")
        header.append(_format_sample_rows(data, sample_rows))
        return "\n".join(header)

    if isinstance(data, list):
        header.append(f"  shape: array (len={len(data)})")
        sample = [_truncate_value(x, MAX_VALUE_UNITS) for x in data[:sample_rows]]
        header.append(
            "  sample: "
            + json.dumps(sample, ensure_ascii=False, default=str)
        )
        return "\n".join(header)

    if isinstance(data, dict):
        header.append(f"  shape: object with {len(data)} top-level keys")
        for k, v in list(data.items())[:MAX_JSON_TOPLEVEL_KEYS]:
            preview = json.dumps(
                _truncate_row(v, MAX_JSON_DICT_PREVIEW_UNITS),
                ensure_ascii=False,
                default=str,
            )
            preview = _truncate_line(preview, max_units=MAX_JSON_DICT_PREVIEW_UNITS)
            header.append(f"    - {k} ({_infer_type(v)}): {preview}")
        return "\n".join(header)

    header.append(
        f"  scalar value ({_infer_type(data)}): "
        f"{_truncate_line(repr(data), MAX_JSON_SCALAR_UNITS)}"
    )
    return "\n".join(header)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def describe_csv(
    base_path: str,
    rel_path: str,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    scan_rows: int = CSV_SCAN_ROWS,
) -> str:
    """Describe a CSV file: columns + sample rows. Output shows only ``rel_path``."""
    full = _resolve(base_path, rel_path)
    size = os.path.getsize(full)
    header_lines = [f"[CSV] {rel_path}", f"  size: {size:,} bytes"]

    with open(full, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = reader.fieldnames or []
        rows: list[dict] = []
        total = 0
        for i, row in enumerate(reader):
            total += 1
            if i < scan_rows:
                rows.append(row)

    def _infer_cell(v: str) -> str:
        if v is None or v == "":
            return "null"
        try:
            int(v)
            return "int"
        except (TypeError, ValueError):
            pass
        try:
            float(v)
            return "float"
        except (TypeError, ValueError):
            pass
        return "str"

    cols: "OrderedDict[str, dict]" = OrderedDict(
        (c, {"types": set(), "nulls": 0, "examples": []}) for c in fieldnames
    )
    for row in rows:
        for c in fieldnames:
            v = row.get(c)
            t = _infer_cell(v)
            info = cols[c]
            info["types"].add(t)
            if t == "null":
                info["nulls"] += 1
            elif len(info["examples"]) < 3:
                info["examples"].append(v)

    header_lines.append(
        f"  rows (scanned/total seen): {len(rows)}/{total}    columns: {len(fieldnames)}"
    )
    header_lines.append(f"  delimiter: {dialect.delimiter!r}")
    header_lines.append("  columns:")
    header_lines.append(_format_columns_table(cols))
    header_lines.append(f"  sample rows (first {sample_rows}):")
    header_lines.append(_format_sample_rows(rows, sample_rows))
    return "\n".join(header_lines)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def describe_sqlite(base_path: str, rel_path: str, sample_rows: int = DEFAULT_SAMPLE_ROWS,
                    collapse_keys: "set[str] | None" = None) -> str:
    """Describe a SQLite database. Output shows only ``rel_path``.

    ``collapse_keys`` (软过滤): 命中 ``{rel_path}::{表名}`` 的表只渲染一行折叠提示,
    不展开列/样例 (见 ``describe_context_dir``).
    """
    full = _resolve(base_path, rel_path)
    size = os.path.getsize(full)
    out = [f"[SQLite] {rel_path}", f"  size: {size:,} bytes"]
    norm_rel = _norm_key(rel_path)

    conn = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [r["name"] for r in cur.fetchall()]
        out.append(f"  tables ({len(tables)}): {', '.join(tables) or '(none)'}")
        for t in tables:
            out.append("")
            # 软过滤: 该表被判无关 → 一行折叠, 不展开 schema/样例.
            if collapse_keys and f"{norm_rel}::{t}" in collapse_keys:
                out.append(
                    f"  ── Table: {t}  —— (table_relevance 判为无关, schema 已折叠省 token; "
                    f"如需可用 explore_data 查询)"
                )
                continue
            out.append(f"  ── Table: {t}")
            try:
                cur.execute(f'SELECT COUNT(*) AS n FROM "{t}"')
                n_rows = cur.fetchone()["n"]
            except sqlite3.Error as e:
                n_rows = f"?? ({e})"
            out.append(f"    rows: {n_rows}")

            # Collect one example value per column for the schema listing
            try:
                cur.execute(f'SELECT * FROM "{t}" LIMIT 1')
                ex_row = cur.fetchone()
                col_examples: dict = dict(ex_row) if ex_row else {}
            except sqlite3.Error:
                col_examples = {}

            cur.execute(f'PRAGMA table_info("{t}")')
            info = cur.fetchall()
            out.append(f"    columns: {len(info)}")
            name_w = max((len(c["name"]) for c in info), default=4)
            for c in info:
                pk = " PK" if c["pk"] else ""
                nn = " NOT NULL" if c["notnull"] else ""
                dflt = (
                    f" DEFAULT {c['dflt_value']}"
                    if c["dflt_value"] is not None
                    else ""
                )
                ex_val = col_examples.get(c["name"])
                if ex_val is not None:
                    eg = f"  e.g. {repr(_truncate_line(str(ex_val), max_units=MAX_COLUMN_EXAMPLE_UNITS))}"
                else:
                    eg = ""
                out.append(
                    f"      - {c['name'].ljust(name_w)}  "
                    f"{(c['type'] or '').ljust(8)}{pk}{nn}{dflt}{eg}"
                )

            try:
                cur.execute(f'SELECT * FROM "{t}" LIMIT ?', (sample_rows,))
                rows = [dict(r) for r in cur.fetchall()]
            except sqlite3.Error as e:
                rows = []
                out.append(f"    sample error: {e}")
            out.append(f"    sample rows (first {sample_rows}):")
            out.append(
                "\n".join(
                    f"      row {i}: "
                    + json.dumps(_truncate_row(r, MAX_VALUE_UNITS), ensure_ascii=False, default=str)
                    for i, r in enumerate(rows, 1)
                )
                or "      (no rows)"
            )
    finally:
        conn.close()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Markdown / plain text
# ---------------------------------------------------------------------------

def describe_markdown(
    base_path: str,
    rel_path: str,
    head_lines: int = DEFAULT_MARKDOWN_HEAD_LINES,
    line_max_units: int = MAX_MARKDOWN_LINE_UNITS,
) -> str:
    """Describe a markdown / text file. Output shows only ``rel_path``."""
    full = _resolve(base_path, rel_path)
    size = os.path.getsize(full)
    with open(full, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()
    headings = [ln for ln in lines if ln.lstrip().startswith("#")]

    ext = os.path.splitext(rel_path)[1].lower().lstrip(".") or "txt"
    out = [
        f"[{ext.upper()}] {rel_path}",
        f"  size: {size:,} bytes    lines: {len(lines)}",
        f"  headings ({len(headings)}):",
    ]
    for h in headings[:MAX_MARKDOWN_HEADINGS]:
        out.append(f"    {h.strip()}")
    if len(headings) > MAX_MARKDOWN_HEADINGS:
        out.append(f"    ... (+{len(headings) - MAX_MARKDOWN_HEADINGS} more)")
    out.append(f"  preview (first {head_lines} lines):")
    for ln in lines[:head_lines]:
        out.append(f"    | {_truncate_line(ln, max_units=line_max_units)}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {
    ".json": describe_json,
    ".csv": describe_csv,
    ".db": describe_sqlite,
    ".sqlite": describe_sqlite,
    ".sqlite3": describe_sqlite,
    ".md": describe_markdown,
    ".markdown": describe_markdown,
    ".txt": describe_markdown,
}


def _norm_key(s: str) -> str:
    """折叠 key 归一: 统一分隔符为 ``/``, 便于跨来源 (Path vs os.path) 比较."""
    return s.replace(os.sep, "/")


def _collapsed_line(rel_path: str, kind: str, extra: str = "") -> str:
    """无关表的一行折叠提示 (软过滤): 告知存在 + 可用 explore_data 找回, 不展开 schema."""
    return (
        f"[{kind}] {rel_path}  —— (table_relevance 判为与本任务无关, schema 已折叠以省 token; "
        f"如确需可用 explore_data 查询){(' ' + extra) if extra else ''}"
    )


def describe_file(base_path: str, rel_path: str, collapse_keys: "set[str] | None" = None, **kwargs) -> str:
    """Dispatch to the right ``describe_*`` based on ``rel_path``'s extension.

    ``collapse_keys`` (软过滤): 见 ``describe_context_dir``. csv/json 整文件折叠;
    sqlite 把 keys 透传给 ``describe_sqlite`` 做逐表折叠.
    """
    ext = os.path.splitext(rel_path)[1].lower()
    fn = _DISPATCH.get(ext)
    if fn is None:
        full = _resolve(base_path, rel_path)
        size = os.path.getsize(full) if os.path.exists(full) else -1
        return f"[UNSUPPORTED {ext or 'no-ext'}] {rel_path}  size={size}"

    norm = _norm_key(rel_path)
    if ext in (".db", ".sqlite", ".sqlite3"):
        # sqlite: 逐表折叠, 把 collapse_keys 透传 (describe_sqlite 内部按 rel::table 判).
        return describe_sqlite(base_path, rel_path, collapse_keys=collapse_keys, **kwargs)
    if collapse_keys and norm in collapse_keys:
        # csv/json 整文件被判无关 → 一行折叠.
        return _collapsed_line(rel_path, ext.lstrip(".").upper())
    return fn(base_path, rel_path, **kwargs)


def describe_context_dir(
    base_path: str,
    rel_dir: str,
    skip_knowledge: bool = False,
    collapse_keys: "set[str] | None" = None,
) -> str:
    """Walk a context directory and describe every file in it.

    ``base_path + rel_dir`` is the actual directory we walk; output paths are
    rendered relative to ``base_path`` (i.e. starting with ``rel_dir``).

    Args:
        base_path: Absolute prefix used to open files.
        rel_dir: Path relative to ``base_path`` of the context directory.
        skip_knowledge: If ``True``, skip any file named ``knowledge.md``
            (regardless of subdirectory depth).
        collapse_keys: 可选的"折叠集" —— 由 ``table_relevance_agent`` 判为无关的表的
            describe key (软过滤). csv/json 的 key 是其相对 ``base_path`` 的路径;
            sqlite 表的 key 是 ``{rel_path}::{表名}``. 命中的表只渲染**一行可见提示**
            (不展开列/样例), 让 solver 知道表存在且可用 explore_data 找回, 但不占 prompt.
            ``None`` (默认) 表示不折叠, 全部详细展示.
    """
    full_dir = _resolve(base_path, rel_dir)
    chunks = [f"# Context directory: {rel_dir}", ""]
    for root, _dirs, files in os.walk(full_dir):
        for name in sorted(files):
            if name.startswith(".") or name == ".DS_Store":
                continue
            if skip_knowledge and name == "knowledge.md":
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, base_path)
            try:
                chunks.append(describe_file(base_path, rel, collapse_keys=collapse_keys))
            except Exception as e:  # noqa: BLE001
                chunks.append(f"[ERROR] {rel}: {e!r}")
            chunks.append("")
    return "\n".join(chunks)


def read_knowledge(
    base_path: str,
    rel_dir: str,
    as_hashline: bool = True,
    hashline_limit: int = 100000,
) -> str:
    """Read the full content of ``knowledge.md`` inside a context directory.

    Searches for ``knowledge.md`` directly under ``base_path/rel_dir``.  If
    the file does not exist, returns an empty string.

    Args:
        base_path: Absolute prefix used to open files.
        rel_dir: Path of the context directory relative to ``base_path``
                 (e.g. ``"context"``).
        as_hashline: 若为 ``True`` (默认), 将文件渲染成 hashline 格式
            (``{line_num}:{hash}|{line}``), 这样 solver 在后续看到
            knowledge.md 内容时, 每一行都自带行号与 2 字符行 hash, 便于
            它在生成的回复 / SQL 注释 / post-check 时, 通过
            ``line_num:hash`` 反向引用回 knowledge.md 的原文行
            (与 ``hashline_edit`` / ``format_hashline_output`` 用同一套
            约定).  注意: 调用前必须已 import ``agents_v2.hashline_patch``,
            否则会回落到原版十六进制 hash, 弱模型容易把 ``"58"`` 当成
            int.  若为 ``False``, 返回原始 raw 文本.
        hashline_limit: 传给 ``format_hashline_output`` 的 ``limit``
            参数, 默认 100000 行 (knowledge.md 通常远小于该值, 因此等
            价于全文).

    Returns:
        - ``as_hashline=True``: hashline 格式字符串, 形如
          ``1:MQ|# Title\n2:VR|...``.
        - ``as_hashline=False``: 原始 raw 文本.
        - 文件不存在时一律返回 ``""``.
    """
    full_dir = _resolve(base_path, rel_dir)
    knowledge_path = os.path.join(full_dir, "knowledge.md")
    if not os.path.isfile(knowledge_path):
        logger.error("knowledge.md not found: {}", knowledge_path)
        return ""
    with open(knowledge_path, "r", encoding="utf-8") as f:
        raw = f.read()

    if not as_hashline:
        return raw

    # 延迟 import: 避免在不需要 hashline 的场景下强制依赖
    # pydantic_ai_backends; 同时让调用方能在更早处触发
    # hashline_patch monkey-patch (alphabet 改写).
    from pydantic_ai_backends.hashline import format_hashline_output

    return format_hashline_output(raw, offset=0, limit=hashline_limit)


def generate_dir_tree(task_dir) -> str:
    """动态生成任务目录的真实树形结构字符串，用于替换 system_instruction 中的静态示意目录。

    扫描 ``task_dir`` 下的实际文件，生成类似 ``tree`` 工具风格的文本。
    只展示 ``context/`` 和 ``workdir/`` 两个子目录的内容；其余根文件（task.json 等）直接列出。

    Args:
        task_dir: 任务根目录，可以是 str 或 Path。

    Returns:
        形如以下格式的字符串::

            ./
            ├── context/
            │   ├── knowledge.md
            │   ├── csv/
            │   │   └── orders.csv
            │   └── db/
            │       └── sales.db
            ├── task.json
            └── workdir/
                └── solver.py
    """
    import os
    from pathlib import Path

    task_dir = Path(task_dir)

    lines: list[str] = ["./"]

    def _tree_lines(directory: Path, prefix: str) -> list[str]:
        """递归收集目录条目，返回带 tree 前缀的行列表。"""
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return []
        result = []
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            if entry.is_dir():
                result.append(f"{prefix}{connector}{entry.name}/")
                result.extend(_tree_lines(entry, child_prefix))
            else:
                result.append(f"{prefix}{connector}{entry.name}")
        return result

    lines.extend(_tree_lines(task_dir, ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default: describe all task contexts under input_demo
        base = (
            "/Users/zhe/Documents/work/20260400-data-agent/"
            "public/input"
        )
        for task in sorted(os.listdir(base)):
            task_base = os.path.join(base, task)
            ctx_full = os.path.join(task_base, "context")
            if os.path.isdir(ctx_full):
                print("=" * 80)
                print(describe_context_dir(task_base, "context"))
    elif len(sys.argv) == 2:
        # Treat single arg as a full path: split into (dirname, basename) for
        # files, or (parent, name) for directories.
        p = sys.argv[1]
        if os.path.isdir(p):
            base = os.path.dirname(p.rstrip(os.sep)) or "/"
            rel = os.path.basename(p.rstrip(os.sep))
            print(describe_context_dir(base, rel))
        else:
            base = os.path.dirname(p) or "."
            rel = os.path.basename(p)
            print(describe_file(base, rel))
    else:
        # Two args: base_path rel_path
        base, rel = sys.argv[1], sys.argv[2]
        full = _resolve(base, rel)
        if os.path.isdir(full):
            print(describe_context_dir(base, rel))
        else:
            print(describe_file(base, rel))
