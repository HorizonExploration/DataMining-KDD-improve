"""doc_extractor.py — 按 DocSchema 把 doc/*.md 抽成结构化表.

设计依据
---------
经过 ``analysis/doc_structure_findings.md`` 的全量扫描, 所有 hard / extreme task
的 doc 都遵循 "一行一条 record" 不变量. 因此本模块走最简方案:

    schema (来自 schema_inference.DocSchema)
        + 该 doc 的每一行 (忽略空行 / 标题行)
        → 并发 worker, 每行一次 LLM 调用
        → 受 schema 约束的 RecordValues
        → 聚合为 ExtractionResult

worker 输出 schema (RecordValues / ExtractionRecord)
- ``values`` 字典: 键来自 schema 列, 值统一为字符串 (数字、日期也保留为字符串,
  由下游统一 dtype 转换).
- ``extras`` 字典: worker 在该行发现的、但不在 schema 里的字段, 作为兜底反馈.
- ``error``: 抽取失败时的错误摘要 (LLM/解析异常).
- ``line_no``: 1-based 行号, 同时也是 record 的稳定 trace id.

Public API
----------
- ``extract_doc_async(doc_path, schema, *, knowledge_md_path=None, model=None,
  concurrency=6, line_range=None)`` → ``ExtractionResult``
- ``extract_doc(...)``  同步包装
- ``ExtractionResult.to_dataframe()`` / ``to_sqlite(db_path)``
- CLI:
    python -m tools_v2.doc_extractor \\
        <doc.md> <schema.json> [--knowledge <knowledge.md>] [--out result.json]
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from tools_v2.schema_inference import (
    DocColumn,
    DocSchema,
    _read_non_empty_lines,
)


# ---------------------------------------------------------------------------
# 调参
# ---------------------------------------------------------------------------

# 单行截断 (避免 prompt 过长 / 个别极端行打爆模型). 与采样器保持一致.
MAX_LINE_CHARS = 1500

# 默认并发上限. model_code_a3b 是内部 vLLM, 6 比较稳妥.
DEFAULT_CONCURRENCY = 6

# 单个 worker 的超时时间 (秒). 超时按 error 处理, 不阻塞其他行.
WORKER_TIMEOUT_S = 90

# knowledge.md 注入 prompt 时的最大长度 (字符).
MAX_KNOWLEDGE_CHARS = 6000


# ---------------------------------------------------------------------------
# Pydantic 输出 schema (worker 单行)
# ---------------------------------------------------------------------------


class RecordValues(BaseModel):
    """LLM worker 在一行 doc 上的抽取输出.

    ``values`` 的键必须来自调用者提供的 schema 列名;
    ``extras`` 用来收集不在 schema 中、但又确实出现在该行里的字段.
    所有值统一为字符串.
    """

    values: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "schema 列名 → 该行抽取出的值 (字符串). 缺失列填空字符串 ''."
            " 对于修正值表述 ('initially X; corrected to Y'), 取后者."
            " 对于明确缺失值 ('not available' / 'NaN' / 'None' / 'not specified'"
            " / 'pending'), 也填 ''."
        ),
    )
    extras: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "schema 中没声明、但在该行里出现的字段. 键用蛇形或临床缩写命名,"
            " 值字符串. 没有就留空 dict. 这些是给上游补充 schema 的反馈."
        ),
    )


# ---------------------------------------------------------------------------
# 抽取结果数据类
# ---------------------------------------------------------------------------


def _merge_by_pk(df, pk_cols: list[str], value_cols: list[str]):
    """同 PK 多行合并: last-non-empty-wins.

    doc 中同一 entity 经常被多段叙述 (基础 / 分类 / 属性 / 修正 ...),
    每段只填部分字段, 其它列空. 默认按 ``__line_no`` 升序遍历, 后面非空字段
    覆盖前面的空字段.

    输出列:
    - ``__line_no`` (int): 该 entity 在 doc 中第一次被叙述的行号 (group min)
    - ``__line_nos`` (str): 参与合并的所有行号, 逗号分隔升序; 用于 SQL 审计 /
      原文回溯, 例如 ``SELECT __line_nos FROM legalities WHERE id='1655'``.

    PK 全空的行直接丢弃 (是 doc 头部介绍段, 没有 record).
    """
    import pandas as pd

    pk_cols = [c for c in pk_cols if c in df.columns]
    if not pk_cols:
        return df

    # PK 全空行丢弃
    pk_empty_mask = df[pk_cols].apply(
        lambda s: s.astype(str).str.strip().eq(""), axis=0
    ).all(axis=1)
    df = df[~pk_empty_mask].copy()
    if df.empty:
        return df

    df = df.sort_values("__line_no").reset_index(drop=True)

    # 手动 group by PK (避免 pandas group-key 列 dropping 兼容性问题)
    out_rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(pk_cols, dropna=False, sort=False):
        line_nos_sorted = sorted(int(x) for x in g["__line_no"].tolist())
        merged_row: dict[str, Any] = {
            "__line_no": line_nos_sorted[0],
            "__line_nos": ",".join(str(n) for n in line_nos_sorted),
        }
        # PK 值从 group key 直接取, 类型与 df 一致
        if not isinstance(keys, tuple):
            keys = (keys,)
        for pk, v in zip(pk_cols, keys):
            merged_row[pk] = "" if v is None else str(v)
        for col in value_cols:
            if col in pk_cols:
                continue
            chosen = ""
            for v in g[col]:
                s = "" if v is None else str(v).strip()
                if s != "":
                    chosen = s
            merged_row[col] = chosen
        out_rows.append(merged_row)

    merged = pd.DataFrame(
        out_rows, columns=["__line_no", "__line_nos", *value_cols]
    )
    return merged


def _cast_columns(df, col_dtypes: dict[str, str]):
    """按 schema dtype 把字符串列转成对应类型. 转失败保留原字符串."""
    import pandas as pd

    out = df.copy()
    for col, dtype in col_dtypes.items():
        if col not in out.columns:
            continue
        s = out[col]
        # 把空字符串先视为缺失值, 避免 cast 报错; cast 后再决定 NaN 怎么写回.
        s_strip = s.astype(str).str.strip().replace("", None)
        if dtype == "int":
            # 先转 float 再 round 后转 Int64: LLM 偶发把 '1.5' / '3.0' 抽进声明为
            # int 的列, float64 直接 .astype('Int64') 会触发 "cannot safely cast"
            # 而整列抛错. round() 后非整数被就近取整, NaN 安全保留为 <NA>.
            num = pd.to_numeric(s_strip, errors="coerce")
            casted = num.round().astype("Int64")
            out[col] = casted
        elif dtype == "float":
            out[col] = pd.to_numeric(s_strip, errors="coerce")
        elif dtype == "date":
            casted = pd.to_datetime(s_strip, errors="coerce")
            out[col] = casted.dt.strftime("%Y-%m-%d")
        elif dtype == "bool":
            mapping = {
                "true": True, "false": False, "1": True, "0": False,
                "yes": True, "no": False, "y": True, "n": False,
            }
            out[col] = s_strip.str.lower().map(mapping)
        else:
            # str / 未知类型: 保留 strip 后的字符串, 空值用 ""
            out[col] = s.astype(str).str.strip()
    return out


def _enforce_enum(df, enum_map: dict[str, list[str]]):
    """对声明了 enum_values 的列, 把不在 enum 中的值统一改为空 (兜底).

    case-insensitive 匹配优先 (LLM 偶发大小写漂移). 仍不匹配的整体改空,
    防止脏值流入 SQLite 影响下游 SQL.
    """
    out = df.copy()
    for col, allowed in enum_map.items():
        if col not in out.columns:
            continue
        allowed_set = set(allowed)
        lower_map = {a.lower(): a for a in allowed}

        def _norm(v):
            if v is None:
                return ""
            s = str(v).strip()
            if s == "" or s.lower() in {"nan", "<na>", "none"}:
                return ""
            if s in allowed_set:
                return s
            return lower_map.get(s.lower(), "")

        out[col] = out[col].map(_norm)
    return out


@dataclass
class ExtractionRecord:
    """单条 record 的抽取结果."""

    line_no: int
    values: dict[str, str] = field(default_factory=dict)
    extras: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "line_no": self.line_no,
            "values": self.values,
            "extras": self.extras,
            "error": self.error,
        }


@dataclass
class ExtractionResult:
    """整份 doc 的抽取结果."""

    doc_path: str
    table_name: str
    schema: DocSchema
    records: list[ExtractionRecord] = field(default_factory=list)

    # ---- 统计 / 反馈 -----------------------------------------------------

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.records if r.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.records if not r.ok)

    def extras_frequency(self) -> dict[str, int]:
        """统计 extras 中各 key 出现频次, 用于发现 schema 漏掉的列."""
        freq: dict[str, int] = {}
        for r in self.records:
            for k in r.extras.keys():
                freq[k] = freq.get(k, 0) + 1
        return dict(sorted(freq.items(), key=lambda kv: -kv[1]))

    # ---- 导出 ------------------------------------------------------------

    def to_dataframe(self, *, dedupe_by_pk: bool = True, cast_dtypes: bool = True):
        """聚合为 pandas DataFrame.

        固定输出两列用于原文回溯 (无需 LLM, 纯算法):
        - ``__line_no`` (int): 该 record 在 doc 中首次出现的行号.
        - ``__line_nos`` (str): 参与该 record 的全部行号, 升序逗号分隔.
          单段 record 与 ``__line_no`` 相同; 多段合并 record 列出全部段落
          行号, 例如 '120,143'. 用于 ``WHERE __line_nos LIKE '%,143%'`` 这类
          审计查询.

        参数:
            dedupe_by_pk: 同一 PK 在 doc 不同段被多次叙述时, 按 last-non-empty-wins
                合并. doc 内多段叙述同一 entity 是 hard task 的常见情况
                (member.md / superhero.md / races.md / legalities.md 全部如此).
                空字符串视为缺失. 主键全部为空的行会被丢弃.
                默认 True.
            cast_dtypes: 按 ``schema.columns[i].dtype`` 把字符串值转成对应类型
                ('int' / 'float' / 'date' / 'bool' / 其他=str). 转失败的格保留
                原字符串. 默认 True.
        """
        import pandas as pd

        col_names = [c.name for c in self.schema.columns]
        col_dtypes = {c.name: c.dtype for c in self.schema.columns}
        pk_cols = list(self.schema.primary_key) or [col_names[0]]

        rows: list[dict[str, Any]] = []
        for r in self.records:
            if not r.ok:
                continue
            row = {"__line_no": r.line_no, "__line_nos": str(r.line_no)}
            for cn in col_names:
                row[cn] = r.values.get(cn, "")
            rows.append(row)
        df = pd.DataFrame(rows, columns=["__line_no", "__line_nos", *col_names])

        if dedupe_by_pk and not df.empty:
            df = _merge_by_pk(df, pk_cols=pk_cols, value_cols=col_names)

        # enum 兜底放在 cast 之前: enum_values 是字符串列表, 与 merge 后的
        # 字符串值直接匹配; 若放在 _cast_columns 之后, Int64 经 .map() 会变
        # float (1→1.0), str(1.0)='1.0' 不匹配 enum '1' 导致误清空.
        enum_map = {
            c.name: list(c.enum_values) for c in self.schema.columns if c.enum_values
        }
        if enum_map and not df.empty:
            df = _enforce_enum(df, enum_map)

        if cast_dtypes and not df.empty:
            df = _cast_columns(df, col_dtypes)

        return df

    def to_sqlite(
        self,
        db_path: str | Path,
        *,
        if_exists: str = "replace",
        dedupe_by_pk: bool = True,
        cast_dtypes: bool = True,
    ) -> None:
        """写入 SQLite 表; 表名取 ``schema.table_name``."""
        import sqlite3

        df = self.to_dataframe(dedupe_by_pk=dedupe_by_pk, cast_dtypes=cast_dtypes)
        with sqlite3.connect(str(db_path)) as conn:
            df.to_sql(self.table_name, conn, if_exists=if_exists, index=False)
        logger.info(
            "[doc_extractor] wrote {} rows to {} (table={}, dedupe={}, cast={})",
            len(df),
            db_path,
            self.table_name,
            dedupe_by_pk,
            cast_dtypes,
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "doc_path": self.doc_path,
            "table_name": self.table_name,
            "schema": self.schema.model_dump(),
            "stats": {
                "n_total": self.n_total,
                "n_ok": self.n_ok,
                "n_failed": self.n_failed,
                "extras_frequency": self.extras_frequency(),
            },
            "records": [r.to_jsonable() for r in self.records],
        }


# ---------------------------------------------------------------------------
# Prompt + Agent 构造
# ---------------------------------------------------------------------------


_EXTRACT_INSTRUCTIONS_TEMPLATE = """\
你是结构化数据抽取助手. 用户会给你:

1. 一个表的 schema (列名 / 类型 / 单位 / 别名 / 是否可空), 这个 schema 是冻结的.
2. (可选) knowledge.md 片段, 提供业务背景.
3. **一行** 来自 doc 的自然语言叙述 (一行 = 一条 record).

你的任务: 把这一行里的字段值抽出来, 按 schema 列名填到 ``values`` 字典里.
不在 schema 中、但确实出现在该行里的字段, 放到 ``extras`` 字典里 (键用蛇形命名).

**硬性规则**:
- ``values`` 字典的键 **只能是 schema 中声明的列名**, 一字不差. 多余的列名 = 失败.
- 所有值统一为字符串 (数字、日期都用字符串). 后续会统一转型, 你不需要做类型转换.
- **若某列声明了 `enum_values`** (在 schema 描述中标了 `enum_values (only allowed)=[...]`),
  抽取的值**必须严格匹配** enum_values 中某一项, 一字不差(大小写敏感). doc 里
  写的 'female' / 'Commander format' 等变体, 必须先归一化到 enum_values 中
  的对应字面量再填; 找不到匹配项就填 ''. 严禁创造新的枚举值.
- 修正值表述 ("initially X; corrected to Y" / "first reported as A, later confirmed as B"
  / "preliminary X, finalized at Y") 一律取 **后者** (corrected/finalized 的那个值).
- 明确缺失 ("not available" / "NaN" / "None" / "not specified" / "pending" /
  "missing" / "未知") 填空字符串 ''.
- 该行原文没有提及某列时, 该列必须填 ''. **绝不允许从其它列推断、拷贝、
  常识填充**.
- 日期模糊 ("mid-August of 1991" / "the tenth of February, 1986" / "late January") 应解析为
  ISO 格式 'YYYY-MM-DD'. 月日不确定时取月初 (e.g. "mid-August 1991" → "1991-08-15";
  "late January 1985" → "1985-01-25"; "early March 1990" → "1990-03-05"; 仅有年份
  ("in 1985") 取 "1985-01-01"). 实在无法定到月份就填 ''.
- 主键缺失 → 主键列填 '' (上游会标记该行 invalid).
- 噪声内容 (借阅记录、餐厅偏好、社团活动等与 schema 列无关的描述) 一律忽略,
  不要塞进 extras.

如果该行只是文档介绍 / 章节标题 / 概览段落, 没有 record 字段可抽, ``values`` 留空.

只输出符合 RecordValues 的 JSON, 不要解释.

---
## SCHEMA (table = {table_name}, primary_key = {primary_key})

{schema_columns_block}

{primary_key_pattern_block}

{schema_notes_block}

{knowledge_block}
"""


def _format_schema_columns(schema: DocSchema) -> str:
    lines: list[str] = []
    for c in schema.columns:
        nullable = "nullable" if c.nullable else "NOT NULL"
        unit = f", unit='{c.unit}'" if c.unit else ""
        aliases = (
            f", aliases={c.aliases!r}" if c.aliases else ""
        )
        enum = (
            f", **enum_values (only allowed)={c.enum_values!r}**"
            if c.enum_values
            else ""
        )
        lines.append(
            f"- {c.name}: {c.dtype}{unit}{aliases}{enum} ({nullable})"
        )
    return "\n".join(lines)


def _build_extract_instructions(
    schema: DocSchema, knowledge_md_text: Optional[str]
) -> str:
    pk_block = (
        f"## 主键识别模式\n{schema.primary_key_pattern}\n"
        if schema.primary_key_pattern
        else ""
    )
    notes_block = (
        f"## schema notes (来自 schema_inference)\n{schema.notes}\n"
        if schema.notes
        else ""
    )
    knowledge_block = ""
    if knowledge_md_text:
        excerpt = knowledge_md_text.strip()
        if len(excerpt) > MAX_KNOWLEDGE_CHARS:
            excerpt = excerpt[:MAX_KNOWLEDGE_CHARS] + "\n...[truncated]"
        knowledge_block = (
            "## knowledge.md (业务背景, 仅供参考, 列名以 schema 为准)\n"
            f"{excerpt}\n"
        )

    return _EXTRACT_INSTRUCTIONS_TEMPLATE.format(
        table_name=schema.table_name,
        primary_key=schema.primary_key,
        schema_columns_block=_format_schema_columns(schema),
        primary_key_pattern_block=pk_block,
        schema_notes_block=notes_block,
        knowledge_block=knowledge_block,
    )


def _build_extract_agent(
    schema: DocSchema,
    knowledge_md_text: Optional[str],
    model: Model,
) -> Agent[None, RecordValues]:
    """构造一个抽取 agent. instructions 静态注入 schema, 跑全文档共用同一个 agent."""
    return Agent(
        model=model,
        output_type=RecordValues,
        instructions=_build_extract_instructions(schema, knowledge_md_text),
    )


def _truncate_line(text: str, limit: int = MAX_LINE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + " ...[truncated]"


# ---------------------------------------------------------------------------
# 核心抽取循环
# ---------------------------------------------------------------------------


async def _run_one_line(
    agent: Agent[None, RecordValues],
    sem: asyncio.Semaphore,
    line_no: int,
    text: str,
    schema_col_names: list[str],
    schema_enum_map: dict[str, set[str]],
) -> ExtractionRecord:
    """对单行执行一次 LLM 抽取, 受 semaphore 限流.

    schema_enum_map: 列名 → 该列 enum_values (大小写敏感的允许集合).
        没有 enum 约束的列不在 map 中. 用于后置兜底: LLM 输出的值若不在
        enum_values 内, 直接改空, 防止脏值流入 SQLite.
    """
    # case-insensitive 列名映射 (lowercase → 规范拼写). 用于把 LLM 偶发产出
    # 的大小写漂移变体 (如 'igG' / 'Date') 归并回 schema 的官方拼写.
    schema_col_lookup = {n.lower(): n for n in schema_col_names}
    user_prompt = f"## 待抽取的一行 (L{line_no})\n{_truncate_line(text)}"
    async with sem:
        try:
            result = await asyncio.wait_for(
                agent.run(user_prompt),
                timeout=WORKER_TIMEOUT_S,
            )
            rv: RecordValues = result.output

            # 1) values: 把 LLM 输出的键按 case-insensitive 映射回官方列名;
            #    不在 schema 里的挪进 extras (防御性: LLM 偶发越界).
            cleaned_values: dict[str, str] = {}
            extras_raw: dict[str, str] = dict(rv.extras)
            for k, v in rv.values.items():
                canonical = schema_col_lookup.get(k.lower())
                if canonical:
                    cleaned_values[canonical] = "" if v is None else str(v)
                else:
                    extras_raw.setdefault(k, "" if v is None else str(v))

            # 2) extras: 反向也做一次, 把 LLM 误塞到 extras 里的合法字段
            #    (常见于大小写漂移) 挪回 values.
            extras_clean: dict[str, str] = {}
            for k, v in extras_raw.items():
                canonical = schema_col_lookup.get(k.lower())
                if canonical and canonical not in cleaned_values:
                    cleaned_values[canonical] = "" if v is None else str(v)
                elif canonical:
                    # values 里已经填过了, 这条 extras 视为冗余 drop
                    continue
                else:
                    extras_clean[k] = "" if v is None else str(v)

            # 3) 字典补齐 (#3): 缺失列填空字符串, 保证形态一致.
            for col in schema_col_names:
                cleaned_values.setdefault(col, "")

            # 4) enum 后置兜底: 若 LLM 输出的值不在该列 enum_values 中,
            #    尝试 case-insensitive 匹配后归一; 仍不匹配就改空.
            for col, allowed in schema_enum_map.items():
                v = cleaned_values.get(col, "")
                if not v:
                    continue
                if v in allowed:
                    continue
                lower_map = {a.lower(): a for a in allowed}
                cleaned_values[col] = lower_map.get(v.strip().lower(), "")

            return ExtractionRecord(
                line_no=line_no,
                values=cleaned_values,
                extras=extras_clean,
            )
        except asyncio.TimeoutError:
            logger.warning("[doc_extractor] L{} timeout after {}s", line_no, WORKER_TIMEOUT_S)
            return ExtractionRecord(line_no=line_no, error=f"timeout after {WORKER_TIMEOUT_S}s")
        except Exception as e:  # noqa: BLE001 — 任意异常都隔离到本行
            err_msg = f"{type(e).__name__}: {e}"
            # LLM/解析异常不打全栈, 否则汇总日志会爆.
            logger.warning("[doc_extractor] L{} failed: {}", line_no, err_msg[:300])
            return ExtractionRecord(line_no=line_no, error=err_msg[:500])


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


async def extract_doc_async(
    doc_path: str | Path,
    schema: DocSchema,
    model: Model,
    *,
    knowledge_md_path: str | Path | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    line_range: tuple[int, int] | None = None,
) -> ExtractionResult:
    """并发抽取整份 doc.

    参数:
        doc_path: 目标 .md 文件路径.
        schema: schema_inference 推断出的 ``DocSchema``.
        model: pydantic-ai 模型实例 (必传). 推荐 ``model_code_a3b_nothink``.
        knowledge_md_path: 同任务的 ``context/knowledge.md`` (可选, 注入 prompt).
        concurrency: 并发上限, 默认 6.
        line_range: ``(start_line, end_line)`` 1-based 闭区间; 仅抽该范围
            (用于调试 / 增量重抽). None 表示全量.
    """
    doc_path = Path(doc_path)
    if not doc_path.is_file():
        raise FileNotFoundError(f"doc 文件不存在: {doc_path}")

    lines = _read_non_empty_lines(doc_path)
    if not lines:
        logger.warning("[doc_extractor] doc 没有有效行: {}", doc_path)
        return ExtractionResult(
            doc_path=str(doc_path), table_name=schema.table_name, schema=schema
        )

    if line_range is not None:
        lo, hi = line_range
        lines = [s for s in lines if lo <= s.line_no <= hi]

    knowledge_text: Optional[str] = None
    if knowledge_md_path is not None:
        kp = Path(knowledge_md_path)
        if kp.is_file():
            knowledge_text = kp.read_text(encoding="utf-8")
        else:
            logger.warning("[doc_extractor] knowledge.md 不存在: {}", kp)

    agent = _build_extract_agent(
        schema=schema, knowledge_md_text=knowledge_text, model=model
    )
    schema_col_names = [c.name for c in schema.columns]
    schema_enum_map = {
        c.name: set(c.enum_values) for c in schema.columns if c.enum_values
    }

    sem = asyncio.Semaphore(concurrency)
    logger.info(
        "[doc_extractor] doc={} lines={} concurrency={}",
        doc_path.name,
        len(lines),
        concurrency,
    )

    tasks = [
        _run_one_line(
            agent, sem, s.line_no, s.text, schema_col_names, schema_enum_map
        )
        for s in lines
    ]
    records = await asyncio.gather(*tasks)

    result = ExtractionResult(
        doc_path=str(doc_path),
        table_name=schema.table_name,
        schema=schema,
        records=records,
    )
    logger.info(
        "[doc_extractor] doc={} ok={}/{}, failed={}, extras_keys={}",
        doc_path.name,
        result.n_ok,
        result.n_total,
        result.n_failed,
        list(result.extras_frequency().keys())[:10],
    )
    return result


def extract_doc(
    doc_path: str | Path,
    schema: DocSchema,
    model: Model,
    *,
    knowledge_md_path: str | Path | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    line_range: tuple[int, int] | None = None,
) -> ExtractionResult:
    """同步包装. 内部 ``asyncio.run`` 执行异步版本."""
    return asyncio.run(
        extract_doc_async(
            doc_path=doc_path,
            schema=schema,
            model=model,
            knowledge_md_path=knowledge_md_path,
            concurrency=concurrency,
            line_range=line_range,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_schema_from_json(path: Path) -> DocSchema:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return DocSchema.model_validate(raw)


def _main_cli() -> None:
    import argparse

    # CLI 入口才导入预定义模型, 库被 import 时不强制依赖 configs.
    from configs import pydantic_predefined_model

    parser = argparse.ArgumentParser(description="按 DocSchema 抽 doc/*.md.")
    parser.add_argument("doc_path", help="目标 .md 文件路径")
    parser.add_argument(
        "schema_path",
        help="DocSchema JSON 文件路径 (来自 schema_inference 的 model_dump)",
    )
    parser.add_argument(
        "--knowledge", default=None, help="同任务的 context/knowledge.md (可选)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="把完整 ExtractionResult 落到该 JSON 路径 (可选)",
    )
    parser.add_argument(
        "--sqlite",
        default=None,
        help="把抽取结果写入该 SQLite 数据库路径 (可选)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并发上限"
    )
    parser.add_argument(
        "--start", type=int, default=None, help="line_range 起始行 (1-based, 含)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="line_range 结束行 (1-based, 含)"
    )
    args = parser.parse_args()

    schema = _load_schema_from_json(Path(args.schema_path))
    line_range = None
    if args.start is not None and args.end is not None:
        line_range = (args.start, args.end)

    result = extract_doc(
        doc_path=args.doc_path,
        schema=schema,
        model=pydantic_predefined_model.model_code_a3b_nothink,
        knowledge_md_path=args.knowledge,
        concurrency=args.concurrency,
        line_range=line_range,
    )

    print(
        json.dumps(
            {
                "doc": result.doc_path,
                "table": result.table_name,
                "n_total": result.n_total,
                "n_ok": result.n_ok,
                "n_failed": result.n_failed,
                "extras_frequency": result.extras_frequency(),
                "first_3_records": [
                    r.to_jsonable() for r in result.records[:3]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(result.to_jsonable(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("[doc_extractor] full result → {}", args.out)

    if args.sqlite:
        result.to_sqlite(args.sqlite)


if __name__ == "__main__":
    _main_cli()
