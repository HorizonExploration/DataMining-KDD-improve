"""schema_inference.py — 从 doc/*.md 反推 schema.

用途
----
hard / extreme 任务的 ``context/doc/*.md`` 把表格数据 "用自然语言一条一条" 叙述
出来. 下游若要把它结构化成 SQLite 表, 必须先知道:

1. 主键模式 (是 5–7 位 ID? 字符串名字? (ID, Date) 复合键?)
2. 这份 doc 涵盖了哪些列 (字段)
3. 每列的数据类型 / 单位 / 自然语言别名

经过对全量 hard / extreme task 的 doc 文件结构调查 (见
``analysis/doc_structure_findings.md``), 一个关键不变量是:

    所有 hard / extreme task 的 doc 都是 "一行一条 record".

因此本模块走最简方案:**按行采样 → 单次 LLM 推断 schema**, 不做切段.

调用入口
--------
``infer_doc_schema(doc_path, knowledge_md_path=None)`` 返回 ``DocSchema``.

模型
----
统一使用 ``configs.pydantic_predefined_model.model_code_a3b``.
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# 与同目录其他文件保持一致的 import 风格

# ---------------------------------------------------------------------------
# 调参
# ---------------------------------------------------------------------------

# 自适应步长边界. 见 analysis/doc_structure_findings.md.
SHORT_DOC_THRESHOLD = 50      # < 50 行: 全量
MEDIUM_DOC_THRESHOLD = 500    # 50 ~ 500 行: step=5
SHORT_STEP = 1                # 全量
MEDIUM_STEP = 5
LONG_STEP = 10

# 每次额外强制采样的头/中/尾行数.
HEAD_TAIL_MID_EXTRA = 3

# 步长抖动幅度 (±JITTER), 打破系统性共振.
JITTER = 2

# 单行截断 (避免 prompt 太长). doc 中最长单行约 1.5K char.
MAX_LINE_CHARS_FOR_SAMPLE = 1500

# 采样样本数硬上限. 对 ~850 行 doc, 10% ≈ 85 个样本.
MAX_SAMPLES = 120

# 采样样本数下限.
MIN_SAMPLES = 20


# ---------------------------------------------------------------------------
# 输出 schema
# ---------------------------------------------------------------------------


class DocColumn(BaseModel):
    """schema 中的一列."""

    name: str = Field(
        description=(
            "规范化列名 (蛇形 snake_case 或大写缩写). 若 knowledge.md 已经定义"
            " 过同义列, **必须沿用 knowledge.md 的拼写**. 含单位时建议带后缀,"
            " 例如 'weight_kg' / 'birthday_iso' / 'creatinine_mg_dl'."
        )
    )
    dtype: str = Field(
        description=(
            "数据类型, 取下列之一: 'int' / 'float' / 'str' / 'date' / 'bool'."
            " 不确定时取 'str'."
        )
    )
    unit: str = Field(
        default="",
        description=(
            "若该列有单位 (mg/dL / cm / kg / years), 在此填出. 无单位填 ''."
        ),
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "doc 中实际出现的、对应该列的自然语言措辞 (≥0). 例如对 CRE 列可填:"
            " ['creatinine', 'Cr', 'serum creatinine']."
        ),
    )
    enum_values: list[str] = Field(
        default_factory=list,
        description=(
            "若该列只能取有限的离散值 (例如 SEX 只能 'M'/'F'; 卡牌 format 只能"
            " 'commander'/'standard' 等), 在此列出**规范化后**的字面量."
            " doc 中出现的变体 ('female', 'Commander format' 等) 由 aliases"
            " 表达, 由下游抽取阶段做归一化. 空列表 = 无取值约束."
        ),
    )
    nullable: bool = Field(
        default=True,
        description="该列在 doc 中是否会出现缺失 (NaN / None / not specified).",
    )


class DocSchema(BaseModel):
    """从 doc 反推出的 schema."""

    table_name: str = Field(
        description=(
            "推断出的表名 (蛇形). 优先取 doc 的文件名 stem; 若 knowledge.md 提到"
            " 该实体的标准表名, 则改用 knowledge.md 命名."
        )
    )
    primary_key: list[str] = Field(
        description=(
            "主键列名列表. 单列主键就一个元素; 复合主键多个 (例如 ['ID', 'Date'])."
            " 主键名必须出现在 columns 里."
        )
    )
    primary_key_pattern: str = Field(
        description=(
            "doc 中识别主键的自然语言模式描述 (供下游抽取 prompt 使用)."
            " 例如 'patient {ID} on {Date}, where ID is a 5-7 digit number and"
            " Date is in natural language form (mid-August of 1991)'."
        )
    )
    columns: list[DocColumn] = Field(
        description="所有列的 schema 描述, 必须包含主键列."
    )
    notes: str = Field(
        default="",
        description=(
            "其他备注: 如修正值的处理 ('initially X, corrected to Y' 取后者),"
            " 缺失值的常见表述, 日期模糊化策略等. 自然语言描述."
        ),
    )


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------


@dataclass
class _SampledLine:
    line_no: int   # 1-based 行号
    text: str


def _read_non_empty_lines(doc_path: Path) -> list[_SampledLine]:
    """读 doc, 忽略空行 / 仅含 markdown 标题分隔符的行."""
    lines: list[_SampledLine] = []
    with doc_path.open("r", encoding="utf-8") as fh:
        for i, raw in enumerate(fh, start=1):
            stripped = raw.rstrip("\n").strip()
            if not stripped:
                continue
            # 跳过纯标题行 / 横线 (不含数据)
            if re.fullmatch(r"#{1,6}\s+.*", stripped):
                continue
            if re.fullmatch(r"-{3,}", stripped):
                continue
            lines.append(_SampledLine(line_no=i, text=stripped))
    return lines


def _pick_step(n: int) -> int:
    if n < SHORT_DOC_THRESHOLD:
        return SHORT_STEP
    if n < MEDIUM_DOC_THRESHOLD:
        return MEDIUM_STEP
    return LONG_STEP


def _sample_lines(lines: list[_SampledLine], rng: random.Random) -> list[_SampledLine]:
    """按自适应步长 + 头中尾强制 + 抖动采样."""
    n = len(lines)
    if n == 0:
        return []
    step = _pick_step(n)
    indices: set[int] = set()

    # 等间隔主样本 + 抖动
    cursor = 0
    while cursor < n:
        jitter = rng.randint(-JITTER, JITTER) if step > 1 else 0
        idx = max(0, min(n - 1, cursor + jitter))
        indices.add(idx)
        cursor += step

    # 头 / 尾 强制
    for off in range(min(HEAD_TAIL_MID_EXTRA, n)):
        indices.add(off)
        indices.add(n - 1 - off)

    # 中间一段强制
    mid = n // 2
    for off in range(-HEAD_TAIL_MID_EXTRA // 2, HEAD_TAIL_MID_EXTRA // 2 + 1):
        idx = mid + off
        if 0 <= idx < n:
            indices.add(idx)

    sorted_idx = sorted(indices)

    # 数量上限 / 下限
    if len(sorted_idx) > MAX_SAMPLES:
        # 在排好序的索引上等距压缩
        keep_step = len(sorted_idx) / MAX_SAMPLES
        sorted_idx = [sorted_idx[int(i * keep_step)] for i in range(MAX_SAMPLES)]
    if len(sorted_idx) < MIN_SAMPLES and n > MIN_SAMPLES:
        # 加密直到达到下限
        all_idx = list(range(n))
        rng.shuffle(all_idx)
        for i in all_idx:
            if i not in indices:
                sorted_idx.append(i)
                if len(sorted_idx) >= MIN_SAMPLES:
                    break
        sorted_idx.sort()

    return [lines[i] for i in sorted_idx]


def _truncate(text: str, limit: int = MAX_LINE_CHARS_FOR_SAMPLE) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + " ...[truncated]"


def _format_samples(samples: list[_SampledLine]) -> str:
    return "\n".join(
        f"L{s.line_no}: {_truncate(s.text)}" for s in samples
    )


# ---------------------------------------------------------------------------
# Prompt 构造 + LLM agent
# ---------------------------------------------------------------------------


_SCHEMA_INFERENCE_INSTRUCTIONS = """\
你是数据 schema 推断助手. 用户会给你:

1. 一份 doc 文件的若干行采样 (每行通常是一条 record 的自然语言叙述). 行号已加前缀 "L<行号>:".
2. 可选的 knowledge.md 片段, 描述本任务相关实体 / 字段的业务定义.
3. doc 文件名 (用于推断 table_name).

你的任务: 推断这份 doc 对应的结构化 schema, 输出 DocSchema.

约束:
- **优先沿用 knowledge.md 中已声明的字段名**作为最终 column.name. 例如 knowledge.md
  里说 "LDH (integer): Lactate dehydrogenase level", 那么这一列就叫 'LDH', 不要叫
  'lactate_dehydrogenase'.
- knowledge.md 没声明、但 doc 中确实存在的字段, 用 snake_case 或临床缩写命名,
  并把 doc 中出现的措辞放进 aliases 里.
- 主键 (primary_key) 通常是: ID 单列, 或 (ID, Date) 复合键. 也可能是字符串
  类型的标识符 (球队名 / 角色代号 / 分子式). 从 doc 实际叙述方式推断.
- doc 里同一行常包含多个字段, 这是正常的; 一行一条 record.
- **如果某列在 knowledge.md 或 doc 中被声明只能取有限值** (典型: SEX='M'/'F',
  卡牌 format='commander'/'standard' 等), 必须在 enum_values 里把规范化后的
  字面量列出来; doc 中出现的所有变体 (例如 'female' / 'Commander format')
  应该作为 aliases 列出, 而不是新枚举值.
- 修正值表述 ("initially X; corrected to Y") 应取**后者**; 在 notes 里记录这个规则.
- 缺失值常见表述: "not available", "NaN", "None", "not specified", "pending". nullable=True.
- 日期模糊表述 ("mid-August of 1991", "the tenth of February, 1986") 视为可解析为
  ISO 日期, 在 notes 里说明.
- 不要输出与字段无关的"叙事噪声" (图书馆借阅, 餐厅偏好等) 作为列.

只输出符合 DocSchema 的 JSON, 不要解释.
"""


def _build_schema_agent(model) -> Agent[None, DocSchema]:
    """惰性构造一个 pydantic-ai Agent, 输出受 DocSchema 约束.

    参数:
        model: pydantic-ai 兼容的 model 对象.
    """
    return Agent(
        model=model,
        output_type=DocSchema,
        instructions=_SCHEMA_INFERENCE_INSTRUCTIONS,
    )


def _build_user_prompt(
    doc_path: Path,
    samples: list[_SampledLine],
    knowledge_md_text: Optional[str],
) -> str:
    parts: list[str] = []
    parts.append(f"## doc 文件名\n{doc_path.name}\n")
    if knowledge_md_text:
        # 控制 knowledge.md 长度 (一般 < 5K char, 但保险起见截一下)
        knowledge_excerpt = knowledge_md_text.strip()
        if len(knowledge_excerpt) > 8000:
            knowledge_excerpt = knowledge_excerpt[:8000] + "\n...[truncated]"
        parts.append("## knowledge.md (业务字段定义, 字段命名以此为准)\n")
        parts.append(knowledge_excerpt)
        parts.append("")
    parts.append(f"## 采样行 (共 {len(samples)} 行)\n")
    parts.append(_format_samples(samples))
    parts.append("")
    parts.append(
        "请基于以上信息推断 DocSchema. 列必须能覆盖采样行中实际出现的字段;"
        " 主键模式必须能区分每一条 record."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


async def infer_doc_schema_async(
    doc_path: str | Path,
    knowledge_md_path: str | Path | None = None,
    *,
    seed: int = 0,
    model,
) -> DocSchema:
    """从 doc 推断 schema (异步).

    参数:
        doc_path: 目标 .md 文件路径.
        knowledge_md_path: 同任务的 ``context/knowledge.md`` 路径 (可选).
            若提供, 会作为字段命名的金标输入 LLM.
        seed: 采样抖动随机种子, 默认 0 保证可复现.
        model: pydantic-ai 兼容的 model 对象 (必须传入).
    """
    doc_path = Path(doc_path)
    if not doc_path.is_file():
        raise FileNotFoundError(f"doc 文件不存在: {doc_path}")

    lines = _read_non_empty_lines(doc_path)
    logger.debug(
        "[schema_inference] doc={} non_empty_lines={}", doc_path.name, len(lines)
    )
    if not lines:
        raise ValueError(f"doc 没有有效行: {doc_path}")

    rng = random.Random(seed)
    samples = _sample_lines(lines, rng)
    logger.debug(
        "[schema_inference] samples={} (step strategy auto-picked)", len(samples)
    )

    knowledge_text: Optional[str] = None
    if knowledge_md_path is not None:
        kp = Path(knowledge_md_path)
        if kp.is_file():
            knowledge_text = kp.read_text(encoding="utf-8")
        else:
            logger.warning(
                "[schema_inference] knowledge.md 不存在: {}", kp
            )

    prompt = _build_user_prompt(doc_path, samples, knowledge_text)
    agent = _build_schema_agent(model=model)
    result = await agent.run(prompt)
    schema = result.output

    # 简单合法性校验: 主键必须在 columns 中.
    # 1) columns 内部 case-insensitive 去重 (LLM 偶发同列大小写不一: 'date' + 'Date').
    seen_ci: dict[str, DocColumn] = {}
    deduped_cols: list[DocColumn] = []
    for c in schema.columns:
        key = c.name.lower()
        if key in seen_ci:
            # 合并 aliases / enum_values, 保留先到的 name 拼写
            kept = seen_ci[key]
            kept.aliases = list(dict.fromkeys([*kept.aliases, *c.aliases]))
            kept.enum_values = list(
                dict.fromkeys([*kept.enum_values, *c.enum_values])
            )
            logger.warning(
                "[schema_inference] case-insensitive 重复列合并: '{}' & '{}' → '{}'",
                kept.name,
                c.name,
                kept.name,
            )
            continue
        seen_ci[key] = c
        deduped_cols.append(c)
    schema.columns = deduped_cols

    # 2) primary_key 校验 (case-insensitive 匹配后用 schema.columns 的拼写).
    col_lookup_ci = {c.name.lower(): c.name for c in schema.columns}
    rebuilt_pk: list[str] = []
    for pk in schema.primary_key:
        canonical = col_lookup_ci.get(pk.lower())
        if canonical:
            rebuilt_pk.append(canonical)
        else:
            # 自动补齐为 str 列
            logger.warning(
                "[schema_inference] primary_key 中含未声明列, 自动补齐: {}", pk
            )
            schema.columns.append(
                DocColumn(
                    name=pk, dtype="str", unit="", aliases=[], nullable=False
                )
            )
            col_lookup_ci[pk.lower()] = pk
            rebuilt_pk.append(pk)
    schema.primary_key = rebuilt_pk

    # 3) 列名 SQL 友好化: 空格 → 下划线 (#6.6). 原拼写并入 aliases 以便上游
    #    knowledge.md 风格的 'First Date' 仍能被 doc_extractor 识别.
    for c in schema.columns:
        if " " in c.name:
            old_name = c.name
            c.name = c.name.replace(" ", "_")
            if old_name not in c.aliases:
                c.aliases.insert(0, old_name)
            logger.info(
                "[schema_inference] 列名空格归一化: '{}' → '{}'",
                old_name,
                c.name,
            )
    # primary_key 也同步替换
    schema.primary_key = [pk.replace(" ", "_") for pk in schema.primary_key]

    logger.info(
        "[schema_inference] doc={} table={} pk={} cols={}",
        doc_path.name,
        schema.table_name,
        schema.primary_key,
        [c.name for c in schema.columns],
    )
    return schema


def infer_doc_schema(
    doc_path: str | Path,
    knowledge_md_path: str | Path | None = None,
    *,
    seed: int = 0,
    model,
) -> DocSchema:
    """同步包装. 内部 ``asyncio.run`` 执行异步版本."""
    return asyncio.run(
        infer_doc_schema_async(
            doc_path=doc_path,
            knowledge_md_path=knowledge_md_path,
            seed=seed,
            model=model,
        )
    )


# ---------------------------------------------------------------------------
# CLI 入口 (便于本地试跑)
# ---------------------------------------------------------------------------


def _main_cli() -> None:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="从 doc/*.md 推断 schema (单文件)."
    )
    parser.add_argument("doc_path", help="目标 .md 文件路径")
    parser.add_argument(
        "--knowledge",
        default=None,
        help="同任务的 context/knowledge.md 路径 (可选)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="采样随机种子 (默认 0)"
    )
    args = parser.parse_args()

    schema = infer_doc_schema(
        doc_path=args.doc_path,
        knowledge_md_path=args.knowledge,
        seed=args.seed,
    )
    print(_json.dumps(schema.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main_cli()
