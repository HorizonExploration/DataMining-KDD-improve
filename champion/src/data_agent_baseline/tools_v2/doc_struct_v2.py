"""doc_struct_v2.py — 全新的 block-aware doc 结构化工具 (独立实现, 不依赖/不修改旧的 doc_prepare/doc_extractor)。

核心: 输入一个散文体 doc (.md/.txt) + 同 task 的 knowledge.md, 返回一张或多张结构化表。

设计依据 (全部经实测验证, 见 analysis/新doc结构化方案设计_20260615.md):
- doc 多为 "block 结构": 宽表按列分成几组, 每组用一整批 "一记录一段" 的段落叙述,
  同一条记录在不同 block 各出现一次, 每次只讲它在该 block 那几列的值。
- 现有 doc_extractor 假设 "一行一条完整 record", 对 block 结构会漏列 / 逐行抽爆 token。

四阶段流水线:
  Stage1  [1 次 LLM 全文调用]  把 doc 切成物理批次(start/end + 列) + 过渡行。
  Stage1.5[1 次轻量 LLM 调用]  给每批样例, 判 "相邻批是否同属性段" → cont 数组。
           程序融合 (列集合相同 OR 模型判同) → 合并成语义 block。
  Stage3  [每 block 内分小批, 各 1 次 LLM]  抽该 block 每条记录的 [主键, 列...] (JSON 数组省 token)。
           prompt 内置数据纠错陷阱处理 (先错后对取后值)。
  Stage4  [程序]  各 block 按主键 join → 宽表。

关思考: 复用 pydantic-ai 的 model (推荐 model_code_a3b_nothink, SDK 会把 chat_template_kwargs
        展开到请求体顶层, enable_thinking=False 生效, 实测 completion_tokens ~20)。

公共入口:
  - structurize_doc_async(doc_path, *, knowledge_md_path=None, model, ...) -> DocStructResult
  - structurize_doc(...) 同步包装
  - DocStructResult.to_dataframe() / to_sqlite(db_path)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model


# ---------------------------------------------------------------------------
# 调参
# ---------------------------------------------------------------------------

# Stage3 block 内每小批的行数 (reflow 后基本一行一记录, 约等于每批记录数).
# 太大易漏行, 太小调用多. 实测 18 稳定.
BLOCK_BATCH_ROWS = 18

# Stage1 全文调用的输出上限 (关思考后批次清单很短, 8k 充裕).
STAGE1_MAX_TOKENS = 8000
# Stage1.5 cont 数组很短.
STAGE15_MAX_TOKENS = 3000
# Stage3 每小批抽取输出 (~18 条记录 × 几列).
STAGE3_MAX_TOKENS = 8000

# Stage1.5 样例: 每批取前 k 条 + 末 1 条原文, 每条截断长度.
SAMPLE_K = 2
SAMPLE_CHARS = 160

# 小 doc 路由阈值: 行数 <= 此值直接整篇出表 (跳过 block 流程).
SMALL_DOC_LINES = 12

# Stage1.5 / Stage3 内部并发 (block 之间 / 小批之间).
INNER_CONCURRENCY = 3


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------


@dataclass
class BlockInfo:
    """一个语义 block (合并后)。"""
    start_line: int
    end_line: int
    columns: list[str]


@dataclass
class DocStructResult:
    """一份 doc 的结构化结果。

    一份 doc → 一张表 (多 block 按主键 join 成的宽表)。
    """
    doc_path: str
    table_name: str
    primary_key: str
    columns: list[str]                       # 最终宽表的列 (主键在首位)
    rows: list[dict[str, Any]]               # 每行一条记录 (列名 → 值)
    blocks: list[BlockInfo] = field(default_factory=list)
    transition_lines: list[int] = field(default_factory=list)
    error: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def to_dataframe(self):
        import pandas as pd
        if not self.rows:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_sqlite(self, db_path: str | Path, *, if_exists: str = "replace") -> None:
        import sqlite3
        df = self.to_dataframe()
        with sqlite3.connect(str(db_path)) as conn:
            df.to_sql(self.table_name, conn, if_exists=if_exists, index=False)
        logger.info("[doc_struct_v2] wrote {} rows × {} cols to {} (table={})",
                    len(df), len(self.columns), db_path, self.table_name)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "doc_path": self.doc_path,
            "table_name": self.table_name,
            "primary_key": self.primary_key,
            "columns": self.columns,
            "n_rows": self.n_rows,
            "n_blocks": self.n_blocks,
            "blocks": [{"start_line": b.start_line, "end_line": b.end_line,
                        "columns": b.columns} for b in self.blocks],
            "transition_lines": self.transition_lines,
            "error": self.error,
            "rows": self.rows,
        }


# ---------------------------------------------------------------------------
# LLM 调用封装 (pydantic-ai, 关思考靠传入的 model)
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出里提取第一个 JSON 对象 (容忍 ```json 包裹/前后缀)。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def _ask_json(model: Model, prompt: str) -> Optional[dict]:
    """发一次请求, 返回解析后的 JSON dict (失败返回 None)。"""
    agent = Agent(model=model)
    result = await agent.run(prompt)
    return _extract_json(result.output)


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------


def _read_lines(doc_path: Path) -> dict[int, str]:
    """读 doc, 返回 {1-based 行号: 文本} (仅非空行, 跳过纯 markdown 标题/横线)。"""
    out: dict[int, str] = {}
    with doc_path.open("r", encoding="utf-8") as fh:
        for i, raw in enumerate(fh, start=1):
            s = raw.rstrip("\n").strip()
            if not s:
                continue
            if re.fullmatch(r"#{1,6}\s+.*", s) or re.fullmatch(r"-{3,}", s):
                continue
            out[i] = s
    return out


def _numbered(lines: dict[int, str]) -> str:
    return "\n".join(f"{i}: {t}" for i, t in lines.items())


def _norm_col(c: Any) -> str:
    """列名归一: 去括号注释/空白/标点, 用于相邻批次列集合比对。"""
    s = str(c).strip().lower()
    return re.sub(r"[\s()（）/、,，.。:：_\-]", "", s)


def _sample_text(lines: dict[int, str], start: int, end: int) -> list[str]:
    rng = [i for i in range(start, end + 1) if i in lines]
    picks = rng[:SAMPLE_K] + ([rng[-1]] if rng and rng[-1] not in rng[:SAMPLE_K] else [])
    return [lines[i][:SAMPLE_CHARS] for i in picks]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_KNOWLEDGE_HINT = (
    "\n\n# 业务背景 (knowledge.md, 仅供字段命名参考; 它通常只覆盖部分列, 缺的列按文档自行命名)\n{know}\n"
)

STAGE1_PROMPT = """# 任务
下面文档(每行前有行号)把一张很宽的结构化表用散文复述。文档采用【分属性段】写法: 把表的列分成几组, 每组用一整批"一条记录一段"的段落叙述(每段讲一条记录在这组列上的值), 讲完所有记录再换下一组列从头开始。
{know_block}
请从前往后扫描, 把文档切成【物理批次】: 每个批次 = 一段连续的"一记录一段"的段落(遇到无数据的过渡/承接句就断开)。对每个批次输出:
- start_line/end_line: 起止行号
- columns: 该批叙述的【真实结构化字段】列名(有明确取值、每条记录都有的字段)。**务必忽略与数据无关的一次性噪声描述**(如"咖啡机保养""晚宴出席""安保升级""内部培训"这类杂事), 它们不是字段。

注意: 同一组列常被过渡句拆成前后两批(先低编号记录、后高编号记录), 此时如实输出为两个批次即可(后续会合并)。
另外请指出主键: doc 用什么字段标识每一条记录(如"档案N""记录N""unit stay N"里的标识), 给出该字段的列名。

直接输出JSON(不要解释):
{{"primary_key":"主键列名","batches":[{{"start_line":int,"end_line":int,"columns":["..."]}}],"transition_lines":[int]}}

# 文档:
{numbered}"""

STAGE15_PROMPT = """下面是从一份散文文档里按顺序切出的 {n} 个【批次】(batch 0..{nlast})。这份文档把一张宽表分成几个"属性段"叙述: 每个属性段聚焦表的一组字段, 先讲一批记录, 常被过渡句打断后再讲下一批记录(仍是同一组字段)。所以同一个属性段往往对应【相邻的多个批次】。

请逐批判断: 批次 i 与【批次 i-1】是不是在讲【同一组字段/同一类信息】(即同一属性段, 只是覆盖的记录不同)?
- **主要依据每批的 sample 样例原文**: 读两批的样例, 如果它们描述的是同一类属性(比如都在讲"身份编码", 或都在讲"任职年限评分+排名"), 就是同一属性段, 即使具体数值/措辞不同。
- columns_hint 仅供参考(可能混入无关噪声词, 不要被它误导)。
- 典型规律: 文档常把每个属性段恰好拆成相邻两批(先低编号记录、后高编号记录), 所以 cont 常呈现 0,1,0,1... 的规律, 但不绝对, 以实际语义为准。

输出长度【恰好为 {n}】的数组 cont: cont[i]=1 表示批次 i 与 i-1 同属性段(应合并), =0 表示开启新属性段。cont[0]=0。

批次列表:
{items}

直接输出JSON(不要解释), 数组长度正好 {n}: {{"cont":[0,...]}}"""

STAGE3_PROMPT = """从下面文本中抽取【每一条记录】的字段值。文本里每条记录通常占一段(以"档案/记录/单元 N"等开头)。

要抽取的字段(按此顺序): {header}
- 第一个字段是主键(记录标识)。
- 对每条记录, 输出一个数组, 元素顺序与上面字段顺序一一对应。
- 缺失值用 null。数值去掉千分位逗号, 日期保留原样。忽略与字段无关的噪声句(食堂/安保/培训等杂事)。

【重要——数据纠错陷阱】文本会【故意先给一个错误值, 再说明纠正为正确值】, 你必须取【纠正后的最终值】, 绝不能取那个被否定的初始值。常见表述与正确取值:
- "初步录入为 101001039, 经复核后最终校正为 101001038" → 取 101001038
- "曾被误记为 101001410, 根据最新同步数据更新为 101001416" → 取 101001416
- "一份早期文件标记为 90.00, 经审计修正后为 90.97" → 取 90.97
- "初步计算值为 25.00, 剔除无关影响后最终校正为 22.00" → 取 22.00
- "排名曾显示为第 1180 位, 经系统同步后确认为第 1186 位" → 取 1186
- "代码以 755 结尾(如 101004755), 经核实确认为 101004754" → 取 101004754
- "记录显示为 101013838, 与草稿里的 101013830 不同, 现统一为正确值" → 取 101013838(被肯定的那个)
判断要点: 取被"最终确认/校正/修正/更新/确认为"等词修饰的值; 丢弃被"初步/曾/误记/早期/草稿/之前"等词修饰的值。

直接输出JSON(不要解释), rows按记录在文中出现的顺序:
{{"header": {header_json}, "rows": [[值,值,...], ...]}}

文本:
{text}"""

SMALL_DOC_PROMPT = """下面文档(每行前有行号)把一张结构化表用散文复述, 每条记录通常占一段。请抽取所有记录, 还原成表格。
{know_block}
要求: 识别字段(列), 第一个字段是主键(记录标识)。对每条记录输出一个数组, 顺序与 header 一致。缺失用 null。
【数据纠错】若文本先给错误值再纠正(如"初记X后确认为Y"), 取最终值 Y, 不要取被否定的 X。忽略无关噪声句。

直接输出JSON(不要解释):
{{"primary_key":"主键列名","header":["主键列","列2",...],"rows":[[值,...],...]}}

# 文档:
{numbered}"""


def _know_block(knowledge_text: Optional[str]) -> str:
    if not knowledge_text:
        return ""
    excerpt = knowledge_text.strip()
    if len(excerpt) > 6000:
        excerpt = excerpt[:6000] + "\n...[truncated]"
    return _KNOWLEDGE_HINT.format(know=excerpt)


# ---------------------------------------------------------------------------
# Stage 1: 物理批次标注
# ---------------------------------------------------------------------------


async def _stage1_batches(model: Model, lines: dict[int, str],
                          knowledge_text: Optional[str]) -> dict:
    prompt = STAGE1_PROMPT.format(numbered=_numbered(lines),
                                  know_block=_know_block(knowledge_text))
    pred = await _ask_json(model, prompt)
    return pred or {}


# ---------------------------------------------------------------------------
# Stage 1.5: 模型判合并 + 程序融合
# ---------------------------------------------------------------------------


async def _stage15_cont(model: Model, batches: list[dict],
                        lines: dict[int, str]) -> Optional[list[int]]:
    items = []
    for idx, b in enumerate(batches):
        s, e = b.get("start_line"), b.get("end_line")
        samp = _sample_text(lines, s, e) if (s and e) else []
        items.append({"batch": idx, "sample": samp, "columns_hint": b.get("columns", [])})
    n = len(items)
    prompt = STAGE15_PROMPT.format(n=n, nlast=n - 1,
                                   items=json.dumps(items, ensure_ascii=False))
    pred = await _ask_json(model, prompt)
    if not pred or "cont" not in pred:
        return None
    cont = pred["cont"]
    if not isinstance(cont, list) or len(cont) != n:
        return None
    return cont


def _fuse_merge(batches: list[dict], cont: Optional[list[int]]) -> list[BlockInfo]:
    """融合: 相邻批次 (列集合相同 OR 模型 cont=1) → 合并成语义 block。"""
    nb = len(batches)

    def col_key(b):
        return frozenset(_norm_col(c) for c in (b.get("columns") or []))

    cont_ok = bool(cont) and len(cont) == nb
    blocks: list[BlockInfo] = []
    for i, b in enumerate(batches):
        s, e = b.get("start_line"), b.get("end_line")
        cols = b.get("columns") or []
        if i > 0:
            same_cols = col_key(b) == col_key(batches[i - 1]) and len(col_key(b)) > 0
            model_cont = bool(cont[i]) if cont_ok else False
            if same_cols or model_cont:
                # 并入上一个 block: 延伸 end, 合并列 (并集, 保留首见顺序)
                prev = blocks[-1]
                if e:
                    prev.end_line = e
                for c in cols:
                    if c not in prev.columns:
                        prev.columns.append(c)
                continue
        blocks.append(BlockInfo(start_line=s, end_line=e, columns=list(cols)))
    return blocks


# ---------------------------------------------------------------------------
# Stage 3: 按 block 分小批抽取
# ---------------------------------------------------------------------------


async def _stage3_extract_block(model: Model, lines: dict[int, str],
                                block: BlockInfo, pk_name: str,
                                sem: asyncio.Semaphore) -> tuple[list[str], list[list]]:
    cols = block.columns
    data_cols = [c for c in cols if c != pk_name]
    header = [pk_name] + data_cols
    rng = [i for i in range(block.start_line, block.end_line + 1) if i in lines]
    segs = [rng[i:i + BLOCK_BATCH_ROWS] for i in range(0, len(rng), BLOCK_BATCH_ROWS)]

    async def one_seg(seg: list[int]) -> list[list]:
        text = "\n".join(f"{i}: {lines[i]}" for i in seg)
        prompt = STAGE3_PROMPT.format(header=header,
                                      header_json=json.dumps(header, ensure_ascii=False),
                                      text=text)
        async with sem:
            pred = await _ask_json(model, prompt)
        return (pred.get("rows", []) if pred else []) or []

    all_rows: list[list] = []
    for rows in await asyncio.gather(*[one_seg(s) for s in segs]):
        all_rows.extend(rows)
    return header, all_rows


def _norm_pk(v: Any) -> str:
    pk = str(v).strip()
    if re.fullmatch(r"\d+(\.0)?", pk):
        return str(int(float(pk)))
    return pk


# ---------------------------------------------------------------------------
# 小 doc 直出
# ---------------------------------------------------------------------------


async def _small_doc(model: Model, lines: dict[int, str],
                     knowledge_text: Optional[str], table_name: str,
                     doc_path: str) -> DocStructResult:
    prompt = SMALL_DOC_PROMPT.format(numbered=_numbered(lines),
                                     know_block=_know_block(knowledge_text))
    pred = await _ask_json(model, prompt) or {}
    header = pred.get("header") or []
    pk = pred.get("primary_key") or (header[0] if header else "id")
    rows_arr = pred.get("rows") or []
    rows = [dict(zip(header, r)) for r in rows_arr if r]
    return DocStructResult(doc_path=doc_path, table_name=table_name, primary_key=pk,
                           columns=header, rows=rows)


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


async def structurize_doc_async(
    doc_path: str | Path,
    *,
    model: Model,
    knowledge_md_path: str | Path | None = None,
    table_name: str | None = None,
) -> DocStructResult:
    """把一个散文体 doc 结构化成一张表。

    参数:
        doc_path: .md/.txt 文档路径。
        model: pydantic-ai 模型 (推荐 model_code_a3b_nothink, 已关思考)。
        knowledge_md_path: 同 task 的 knowledge.md (可选, 作字段命名参考)。
        table_name: 表名 (缺省取 doc 文件名 stem)。
    """
    doc_path = Path(doc_path)
    if not doc_path.is_file():
        raise FileNotFoundError(f"doc 不存在: {doc_path}")
    table_name = table_name or doc_path.stem

    knowledge_text: Optional[str] = None
    if knowledge_md_path is not None:
        kp = Path(knowledge_md_path)
        if kp.is_file():
            knowledge_text = kp.read_text(encoding="utf-8")

    lines = _read_lines(doc_path)
    if not lines:
        return DocStructResult(doc_path=str(doc_path), table_name=table_name,
                               primary_key="", columns=[], rows=[],
                               error="empty doc")

    # Stage0 路由: 小 doc 直出
    if len(lines) <= SMALL_DOC_LINES:
        logger.info("[doc_struct_v2] {}: 小doc({}行)直出", doc_path.name, len(lines))
        return await _small_doc(model, lines, knowledge_text, table_name, str(doc_path))

    # Stage1
    s1 = await _stage1_batches(model, lines, knowledge_text)
    batches = s1.get("batches", []) or []
    pk_name = s1.get("primary_key") or (
        batches[0].get("columns", ["id"])[0] if batches else "id")
    transition_lines = s1.get("transition_lines", []) or []
    if not batches:
        return DocStructResult(doc_path=str(doc_path), table_name=table_name,
                               primary_key=pk_name, columns=[], rows=[],
                               error="stage1 no batches")

    # Stage1.5 + 融合合并
    cont = await _stage15_cont(model, batches, lines)
    blocks = _fuse_merge(batches, cont)
    logger.info("[doc_struct_v2] {}: {}物理批次 → {}语义block (pk={})",
                doc_path.name, len(batches), len(blocks), pk_name)

    # Stage3 (各 block 并发, block 内分小批)
    sem = asyncio.Semaphore(INNER_CONCURRENCY)
    block_results = await asyncio.gather(*[
        _stage3_extract_block(model, lines, b, pk_name, sem) for b in blocks
    ])

    # Stage4 按主键 join
    table: dict[str, dict[str, Any]] = {}
    all_cols: list[str] = []
    for header, rows in block_results:
        for col in header:
            if col not in all_cols:
                all_cols.append(col)
        for row in rows:
            if not row:
                continue
            pk = _norm_pk(row[0])
            d = table.setdefault(pk, {})
            for col, val in zip(header, row):
                if val not in (None, "", "null") and col not in d:
                    d[col] = val
    # 主键列保证在首位
    if pk_name in all_cols:
        all_cols = [pk_name] + [c for c in all_cols if c != pk_name]
    rows_out = [dict(rec) for rec in table.values()]

    return DocStructResult(
        doc_path=str(doc_path), table_name=table_name, primary_key=pk_name,
        columns=all_cols, rows=rows_out, blocks=blocks,
        transition_lines=transition_lines)


def structurize_doc(
    doc_path: str | Path,
    *,
    model: Model,
    knowledge_md_path: str | Path | None = None,
    table_name: str | None = None,
) -> DocStructResult:
    """同步包装。"""
    return asyncio.run(structurize_doc_async(
        doc_path, model=model, knowledge_md_path=knowledge_md_path,
        table_name=table_name))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main_cli() -> None:
    import argparse
    from configs import pydantic_predefined_model

    parser = argparse.ArgumentParser(description="block-aware doc 结构化 (v2).")
    parser.add_argument("doc_path", help="目标 .md/.txt 文件")
    parser.add_argument("--knowledge", default=None, help="同 task 的 knowledge.md")
    parser.add_argument("--out", default=None, help="结果 JSON 落盘路径")
    parser.add_argument("--sqlite", default=None, help="写入 sqlite 路径")
    parser.add_argument("--table", default=None, help="表名 (默认 doc stem)")
    args = parser.parse_args()

    result = structurize_doc(
        args.doc_path,
        model=pydantic_predefined_model.model_code_a3b_nothink,
        knowledge_md_path=args.knowledge,
        table_name=args.table,
    )
    print(json.dumps({
        "doc": result.doc_path, "table": result.table_name,
        "primary_key": result.primary_key, "n_blocks": result.n_blocks,
        "n_rows": result.n_rows, "columns": result.columns,
        "first_3_rows": result.rows[:3], "error": result.error,
    }, ensure_ascii=False, indent=2))

    if args.out:
        Path(args.out).write_text(
            json.dumps(result.to_jsonable(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    if args.sqlite:
        result.to_sqlite(args.sqlite)


if __name__ == "__main__":
    _main_cli()
