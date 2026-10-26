"""table_relevance_agent.py — 在把结构化表 schema 注入 solver prompt 之前, 判定每张表是否与 task 相关.

动机
----
一个 task 的 ``context/`` 下动辄 8~26 张结构化表 (csv/json/db/sqlite), 但经
``build_table_relevance_gold.py`` 用 opus solver 反推统计: 真正解题用到的表中位数仅 1 张,
**无关表占比中位数 93%**. 这些无关表的 schema 描述 (表名+列名+样例) 会被
``describe_context_dir`` 全量拼进 solver prompt (实测中位 ~1.5万 token), 对弱模型既费 token
又是纯噪声干扰.

本模块类比 ``doc_relevance_agent``: 在 describe 注入之前跑一个轻量判定, 用默认模型
(qwen3.5-35b-a3b) 横向比较 task 的 question/knowledge/视频口径 与每张表的 schema 预览,
输出**逐表相关性判定**, 让下游 describe 把无关表折叠掉.

软过滤 (关键: 与 doc_relevance 的本质区别)
-------------------------------------------
doc 漏判 = 该 doc 不被抽取 → solver **彻底拿不到数据** → 答错 (不可逆, 高危).
表漏判 = 仅仅在 prompt 描述里被折叠, 但 ``explore_data`` 工具仍注册全部表、solver.py
脚手架仍 load 全部表 → solver 随时能 ``SHOW TABLES`` / 查 schema **找回**.
故表漏判代价从"答错"降为"多探一步". 本判定结果**只用于折叠 prompt 描述, 绝不影响
explore_data 注册与 scaffold 读数据**.

召回优先 (代价不对称, 罚漏召回 10 倍)
--------------------------------------
- 漏召回 (相关表判无关 → 描述被折叠): solver 需多花一步 explore 找回, **高惩罚**.
- 多召回 (无关表判相关 → 多留一段描述): 只是少省一点 token, **低惩罚**.
故 prompt 与平票兜底都偏向召回: 拿不准判相关; 全失败时默认全部相关.

_doc_extracted 豁免
-------------------
``_doc_extracted.db`` 的表是 doc 抽取产物, 已由 ``doc_relevance_agent`` 把关, 本模块
**不判定它们** (始终详细展示), 避免对同一份数据双重过滤.

入口
----
``run_table_relevance_async(model, task_dir, ...) -> TableRelevanceResult``
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from tools_v2.datasource_runtime import _quote, build_datasource_state


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 每张表注入 prompt 的样例行数 (judge 只需粗看值的形态, 2 行足够).
DEFAULT_TABLE_SAMPLE_ROWS = 2

# 单元格值最长字符 (防极长文本字段撑爆 token).
DEFAULT_CELL_CHARS = 80

# question/knowledge 注入上限.
DEFAULT_KNOWLEDGE_MAX_CHARS = 8000

# 视频文本注入上限.
DEFAULT_VIDEO_MAX_CHARS = 6000

# 单轮模型调用超时.
DEFAULT_TIMEOUT_S = 90.0

# 投票轮数 (并发跑 N 轮, 逐表多数投票聚合, 提升稳定性).
DEFAULT_VOTE_ROUNDS = 5

# doc 抽取库 stem: 这些表归 doc_relevance 管, 本模块不判定 (始终视为相关).
DOC_EXTRACTED_STEMS = {"_doc_extracted"}


# ---------------------------------------------------------------------------
# 结构化输出 schema
# ---------------------------------------------------------------------------


class TableVerdict(BaseModel):
    """单张表的相关性判定.

    字段顺序刻意为 table_name → reason → relevant: 先论证后结论的思维链,
    避免弱模型 reason/relevant 自相矛盾 (doc_relevance 实测发生过).
    """

    table_name: str = Field(
        description="表名 (canonical), 必须与输入清单中的表名完全一致",
    )
    reason: str = Field(
        description="先写判定依据: 解题为何确需此表 (承载哪些目标指标/字段, 或是哪类实体的"
        "映射桥梁); 或为何无关. 这句话的结论必须与下面的 relevant 一致.",
    )
    relevant: bool = Field(
        description="承接 reason 的结论: 解题确需此表则 true, 否则 false. "
        "**必须与 reason 一致**. 边界拿不准时填 true (偏召回).",
    )


class TableRelevanceOut(BaseModel):
    """模型对一个 task 全部表的逐表判定."""

    verdicts: List[TableVerdict] = Field(
        default_factory=list,
        description="对输入清单中每张表的判定, 每张表一条, table_name 与清单一一对应",
    )


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------


@dataclass
class TableRelevanceResult:
    """一个 task 的表相关性判定结果."""

    task_id: str
    per_table: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    all_tables: List[str] = field(default_factory=list)
    # 始终保留的表 (例如 _doc_extracted), 不参与判定但下游应详细展示.
    always_keep: List[str] = field(default_factory=list)
    # canonical -> {"source_rel": str, "sqlite_table": str|None}, 供构造 describe 折叠 key.
    table_meta: dict[str, dict] = field(default_factory=dict)
    error: str = ""
    vote_tally: dict[str, str] = field(default_factory=dict)
    n_rounds_ok: int = 0

    @property
    def relevant_tables(self) -> List[str]:
        """判为相关的表 (canonical) + always_keep. 顺序保持 all_tables."""
        rel = [t for t in self.all_tables if self.per_table.get(t, True)]
        return rel + [t for t in self.always_keep if t not in rel]

    @property
    def skipped_tables(self) -> List[str]:
        """判为无关 (可折叠) 的表."""
        return [t for t in self.all_tables if not self.per_table.get(t, True)]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "per_table": self.per_table,
            "relevant_tables": self.relevant_tables,
            "skipped_tables": self.skipped_tables,
            "always_keep": self.always_keep,
            "skipped_describe_keys": self.skipped_describe_keys,
            "reasons": self.reasons,
            "all_tables": self.all_tables,
            "vote_tally": self.vote_tally,
            "n_rounds_ok": self.n_rounds_ok,
            "error": self.error,
        }

    @property
    def skipped_describe_keys(self) -> List[str]:
        """被判无关的表, 转成 describe_tool 的折叠 key:
        - csv/json: ``source_rel`` (相对 task_dir 的路径, 如 ``context/csv/x.csv``).
        - sqlite 表: ``{source_rel}::{sqlite_table}`` (一个 db 文件内逐表).
        describe 端按这套 key 决定哪些折叠. 永远不会包含 always_keep / _doc_extracted.
        """
        keys: List[str] = []
        for tbl in self.skipped_tables:
            meta = self.table_meta.get(tbl)
            if not meta:
                continue
            rel = meta.get("source_rel", "")
            sqlite_table = meta.get("sqlite_table")
            keys.append(f"{rel}::{sqlite_table}" if sqlite_table else rel)
        return keys


# ---------------------------------------------------------------------------
# 表预览渲染 (judge 输入)
# ---------------------------------------------------------------------------


def _truncate(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def _render_table_preview(
    state, canonical: str, *, sample_rows: int, cell_chars: int
) -> str:
    """渲染单张表的 schema 预览: 列名+类型 + N 行样例. 失败返回占位."""
    try:
        cols_info = state.conn.execute(f"DESCRIBE {_quote(canonical)}").fetchall()
        col_line = ", ".join(f"{c[0]}:{c[1]}" for c in cols_info)
    except Exception as e:  # noqa: BLE001
        return f"  columns: (无法读取: {e})"
    lines = [f"  columns ({len(cols_info)}): {col_line}"]
    try:
        rows = state.conn.execute(
            f"SELECT * FROM {_quote(canonical)} LIMIT {sample_rows}"
        ).fetchall()
        names = [c[0] for c in cols_info]
        for i, r in enumerate(rows, 1):
            cells = {n: _truncate(v, cell_chars) for n, v in zip(names, r)}
            lines.append(f"  sample{i}: {cells}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


TABLE_RELEVANCE_INSTRUCTION = """你的任务: 判断每张候选数据表 (table) 是否与数据分析任务 (task) **相关**.

## 背景
一个数据分析 task 有一个 question (任务目标) 和若干结构化数据表 (csv/json/数据库表).
后续这些表的 schema (表名+列名+样例) 会被拼进求解程序的提示词. 一个 task 往往挂了
十几到二十几张表, 但真正解题用到的通常只有 1~3 张, 其余是同库里不相关的表. 为降噪 + 省
token, 需要先筛掉与本 task 无关的表.

## 判定标准 (关键): 是否"解题确需使用"这张表
判据**不是**字段/领域沾边, 而是: **要解决这道 task 的问题, 是否确实需要从这张表取数**.

判定前先做一件事 — **拆解题目的产出**: 一道题的答案通常由两部分构成,
(a) **筛选/计算口径** 用到的指标字段, 和 (b) **最终要呈现的输出列** (题目里"列出…/给出…"
后面跟的那些属性). 这两部分各自的字段可能分散在不同的表里. 判定每张表时, 都对照这份拆解
看它是否提供了其中某一项不可替代的字段.

- **该召回 (relevant=true)**:
  1. **承载口径或输出所需字段**: 题目用于筛选/分组/计算的指标, 或最终要呈现的任何一个输出
     列, 其字段出自这张表. **输出列和筛选指标同等重要** —— 一张表哪怕不含任何筛选条件,
     只要它是某个输出列的来源, 漏掉它结果就缺列, 必须 relevant=true.
  2. **不替这道题"脑补"字段归属**: 判断一张表无关时, 依据应是**它的真实列里没有**本题需要
     的字段, 而不是"我猜另一张表里大概也有这一列". 不要因为"别处可能也有"就跳过一张确实
     提供了所需字段的表 —— 是否真有, 以各表列出的列清单为准.
  3. **必经桥梁 (映射/维表)**: 题目给了一个具体实体标识 (人类可读的编号/代码/名称/简称),
     而这张表记录"该类实体 ↔ 内部主键"或"实体 ↔ 其关联对象"的对应. 不经它就无法把题目
     里的实体接到目标数据 → 确需, 召回. **即使它不含目标指标数值, 也判 true** (桥梁的
     价值在于提供映射, 不在于含不含筛选指标).
  4. **视频已点名该表的内容**: 若视频口径 (见下方"视频口径"段) 明确提到/指向某张表承载的
     字段、实体、口径, 即使数据别处也有, 也判相关.
- **不该召回 (relevant=false)**: 与题目目标是两码事, 解题任何一步都用不到它. 判 false 是常态.

## 严禁"沾边即召回" (重点纠正的失败模式)
以下都**不构成**召回理由, 出现这类措辞应判 relevant=false:
- "同属一个数据库 / 同一业务领域" → 无关.
- "可能有联动 / 是先行指标 / 有背景参考价值" → 无关.
- "包含通用标识符 (各种代码/编号), 也许能 join" —— 若题目并不需要经由该表做实体映射,
  仅仅"它也有代码字段"不算确需 → 无关.
- "无法排除其相关性 / 也许能辅助" 这类说不出具体用途的理由 → 无关.
问自己一句: **"为解出这道题, 我必须用这张表吗? 它提供了哪一步不可替代的输入?"**
答不上来"哪一步必须用它", 就判 false.
判完全部表后再反向自查一句: **"题目要呈现的每一个输出列、用到的每一个筛选指标, 其来源表
我都召回了吗?"** —— 若某个所需字段的唯一来源表被判了 false, 那是漏召回, 必须纠回 true.

## 召回优先 (代价不对称, 但仅用于边界拿捏)
- 漏判 (相关却判无关 → 描述被折叠): 求解程序需多花一步去找回, **代价较高**.
- 多判 (无关却判相关 → 多留一段描述): 多花一点 token, 代价低.
- 因此**在"确需 vs 不确定"的边界上, 倾向召回**: 真拿不准某表是不是解题必经环节时, 判 true.
- 但这只用于边界; **不要把"沾边"当成"边界"** —— 明显只是领域/字段沾边的, 仍判 false.

## 输入
- task question: 任务目标.
- knowledge (可能为空): 字段语义/业务口径说明.
- 视频口径 (可能为空): 部分 task 带操作讲解视频, 其幻灯片+旁白文本会给出. 很多题真正的
  筛选口径/要用哪张表只在视频里说. **务必结合视频判断该用哪些表**.
- 候选表清单: 每张表给出表名 (canonical) + 列清单 + 几行样例.

## 输出
对清单中**每一张**表输出一条判定 (verdicts 数组), 字段:
- table_name: 表名, 必须与清单中的表名**逐字一致**.
- relevant: true/false.
- reason: 一句话依据.
**必须覆盖清单里的每一张表, 不要遗漏, 不要新增清单外的 table_name.**
"""


def build_table_relevance_agent(model: Any) -> Agent:
    """构造表相关性判定 agent (output_type=TableRelevanceOut)."""
    return Agent(
        model=model,
        instructions=TABLE_RELEVANCE_INSTRUCTION,
        output_type=TableRelevanceOut,
    )


def _build_user_input(
    *,
    question: str,
    knowledge_md: str,
    video_text: str,
    table_blocks: "list[tuple[str, str]]",
    knowledge_max_chars: int,
) -> str:
    """拼装喂给模型的 user_input."""
    kn = (knowledge_md or "").strip()
    if len(kn) > knowledge_max_chars:
        kn = kn[:knowledge_max_chars] + "\n…(knowledge 已截断)"
    kn_section = kn if kn else "(本任务无 knowledge.md)"

    blocks = [f"### 表: {name}\n{preview}" for name, preview in table_blocks]
    tables_section = "\n\n".join(blocks)

    video_section = ""
    if video_text and video_text.strip():
        video_section = (
            f"## 视频口径 (操作讲解视频的幻灯片+旁白文本)\n{video_text.strip()}\n\n"
        )

    return (
        f"## task question\n{question}\n\n"
        f"## knowledge\n{kn_section}\n\n"
        f"{video_section}"
        f"## 候选表清单 (共 {len(table_blocks)} 张)\n{tables_section}"
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run_table_relevance_async(
    *,
    model: Any,
    task_dir: Path,
    question: Optional[str] = None,
    knowledge_md: Optional[str] = None,
    video_text: Optional[str] = None,
    sample_rows: int = DEFAULT_TABLE_SAMPLE_ROWS,
    cell_chars: int = DEFAULT_CELL_CHARS,
    knowledge_max_chars: int = DEFAULT_KNOWLEDGE_MAX_CHARS,
    video_max_chars: int = DEFAULT_VIDEO_MAX_CHARS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    vote_rounds: int = DEFAULT_VOTE_ROUNDS,
    log_path: Optional[Path] = None,
) -> TableRelevanceResult:
    """判定 ``task_dir/context/`` 下每张结构化表是否与 task 相关.

    Args:
        model: pydantic-ai 兼容模型 (推荐 build_model_nothink()).
        task_dir: task 根目录 (含 context/).
        question: 为 None 时从 task_dir/task.json 读取.
        knowledge_md: 为 None 时从 task_dir/context/knowledge.md 读取.
        video_text: 为 None 时从 task_dir/workdir/_video_frames 自动读取 (无视频则空).
        sample_rows / cell_chars: 每张表注入的样例行数 / 单元格字符上限.
        vote_rounds: 并发投票轮数. 超时/异常的轮自动忽略; 全失败则全判相关兜底.
        log_path: 若提供, 落盘第一轮成功的判定消息历史 (审计).

    Returns:
        TableRelevanceResult. 无表时返回空结果. 所有轮失败时 error 非空且全判相关兜底.
    """
    task_id = task_dir.name
    result = TableRelevanceResult(task_id=task_id)

    if not (task_dir / "context").is_dir():
        logger.debug("[table_relevance] no context for {}, skip", task_id)
        return result

    # 1) 构造物理表清单 (排除 _doc_extracted → always_keep)
    try:
        state = build_datasource_state(task_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("[table_relevance] build_datasource_state failed for {}: {}", task_id, e)
        result.error = f"datasource: {type(e).__name__}: {e}"
        return result

    judged_tables: list[str] = []
    for t in state.tables:
        if Path(t.source_rel).stem in DOC_EXTRACTED_STEMS:
            if t.canonical not in result.always_keep:
                result.always_keep.append(t.canonical)
            continue
        if t.canonical not in judged_tables:
            judged_tables.append(t.canonical)
            result.table_meta[t.canonical] = {
                "source_rel": t.source_rel,
                "sqlite_table": t.sqlite_table,
                "source_type": t.source_type,
            }

    result.all_tables = judged_tables
    if not judged_tables:
        logger.debug("[table_relevance] no judgeable tables for {}, skip", task_id)
        return result

    # question / knowledge 兜底读取
    if question is None:
        try:
            import json

            question = json.loads(
                (task_dir / "task.json").read_text(encoding="utf-8")
            ).get("question", "")
        except Exception as e:  # noqa: BLE001
            logger.warning("[table_relevance] read task.json failed for {}: {}", task_id, e)
            question = ""
    if knowledge_md is None:
        kp = task_dir / "context" / "knowledge.md"
        knowledge_md = kp.read_text(encoding="utf-8") if kp.is_file() else ""

    # video_text 兜底: 复用 doc_relevance 的 _load_video_text (口径一致).
    if video_text is None:
        try:
            from agents_v2.doc_relevance_agent import _load_video_text

            video_text = _load_video_text(task_dir, video_max_chars)
        except Exception as e:  # noqa: BLE001
            logger.debug("[table_relevance] load video text failed: {}", e)
            video_text = ""

    # 2) 渲染每张表预览
    table_blocks: list[tuple[str, str]] = []
    for canonical in judged_tables:
        preview = _render_table_preview(
            state, canonical, sample_rows=sample_rows, cell_chars=cell_chars
        )
        table_blocks.append((canonical, preview))

    user_input = _build_user_input(
        question=question or "",
        knowledge_md=knowledge_md or "",
        video_text=video_text or "",
        table_blocks=table_blocks,
        knowledge_max_chars=knowledge_max_chars,
    )

    agent = build_table_relevance_agent(model)

    # 默认全判相关兜底 (偏召回): 任何一轮成功后用投票结果覆盖.
    result.per_table = {t: True for t in judged_tables}
    valid = set(judged_tables)
    n_rounds = max(1, int(vote_rounds))

    async def _one_round(idx: int):
        try:
            run = await asyncio.wait_for(agent.run(user_input), timeout=timeout_s)
            out: TableRelevanceOut = run.output
        except asyncio.TimeoutError:
            return None, f"timeout>{timeout_s}s"
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
        verdicts = {v.table_name: v for v in out.verdicts if v.table_name in valid}
        return verdicts, run

    # 投票: 循环并发补齐, 直到攒够 n_rounds 个成功轮. 每批只补还差的个数, 仍并发执行.
    ok_verdicts: list[dict] = []
    first_ok_run = None
    last_error = ""
    max_batches = n_rounds + 2
    batches = 0
    while len(ok_verdicts) < n_rounds and batches < max_batches:
        need = n_rounds - len(ok_verdicts)
        batches += 1
        rounds = await asyncio.gather(*[_one_round(i) for i in range(need)])
        for verdicts, run_or_err in rounds:
            if verdicts is None:
                last_error = run_or_err
                continue
            ok_verdicts.append(verdicts)
            if first_ok_run is None:
                first_ok_run = run_or_err
    if len(ok_verdicts) < n_rounds and ok_verdicts:
        logger.warning(
            "[table_relevance] {} 仅攒到 {}/{} 个成功轮, 用现有票投票",
            task_id, len(ok_verdicts), n_rounds,
        )

    result.n_rounds_ok = len(ok_verdicts)

    if not ok_verdicts:
        result.error = last_error or "all rounds failed"
        logger.warning(
            "[table_relevance] {} all {} rounds failed ({}), all tables default to relevant",
            task_id, n_rounds, result.error,
        )
    else:
        # 逐表多数投票. 某表所有成功轮都没判到 → 偏召回默认相关.
        for tbl in valid:
            judged = [vd[tbl] for vd in ok_verdicts if tbl in vd]
            if not judged:
                result.per_table[tbl] = True
                result.vote_tally[tbl] = "0/0(默认相关)"
                result.reasons.setdefault(tbl, "所有轮均未判定, 按召回优先默认相关")
                continue
            yes = sum(1 for v in judged if v.relevant)
            n_j = len(judged)
            relevant = (yes * 2 >= n_j)  # 偏召回平票: yes>=no 即相关
            result.per_table[tbl] = relevant
            result.vote_tally[tbl] = f"{yes}/{n_j}"
            picked = next(
                (v.reason for v in judged if bool(v.relevant) == relevant and v.reason),
                "",
            )
            result.reasons[tbl] = picked or (judged[0].reason if judged else "")
        result.error = ""
        if log_path is not None and first_ok_run is not None:
            try:
                from agents_v2.agent_util import save_history

                save_history(first_ok_run.all_messages(), log_path)
            except Exception as e:  # noqa: BLE001
                logger.debug("[table_relevance] log save failed: {}", e)

    logger.info(
        "[table_relevance] task={} {} judgeable table(s) [{}轮投票]: relevant={}, skipped={}, always_keep={}",
        task_id,
        len(judged_tables),
        result.n_rounds_ok,
        [t for t in judged_tables if result.per_table.get(t, True)],
        result.skipped_tables,
        result.always_keep,
    )
    return result


def run_table_relevance(
    *,
    model: Any,
    task_dir: Path,
    **kwargs: Any,
) -> TableRelevanceResult:
    """``run_table_relevance_async`` 的同步封装."""
    return asyncio.run(
        run_table_relevance_async(model=model, task_dir=task_dir, **kwargs)
    )
