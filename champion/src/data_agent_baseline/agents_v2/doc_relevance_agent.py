"""doc_relevance_agent.py — 在 doc_prepare 之前, 判定每个 doc 是否与 task 相关.

动机
----
``doc_prepare`` (把 ``context/doc/*.md`` 用 LLM 抽成结构化表) 是整个 agent 的
耗时瓶颈: 每个 task 的 doc/ 动辄 5~10 个文件, 但实际与 question 相关的往往只有
1~2 个 (经 experiments/20260610-opus-result 统计, 269 个 doc 中真正被 solver 消费的仅 21 个).

本模块在 doc_prepare **之前** 跑一个轻量判定: 用默认模型 (qwen3.5-35b-a3b)
横向比较 task 的 question/knowledge 与每个 doc 的内容, 输出**逐 doc 的相关性
判定**, 让 doc_prepare 只处理相关 doc, 跳过无关 doc.

判定标准 (重要)
---------------
判据是 **doc 内容是否与 task 任务相关**, 而非 "solver 是否实际消费了它":
即使同样的数据能从 csv/json/sqlite 等其他来源获得, 只要 doc 内容与任务相关
(涉及题目需要的实体/字段/指标/口径), 仍判为相关 → 需要提取.

召回优先 (代价不对称)
---------------------
- 漏召回 (把相关 doc 判成无关 → 被跳过): solver 拿不到数据, 直接答错, **高惩罚**.
- 多召回 (把无关 doc 判成相关 → 多抽一个): 只是多花一点抽取时间, **低惩罚**.
故 prompt 与后处理都明确偏向召回: 模型拿不准时判相关; 模型遗漏未提及的 doc
默认按相关兜底.

为什么不依赖 _doc_extracted.db
------------------------------
本模块必须在 doc_prepare 之前运行, 此时 ``_doc_extracted.db`` 还不存在.
因此 doc 内容一律从原文读取 (md 直接读; pdf 用 ``pdf_reflow`` 纯本地 reflow 取
预览), 不读结构化表.

入口
----
``run_doc_relevance(model, task_dir, ...) -> DocRelevanceResult``
(同步封装 ``run_doc_relevance_async``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 每个 doc 注入 prompt 的预览行数 (叙事体 doc 第一行通常是信息量最高的标题).
DEFAULT_DOC_PREVIEW_LINES = 40

# 单行最长字符数, 防止极长行 (如压平的表格) 撑爆 token.
DEFAULT_DOC_PREVIEW_LINE_CHARS = 400

# question/knowledge 注入上限, 防超长 (knowledge.md 偶尔很大).
DEFAULT_KNOWLEDGE_MAX_CHARS = 8000

# 视频文本 (timeline) 注入上限. timeline 通常仅 20~40 行, 但仍设上限兜底.
DEFAULT_VIDEO_MAX_CHARS = 6000

# 单次(单轮)模型调用超时 (秒). 自建模型正常 1~5s (带 hiccup 视频的大 prompt 也
# 远低于此), 超时基本意味着这次调用 hang 了. 多轮投票并发执行, 单轮超时不影响其余轮.
DEFAULT_TIMEOUT_S = 60.0

# 投票轮数: 并发跑 N 轮独立判定, 再逐 doc 多数投票聚合, 提升结果正确性与稳定性
# (解决弱模型边界 case 的抖动, 如 task_44 跨次 4:1 波动). 失败/超时的轮自动忽略,
# 只要有 >=1 轮成功即可投票; 全部失败才退化为全判相关兜底 (偏召回, 不漏).
DEFAULT_VOTE_ROUNDS = 5


# ---------------------------------------------------------------------------
# 结构化输出 schema
# ---------------------------------------------------------------------------


class DocVerdict(BaseModel):
    """单个 doc 的相关性判定.

    字段顺序刻意为 doc_name → reason → relevant: 让模型先写出判定依据 (reason),
    再据此给出布尔结论 (relevant), 形成"先论证后结论"的思维链, 避免弱模型出现
    "reason 说该召回、relevant 却填 false" 的字段与理由自相矛盾 (实测发生过).
    """

    doc_name: str = Field(
        description="doc 文件的 stem (不含扩展名), 必须与输入清单中的 stem 完全一致",
    )
    reason: str = Field(
        description="先写判定依据: 解题为何确需此 doc (承载哪些实体/字段/指标, 或是哪类实体的"
        "映射桥梁, 或被视频点名); 或为何无关. 这句话的结论必须与下面的 relevant 一致.",
    )
    relevant: bool = Field(
        description="承接上面 reason 的结论: 解题确需此 doc 则 true, 否则 false. "
        "**必须与 reason 一致** —— 若 reason 论证了'确需/应召回', 这里就填 true, 不要填 false. 边界拿不准时填 true.",
    )


class DocRelevanceOut(BaseModel):
    """模型对一个 task 全部 doc 的逐文件判定."""

    verdicts: List[DocVerdict] = Field(
        default_factory=list,
        description="对输入清单中每个 doc 的判定, 每个 doc 一条, doc_name 与清单一一对应",
    )


# ---------------------------------------------------------------------------
# 最终结果数据类 (Q1: 同时给逐 doc 判定 + 必需列表)
# ---------------------------------------------------------------------------


@dataclass
class DocRelevanceResult:
    """一个 task 的 doc 相关性判定结果.

    同时提供两种视图 (Q1 要求两种都输出):
    - ``per_doc``: {stem: relevant_bool}, 逐文件二分类, 直接对接 doc_prepare 的
      "处理/跳过" 开关.
    - ``relevant_docs``: 相关 doc 的 stem 列表 (= per_doc 中 True 的那些).
    """

    task_id: str
    per_doc: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    all_doc_stems: List[str] = field(default_factory=list)
    error: str = ""
    # 投票统计: {stem: "yes/total"}, 记录该 doc 在多少轮里被判相关 (审计/调试用).
    vote_tally: dict[str, str] = field(default_factory=dict)
    n_rounds_ok: int = 0  # 成功参与投票的轮数

    @property
    def relevant_docs(self) -> List[str]:
        """相关 doc 的 stem 列表 (保持 all_doc_stems 的顺序)."""
        return [s for s in self.all_doc_stems if self.per_doc.get(s, True)]

    @property
    def skipped_docs(self) -> List[str]:
        """被判为无关 (可跳过) 的 doc stem 列表."""
        return [s for s in self.all_doc_stems if not self.per_doc.get(s, True)]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "per_doc": self.per_doc,
            "relevant_docs": self.relevant_docs,
            "skipped_docs": self.skipped_docs,
            "reasons": self.reasons,
            "all_doc_stems": self.all_doc_stems,
            "vote_tally": self.vote_tally,
            "n_rounds_ok": self.n_rounds_ok,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# doc 发现与预览
# ---------------------------------------------------------------------------


def _doc_stem_to_paths(doc_dir: Path) -> "dict[str, list[Path]]":
    """扫描 doc_dir, 按 stem 聚合 md/pdf (同名算同一个文档).

    返回 {stem: [path, ...]}, 优先级 md > pdf (读预览时优先用 md).
    """
    stem_map: dict[str, list[Path]] = {}
    for p in sorted(doc_dir.iterdir()):
        if p.name.startswith(".") or p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext not in (".md", ".markdown", ".pdf", ".txt"):
            continue
        stem_map.setdefault(p.stem, []).append(p)
    # 每个 stem 内部排序: md/txt 在前, pdf 在后
    for stem in stem_map:
        stem_map[stem].sort(key=lambda p: 0 if p.suffix.lower() != ".pdf" else 1)
    return stem_map


def _read_doc_preview(
    paths: List[Path],
    *,
    preview_lines: int,
    line_chars: int,
) -> str:
    """读一个 doc (md 优先, 否则 pdf reflow) 的前 N 行作为预览.

    任意失败返回带错误标记的占位串, 不抛异常 (判定阶段绝不阻塞主流程).
    """
    for p in paths:
        try:
            if p.suffix.lower() == ".pdf":
                # pdf: 纯本地 reflow 取文本 (扫描件可能抽不出, 走 except)
                from tools_v2.pdf_reflow import is_text_pdf, reflow_pdf_to_text

                if not is_text_pdf(p):
                    continue
                text = reflow_pdf_to_text(p)
            else:
                text = p.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.debug("[doc_relevance] preview read failed {}: {}", p.name, e)
            continue

        lines = text.splitlines()
        preview = []
        for ln in lines[:preview_lines]:
            ln = ln.rstrip()
            if len(ln) > line_chars:
                ln = ln[:line_chars] + " …(truncated)"
            preview.append(ln)
        body = "\n".join(preview).strip()
        suffix = "" if len(lines) <= preview_lines else f"\n…(共 {len(lines)} 行, 仅展示前 {preview_lines} 行)"
        return body + suffix if body else "(空文档)"

    return "(无法读取预览: 可能是扫描件/图片型 pdf)"


def _load_video_text(task_dir: Path, max_chars: int) -> str:
    """读取 task 的视频讲解文本 (ASR 旁白 + hiccup 版面树), 无则返回空串.

    与 solver agent 引入视频的口径一致: 真正的筛选阈值/字段名/去重规则常藏在 hiccup
    版面树的小字里, 仅 ASR 旁白 (timeline) 是粗线条, 不够. 故这里渲染 **旁白 + hiccup**
    的纯文本 (图片用占位行, 不做多模态 —— 本判定用轻量默认模型, 文本口径已足够判断
    "该用哪些 doc").

    取数优先级:
    1. ``workdir/_video_frames/video_input.json`` (solver 视频预处理的落盘产物, 含
       frames/transcript/hiccup): 用 ``interleave`` 重建 events 后
       ``events_to_text(with_hiccup=True)``. 秒级, 不重抽帧/不重转写, 也不读图片
       (故对 json 里残留的绝对图片路径不敏感).
    2. 兜底 ``workdir/_video_frames/timeline.txt`` (仅旁白, 无 hiccup): 仅在缺
       video_input.json 时用.
    3. 都没有 → 空串 (无视频, 或视频尚未预处理).
    """
    vframes = task_dir / "workdir" / "_video_frames"
    vinput = vframes / "video_input.json"
    text = ""
    if vinput.is_file():
        try:
            import json

            from video.build_video_input import (
                SlideFrame,
                events_to_text,
                interleave,
            )

            saved = json.loads(vinput.read_text(encoding="utf-8"))
            frames = [
                SlideFrame(
                    index=f["index"], t=f["t"], path=f["path"],
                    burst_start=f["burst_start"], burst_end=f["burst_end"],
                )
                for f in saved.get("frames", [])
            ]
            # interleave 只组装 events(挂 hiccup), 不读图片 —— 故 json 里残留的
            # /tmp 绝对图片路径不影响; events_to_text 用占位行表示帧, 同样不读图.
            events = interleave(
                frames,
                saved.get("transcript"),
                hiccup_map=saved.get("hiccup") or {},
            )
            text = events_to_text(events, with_hiccup=True).strip()
        except Exception as e:  # noqa: BLE001
            logger.debug("[doc_relevance] rebuild video text failed: {}", e)
            text = ""
    if not text:
        tl = vframes / "timeline.txt"
        if tl.is_file():
            try:
                text = tl.read_text(encoding="utf-8").strip()
            except Exception as e:  # noqa: BLE001
                logger.debug("[doc_relevance] read timeline.txt failed: {}", e)
                text = ""
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(视频文本已截断)"
    return text


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


DOC_RELEVANCE_INSTRUCTION = """你的任务: 判断每个候选文档 (doc) 是否与数据分析任务 (task) **相关**.

## 背景
一个数据分析 task 有一个 question (任务目标) 和若干数据源. 其中 doc 是非结构化文档
(叙事体的中文/英文报告), 后续会被昂贵的 LLM 流程抽成结构化表供查询. 为节省开销,
需要先筛掉与本 task 无关的 doc.

## 判定标准 (关键): 是否"解题确需使用"这个 doc
判据**不是**字段/领域沾边, 而是: **要解决这道 task 的问题, 是否确实需要用到这个 doc 的内容**.
- **该召回 (relevant=true)**: 解题链路上确实要从这个 doc 取数据/口径/映射才能得到答案. 包括:
  1. **直接承载答案所需数据**: 题目要查/统计/筛选的目标指标、字段、记录就在这个 doc 里.
  2. **必经桥梁 (映射/维表)**: 题目提到一个具体实体 (某个对象的人类可读编号/代码/名称/简称等),
     而这个 doc 记录的正是 "人类可读标识 ↔ 内部主键" 或 "实体 ↔ 其关联对象" 的对应关系.
     不经它就无法把题目里的那个实体接到结构化数据 → 确需, 召回.
     · **关键判定动作 (桥梁型最易被漏判, 务必执行)**: 先看 question 里有没有一个**具体实体标识**
       (任何人类可读的编号/代码/名称/简称, 用来指定"哪一个"对象). 如果有, 再看是否存在某个 doc
       专门登记"该类实体 ↔ 其内部主键 / 关联对象"的对应. 若有, 该 doc 往往是把题目实体接入结构化
       数据的**入口**, 哪怕它**完全不含**题目要求的那个目标指标 (数值/时间/分类等), 也应 relevant=true.
       **不要因为"它不含目标指标数值 / 无法单独给出答案"就判无关** —— 桥梁的价值在于提供映射让你
       能据此去别的表取数, 不在于它自己含不含那个指标.
  3. **视频已点名该 doc 的内容**: 若视频口径 (见下方"视频口径"段) 明确提到/指向了某个 doc 承载的
     字段、实体、表或口径, 即使这些数据别处也有, 也判相关 (视频点名 = 解题要用它).
- **不该召回 (relevant=false)**: 与题目目标是两码事, 解题任何一步都用不到它. 判 false 是常态, 不要犹豫.

## 严禁"沾边即召回" (重点纠正的失败模式)
以下都**不构成**召回理由, 出现这类措辞时应判 relevant=false:
- "同属一个数据库 / 同一领域" (仅仅因为都属于同一业务大类) → 无关.
- "可能有联动 / 是先行指标 / 有背景参考价值" (题目问 A 指标, 该 doc 是另一个仅在宏观上相关的
  B 指标) → 无关.
- "包含通用标识符 (各种代码/编号), 也许能用来 join" —— 若题目并不需要经由该 doc 做实体映射,
  仅仅"它也有代码字段"不算确需 → 无关.
- "无法排除其相关性 / 也许能辅助" 这类说不出具体用途的理由 → 无关.
判断时问自己一句: **"为解出这道题, 我必须打开这个 doc 吗? 它提供了哪一步不可替代的输入?"**
答不上来"哪一步必须用它", 就判 false.

## 召回优先 (代价不对称, 但仅用于边界拿捏)
- 漏判 (相关却判无关 → 被跳过): 解题拿不到数据, 直接答错, **代价极高**.
- 多判 (无关却判相关 → 多抽一个): 多花一点时间, 代价低.
- 因此**在"确需 vs 不确定"的边界上, 倾向召回**: 当你真的拿不准某 doc 是不是解题必经环节时, 判 true.
- 但这只用于边界; **不要把"沾边"当成"边界"** —— 明显只是领域/字段沾边的, 仍判 false.

## 输入
- task question: 任务目标.
- knowledge (可能为空): 字段语义/业务口径说明, 帮助你理解 question 涉及哪些实体和字段.
- 视频口径 (可能为空): 部分 task 带操作讲解视频, 其幻灯片+旁白文本会一并给出. 很多题的真正
  筛选口径/统计维度/要用哪张表只在视频里说 (question 文本本身看不出). **务必结合视频判断该用哪些 doc**;
  视频里点名/指向的字段、实体、表所对应的 doc, 判相关.
- 候选 doc 清单: 每个 doc 给出 stem (文件名, 不含扩展名) 和内容预览 (前若干行,
  叙事体 doc 的首行通常是信息量最高的标题, 已点明该文档讲的是什么实体/指标).

## 输出
对清单中**每一个** doc 输出一条判定 (verdicts 数组), 字段:
- doc_name: 该 doc 的 stem, 必须与清单中的 stem **逐字一致** (含中文原名).
- relevant: true/false.
- reason: 一句话依据 (该 doc 含题目需要的什么; 或为何无关).
**必须覆盖清单里的每一个 doc, 不要遗漏, 不要新增清单外的 doc_name.**
"""


def build_doc_relevance_agent(model: Any) -> Agent:
    """构造 doc-relevance 判定 agent (output_type=DocRelevanceOut)."""
    return Agent(
        model=model,
        instructions=DOC_RELEVANCE_INSTRUCTION,
        output_type=DocRelevanceOut,
    )


def _build_user_input(
    *,
    question: str,
    knowledge_md: str,
    video_text: str,
    doc_previews: "list[tuple[str, str]]",
    knowledge_max_chars: int,
) -> str:
    """拼装喂给模型的 user_input."""
    kn = (knowledge_md or "").strip()
    if len(kn) > knowledge_max_chars:
        kn = kn[:knowledge_max_chars] + "\n…(knowledge 已截断)"
    kn_section = kn if kn else "(本任务无 knowledge.md)"

    doc_blocks = []
    for stem, preview in doc_previews:
        doc_blocks.append(f"### doc stem: {stem}\n{preview}")
    docs_section = "\n\n".join(doc_blocks)

    video_section = ""
    if video_text and video_text.strip():
        video_section = (
            f"## 视频口径 (操作讲解视频的幻灯片+旁白文本)\n{video_text.strip()}\n\n"
        )

    return (
        f"## task question\n{question}\n\n"
        f"## knowledge\n{kn_section}\n\n"
        f"{video_section}"
        f"## 候选 doc 清单 (共 {len(doc_previews)} 个)\n{docs_section}"
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run_doc_relevance_async(
    *,
    model: Any,
    task_dir: Path,
    question: Optional[str] = None,
    knowledge_md: Optional[str] = None,
    video_text: Optional[str] = None,
    preview_lines: int = DEFAULT_DOC_PREVIEW_LINES,
    line_chars: int = DEFAULT_DOC_PREVIEW_LINE_CHARS,
    knowledge_max_chars: int = DEFAULT_KNOWLEDGE_MAX_CHARS,
    video_max_chars: int = DEFAULT_VIDEO_MAX_CHARS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    vote_rounds: int = DEFAULT_VOTE_ROUNDS,
    log_path: Optional[Path] = None,
) -> DocRelevanceResult:
    """判定 ``task_dir/context/doc/`` 下每个 doc 是否与 task 相关.

    Args:
        model: pydantic-ai 兼容模型 (推荐默认 qwen3.5-35b-a3b, build_model_nothink()).
        task_dir: task 根目录 (含 context/). 必须在 doc_prepare 之前调用,
            此时 _doc_extracted.db 尚不存在, 故只读 doc 原文.
        question: task question; 为 None 时从 task_dir/task.json 读取.
        knowledge_md: knowledge.md 全文; 为 None 时从 task_dir/context/knowledge.md 读取.
        video_text: 视频讲解文本 (幻灯片+旁白 timeline); 为 None 时自动从
            ``task_dir/workdir/_video_frames/timeline.txt`` 兜底读取 (无视频则为空).
            很多题的筛选口径/该用哪张表只在视频里, 故强烈建议带上.
        preview_lines / line_chars: 每个 doc 注入的预览行数 / 单行字符上限.
        knowledge_max_chars: knowledge 注入上限.
        video_max_chars: 视频文本注入上限.
        timeout_s: **单轮**模型调用超时秒数. 多轮并发, 单轮超时不影响其余轮.
        vote_rounds: 并发投票轮数 (>=1). 跑 N 轮独立判定后逐 doc 多数投票聚合, 提升
            稳定性. 超时/异常/未覆盖的轮自动忽略; 只要 >=1 轮成功即可投票.
        log_path: 若提供, 落盘 (第一轮成功的) 判定消息历史 (审计用).

    Returns:
        DocRelevanceResult. 无 doc 目录 / 无 doc 时返回空结果 (relevant_docs=[]).
        所有轮都失败时 error 非空, 且**所有 doc 默认按相关兜底** (偏召回, 不漏).
    """
    task_id = task_dir.name
    result = DocRelevanceResult(task_id=task_id)

    doc_dir = task_dir / "context" / "doc"
    if not doc_dir.is_dir():
        logger.debug("[doc_relevance] no context/doc for {}, skip", task_id)
        return result

    stem_map = _doc_stem_to_paths(doc_dir)
    if not stem_map:
        logger.debug("[doc_relevance] no doc files in {}, skip", doc_dir)
        return result

    result.all_doc_stems = list(stem_map.keys())

    # question / knowledge 兜底读取
    if question is None:
        try:
            import json

            question = json.loads(
                (task_dir / "task.json").read_text(encoding="utf-8")
            ).get("question", "")
        except Exception as e:  # noqa: BLE001
            logger.warning("[doc_relevance] read task.json failed for {}: {}", task_id, e)
            question = ""
    if knowledge_md is None:
        kp = task_dir / "context" / "knowledge.md"
        knowledge_md = kp.read_text(encoding="utf-8") if kp.is_file() else ""

    # video_text 兜底: 调用方未传时, 从 workdir/_video_frames/timeline.txt 读取 (无则空串).
    if video_text is None:
        video_text = _load_video_text(task_dir, video_max_chars)

    # 读每个 doc 预览
    doc_previews: list[tuple[str, str]] = []
    for stem, paths in stem_map.items():
        preview = _read_doc_preview(
            paths, preview_lines=preview_lines, line_chars=line_chars
        )
        doc_previews.append((stem, preview))

    user_input = _build_user_input(
        question=question or "",
        knowledge_md=knowledge_md or "",
        video_text=video_text or "",
        doc_previews=doc_previews,
        knowledge_max_chars=knowledge_max_chars,
    )

    agent = build_doc_relevance_agent(model)

    # 默认全部按相关兜底 (偏召回): 任何一轮成功后, 用投票结果覆盖.
    result.per_doc = {stem: True for stem in result.all_doc_stems}
    valid_stems = set(result.all_doc_stems)
    n_rounds = max(1, int(vote_rounds))

    async def _one_round(idx: int):
        """跑一轮判定. 成功返回 (verdicts_dict, run); 失败返回 (None, err_str)."""
        try:
            run = await asyncio.wait_for(agent.run(user_input), timeout=timeout_s)
            out: DocRelevanceOut = run.output
        except asyncio.TimeoutError:
            return None, f"timeout>{timeout_s}s"
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
        verdicts = {v.doc_name: v for v in out.verdicts if v.doc_name in valid_stems}
        return verdicts, run

    # 投票: 循环并发补齐, 直到攒够 n_rounds 个【成功】轮就 break (超时/异常的轮不算
    # 数, 继续补). 这样保证投票基于 n_rounds 个有效票, 而非"发了 n_rounds 个里能成几个".
    # 每批只补"还差的个数", 仍并发执行 (墙钟≈单轮, 非 n_rounds×). max_batches 上限防
    # 模型持续失败时死循环.
    ok_verdicts: list[dict] = []
    first_ok_run = None
    last_error = ""
    max_batches = n_rounds + 2  # 允许少量失败重补, 超过即放弃 (拿现有票投或兜底)
    batches = 0
    while len(ok_verdicts) < n_rounds and batches < max_batches:
        need = n_rounds - len(ok_verdicts)
        batches += 1
        rounds = await asyncio.gather(*[_one_round(i) for i in range(need)])
        for verdicts, run_or_err in rounds:
            if verdicts is None:
                last_error = run_or_err  # 这是 err_str
                continue
            ok_verdicts.append(verdicts)
            if first_ok_run is None:
                first_ok_run = run_or_err  # 这是 run 对象
    if len(ok_verdicts) < n_rounds and ok_verdicts:
        logger.warning(
            "[doc_relevance] {} 仅攒到 {}/{} 个成功轮 (已达重补上限), 用现有票投票",
            task_id, len(ok_verdicts), n_rounds,
        )

    result.n_rounds_ok = len(ok_verdicts)

    if not ok_verdicts:
        # 所有轮都失败 → 全判相关兜底 (偏召回, 不漏).
        result.error = last_error or "all rounds failed"
        logger.warning(
            "[doc_relevance] {} all {} vote rounds failed ({}), all docs default to relevant",
            task_id, n_rounds, result.error,
        )
    else:
        # 逐 doc 多数投票. 票数统计基于"成功轮数"为分母; 某 doc 在某轮未被判定
        # (模型遗漏) 视为该轮弃权, 不计入分母, 但若所有成功轮都没判到它 → 默认相关.
        for stem in valid_stems:
            judged = [vd[stem] for vd in ok_verdicts if stem in vd]
            if not judged:
                # 所有成功轮都没判到该 doc → 偏召回默认相关.
                result.per_doc[stem] = True
                result.vote_tally[stem] = f"0/0(默认相关)"
                result.reasons.setdefault(stem, "所有轮均未判定, 按召回优先默认相关")
                continue
            yes = sum(1 for v in judged if v.relevant)
            n_j = len(judged)
            # 多数票决定; 偏召回的平票处理: yes >= no 即判相关 (yes*2 >= n_j).
            relevant = (yes * 2 >= n_j)
            result.per_doc[stem] = relevant
            result.vote_tally[stem] = f"{yes}/{n_j}"
            # reason 取与多数结论一致的某一轮的理由 (更具代表性).
            picked = next(
                (v.reason for v in judged if bool(v.relevant) == relevant and v.reason),
                "",
            )
            result.reasons[stem] = picked or (judged[0].reason if judged else "")
        result.error = ""
        if result.n_rounds_ok < n_rounds:
            logger.info(
                "[doc_relevance] {} voted with {}/{} successful rounds (others timeout/err)",
                task_id, result.n_rounds_ok, n_rounds,
            )
        # 落盘第一轮成功的消息历史 (审计).
        if log_path is not None and first_ok_run is not None:
            try:
                from agents_v2.agent_util import save_history

                save_history(first_ok_run.all_messages(), log_path)
            except Exception as e:  # noqa: BLE001
                logger.debug("[doc_relevance] log save failed: {}", e)

    logger.info(
        "[doc_relevance] task={} {} doc(s) [{}轮投票]: relevant={}, skipped={}, tally={}",
        task_id,
        len(result.all_doc_stems),
        result.n_rounds_ok,
        result.relevant_docs,
        result.skipped_docs,
        result.vote_tally,
    )
    return result


def run_doc_relevance(
    *,
    model: Any,
    task_dir: Path,
    **kwargs: Any,
) -> DocRelevanceResult:
    """``run_doc_relevance_async`` 的同步封装."""
    return asyncio.run(
        run_doc_relevance_async(model=model, task_dir=task_dir, **kwargs)
    )
