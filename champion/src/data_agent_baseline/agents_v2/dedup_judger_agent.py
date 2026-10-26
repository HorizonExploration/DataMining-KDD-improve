"""dedup_judger_agent.py — 判断最终 prediction 是否应按输出行去重.

本模块是一个轻量的口径判定 agent: 只根据 question / knowledge.md / 视频文本
(幻灯片占位 + hiccup 版面树 + ASR 旁白) 直接判断最终 prediction.csv 是否应
按目标输出列去重, 并给出分析。

不读取 gold, 不看 prediction, 不针对任何 task 写特例。

入口:
    run_dedup_judger_async(model, task_dir, vote_rounds=5) -> DedupJudgerResult
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents_v2.agent_util import save_history
from agents_v2.doc_relevance_agent import _load_video_text


DEFAULT_KNOWLEDGE_MAX_CHARS = 12000
DEFAULT_VIDEO_MAX_CHARS = 10000
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_VOTE_ROUNDS = 5


class DedupJudgerOut(BaseModel):
    """单轮模型输出: 直接判断是否应对最终结果去重."""

    should_dedup: bool = Field(
        description=(
            "最终 prediction.csv 是否应按目标输出列去重。true 表示应使用 "
            "SELECT DISTINCT 目标输出列或等价 drop_duplicates；false 表示不应把最终行去重作为兜底。"
        ),
    )
    analysis: str = Field(
        description=(
            "简洁说明判定依据: 最终答案行粒度是什么, 为什么应去重或不应去重。"
        ),
    )


@dataclass
class DedupJudgerResult:
    """多轮投票后的去重判定结果."""

    task_id: str
    should_dedup: bool = False
    reason: str = ""
    vote_rounds: int = 0
    n_rounds_ok: int = 0
    should_dedup_tally: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    error: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "should_dedup": self.should_dedup,
            "reason": self.reason,
            "vote_rounds": self.vote_rounds,
            "n_rounds_ok": self.n_rounds_ok,
            "should_dedup_tally": self.should_dedup_tally,
            "reasons": self.reasons,
            "error": self.error,
        }


DEDUP_JUDGER_INSTRUCTION = """你是数据分析任务的**最终结果去重判定 agent**。

你的任务不是解题, 也不是写 SQL, 而是根据 question / knowledge.md / 视频口径,
直接判断最终 prediction.csv 是否应按**目标输出列**去重, 并给出分析。

## 输入
- question: 任务目标。
- knowledge.md: 字段、实体、业务口径说明, 可能为空。
- 视频口径: 若任务有 briefing.mp4, 会以文本形式给出三类信息:
  1. 幻灯片帧占位与时间;
  2. hiccup 版面树中抽到的画面文字/控件/表格;
  3. ASR 旁白文本。
  视频里可能明确说明输出是 records / export records / list / count / group 等粒度。

## should_dedup=true 的情况
只有当最终答案语义是**唯一对象集合**或**唯一对象答案行集合**时, 才判为 true。

1. 用户问 which / list / 哪些 / 名单 / top N 等, 目标输出是实体、公司、股东、人员、
   基金、证券、患者等对象集合。
   - 例: "which major shareholders qualify" 且输出股东名称。
   - SQL 倾向: SELECT DISTINCT 目标对象列。

2. 用户问对象集合, 且同时要求稳定标识/描述字段/业务属性, 这些目标输出列共同构成
   答案行的唯一组合。若底层表是一对多事件/记录表, 应按完整目标输出列去重。
   - 例: 公司+增发目的、基金+代码、基金+投资目标、基金经理+最高学历。
   - SQL 倾向: SELECT DISTINCT 目标输出列。

## should_dedup=false 的情况
只要最终答案语义不是唯一对象集合/唯一对象答案行, 就不要把最终行去重作为兜底。

1. **事件/记录/明细**: 题目要求 records、triggered records、freeze records、交易记录、
   日志、诊疗记录、非空数据记录等。每行代表一次独立事实或一条原始记录, 同一实体多次出现应保留。
   特别地, "which shareholders/entities have ... records and corresponding amount/shares/date"
   通常是在取命中的事件记录及其金额/股数/日期, 不是唯一实体名录; 除非题目明确只要 distinct names。

2. **指标观测序列**: 题目要求回报率、GDP、成交量、存款余额、资产负债表科目、宏观指标、
   历史数据、这些年来的数据等。重复数值和空值可能是不同观测点, 不应为了唯一值或唯一行而去重。

3. **关系实例**: 题目要求 A-B 关系实例, 如基金经理管理的基金简称和风险等级、人员-证券、
   公司-项目等。关系本身是答案粒度, 不应折叠成某一侧实体名单。
   只要题目语义是"满足条件的 A 所管理/持有/关联的 B 及其属性", 就按关系实例处理;
   即使最终输出列只投影 B+属性, 也不要因为看起来像对象+属性而判 true。
   即使每个 B 当前只有一个稳定属性值, 也不能把这种"由 A 筛选后投影 B"的关系结果
   改判成唯一对象答案行。

4. **聚合/单点结果**: 题目要求 count / sum / avg / max / min / group by / 分布 / TopN 聚合排名 /
   first / last / 某对象某属性。SQL 内部可能需要 COUNT(DISTINCT ...) 或 GROUP BY, 但这不是最终
   prediction 行去重问题。

## 关键判定原则
- 看**最终 prediction.csv 的目标输出行粒度**, 不看 SQL 表面词。
- "which/list/哪些" 不自动等于去重:
  - "which shareholders qualify" 且只输出股东名 -> true。
  - "which companies ... and their purposes" -> true, 按公司+目的去重。
  - "which records ... return date and amount" -> false, 这是记录明细。
  - "which shareholders have freeze records and corresponding shares involved" -> false, 这是冻结记录明细。
  - "how many distinct patients" -> false, DISTINCT 属于 COUNT 内部逻辑。
- 回报率、GDP、成交量、余额等数值指标, 即使能想象出对应实体/时间维度, 也优先判 false。
- "他们管理的/持有的/关联的"对象和属性, 优先判 false, 因为答案粒度是关系实例或关系投影。
  典型 false 例子: "哪些基金经理满足要求？请列出他们管理的基金简称和风险等级。"
- most frequently / average / highest / count 这类由聚合得出的结果, 判 false; 用聚合逻辑解决,
  不用最终行去重兜底。
- 视频若明确说 "pull/export triggered records/list of records", 优先判 false。
- 视频若明确说 "unique/distinct names/entities" 或仅要求实体名单, 优先判 true。
- 不要根据某个已知 task 或 gold 反推; 只能依据 question / knowledge / 视频口径做通用判断。

## 输出
严格按 schema 输出:
- should_dedup: boolean;
- analysis: 简洁说明依据。不要输出 markdown, 不要添加额外字段。
"""


def build_dedup_judger_agent(model: Any) -> Agent:
    """构造 dedup judger agent (output_type=DedupJudgerOut)."""
    return Agent(
        model=model,
        instructions=DEDUP_JUDGER_INSTRUCTION,
        output_type=DedupJudgerOut,
    )


def _read_question(task_dir: Path) -> str:
    try:
        return json.loads((task_dir / "task.json").read_text(encoding="utf-8")).get("question", "")
    except Exception as e:  # noqa: BLE001
        logger.warning("[dedup_judger] read task.json failed for {}: {}", task_dir, e)
        return ""


def _read_knowledge(task_dir: Path, max_chars: int) -> str:
    kp = task_dir / "context" / "knowledge.md"
    if not kp.is_file():
        return ""
    text = kp.read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(knowledge 已截断)"
    return text


def _build_user_input(*, question: str, knowledge_md: str, video_text: str) -> str:
    knowledge_section = knowledge_md.strip() if knowledge_md.strip() else "(本任务无 knowledge.md)"
    video_section = video_text.strip() if video_text.strip() else "(本任务无视频口径或视频尚未预处理)"
    return (
        f"## question\n{question}\n\n"
        f"## knowledge.md\n{knowledge_section}\n\n"
        f"## 视频口径文本（三件套: 幻灯片/版面hiccup/旁白）\n{video_section}"
    )


def _vote_should_dedup(preds: list[DedupJudgerOut]) -> bool:
    """多数投票选择是否去重; 平票时保守选择 False, 避免误删记录。"""
    if not preds:
        return False
    cnt = Counter(p.should_dedup for p in preds)
    max_votes = max(cnt.values())
    tied = {flag for flag, n in cnt.items() if n == max_votes}
    if len(tied) > 1:
        return False
    return next(iter(tied))


async def run_dedup_judger_async(
    *,
    model: Any = None,
    task_dir: Path,
    question: Optional[str] = None,
    knowledge_md: Optional[str] = None,
    video_text: Optional[str] = None,
    knowledge_max_chars: int = DEFAULT_KNOWLEDGE_MAX_CHARS,
    video_max_chars: int = DEFAULT_VIDEO_MAX_CHARS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    vote_rounds: int = DEFAULT_VOTE_ROUNDS,
    log_path: Optional[Path] = None,
) -> DedupJudgerResult:
    """对单个 task 判断最终 prediction 是否应按输出行去重.

    Args:
        model: pydantic-ai 模型。若为 None, 默认调用
            ``tools_v2.general_tools.build_model_nothink()`` 构造 no-think Qwen3.5。
        task_dir: task 根目录。
        question / knowledge_md / video_text: 可显式传入; 缺省时从 task_dir 读取。
        video_text: 推荐传入视频三件套文本; 为 None 时自动从
            ``workdir/_video_frames/video_input.json`` 重建, 再退化到 timeline.txt。
        vote_rounds: 3 或 5 均可; 默认 5。多轮并发投票, 失败轮会被忽略。
        log_path: 若提供, 保存第一轮成功的消息历史。

    Returns:
        DedupJudgerResult. 全部失败时 error 非空, 默认保守返回 should_dedup=False。
    """
    task_id = task_dir.name
    result = DedupJudgerResult(task_id=task_id, vote_rounds=max(1, int(vote_rounds)))

    if model is None:
        from tools_v2.general_tools import build_model_nothink

        model = build_model_nothink()

    if question is None:
        question = _read_question(task_dir)
    if knowledge_md is None:
        knowledge_md = _read_knowledge(task_dir, knowledge_max_chars)
    elif len(knowledge_md) > knowledge_max_chars:
        knowledge_md = knowledge_md[:knowledge_max_chars] + "\n…(knowledge 已截断)"
    if video_text is None:
        video_text = _load_video_text(task_dir, video_max_chars)
    elif len(video_text) > video_max_chars:
        video_text = video_text[:video_max_chars] + "\n…(视频文本已截断)"

    user_input = _build_user_input(
        question=question or "",
        knowledge_md=knowledge_md or "",
        video_text=video_text or "",
    )
    agent = build_dedup_judger_agent(model)

    n_rounds = result.vote_rounds

    async def _one_round():
        try:
            run = await asyncio.wait_for(agent.run(user_input), timeout=timeout_s)
            return run.output, run
        except asyncio.TimeoutError:
            return None, f"timeout>{timeout_s}s"
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"

    rounds = await asyncio.gather(*[_one_round() for _ in range(n_rounds)])
    preds: list[DedupJudgerOut] = []
    first_ok_run = None
    errors: list[str] = []
    for out, run_or_err in rounds:
        if out is None:
            errors.append(str(run_or_err))
            continue
        preds.append(out)
        if first_ok_run is None:
            first_ok_run = run_or_err

    if first_ok_run is not None and log_path is not None:
        try:
            save_history(first_ok_run.all_messages(), log_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("[dedup_judger] log save failed for {}: {}", task_id, e)

    if not preds:
        result.should_dedup = False
        result.reason = "dedup judger 全部轮次失败, 保守不做最终行去重"
        result.error = "; ".join(errors) if errors else "all rounds failed"
        return result

    should_dedup = _vote_should_dedup(preds)
    result.should_dedup = should_dedup
    result.n_rounds_ok = len(preds)
    result.should_dedup_tally = {str(k).lower(): v for k, v in Counter(p.should_dedup for p in preds).items()}
    result.reasons = [p.analysis for p in preds]
    # 取属于获胜布尔值的第一条分析; 若没有则取第一轮分析。
    result.reason = next((p.analysis for p in preds if p.should_dedup == should_dedup), preds[0].analysis)
    if errors:
        result.error = "ignored failed rounds: " + "; ".join(errors)
    return result


def run_dedup_judger(
    *,
    model: Any = None,
    task_dir: Path,
    **kwargs: Any,
) -> DedupJudgerResult:
    """同步封装, 便于脚本/调试调用."""
    return asyncio.run(run_dedup_judger_async(model=model, task_dir=task_dir, **kwargs))
