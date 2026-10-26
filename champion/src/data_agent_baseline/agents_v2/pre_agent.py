"""Pre-agent: 在主 solver 启动前, 从 knowledge.md 中预抽取与 question
相关的 use cases, 同时对任务做结构化预测 (输出列名, 行数限制, 任务类型,
任务摘要), 作为提示喂给 solver, 提升 SQL 生成质量.

调用入口:
    run_pre_agent(model, task_question, knowledge_md, log_path=..., vote_rounds=1)
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents_v2.agent_util import save_history


PRE_AGENT_INSTRUCTION = """
你的任务是基于任务目标(`question`)从知识库(`knowledge.md`)中抽取信息, 并对任务做结构化预测,
为后续 SQL 生成做准备.

## 你需要产出
1. **related_use_cases**: 从 knowledge.md 的 "Exemplar Use Cases" / "Examples" 等章节中
   找出与 question **SQL pattern 强对齐**的案例, 直接摘录原文 (含 SQL 与 explanation).
   多条按相关度降序排列.

   **判断"强对齐"的标准** (必须同时满足才摘):
   - case 的 SQL 形态 (聚合 / 排序 / join / 子查询) 与本题应有形态一致
   - case 涉及的表 / 字段 与本题用到的表 / 字段重叠

   **负向规则 (重要)**:
   - 不要因为 domain 沾边 (都关于 F1, school, post 等) 就摘. 例如本题问 "lowest cost",
     而 use case 讲 "budget utilization 比率" → SQL pattern 不一致, 不要摘.
   - 本题问"按元素过滤 bond", use case 讲"carcinogenic 三键" → 表对但逻辑不同, 不要摘.
   - 若实在没有强相关 use case, **宁可留空字符串**, 也不要硬塞一条 domain 沾边但解题
     pattern 不同的 case (这会误导 solver).

2. **related_field_constraints**: 从 knowledge.md 中摘录与本题字段相关的**非 use-case**信息,
   补足 use case 缺失的 schema 知识. 包含但不限于以下章节:
   - "Core Entities & Fields" / "Schema" / 字段定义段落: 摘录本题用到的字段的语义说明,
     特别是**编码映射** (e.g. Thrombosis: 1=most severe, 2=severe; SEX: M/F; Admission: +/-),
     **取值范围** (e.g. WBC normal range 3.5-9.0; UA > 8.0 abnormal; LDH > 500 abnormal).
   - "Constraints & Conventions": 大小写敏感 / 日期格式 / 单位换算 / 计算公式.
   - **doc 文件指针**: 若 knowledge.md 提到某些 ID 映射 / 实体定义在 `context/doc/xxx.md` 这种
     非结构化文档里 (比如 event_id↔event_name 映射, league_id 映射, raceId 映射), 必须摘出来,
     提醒 solver 去读对应 doc 文件.
   - 散文形式的维表说明 (e.g. major.md 里描述的 major 列表需要 regex 抽取).

   摘录时直接保留原文段落 / bullet, 多条用空行分隔. 若确实没有相关字段说明则留空字符串.
   **关键: 即使 related_use_cases 为空, related_field_constraints 也可以非空, 二者独立判断.**

3. **output_columns**: 预测最终 prediction.csv 应该包含的英文列名集合.
   - 命名应符合 SQL 变量名规则 (英文 + 数字 + 下划线). 列名本身不重要 (评测看的是值是否匹配),
     但**列覆盖**很重要: 缺一列扣分较多, 多一列扣分较少.
   - **摇摆原则**: 当不确定一列该不该输出时, **倾向于加上**, 不要漏.
   - **拆分原则**: 题目要求 "full name" / "name" 等可能由多个底层字段组成的字段时,
     若 schema 中存在 first_name + last_name 这种拆分字段, 优先把**两列分别**输出
     (例如 ["first_name", "last_name"]), 而不是合并为单列 "full_name".
     同理 "address" 若拆分为 street + city + state 等也分别输出.
   - 题目要求 "list ID, sex, diagnosis"  → ["ID", "SEX", "Diagnosis"]
   - 题目要求 "how many ..."             → ["count"] (单一计数列)
   - 题目要求 "what is the average X"    → ["avg_X"]
   - 题目要求 "list top 3 X by Y"        → ["X", "Y"] (X 是被列举对象, Y 是排序依据通常也要包含)
   - 涉及百分比/比率: 通常一列即可, 例如 ["percentage"] / ["ratio"].
   - 当题目同时问多个属性 (e.g. "name and funding type" / "ID and date"): 务必都列出.
   - 列名歧义/可选时, **倾向于把题目明确提到的字段都加进来**, 不要漏掉.

4. **row_limit**: 预测最终输出的行数上限.
   判定原则: 优先看 SQL **形态**, 而不是看题目语义"听起来"是不是唯一.
   - 真正的标量聚合 (顶层是 COUNT/SUM/AVG/MIN/MAX/比率/百分比, 且无 GROUP BY): row_limit=1
     · 题面常带 "how many", "what is the total/average/sum", "what percentage", "what ratio".
   - 题目显式给出固定行数: "top N" / "list 3" / "the N largest" → row_limit=N
   - **极值题 (lowest / highest / most / fastest / best / largest / smallest / Nth / second / third 等)**:
     默认 row_limit=-1 (unlimited). 因为正确写法是 `WHERE col = (SELECT MIN/MAX ...)`,
     可能命中并列多行. 仅当题目用 "the one" / "top 1" 等显式唯一化语句时, 才设 1.
   - **过滤后取列 (无聚合, 无 GROUP BY)**: 即使题面是 "what is X of Y" / "state the date when ..."
     看起来语义唯一, 也设 row_limit=-1. 因为 SQL 形态决定行数, 过滤命中数取决于数据.
   - 其他不确定时倾向 -1 (unlimited).

5. **task_type**: 任务大类, 取以下之一:
   - "lookup_filter" : 简单 SELECT + WHERE, 无聚合 (e.g. "list patients where ...")
   - "aggregate"     : 顶层 COUNT/SUM/AVG/MIN/MAX 等聚合产出标量或几行
   - "ranking"       : top-N / 排序取前几名
   - "ratio"         : 比率/百分比/对比
   - "join_complex"  : 多表 join + 子查询/HAVING 等组合逻辑
   - "other"         : 不在以上类别

6. **task_summary**: 用一句中文 (≤ 60 字) 概括题目要求, 不要照抄英文原文.

## 输出要求
严格按照下方 schema 输出, 不要添加额外解释 / markdown 代码块.
"""


TaskType = Literal[
    "lookup_filter",
    "aggregate",
    "ranking",
    "ratio",
    "join_complex",
    "other",
]


class PreAgentOut(BaseModel):
    related_use_cases: str = Field(
        default="",
        description="从 knowledge.md 的 Exemplar Use Cases 中摘录与本题 SQL pattern 强对齐的案例; 无强相关时留空",
    )
    related_field_constraints: str = Field(
        default="",
        description="从 knowledge.md 的 Core Entities & Fields / Constraints 章节摘录的字段语义、编码映射、阈值、doc 文件指针等; 无相关时留空",
    )
    output_columns: List[str] = Field(
        default_factory=list,
        description="预测 prediction.csv 应该包含的英文列名 (case-insensitive)",
    )
    row_limit: int = Field(
        default=-1,
        description="预测最终输出的行数上限; -1 表示 unlimited",
    )
    task_type: TaskType = Field(
        default="other",
        description="任务大类: lookup_filter | aggregate | ranking | ratio | join_complex | other",
    )
    task_summary: str = Field(
        default="",
        description="用一句中文概括题目要求 (≤ 60 字)",
    )


def build_pre_agent(model: Any) -> Agent:
    """构造 pre-agent 实例 (output_type=PreAgentOut)."""
    return Agent(
        model=model,
        instructions=PRE_AGENT_INSTRUCTION,
        output_type=PreAgentOut,
    )


def _vote_aggregate(preds: List[PreAgentOut]) -> PreAgentOut:
    """把多轮 PreAgentOut 通过简单投票聚合成一份."""
    if len(preds) == 1:
        return preds[0]

    # output_columns: 取出现频次 > 一半的列, 按首次出现顺序保留
    flat_lower_per_pred: List[List[str]] = [[c.strip() for c in p.output_columns] for p in preds]
    name_freq: Counter = Counter()
    canonical: dict[str, str] = {}  # lower -> first seen original
    order: List[str] = []
    for cols in flat_lower_per_pred:
        seen_in_round = set()
        for c in cols:
            key = c.lower()
            if key in seen_in_round:
                continue
            seen_in_round.add(key)
            if key not in canonical:
                canonical[key] = c
                order.append(key)
            name_freq[key] += 1

    threshold = max(1, len(preds) // 2 + 1)  # 多数投票
    voted_cols = [canonical[k] for k in order if name_freq[k] >= threshold]
    if not voted_cols:
        # fallback: 取第一份预测
        voted_cols = preds[0].output_columns

    # row_limit: 多数投票, 平票取最常见
    rl_freq = Counter(p.row_limit for p in preds)
    voted_rl = rl_freq.most_common(1)[0][0]

    # task_type: 多数投票
    tt_freq = Counter(p.task_type for p in preds)
    voted_tt = tt_freq.most_common(1)[0][0]

    # related_use_cases / related_field_constraints / task_summary: 取最长的那份(信息量近似最多)
    longest_uc = max(preds, key=lambda p: len(p.related_use_cases or "")).related_use_cases
    longest_fc = max(preds, key=lambda p: len(p.related_field_constraints or "")).related_field_constraints
    longest_sum = max(preds, key=lambda p: len(p.task_summary or "")).task_summary

    return PreAgentOut(
        related_use_cases=longest_uc,
        related_field_constraints=longest_fc,
        output_columns=voted_cols,
        row_limit=voted_rl,
        task_type=voted_tt,
        task_summary=longest_sum,
    )


async def run_pre_agent(
    model: Any,
    task_question: str,
    knowledge_md: str,
    skill_info: str = "",
    log_path: Optional[Path] = None,
    vote_rounds: int = 1,
) -> PreAgentOut:
    """跑 pre-agent, 返回结构化的 PreAgentOut.

    Args:
        model: pydantic_ai 模型实例.
        task_question: 当前任务的 question 字段.
        knowledge_md: knowledge.md 全文.
        skill_info: bird 数据集 SQL skill 内容, 为空时不注入.
        log_path: 若提供, 把第一轮 ``result.all_messages()`` 落到该路径.
        vote_rounds: 多轮投票次数 (≥ 1). 当 > 1 时并发跑多轮, 然后做投票聚合.
    """
    if not isinstance(vote_rounds, int) or vote_rounds < 1:
        print(f"[pre-agent] invalid vote_rounds={vote_rounds!r}, fallback to 1")
        vote_rounds = 1
    agent = build_pre_agent(model)
    skill_info_section = ""
    if skill_info:
        skill_info_section = f"\n\n## SQL技巧速查\n以下是SQL相关技巧和注意事项,分析question时请参考:\n{skill_info}"
    user_input = (
        f"## question\n{task_question}\n\n"
        f"## knowledge.md\n{knowledge_md}"
        f"{skill_info_section}"
    )

    async def _one_round() -> tuple[PreAgentOut, Any]:
        result = await agent.run(user_input)
        return result.output, result

    runs = await asyncio.gather(*[_one_round() for _ in range(vote_rounds)])
    preds = [r[0] for r in runs]

    if log_path is not None:
        try:
            # 落第一轮的消息历史即可 (其他轮 input 完全相同).
            save_history(runs[0][1].all_messages(), log_path)
        except Exception as e:
            print(f"[pre-agent] log save failed: {e}")

    return _vote_aggregate(preds)
