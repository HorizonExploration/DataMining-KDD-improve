"""plan_agent.py — 在 solver 启动前, 产出一份"解题规划"(markdown plan).

动机
----
对照 ``experiments/20260610-opus-result/`` 的 opus 手工解 (53/60 满分), 每题都有一份
``GUIDE.md``: 它本质上是 solver 之前的"plan"——把口径来源 / 数据源字段映射 / 解题路径 /
坑与干扰 / 输出形态先想清楚, 再写 SQL. baseline 失败分析 (analysis/...) 显示最大失分块
(单位标度 / 视频口径 / 去重空值) 恰恰是缺这套显式规划导致的.

本 agent 把 ``SOLVING_GUIDE.md`` 的规划维度提炼成指令, 并**带 explore_data 工具**让它
主动查数验证 (去重粒度 / 单位量纲 / join 基数 / 预期行数都得实查, 不能空想), 最终产出
一份 markdown plan (仿 GUIDE.md 结构), 供后续 solver 消费 / 人工评测.

与 solver/doc_relevance 的关系
------------------------------
- 复用 ``build_explore_tool`` (与 solver 同一只读 SQL 工具, 自动注册 csv/json/sqlite +
  doc 抽取表 ``_doc_extracted.db``).
- 复用 ``doc_relevance_agent._load_video_text`` 载入视频 timeline+hiccup 文本.
- 输出是**自由 markdown 文本** (``output_type=str``), 不做结构化 schema / 投票.
- **不写文件、不需 execute/hashline**, 故用普通 ``pydantic_ai.Agent`` 而非 deep agent.

入口
----
``run_plan_agent_async(model, task_dir, ...) -> PlanResult``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic_ai import Agent, Tool, capture_run_messages
from pydantic_ai.usage import UsageLimits

import tools_v2.describe_tool
from agents_v2.agent_util import save_history
from agents_v2.doc_relevance_agent import _load_video_text
from tools_v2.explore_tool import build_explore_tool
from tools_v2.table_profile import build_table_profiles


# plan agent 最大请求轮数 (带工具的多轮 ReAct, 给足探查空间).
DEFAULT_PLAN_REQUEST_LIMIT = 40

# 视频文本注入上限 (与 doc_relevance 同口径).
DEFAULT_VIDEO_MAX_CHARS = 8000


# ---------------------------------------------------------------------------
# system instruction
# ---------------------------------------------------------------------------

PLAN_AGENT_INSTRUCTION = """你是数据分析任务的**解题规划 agent**. 你的产出不是答案, 而是一份给后续 solver 用的
**解题规划 (plan, markdown 格式)**: 把"用哪些表、哪些字段、怎么过滤、是否去重、单位怎么换、
有哪些坑"先想清楚并**用数据验证过**, solver 据此直接写 SQL, 不必再摸索.

## 最重要的工作方式: 先查数, 再下结论
你有一个 `explore_data` 工具 (只读 SQL, 后端 DuckDB, 已把所有数据源 csv/json/sqlite 以及
doc 抽取库 `_doc_extracted.db` 注册成干净表名). **凡是"看列名猜不出、必须看数据才知道"的判断,
都必须先用 explore_data 实查, 不允许凭直觉写进 plan**. 典型必须查的:
- **去重粒度**: 同一个键 (如 CompanyCode/InnerCode) 在表里是 1 条还是多条?
  实查 `SELECT COUNT(*), COUNT(DISTINCT key) FROM t` —— 决定该用 count(*) 还是 count(distinct).
- **单位/量纲**: 同一列的值是否混了不同标度 (有的"百万为单位的小数"、有的"原始整数")?
  实查 `SELECT MIN(col), MAX(col), col FROM t ...` 看数量级是否自洽, 决定是否 ×100/×1e6 换算.
  **不要盲信 knowledge.md 声明的单位**, 以数据内部一致性 + doc 原文为准.
- **join 基数**: 关联键是 1:1 还是 1:多? 限定"宇宙"(如档案表登记的基金) 后命中几行?
- **预期行数**: 按你规划的过滤条件实跑一次, 看命中几行, 作为 solver 的 self-check 锚点.
- **孤儿行/空值**: doc 抽取表常有"只含 ID 的叙述片段行", 用 `WHERE 关键列 <> ''` 实查真实记录数.

开始时先用 `\\tables` 看有哪些表 (含行数/列名/出处), 再 `\\schema <表名>` 看字段+样本.

## 你必须规划的维度 (按此组织 plan 的章节)
1. **任务复述 + 口径来源**: 一句话复述 question 要什么. 标注解题口径 (阈值/年份/分组/去重等)
   来自哪里: question 文本自带 / knowledge.md / **视频** / doc 原文. 视频题务必从下方"视频口径"
   文本里把每条口径连同**出处帧** (slide_xx) 抽出来列成表.
2. **数据源与表/字段映射**: 目标指标落在哪张表的哪一列. **显式判断结构化表里到底有没有这个字段**——
   若结构化表根本没有 question/看板要的指标 (字段名相近但语义不同: 是否年化/是否含基准/统计周期不同),
   不要勉强用相近字段或 join 凑数; 此时若视频看板已直接给出该指标结果, 就以看板为权威来源.
3. **筛选/过滤条件**: 每个 WHERE 条件及其依据 (阈值数值、年份口径、批次状态 ACTIVE/ARCHIVED、
   严格大于 `>` 还是 `>=`). 注意视频里常有 ACTIVE 批次 vs 已锁 ARCHIVED 批次的区分.
4. **去重规则**: count(*) 还是 count(distinct)? 公司数 vs 记录数? 并列是否保留?
   **这个没有默认值, 必须按 question 语义 + 视频规则 + 实查结果判断** (有的题要去重, 有的题 gold 就是没去重).
5. **单位/量纲**: 取数后是否需要换算? 倍率依据是什么 (实查得到的数量级关系)?
6. **join 键与基数**: 用哪个键关联 (注意 InnerCode vs 交易代码 等易混键); 宇宙如何限定.
7. **干扰项**: 列出会误导的陷阱——名字相似但语义不同的字段 (数值甚至相同)、相似指标
   (年化 vs 累计 vs N年窗口)、一级 vs 二级行业、视频里的"仅供参考/边界样本/红线以下"干扰帧、
   doc 原文里混入的无关噪声 (办公室琐事等).
8. **输出列 + 预期行数**: 只输出 question 明确要求的列 (多余列会被轻罚); 是否保留 NULL 行
   (不要擅自 dropna); 预期行数/列数 (作为 self-check 锚点).
9. **数据质量风险**: 若发现 doc 抽取丢精度 (取整/降精度)、记录被拆段导致 NaN、字段像把
   档案号错当业务值等, 在此标注, 提示 solver 回看对应 `doc/*.md` 原文核对.

## 严格准则 (必须遵守)
- **禁止查看 gold / 答案文件**, 也禁止为了凑某个预期答案而引入无数据依据的 tie-break / 排序 /
  过滤 / 魔法值. plan 里每一条规划都要能用 question/context/视频/knowledge + 实查数据独立解释.
- 数据本身有歧义时 (如"最后一条"在最晚时间戳上有多条并列), 如实记录"数据无法唯一确定", 给出
  基于口径最合理的处理, 不要臆造.

## 输出格式
直接输出 markdown plan 正文 (用上面 9 个维度作为 `##` 二级标题章节). 不要包代码块, 不要写
寒暄/解释你在做什么. 在涉及具体数值/行数/单位的结论处, 简要标注你是通过哪条 explore_data
查询验证的 (便于审计).
"""


# ---------------------------------------------------------------------------
# 静态版 instruction (无 explore_data 工具, 列画像已预计算注入)
# ---------------------------------------------------------------------------

PLAN_AGENT_INSTRUCTION_STATIC = """你是数据分析任务的**解题规划 agent**. 你的产出**不是答案, 也不是替 solver 把题做出来**,
而是一份给后续 solver 的**注意事项清单 (markdown)**: 提前点出"这道题容易在哪些地方栽跟头、
该往哪些维度多想一层", 让 solver 解题时不漏掉关键口径、不踩常见坑. 最终答案仍由 solver
自己对数据分析得出, 你只负责"提醒", 不负责"求解".

## 定位
- 你的任务是**识别注意点 / 风险**, 提醒 solver 该核对什么, 而不是给出精确结论 (具体取哪张表、
  用什么聚合、得几行, 都由 solver 查数后定).
- 你**没有数据探查工具**. 下方 context 给了每个结构化表的**列画像** (每列 distinct 数 / 非空数 /
  数值列 min-max / 样例值), 用它来**发现风险信号** (有空值? 有重复? 数量级可疑? 列值杂乱?), 而非算答案.
- 给的应是**通用避坑方法论** (适用于任何同类题), 不要写死针对本题数据的具体规则或魔法值.

## 画像里的风险信号怎么读
- **可能需要去重**: 某列 `distinct < rows` → 有重复, 提示 solver 留意"按此列计数时 count(*) vs
  count(distinct) 的语义差别".
- **量纲可疑**: 数值列 min/max 数量级若与 question/视频阈值差很多 → 提示"单位存疑, 取数前核对量纲,
  必要时回看 doc 原文".
- **孤儿行/空值**: 某关键列 `nulls` 占比高 (doc 抽取表常见"只含 ID 的叙述片段行") → 提示"留意真实
  记录数可能远小于总行数".
- **列值可能不可靠**: 画像只给 distinct 数, 看不出列值是否杂乱 —— 若某分类列 distinct 数异常或样例值
  看着混乱, 提示 solver "该列可能抽取错乱, 用前先抽样核对, 或从更可靠字段派生".

## 你要点出的注意点维度 (按此组织 markdown 章节, 每条都是"提醒")
1. **任务要点 + 口径来源**: 一句话复述 question. 指出关键口径 (阈值/年份/分组/去重等) 可能来自
   哪里 (question / knowledge / **视频** / doc). 视频题提示"先去视频 timeline/hiccup 找全筛选阈值/
   年份/分组/去重规则, 连同出处帧 slide_xx", 因为这些 question 文本里往往没有.
2. **数据源与字段映射注意点**: 目标指标大概在哪张表哪列. 提示"字段名相似但口径不同 (是否年化/是否
   含基准/统计周期不同) 的, 按语义匹配而非名称". **特别注意**: 有些视频题的答案 (看板上的数值/榜单/
   结果行) **直接在视频里, 未必有对应的结构化表或 doc** —— 若结构化数据里找不到语义对应的字段/记录,
   提示 solver "该指标可能只存在于视频看板, 应直接以视频内容为数据源, 不要勉强用相近字段或 join 凑".
3. **筛选条件注意点**: 提示有哪些过滤维度要落实 (阈值、年份口径、批次状态等), 提醒 solver 去
   question/视频里确认精确取值.
4. **去重注意点**: 提示"这是算实体数还是记录数? 是否需要去重? 并列是否保留?" —— 让 solver 按语义定.
5. **单位/量纲注意点**: 提示哪些数值列单位存疑、需要核对 (结合画像数量级信号).
6. **关联/宇宙注意点**: 提示用哪个键关联、有无易混键 (如内部编码 vs 交易代码); 若 question/视频指定了
   统计"宇宙"(某登记范围、某批次), 提示 solver 以指定宇宙为准.
7. **干扰项**: 点出会误导的陷阱类型 —— 名字相似但语义不同的字段、相似指标 (年化 vs 累计 vs N年窗口)、
   一级 vs 二级分类、视频里的"仅供参考/边界样本"干扰帧、doc 原文混入的无关噪声.
8. **输出形态注意点**: 提示"只输出 question 明确要求的列 (多余列会被轻罚)"; 留意输出是单值/单行还是整列
   多行, 由 question 语义决定; 空值是否保留.
9. **数据质量风险**: 从画像看出的异常 (某列像把档案号错当业务值、大量异常空值、数量级离群、分类列疑似
   错乱等) 列为风险提示, 建议 solver 回看对应 `doc/*.md` 原文核对.

## 准则
- **禁止查看 gold / 答案**, 禁止反推或臆造. 你点出的每条注意点都要能用 question/context/视频/knowledge/
  画像独立解释, 且是这道题真实存在、对同类题也通用的风险.
- 不确定的地方如实写"需 solver 核对".

## 输出格式
直接输出 markdown 注意事项清单 (用上面 9 个维度作为 `##` 二级标题章节). 措辞用提示性语气. 不要包代码块, 不要寒暄.
"""


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------


@dataclass
class PlanResult:
    """一个 task 的解题规划结果."""

    task_id: str
    plan_md: str = ""
    error: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "plan_md": self.plan_md,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# agent 构造
# ---------------------------------------------------------------------------


def build_plan_agent(
    task_dir: Path,
    *,
    model: Any,
    use_tool: bool = True,
) -> Agent:
    """为单个 task 构造一个 plan agent.

    Args:
        task_dir: 任务根目录 (含 context/; doc 抽取库已在 context/db/).
        model: pydantic_ai 模型实例 (推荐 build_model(), 带思维链).
        use_tool: True (默认) 挂 explore_data 工具, agent 多轮 ReAct 主动查数;
            False 为**静态版**: 不挂工具, 列画像已预计算注入 context, agent 单轮产出
            (省掉工具往返, 显著更快, 代价是无法做画像未覆盖的临场验证).

    Returns:
        配置好 instructions (+ 可选 explore_data 工具) 的 ``pydantic_ai.Agent``.
    """
    if use_tool:
        explore_fn = build_explore_tool(task_dir)
        return Agent(
            model=model,
            instructions=PLAN_AGENT_INSTRUCTION,
            tools=[Tool(explore_fn, takes_ctx=False)],
        )
    return Agent(
        model=model,
        instructions=PLAN_AGENT_INSTRUCTION_STATIC,
    )


def _build_user_input(
    *,
    question: str,
    knowledge_md: str,
    context_desc: str,
    video_text: str,
    table_profiles: str = "",
    use_tool: bool = True,
) -> str:
    """拼装喂给 plan agent 的 user_input."""
    video_section = ""
    if video_text and video_text.strip():
        video_section = (
            "## 视频口径 (操作讲解视频的幻灯片+旁白文本, 含 hiccup 版面树)\n"
            "真正的筛选阈值/统计年份/分组维度/去重规则常只在视频里, question 文本看不出. "
            "请逐条抽取并标注出处帧 (slide_xx).\n"
            f"{video_text.strip()}\n\n"
        )

    if use_tool:
        # 带工具版: context 概览只给 schema+少量样本, 细节让 agent 自己 explore_data 查.
        data_section = (
            "## context 数据源概览 (列名 + 少量样本; 详细探查请用 explore_data 的 \\tables / \\schema)\n"
            f"{context_desc}\n\n"
            "现在请先用 explore_data 探查验证, 再产出这道题的解题规划 (markdown)."
        )
    else:
        # 静态版: 列画像 (distinct/null/min-max) 已预计算, 直接推理, 不调工具.
        profile_block = table_profiles.strip() or "(本任务无结构化表)"
        data_section = (
            "## 结构化表列画像 (已预计算: 每列 distinct 数 / 非空数 / 数值列 min-max / 样例值)\n"
            "这是判断去重粒度、单位量纲、空值孤儿行的关键依据, 直接据此推理.\n"
            f"{profile_block}\n\n"
            "## context 其他文件概览 (doc/markdown 等非结构化预览)\n"
            f"{context_desc}\n\n"
            "现在请直接基于以上画像与 context 产出这道题的解题规划 (markdown), 不要请求任何工具."
        )

    return (
        f"## question\n{question}\n\n"
        f"## knowledge.md (字段语义/单位/业务口径)\n{knowledge_md}\n\n"
        f"{video_section}"
        f"{data_section}"
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run_plan_agent_async(
    *,
    model: Any,
    task_dir: Path,
    question: Optional[str] = None,
    knowledge_md: Optional[str] = None,
    video_text: Optional[str] = None,
    request_limit: int = DEFAULT_PLAN_REQUEST_LIMIT,
    video_max_chars: int = DEFAULT_VIDEO_MAX_CHARS,
    use_tool: bool = True,
    log_path: Optional[Path] = None,
) -> PlanResult:
    """跑 plan agent, 返回 ``PlanResult``.

    缺省入参从 ``task_dir`` 兜底读取 (对齐 ``run_doc_relevance_async`` 的风格), 因此既能在
    主流程里被显式喂入, 也能由独立 runner 直接吃一个已跑过的 ``temp_xxx/task_xxx`` 目录
    (此时 video/doc 抽取/scaffold 都已就绪).

    Args:
        model: pydantic_ai 模型实例.
        task_dir: task 根目录 (含 context/, 通常还有 workdir/_video_frames/).
        question: task question; 为 None 时从 task_dir/task.json 读取.
        knowledge_md: knowledge.md (hashline 渲染); 为 None 时从 context/ 读取.
        video_text: 视频讲解文本; 为 None 时从 workdir/_video_frames/ 兜底载入 (无视频则空).
        request_limit: agent 最大请求轮数.
        video_max_chars: 视频文本注入上限.
        use_tool: True (默认) 挂 explore_data 工具多轮查数; False 走静态版 (列画像预计算注入, 单轮产出).
        log_path: 若提供, 落盘对话历史 (审计).

    Returns:
        PlanResult. 失败时 error 非空, plan_md 尽力保留已生成内容.
    """
    task_id = task_dir.name
    result = PlanResult(task_id=task_id)

    # ---- 兜底读取输入 ----
    if question is None:
        try:
            question = json.loads(
                (task_dir / "task.json").read_text(encoding="utf-8")
            ).get("question", "")
        except Exception as e:  # noqa: BLE001
            logger.warning("[plan] read task.json failed for {}: {}", task_id, e)
            question = ""

    if knowledge_md is None:
        knowledge_md = tools_v2.describe_tool.read_knowledge(task_dir, "context")

    if video_text is None:
        video_text = _load_video_text(task_dir, video_max_chars)

    context_desc = tools_v2.describe_tool.describe_context_dir(
        task_dir, "context", skip_knowledge=True
    )

    # 静态版预计算列画像 (去重/量纲/空值统计); 带工具版让 agent 自己查, 不预算.
    table_profiles = ""
    if not use_tool:
        try:
            table_profiles = build_table_profiles(task_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("[plan] build_table_profiles failed for {}: {}", task_id, e)

    user_input = _build_user_input(
        question=question or "",
        knowledge_md=knowledge_md or "",
        context_desc=context_desc,
        video_text=video_text or "",
        table_profiles=table_profiles,
        use_tool=use_tool,
    )

    agent = build_plan_agent(task_dir, model=model, use_tool=use_tool)

    with capture_run_messages() as captured_msgs:
        try:
            run = await agent.run(
                user_input,
                usage_limits=UsageLimits(request_limit=request_limit),
            )
            result.plan_md = run.output or ""
        except Exception as e:  # noqa: BLE001
            result.error = f"{type(e).__name__}: {e}"
            logger.exception("[plan] task {} failed: {}", task_id, e)
        finally:
            if log_path is not None and captured_msgs:
                try:
                    save_history(list(captured_msgs), log_path)
                except Exception as e:  # noqa: BLE001
                    logger.debug("[plan] log save failed for {}: {}", task_id, e)

    logger.info(
        "[plan] task={} done: plan_len={} chars, error={!r}",
        task_id, len(result.plan_md), result.error,
    )
    return result
