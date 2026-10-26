"""fanout_struct.py — fan-out 式 doc 结构化引擎 (从 experiments/doc/scripts/dialog_doc_struct.py 移植).

流程 (去掉了原 dialog 的 audit/fixer 收尾阶段):
  Phase 1  planner agent  → JSON {sections, record_universe, schema, column_normalize?}   (1 次 chat, 失败重试 1 次)
  Phase 2  代码校验 plan
  Phase 3  asyncio.gather 并发拉起 N 个 section worker, 每个独立 chat → 一张 markdown 表
  Phase 4  每张表代码校验 (列名/行数/universe), 失败让该 worker 单独重试 1 次
  Phase 5  按 record_id outer-join 所有表 → 一张宽表 (columns, rows)
  Phase 6  对 plan.column_normalize 中的列做归一化, 把 doc 原文格式归一到入库目标格式
           (例如 "166.097944%" → "166.097944"). planner 只声明 doc_format/target_format (自然语言+示例),
           不再写转换函数; 具体转换逻辑下游针对该列全量数据生成. worker 不可见这部分规则.

每个 worker 单轮对话, 不维护多轮历史; 上下文 = system + doc + plan summary + section spec.
backend 走 pydantic-ai 的 ``Agent`` (output_type=str), 由调用方注入 model
(主流程注入 qwen nothink), 不再依赖裸 AsyncOpenAI/AsyncAnthropic.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import queue as _queue
import re
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.openai import OpenAIChatModelSettings

# worker 在 plan.columns 之外额外输出的"来源行号"列名. 双下划线前缀, 在 assemble_wide
# 阶段被剥进 provenance sidecar, **绝不进入主表 / DB / prediction.csv** —— 否则评分时
# 它会变成一个匹配不上 gold 的 extra 列, 倒扣 penalty. 仅供后处理把可疑 cell 映射回 doc 原文.
SRC_LINE_COL = "__src_line"

# ───────────────────── 模块级模型 (由调用方注入, 各 agent 直接取用) ─────────────────────
# planner / section worker / transform 生成器分散在多个函数里, 早先靠层层透传 model 参数,
# 嵌套很深. 改为 contextvar 注入: 调用方 (doc_prepare) 在跑流程前用 set_models() 注入一次,
# 各 agent 函数直接 _model_nothink()/_model_think() 取用, 不再透传.
#   MODEL_NOTHINK: planner / section worker 用 (主流程的 qwen nothink, 纯抽取不需思维链).
#   MODEL_THINK:   Phase 5.5 transform 代码生成用 (推理任务, 建议 think 模型); 未注入时退回 nothink.
#
# **为什么用 contextvar 而非普通全局**: 主流程并发处理多个 task, 普通全局是进程内共享单例,
# task A set 的模型会被并发的 task B set 冲掉, A 之后 await 读到的就是 B 的模型. ContextVar
# 在 asyncio 下每个 Task 有独立 context 副本: 在某 task 内 set 只影响该 task 自己的 context,
# 不波及兄弟 task; 而 structurize_doc 内 asyncio.gather 出去的 section workers 继承创建时的
# context (那时模型已注入), 因此整条调用链都读到本 task 注入的那对模型, 实现 task 间隔离.
import contextvars

_MODEL_NOTHINK: contextvars.ContextVar = contextvars.ContextVar("fanout_model_nothink", default=None)
_MODEL_THINK: contextvars.ContextVar = contextvars.ContextVar("fanout_model_think", default=None)


def set_models(*, model_nothink: Any, model_think: Any = None) -> None:
    """注入当前 task context 的模型. model_think 为 None 时 transform 生成退回 model_nothink.

    用 ContextVar.set, 只影响当前 asyncio Task 的 context 副本, 并发 task 间互不干扰.
    应在每个 task 自己的协程内 (跑 structurize_doc 之前) 调用一次.
    """
    _MODEL_NOTHINK.set(model_nothink)
    _MODEL_THINK.set(model_think if model_think is not None else model_nothink)


def _model_nothink() -> Any:
    m = _MODEL_NOTHINK.get()
    if m is None:
        raise RuntimeError("fanout 模型未注入: 请先在本 task 协程内调用 set_models(model_nothink=...)")
    return m


def _model_think() -> Any:
    m = _MODEL_THINK.get()
    if m is None:
        raise RuntimeError("fanout 模型未注入: 请先在本 task 协程内调用 set_models(model_nothink=...)")
    return m

# ───────────────────── prompts ─────────────────────

PLANNER_SYSTEM = """你是 doc 结构化的 planner. 你只产出**一份 JSON 计划**, 不抽数据.

读完整文档, 输出唯一一个 ```json``` 代码块, 形如:
{
  "sections": [
    {
      "id": 1,
      "title": "简短中文标题",
      "start_marker": "本 section 第一句的 8~30 字原文(必须能在 doc 中精确匹配)",
      "end_marker":   "本 section 最后一句的 8~30 字原文(必须能在 doc 中精确匹配)",
      "start_line":   12,
      "end_line":     63,
      "columns": ["record_id", "<本 section 涉及的列>", ...]
    }
  ],
  "record_universe": ["65", "111", ...],
  "schema": {
    "record_id": {
      "doc_meaning": "主键, 字符串",
      "task_meaning": "主键, 字符串"
    },
    "<col_name>": {
      "doc_meaning": "这列在 doc 散文里的真实形态 (含 doc 用的单位 / 写法 / 是否带符号), 给 worker 抽取参考",
      "task_meaning": "这列入库后的目标语义 (含 task 约束的单位 / 格式), 来自 knowledge 摘录 / 视频口径 / 题面约束"
    }
  },
  "global_notes": "全文级总结 / 结论 / 不挂在某条 record 上的内容. 没有就空字符串.",
  "column_normalize": {
    "<col_name>": {
      "doc_format": "doc 原文里这列大致长什么样 (举一两个具体值, 注意标点/单位)",
      "target_format": "用自然语言详细描述入库目标格式 + 两个从 doc_format 转换过去的示例 (如 \\"'1.324 亿元' → '132400000'\\")",
      "reason": "为什么要转 (引用 knowledge 里的依据)"
    }
  }
}

先理解这套架构 (决定你该怎么切 section):
  - 你切出的**每个 section 都会变成一张独立子表**: 后续 worker 会对 record_universe 里
    **每个 id 抽恰好一行**, 同一 section 内一行就是一条 record, 一条 record 可有多列.
  - **各 section 子表之间靠 record_id 做 outer-join** 拼成最终宽表. 所以每个 section 描述的是
    "**同一组 record 的一类并列属性**", section 之间用 record_id 关联.
  - **推论**: 某内容如果伴随每条 record、且在每个属性块里各自出现一次 (典型: 行政噪音 / 局部备注),
    它本身就**属于各 section 内部**, 应作为相关数据 section 的尾列 (`<section前缀>_noise`).
    **严禁把这种内容跨 section 聚合成一张只有它一列的独立子表** —— 因为同一 record 会在多个属性块
    各有一句来源, 无法塞进"一行一 record"的子表, 会爆出重复行. noise 不可跨 section 聚合.

要点:
  - section 的切分以"实体属性维度"为准 (如 基本信息 / 时间戳 / 静态数值 / 季度变化 / 半年变化).
  - **结论 / 总览 / 红旗警示这种"全文级、不属于某条 record"的内容, 不要切成 section,
    直接放到顶层 `global_notes` 字段里**.
  - 过渡句 ("在完成对 X 的审计后, 现在转向 Y...") 必须明确归属(默认到下一 section 开头), 不产生新行.
  - **行号定位 (start_line / end_line)**: doc 每行开头都有 `NNNN\t` 形式的行号前缀 (1-based, Tab
    分隔). 除了 start_marker / end_marker 两段原文, 你还要给每个 section 标出它覆盖的行号区间:
    start_line = 本 section 第一行的行号 (即 start_marker 所在行), end_line = 最后一行的行号
    (即 end_marker 所在行, 含). 两者都是整数, 不带前导零也行 (写 12 / 63 即可).
    约束: 1 ≤ start_line ≤ end_line; 各 section 区间**互不重叠**, 按 id 顺序紧邻拼接, 合起来
    覆盖正文主体 (顶层 global_notes / 纯过渡句对应的行可以不被任何 section 覆盖).
    注意: 行号是**物理行** (按换行切), 与 start_marker / end_marker 指向同一行, 二者必须自洽.
  - record_universe: 全文所有出现过的 record_id, 字符串列表, **去重 + 按出现顺序**.
  - schema: 全 doc 的统一列定义. record_id 必须是第一列且为主键.
    每个列的值必须是 dict, 含两个字段:
      * doc_meaning: 这列在 doc 散文里的真实形态. 写明 doc 用的单位 / 数值是否带符号 / 是否裸数 等
        worker 抽取时需要的描述. 来源: doc 正文样例.
      * task_meaning: 这列入库后的目标语义与口径. 写明 task 约束的单位 / 格式 / ratio vs % 等. 来源:
        knowledge 全局口径 (task 整库声明) / knowledge 摘录 (该 doc 字段细节) / 视频口径 / 题面约束.
        **优先级**: 若上述任一来源声明了某种"全表/全 task 默认"的口径 (如默认单位/默认时区/默认主键约定),
        它就是 task_meaning 的依据, **优先级高于** doc 散文里隐含的口径. 当 doc 散文使用了与该默认不同的
        单位/格式时, doc_meaning 写 doc 实际样子, task_meaning 写默认口径, 两者不一致就需要 column_normalize
        做实质换算 (而不是只剥掉单位字符串).
    若 knowledge 全局口径与摘录都没规定单位/格式 (与本列无关), doc_meaning 与 task_meaning 写**完全相同**的描述.
    举例 (仅说明机制, 不针对具体 task):
      * 若 doc 写"166.097944%"且 knowledge 标注"% 单位, 数值不含百分号":
        doc_meaning = "doc 写成带百分号的字符串, 如 '166.097944%'"
        task_meaning = "数值, 单位 %, 不含百分号 (如 '166.097944'); 10 表示 10%"
      * 若 knowledge 把某列标注为 (ratio) 而 doc 写百分比:
        doc_meaning = "doc 写成带百分号字符串, 如 '98.3%'"
        task_meaning = "0~1 之间的小数, 如 '0.983'"
      * 若 knowledge 没说单位 (摘录为空):
        doc_meaning = task_meaning = "近一年累计回报率"
  - 修正历史: 凡是 doc 里有"原 X, 修正为 Y"的字段, schema 应配套 raw + final 双列 (例如
    internal_code_raw / internal_code_final).
  - **同义异形名称/标识分列 (重要)**: 同一条 record 的某种属性可能在 doc 中常以**多种并存的
    表达形态**出现 —— 例如名称 / 标识这一类, 全称 vs 简称、机构正式名 vs 交易简称、中文名 vs 英文名、不同口径的
    代码 (内部码 / 对外证券码 / 交易所码). 只要 doc 正文里这些形态**各自独立出现**, 就必须**各拆成
    一列**, 不可合并成一列, 也不可只挑一种丢弃其余. 命名要体现形态差异 (如 full_name / fund_abbr /
    secu_abbr / trade_abbr / chi_name / eng_name / internal_code / secu_code), 并在 task_meaning 里
    **写清每列对应 knowledge 中的哪个具体概念/字段**. 即使是简称,也可能存在多个, 
    例如 (基金简称 vs 证券简称) , 必须分辨清楚分别落列, 
  - 成对列建议: 同一指标若在 doc 中**频繁同时**出现"排名 / 同类均值 / 中位数 / 分位数"等
    衍生表述, 可以拆成 X / X_rank / X_avg / X_median 独立列, 方便下游使用. 但若 doc 只是
    偶尔提一两次, 不要为此额外加列, 保持 schema 紧凑.
  - 短枚举字段建议: 若 doc 中**所有 record 普遍**带有某种"少数取值"的状态描述 (例如 在职
    / 离任, 高 / 中 / 低风险, Fund Manager / Investment Director 这种), 可以单独成列;
    若只在个别 record 提及或语义模糊, 留在 noise 即可.
  - 比例/率类字段建议: 若 doc 中同一 record 同时出现两种以上的"比例/率"数值 (例如同时有
    折溢价比和日增长率), 应分列; 若只有一种, 一列即可.
  - 噪音/行政描述 (办公室/培训/演习这类与核心实体无关的句子): 它伴随每条 record 在**各属性块内部**
    各自出现, 属于 section 内部内容. 在 schema 中给一个 noise 列, 并把它作为**每个数据 section
    的尾列** `<section前缀>_noise` (例如 basic_info_noise / profit_noise). **严禁单独切一个只含
    noise 列的 section, 也严禁把全文噪音聚合到一个 noise 列** —— 同一 record 在多个属性块各有一句
    噪音, 跨 section 聚合必然爆出重复行 (见上文架构推论).
    保留 noise 尾列的意义: 它给 worker 一个明确的"垃圾桶", 逼 worker 把该 record 段落里那句闲话
    显式识别并隔离到 noise 列, 从而**降低闲话污染真实数据列的概率**.
  - **每个 section 的输出行数恒等于 |record_universe|**: 该 section 没有数据的 record,
    worker 会整行填 NULL. 因此 planner 不需要估"该 section 涉及多少 record".
  - 列名一律 snake_case 英文 + 数字, 不出现空格/中文.
  - 若 user prompt 中提供了 "knowledge 摘录": 这是从 task 的 knowledge.md 中按当前 doc 名 grep 出来的相关片段,
    用于**列命名 / 单位 / 枚举 / 阈值口径的对齐**, 让你产出的 schema 列名与 task 通用概念一致.
    **重要前提**: knowledge 摘录通常**并不完整** —— 它一般只覆盖该 doc 的**一部分列** (经验上常常只覆盖
    约 1/4 的列), 甚至可能是空占位. 因此摘录的角色是"已知列的命名/口径锚点", **不是 doc 字段的全集**.
    你的任务是: 以 **doc 正文实际出现的字段为准**, 把每个字段都建成列; 摘录里能对上的, 复用其命名与口径,
    并在 task_meaning 里标注对应关系; 摘录里**没提到但 doc 正文确实出现**的字段, **必须照样建列、不可遗漏**,
    自行按语义推断一个 snake_case 列名, 并尽量在 task_meaning 里推断它对应 knowledge 的哪个概念 (推断不出
    就如实写 doc 语义). 宁可多保留一个 doc 里真实存在的字段, 也不要因为摘录没写就把它丢掉.
    规则:
      * user prompt 顶部会告诉你 "当前 doc 名". 摘录通常采用 markdown 小节 + 字段表的结构,
        例如 `### 2.2 ... (<当前 doc 名>)` 这种标题, 紧跟一张 `| Column | Semantic Definition |`
        风格的表; 表格首列就是该 doc 的字段名/列名 (可能是反引号包的英文 `qdiinv`, 也可能是
        裸中文 `社会消费品零售总额(百万元)`, 还可能是普通英文标识符). 对于**摘录里出现、doc 正文也确实有**
        的字段, 优先复用摘录里的字段名作为 schema 列名 (命名锚点); 但这只是"已知列"的命名来源, **不限制**
        schema 只能有这些列 —— doc 正文里摘录未覆盖的字段仍要照建 (见上文前提).
      * 同一份摘录里通常也夹带邻表 / join 目标的字段 (例如 join 说明、SQL 例子), 不要把它们
        并入当前 doc 的 schema. 判断办法是看小节标题对应的是哪张表.
      * 字段是中文时, 转 snake_case 时按语义译成简短英文 (`社会消费品零售总额` → `total_retail_sales`),
        括号内的单位/限定 (`(百万元)` / `(元/人)`) 不进列名, 而是写到 schema 的列含义里, 并在 worker 抽数时保留 doc 原文数字.
      * 字段是英文标识符时 (反引号包或裸名), 原样照搬, 不要改大小写, 不要再翻译.
      * 单位 / 枚举值 / 阈值口径以摘录为准 (例如 "亿元", "百分比数值如 2.5 表示 2.5%", 枚举值 `期末累计`).
      * 摘录**只是命名/口径参考, 不是数据源**. 严禁把摘录里的 schema 行、SQL 例子、字段值当作 doc 中的 record 抽出来.
      * record_universe 仍只从 doc 正文统计, 不从摘录里凑.
      * 若摘录里给出的字段在 doc 正文里完全没出现, 不要硬加; 以 doc 实际内容为准.
      * 摘录可能为空 / 极短 / 全是无关引用 (task 的 knowledge.md 可能是空占位). 此时退回按 doc 自洽切 schema.
  - **column_normalize**: 仅当某列的 doc_meaning 与 task_meaning 不一致时使用, 用于在 worker 抽完之后做后处理归一化.
    worker **不会看到** column_normalize, 也不会看到 task_meaning, 仍按 doc_meaning 描述的形态忠实抽取;
    后处理阶段会按你这里的描述把 doc_meaning 形态改写为 task_meaning 形态 (具体转换逻辑由下游针对该列**全量数据**生成,
    你**不需要、也不要**写任何转换代码 / 函数, 只需把"要做什么归一 + 目标长什么样"讲清楚).
    每条规则三个字段:
      * doc_format: 等价于 schema[col].doc_meaning 中描述的形态, 给一两个具体值 (如 '166.097944%').
      * target_format: 等价于 schema[col].task_meaning 中描述的目标形态. **用自然语言比较详细地描述目标格式**:
          - 讲清要做哪类归一 (例如: 时间统一转 ISO 'YYYY-MM-DD' / 金额单位统一, 亿元万元等换算到其他量级 /
            去掉百分号只保留数值 / 百分比转成 0~1 之间的小数 等);
          - 讲清边界细节 (是否保留小数位、是否保留负号、是否需要四舍五入、千分位逗号如何处理等);
          - **并给出两个从 doc_format 转换到目标格式的具体示例**, 形如
            "'1.324 亿元' → '13240(万元)' " 和 "'6597 万元' → '65970000(元)'"; 示例要能体现这一列真实的换算规律.
            转换后的示例中的括号中的单位仅用于理解, 不真实输出到行的结果中, 行的结果通常都是不带单位的, 单位在列的描述中说明.
      * reason: 一句话解释**为什么需要归一** (即两份 meaning 为什么不一致), 必须引用 knowledge 摘录里的具体表述
        (例如 "knowledge §X.Y 标注单位为 ...").
    判断要点 (避免过度归一):
      * doc_meaning == task_meaning 时, **不要**给该列加 column_normalize.
      * 只为 schema 里**实际存在**的列写规则; 不要给 record_id / noise 列加规则.
      * **doc 中可能会声明的特定数据格式, 不可以被清洗**: 例如 doc 正文
        对某字段明确说明它采用某种特殊格式 (例如"该字段采用字符间隔的格式", 名称/代码中"保留
        前导零", "以 X 作为分隔符"等), 这种格式 (含其中的空格 / 分隔符 / 补位字符) 是该字段
        **真实值的组成部分**, 不是排版错误. 此时 doc_meaning 与 task_meaning 都应说明这种情况,
        不应错误地转换. 典型反例: doc 说明证券简称"采用字符间隔
        格式"而出现"王 府 井"这类带空格的值 —— 这是 doc 有意保留的真实简称, 去空格
        会破坏列签名匹配, 属于错误归一. doc 未声明要保留原样的数值列, 仍按正常规则处理.
    若没有任何列需要归一, `column_normalize` 写成 `{}` 或省略.

输出要求: 只发**一个** json 块, 块外不加任何解释文字.
**重要**: JSON 字符串值内若需要引号, 必须用转义 `\\"` (或干脆改用中文「」/『』包裹), 严禁出现裸 `"` 导致解析失败.
"""

WORKER_SYSTEM = """你是 doc 结构化的 section worker. 单次任务: 把指定 section 抽成 markdown 表.

输入: 完整文档 (每行开头有 `NNNN\\t` 形式的 1-based 行号前缀) + 一份全局 plan JSON + 你被分配的 section_id.
输出规范:
  - 表头 = **第一列固定为 `__src_line`** (来源行号), 其后接 plan.sections[<section_id>].columns
    的全部列, 顺序一致、列名一字不差. 即实际表头形如 `| __src_line | record_id | ... |`,
    比 plan.columns 在**最前面**多一列.
  - 表头下一行是 |---|---|... 分隔.
  - **每行列数必须 = 表头列数 (含首列 __src_line)**, 缺值写 NULL, 不允许省略竖线.
  - **行数必须 = |plan.record_universe|, 即每个 universe id 都要出现一行**.
    没数据的 record_id 整行除主键外全填 NULL, 不能漏行也不能多行.
  - 行的顺序与 plan.record_universe 一致.
  - 不要在表前后再写多余解释; 至多在表前一行注一句"覆盖 N 个 record".

`__src_line` 列填法 (放在每行最前面, 体现"先定位原文行、再抽该行字段"的顺序):
  - 填该 record 在**本 section** 对应原文所在的行号 (doc 行首 `NNNN\\t` 里的那个数字), **纯整数**,
    不要带前导零/引号/`L`/区间, 例如 `57`.
  - 先按行号锚定该 record 在本 section 的那一行, 再从该行抽后面各列, 能减少抽错行/串行.
  - 本 section 一条 record 的内容通常落在某一物理行 (出现该 record_id 的那行); 若跨多行,
    填该 record 数据的**主行号** (含其主键与主要数值那行).
  - 该 record 在本 section 没有任何内容 (其余列全 NULL) 时, `__src_line` 也填 NULL.
  - 这列仅用于审计回溯, 不影响数据语义; 但务必如实填你实际抽取所依据的那一行, 不要编造.

抽取要点:
  - 中文数字 → 阿拉伯数字归一. 其他数值**保留 doc 原文形式**, 不要主动加/去单位、不要改
    小数位、不要把百分号去掉. 列签名匹配对格式敏感, 改格式会破坏下游评测.
  - **doc 显式声明的特殊格式必须原样保留**: 当 doc 正文对某字段明确说明它采用某种特殊书写
    格式时 (例如"该字段采用字符间隔的格式""保留前导零""以 X 作为分隔符"等), 这种格式是该
    字段真实值的组成部分, 抽取时必须**逐字符照搬 doc 给出的形态** (含其中的空格 / 分隔符 /
    补位字符), 不得把它当成排版错误去"纠正"或清洗. 判断依据是 **doc 是否有此类明确说明**;
    doc 没有特别说明的字段按常规抽取, 不要臆造保留. 列签名匹配逐字符敏感, 擅自清洗会失分.
  - 修正历史: doc 中"原 X, 后修正为 Y", 当前列填 Y; raw 列填 X (或 X -> Y).
  - 成对列 (X / X_rank / X_avg 等): 当 doc 写"指标 A 为 50亿元, 同类排名第 12, 同类均值
    38亿元"时, 按列名分别落到 a / a_rank / a_avg, 不要把排名或均值丢进 noise.
  - 短枚举字段 (在职/离任、高/中/低、Fund Manager 等): doc 里出现就抽到对应列, 不要因为
    它语义短或者像"客套话"就当噪音忽略. NULL 仅留给 doc 里确实没提到的 record.
  - 比例/率类字段: 按 schema 定义的列名匹配 doc 里的说法 (折溢价比 → discount_ratio,
    日增长率 → nav_daily_growth_rate), 不要按列出现顺序硬塞数字, 防止两个比例列错位.
  - 噪音/行政描述: 不进核心数据列, 单独到 noise 列; section 间过渡句不产生新行.
  - 同一 record_id 在该 section 没有内容时: 整行仍要出现, 该 section 的非 record_id 列填 NULL.
"""

# ───────────────────── knowledge excerpt ─────────────────────


def number_doc_lines(text: str) -> str:
    """给 doc 每行开头加 1-based 行号前缀, 形如 ``0001\\t正文``.

    用途: 让 planner / worker 能用行号锚定 section 区间与逐 record 来源, 后处理也能
    按行号把"可疑 cell"映射回 doc 原文 (provenance 回链).

    格式约定 (与 Read / ``cat -n`` 同源, 模型最熟):
      - 纯数字, 无 ``L`` / ``#`` 之类字母前缀 —— 避免模型把前缀抄进"行号列".
      - 零填充到定宽, 宽度 = max(4, 总行数位数). 默认至少 4 位 (≤9999 行对齐),
        行数更多时按实际位数自动加宽, 不会溢出截断.
      - 分隔符用 Tab: 回来侧 worker 报的是纯整数行号, 解析永远 ``int(cell)``,
        不依赖分隔符在正文里的唯一性; 而 ``|`` 会和 worker 输出的 markdown 表边框
        语义打架, ``:`` 在中文正文里太常见, 都不合适.

    行号从 1 开始 (与人类/Read 直觉一致). ``int("0057") == 57``, 下游解析天然兼容零填充.
    空串返回空串.
    """
    if not text:
        return text
    lines = text.split("\n")
    width = max(4, len(str(len(lines))))
    return "\n".join(f"{i:0{width}d}\t{ln}" for i, ln in enumerate(lines, start=1))


def grep_knowledge_excerpt(
    knowledge_md_path: Optional[Path],
    doc_stem: str,
    *,
    before: int = 5,
    after: int = 10,
    max_chars: int = 6000,
) -> str:
    """从 knowledge.md grep doc_stem, 取每个命中行的 ±窗口, 合并去重后返回纯文本.

    用于在 planner 之前给 LLM 注入"该 doc 在 task knowledge 里的相关上下文",
    帮助 schema 列命名向 task 通用概念对齐 (avoid `qdiinv` ↔ `qdii_scale` 漂移).

    匹配规则:
      - 大小写不敏感的子串匹配 (`doc_stem.lower() in line.lower()`).
        knowledge.md 里常写 `\\`mf_xxx\\`` (反引号包裹), 子串匹配天然兼容.
      - 命中行 i → 保留 [i - before, i + after] 闭区间内所有行.
      - 多次命中的窗口集合做 union, 再按连续段切, 段间 `\\n\\n---\\n\\n` 分隔.
        这样既去重又保留段内连续上下文.
      - 总输出超 max_chars 截断, 末尾追加 "... (truncated)" 提示.

    边界:
      - knowledge_md_path 为 None / 文件不存在 / doc_stem 空 → 返回 "" (调用方
        据此决定走 fallback 纯 doc planner, 不要往 prompt 塞空字符串).
      - 0 命中也返回 "".
    """
    if not knowledge_md_path or not Path(knowledge_md_path).is_file():
        return ""
    if not doc_stem:
        return ""

    # 防御: 调用方手滑传 doc_path.name (带 .pdf/.md 后缀) 时, knowledge.md 里
    # 通常只写裸名 (如 `mf_xxx`), 子串匹配会因后缀差异 0 命中. 这里统一剥成 stem.
    key = Path(doc_stem).stem.lower()

    lines = Path(knowledge_md_path).read_text(encoding="utf-8").splitlines()

    keep: set[int] = set()
    for i, ln in enumerate(lines):
        if key in ln.lower():
            lo = max(0, i - before)
            hi = min(len(lines), i + after + 1)
            keep.update(range(lo, hi))
    if not keep:
        return ""

    blocks: list[str] = []
    cur: list[int] = []
    for i in sorted(keep):
        if cur and i != cur[-1] + 1:
            blocks.append("\n".join(lines[j] for j in cur))
            cur = []
        cur.append(i)
    if cur:
        blocks.append("\n".join(lines[j] for j in cur))

    out = "\n\n---\n\n".join(blocks)
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "\n... (truncated)"
    return out


def read_knowledge_global(
    knowledge_md_path: Optional[Path],
    *,
    head_lines: int = 80,
    max_chars: int = 4000,
) -> str:
    """读 knowledge.md 顶部前 N 行作为"全局口径"块.

    用途: 与 grep_knowledge_excerpt 互补. 后者按 doc stem 命中具体字段块, 但 task 整库
    级别的声明 (Database Identifier / Governing Principle / 默认单位 "百万元" / 默认时区
    / Classification 等) 通常写在 knowledge.md 开头, 这些声明里**通常不会出现 doc stem**,
    grep 摘录就漏掉了 (典型例子: task_58 ed_grossdomesticproduct 抽完仍是亿元, 因为 §1 的
    "默认百万元" 声明离 §2.2 的字段表 30+ 行, ±窗口够不着).

    机制简单: 直接读前 head_lines 行, 截到 max_chars. 不做 grep, 不解析 markdown 结构.
      - knowledge_md_path None / 文件不存在 → 返回 "" (planner 端跳过该 block).
      - 文件长度不足 head_lines → 返回全部.
      - max_chars 截断时末尾加 "... (truncated)".

    与 grep_knowledge_excerpt 的关系:
      - 这两个 block 在 planner user prompt 中**并列**, 互不替代;
      - 全局块给"整库口径", 摘录块给"该 doc 字段细节";
      - 两者可能有少量重叠 (前几行也提到当前 doc 时), 不去重, planner 自行权衡.
    """
    if not knowledge_md_path or not Path(knowledge_md_path).is_file():
        return ""
    lines = Path(knowledge_md_path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return ""
    head = "\n".join(lines[:head_lines])
    if len(head) > max_chars:
        head = head[:max_chars].rstrip() + "\n... (truncated)"
    return head


# ───────────────────── backend (pydantic-ai) ─────────────────────


async def _chat(
    model: Any,
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float = 0.4,
) -> tuple[str, str]:
    """单轮对话: (text, finish_reason).

    用 pydantic-ai Agent(output_type=str) 包一次无工具的纯文本对话. finish_reason
    无法跨 provider 可靠取得, 统一返回空串; 下游 validate_table 的 length 截断检查
    因此退化为软判断 (空串不触发截断分支).
    """
    agent = Agent(model=model, output_type=str, instructions=system)
    settings = OpenAIChatModelSettings(max_tokens=max_tokens, temperature=temperature)
    result = await agent.run(user, model_settings=settings)
    return (result.output or ""), ""


# ───────────────────── parsing helpers ─────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_MD_TABLE_RE = re.compile(
    r"(^\|[^\n]+\|\n\|[\s\-:|]+\|\n(?:\|[^\n]+\|\n?)+)",
    re.MULTILINE,
)


def extract_json(text: str) -> dict | list | None:
    candidates: list[str] = []
    for m in _JSON_BLOCK_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text.strip())
    if "{" in text and "}" in text:
        try:
            candidates.append(text[text.index("{"): text.rindex("}") + 1])
        except Exception:
            pass
    for raw in candidates:
        for variant in (raw, _heal_quotes_in_strings(raw)):
            try:
                return json.loads(variant)
            except Exception:
                continue
    return None


def _heal_quotes_in_strings(s: str) -> str:
    """把 JSON string 内"未转义的内嵌双引号"转义掉.

    扫一遍, 维护"是否在 string 内", 在 string 内遇到 `"` 时, 看下一个非空字符是不是
    `,` / `}` / `]` / `:`, 是 → 当作 string 结束; 否则 → 视为内嵌引号, 替成 `\"`.
    """
    out = []
    in_str = False
    escape = False
    i = 0
    while i < len(s):
        ch = s[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < len(s) and s[j] in " \t\r\n":
                j += 1
            nxt = s[j] if j < len(s) else ""
            if nxt in ",}]:" or j >= len(s):
                out.append(ch)
                in_str = False
            else:
                out.append("\\\"")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_md_row(line: str) -> list[str]:
    """把一行 markdown 表格切成单元格列表.

    只剥掉**首尾各一个**边框竖线 (markdown 表行形如 ``|a|b|c|``), 不能用
    ``str.strip("|")`` —— 后者会把行尾的 ``||`` (末列为空 + 行尾边框) 整体吞掉,
    导致"最后一列为空"的行少算一列 (常见于 *_noise 这种多数行留空的列), 被误判
    列数不齐而触发无谓重试 / 降级. 这里精确去掉一个边框竖线后再 split.
    """
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def parse_md_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    m = _MD_TABLE_RE.search(text)
    if not m:
        return None
    md = m.group(1)
    lines = [ln.rstrip() for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return None
    header = _split_md_row(lines[0])
    body_lines = lines[2:] if re.fullmatch(r"\|[\s\-:|]+\|", lines[1]) else lines[1:]
    rows = [_split_md_row(ln) for ln in body_lines]
    rows = [r for r in rows if not all(re.fullmatch(r"[\s\-:]*", c) for c in r)]
    return header, rows


def md_table_to_tsv(header: list[str], rows: list[list[str]]) -> str:
    out = ["\t".join(header)]
    for r in rows:
        out.append("\t".join(r))
    return "\n".join(out) + "\n"


# ───────────────────── validation ─────────────────────


def validate_plan(plan) -> list[str]:
    fails: list[str] = []
    if not isinstance(plan, dict):
        return ["plan 不是 JSON object"]
    for k in ("sections", "record_universe", "schema"):
        if k not in plan:
            fails.append(f"缺少顶层字段 {k}")
    if fails:
        return fails
    if not isinstance(plan["sections"], list) or not plan["sections"]:
        fails.append("sections 必须是非空列表")
    if not isinstance(plan["record_universe"], list) or not plan["record_universe"]:
        fails.append("record_universe 必须是非空列表")
    if not isinstance(plan["schema"], dict) or "record_id" not in plan["schema"]:
        fails.append("schema 必须是 dict 且包含 record_id 列")
    else:
        for col, spec in plan["schema"].items():
            if not isinstance(spec, dict):
                fails.append(f"schema[{col!r}] 必须是 dict, 含 doc_meaning + task_meaning 两个字段"); continue
            for k in ("doc_meaning", "task_meaning"):
                v = spec.get(k)
                if not isinstance(v, str) or not v.strip():
                    fails.append(f"schema[{col!r}].{k} 必须是非空字符串")
    universe = {str(x) for x in plan.get("record_universe", [])}
    if len(universe) != len(plan.get("record_universe", [])):
        fails.append("record_universe 含重复 id")
    for i, s in enumerate(plan.get("sections", [])):
        if not isinstance(s, dict):
            fails.append(f"sections[{i}] 不是 object"); continue
        for k in ("id", "title", "start_marker", "end_marker", "start_line", "end_line", "columns"):
            if k not in s:
                fails.append(f"sections[{i}] 缺字段 {k}")
        # start_line / end_line: 与 start_marker/end_marker 并存的行号定位. 本步只做
        # 单 section 自洽校验 (整数 + 1≤start≤end), 暂不强制跨 section 不重叠/全覆盖,
        # 待 planner 实际产出质量验证后再在 validate 收紧 (那时即可结构上掐死 noise section).
        sl, el = s.get("start_line"), s.get("end_line")
        if "start_line" in s and not (isinstance(sl, int) and not isinstance(sl, bool) and sl >= 1):
            fails.append(f"sections[{i}].start_line 必须是 ≥1 的整数, got {sl!r}")
        if "end_line" in s and not (isinstance(el, int) and not isinstance(el, bool) and el >= 1):
            fails.append(f"sections[{i}].end_line 必须是 ≥1 的整数, got {el!r}")
        if (isinstance(sl, int) and not isinstance(sl, bool)
                and isinstance(el, int) and not isinstance(el, bool) and sl > el):
            fails.append(f"sections[{i}] start_line({sl}) > end_line({el})")
        if "columns" in s:
            cols = s["columns"]
            if not isinstance(cols, list) or not cols or cols[0] != "record_id":
                fails.append(f"sections[{i}].columns 第一列必须是 record_id")
            for c in cols:
                if not isinstance(c, str) or not re.fullmatch(r"[a-z0-9_]+", c) or "__" in c:
                    fails.append(f"sections[{i}].columns 含非 snake_case 列名: {c}")
    return fails


def validate_table(
    text: str,
    finish_reason: str,
    section: dict,
    universe: set[str],
) -> tuple[list[str], list[str], list[list[str]], dict]:
    """校验 worker 输出. 返回 (fails, header, rows, diag).

    diag: 结构化诊断, 供重试 hint 精确判断该提示什么 (不靠 parse 中文 fail 文案):
      {"truncated": bool, "no_table": bool, "header_mismatch": bool,
       "bad_col_rows": int, "n_rows": int, "n_universe": int,
       "dup_ids": [...], "missing_ids": [...], "extra_ids": [...]}
    """
    fails: list[str] = []
    diag: dict = {
        "truncated": False, "no_table": False, "header_mismatch": False,
        "bad_col_rows": 0, "n_rows": 0, "n_universe": len(universe),
        "dup_ids": [], "missing_ids": [], "extra_ids": [],
    }
    if finish_reason == "length":
        diag["truncated"] = True
        fails.append("回复因 max_tokens 截断")
    parsed = parse_md_table(text)
    if parsed is None:
        diag["no_table"] = True
        fails.append("没有解析到合法 markdown 表格")
        return fails, [], [], diag
    header, rows = parsed
    diag["n_rows"] = len(rows)
    # worker 被要求把 __src_line 作为**第一列**, 其后接 plan.columns. 但这是审计列, 缺了不该让
    # 真实数据降级: 同时接受"带行号列(首列)"和"不带"两种表头. record_id 的 universe 校验按
    # 表头里 record_id 的实际位置取列, 不再假设它在第 0 列 (加了 __src_line 后它在第 1 列).
    base_cols = section["columns"]
    if header != base_cols and header != [SRC_LINE_COL] + base_cols:
        diag["header_mismatch"] = True
        fails.append(
            f"表头不匹配. expected={base_cols} (可在最前追加 {SRC_LINE_COL!r} 作首列), got={header}"
        )
    bad_col = [i for i, r in enumerate(rows) if len(r) != len(header)]
    if bad_col:
        diag["bad_col_rows"] = len(bad_col)
        fails.append(f"列数不齐: 第 {bad_col[:5]} 行(共 {len(bad_col)} 行)列数 != {len(header)}")
    if len(rows) != len(universe):
        fails.append(f"行数 {len(rows)} != |record_universe| {len(universe)} (每个 universe id 都必须有一行)")
    rid_idx = header.index("record_id") if "record_id" in header else 0
    if rows:
        first_col = [r[rid_idx] for r in rows if len(r) > rid_idx]
        seen = set(first_col)
        extra = sorted(seen - universe)
        if extra:
            diag["extra_ids"] = extra
            fails.append(f"出现 universe 之外的 id: {extra[:8]}")
        missing = sorted(universe - seen)
        if missing:
            diag["missing_ids"] = missing
            fails.append(f"漏掉了 universe 中的 id: {missing[:8]} (共 {len(missing)})")
        if len(first_col) != len(seen):
            from collections import Counter
            dup = [k for k, c in Counter(first_col).items() if c > 1]
            diag["dup_ids"] = dup
            fails.append(f"record_id 列出现重复: {dup[:8]}")
    return fails, header, rows, diag


# ───────────────────── stages ─────────────────────


async def run_planner(
    doc_text: str,
    *,
    doc_name: str = "",
    knowledge_excerpt: str = "",
    knowledge_global: str = "",
    trace: Optional[list[dict]] = None,
) -> dict:
    def _wrap_user(text_for_retry: str = "") -> str:
        parts: list[str] = []
        if doc_name:
            parts.append(f"当前 doc 名: {doc_name}")
        if knowledge_global:
            parts.append(
                "===== knowledge 全局口径 (task 整库级别声明: 默认单位 / 主键约定 / 时区 / 业务术语等;\n"
                "  适用于本 task 所有 doc, 优先级高于 doc 散文里隐含的口径) =====\n"
                + knowledge_global
                + "\n===== 全局口径结束 ====="
            )
        if knowledge_excerpt:
            parts.append(
                "===== knowledge 摘录 (用于列命名/口径对齐, 非数据源, 严禁据此造行;\n"
                "  只复用属于上述当前 doc 小节的字段; 邻表字段仅供理解 join) =====\n"
                + knowledge_excerpt
                + "\n===== 摘录结束 ====="
            )
        if text_for_retry:
            parts.append(text_for_retry)
        parts.append(
            "以下是完整文档. 输出唯一一个 plan JSON 代码块, 块外无文字.\n\n"
            "===== 文档开始 =====\n" + doc_text + "\n===== 文档结束 ====="
        )
        return "\n\n".join(parts)

    user = _wrap_user()

    fails: list[str] = []
    for attempt in range(2):
        reply, _fr = await _chat(_model_nothink(), PLANNER_SYSTEM, user, max_tokens=32768)
        plan = extract_json(reply)
        fails = validate_plan(plan) if plan is not None else ["JSON 无法解析"]
        if trace is not None:
            trace.append({
                "role": "planner", "attempt": attempt,
                "reply": reply, "fails": fails, "ok": not fails,
            })
        if not fails:
            logger.info("[fanout] planner OK on attempt {}", attempt)
            return plan
        logger.warning("[fanout] planner FAIL on attempt {}: {}", attempt, fails)
        if attempt == 0:
            user = _wrap_user(
                "你上一次输出未通过校验. 失败原因:\n- "
                + "\n- ".join(fails)
                + "\n\n请重新输出, 严格遵循 system 中的 schema 要求."
            )
    raise RuntimeError(f"planner failed after retry: {fails}")


def _section_user_prompt(doc_text: str, plan: dict, section: dict) -> str:
    # worker 只看 doc 端口径 (doc_meaning), 不看 task_meaning, 也不看 column_normalize.
    # 单位/格式归一统一交给 Phase 6 后处理. 这里把 dict-spec 扁平成 {col: doc_meaning},
    # 让 worker prompt 与改造前的旧 schema 形态保持一致, WORKER_SYSTEM 无需改动.
    worker_schema = {
        col: (spec.get("doc_meaning", "") if isinstance(spec, dict) else str(spec))
        for col, spec in plan["schema"].items()
    }
    plan_summary = {
        "schema": worker_schema,
        "record_universe": plan["record_universe"],
        "this_section": section,
    }
    return (
        "你是 section_id=" + str(section["id"]) + " 的 worker.\n"
        "下面是完整文档 + 全局 plan + 你的 section spec.\n"
        "按 system 规则输出 markdown 表.\n\n"
        "===== plan =====\n"
        + json.dumps(plan_summary, ensure_ascii=False, indent=2)
        + "\n\n===== 文档开始 =====\n" + doc_text + "\n===== 文档结束 ====="
    )


def _retry_hint_for_diag(diag: dict) -> str:
    """按结构化诊断 diag (validate_table 返回) 追加针对性纠错提示, 仅在重试时注入.

    不再 parse 中文 fail 文案 —— 直接读 diag 的结构化字段, 精确区分两种独立问题:
      - dup_ids 非空: 同一 record_id 出现多行. 通常是把"同一 record 分散在 doc 多行/多处"
        的内容各开了一行 (或把别的 section 数据误抽进来). 提示合并成一行.
      - missing_ids 非空: 真的漏抽了某些 id. 提示这些 record 必须补上 (找不到数据也保留 NULL 行).
    注意: "行数 != universe" 本身不单独触发提示 —— 行数偏多基本是 dup 的副作用 (由 dup 分支
    覆盖), 行数偏少则必然伴随 missing_ids. 这样避免上一版"行数不等就提示抽取遗漏"的误导
    (dup 导致 51 行、id 其实全在时, 不该提示遗漏).
    各提示带上 diag 里的具体 id (已 [:8] 截断显示).
    """
    hints: list[str] = []
    dup = diag.get("dup_ids") or []
    missing = diag.get("missing_ids") or []
    if dup:
        hints.append(
            f"【提示】出现重复 record_id (重复的 id: {dup[:8]}).首先检查是否将其他section的数据错误提取. "
            "如果不是,检查是否同一条 record 的内容在 doc 中分散于多行/多处, "
            "你为它开了多行. 请把这些信息**合并进同一行**: 一条 record 占且仅占一行, 同一列出现"
            "多个值时以该 record 主行 (含主键与主要数值那行) 为准, 不要为此多开一行."
        )
    if missing:
        hints.append(
            f"【提示】漏掉了 record_universe 中的 id (共 {len(missing)} 个: {missing[:8]}), "
            "行数必须恒等于 |record_universe|.首先检查是否有抽取遗漏. "
            "如果不是,严谨地确认是否这些 record 在本 section 确实没有任何数据 "
            "(罕见), 也应创建该行: 行号填0, record_id 照填, 其余列全 NULL, 不要删行. "
        )
    return ("\n\n" + "\n".join(hints)) if hints else ""


async def run_section_worker(
    doc_text: str,
    plan: dict,
    section: dict,
    universe: set[str],
    trace: Optional[list[dict]] = None,
) -> tuple[dict, list[str], list[list[str]], list[str]]:
    user = _section_user_prompt(doc_text, plan, section)

    last_fails: list[str] = []
    last_reply = ""
    for attempt in range(2):
        reply, fr = await _chat(_model_nothink(), WORKER_SYSTEM, user, max_tokens=32768)
        last_reply = reply
        fails, header, rows, diag = validate_table(reply, fr, section, universe)
        if trace is not None:
            trace.append({
                "role": "section", "section_id": section.get("id"), "attempt": attempt,
                "reply": reply, "fails": fails, "ok": not fails, "n_rows": len(rows),
            })
        if not fails:
            logger.info("[fanout] section {} OK on attempt {}, rows={}", section["id"], attempt, len(rows))
            return section, header, rows, []
        last_fails = fails
        logger.warning("[fanout] section {} FAIL on attempt {}: {}", section["id"], attempt, fails)
        if attempt == 0:
            user = (
                "你上一次输出未通过校验. 失败原因:\n- "
                + "\n- ".join(fails)
                + _retry_hint_for_diag(diag)
                + "\n\n请只输出修正后的 markdown 表, 不要解释.\n\n"
                + _section_user_prompt(doc_text, plan, section)
            )
    logger.warning("[fanout] section {} giving up after retries, keep best-effort", section["id"])
    parsed = parse_md_table(last_reply)
    header, rows = parsed if parsed else (section["columns"], [])
    return section, header, rows, last_fails


# ───────────────────── assembly ─────────────────────


def assemble_wide(
    plan: dict, section_results: list[tuple]
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, str]]]:
    """按 record_id outer-join 所有 section 表, 列顺序 = schema 顺序.

    返回 (final_cols, rows, provenance):
      - rows 为 list[dict], 每个 dict 是 列名→值, 便于下游直接构造 DataFrame;
        **不含 __src_line** —— 该列被剥进 provenance, 绝不进主表/DB/prediction.csv.
      - provenance: {record_id: {section_id(str): src_line}} —— 每条 record 在各 section
        的来源行号, 供后处理把可疑 cell 映射回 doc 原文. worker 没给行号的 cell 不收录.

    record_id 与 __src_line 都按各 section 的**表头名**定位 (不假设列位置), 因为带行号时
    record_id 不再在第 0 列.
    """
    schema_cols = list(plan["schema"].keys())
    universe = [str(x) for x in plan["record_universe"]]
    table: dict[str, dict[str, str]] = {rid: {"record_id": rid} for rid in universe}
    provenance: dict[str, dict[str, str]] = {}
    for section, header, rows, _ in section_results:
        sec_id = str(section.get("id", ""))
        rid_idx = header.index("record_id") if "record_id" in header else 0
        src_idx = header.index(SRC_LINE_COL) if SRC_LINE_COL in header else None
        for r in rows:
            if len(r) <= rid_idx:
                continue
            rid = r[rid_idx]
            if rid not in table:
                table[rid] = {"record_id": rid}
            for c, v in zip(header, r):
                if c == "record_id" or c == SRC_LINE_COL:
                    continue
                if v and v != "NULL":
                    table[rid][c] = v
            # 收 provenance: 行号非空才记 (worker 对空行填 NULL).
            if src_idx is not None and len(r) > src_idx:
                sv = r[src_idx]
                if sv and sv != "NULL":
                    provenance.setdefault(rid, {})[sec_id] = sv
    final_cols = ["record_id"] + [c for c in schema_cols if c != "record_id"]
    extra_cols: list[str] = []
    for rid_data in table.values():
        for k in rid_data:
            if k not in final_cols and k not in extra_cols and k != SRC_LINE_COL:
                extra_cols.append(k)
    final_cols += extra_cols
    rows_out = [{c: table[rid].get(c, "") for c in final_cols} for rid in universe]
    return final_cols, rows_out, provenance


# ───────────────────── column_normalize 后处理 ─────────────────────


# transform_py 沙箱可用的 builtins (白名单). 故意不放 open/eval/exec/compile/
# globals/locals/getattr/setattr/delattr/vars/dir, 杜绝从字符串抽出文件读写或反射逃逸.
# __import__ 用受控版本 (_safe_import) 替换: 只放行 re / math 两个纯计算模块, 其余一律拒绝.
# 放行原因: 日期/数值归一几乎离不开正则, 模型高频写 `import re`; 禁掉会让整列 normalize 直接
# ImportError 全失败 (实测 enddate 列 50 行全 fail). re/math 无文件/网络/反射能力, 无逃逸风险.
_TRANSFORM_ALLOWED_MODULES = {"re", "math"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """受控 __import__: 仅放行 _TRANSFORM_ALLOWED_MODULES 中的纯计算模块."""
    root = name.split(".", 1)[0]
    if root not in _TRANSFORM_ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed in transform sandbox")
    return __import__(name, globals, locals, fromlist, level)


_TRANSFORM_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min, "range": range,
    "round": round, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
    # ord/chr 让 transform 可以做字符级处理而无需 import re
    "ord": ord, "chr": chr,
    # 类型守卫/集合构造: 模型生成的 transform 常写 isinstance(v, str) 之类,
    # 缺失会让整列 normalize 抛 NameError 而全部退化. 这些都无逃逸风险.
    "isinstance": isinstance, "type": type, "set": set, "dict": dict,
    "frozenset": frozenset, "reversed": reversed,
    # 受控 import: 只放行 re / math (见 _safe_import). 让 `import re` 这类正则归一可正常工作.
    "__import__": _safe_import,
}


def _sanitize_normalize_rules(
    plan: dict,
    schema_cols: list[str],
    transforms: dict[str, str],
) -> dict[str, dict]:
    """挑出可执行的归一化规则, 返回 {col: rule_dict (含已编译 _fn)}.

    transforms 是 Phase 5.5 (generate_transforms) 针对该列全量数据生成的 {col: transform_py 源码}.
    放过: column 在 schema 里 + 在 plan["column_normalize"] 声明 + transforms 里有对应代码 + 能编译.
    其他 (record_id / 不在 schema 的列 / 未声明 / 无生成代码 / 编译失败) 全部丢弃, 仅 warn 不报错,
    保证后处理永远不会让原 rows 退化.
    """
    raw = plan.get("column_normalize")
    if not isinstance(raw, dict) or not raw or not transforms:
        return {}
    schema_set = set(schema_cols)
    out: dict[str, dict] = {}
    for col, spec in raw.items():
        if col == "record_id" or col not in schema_set:
            logger.warning("[fanout] normalize: drop col={!r} (not in schema or is record_id)", col)
            continue
        if not isinstance(spec, dict):
            logger.warning("[fanout] normalize: drop col={!r} (rule not dict)", col); continue
        code = transforms.get(col)
        if not isinstance(code, str) or "def transform" not in code:
            logger.warning("[fanout] normalize: drop col={!r} (no generated transform)", col); continue
        try:
            fn = _compile_transform(code)
        except Exception as e:
            logger.warning("[fanout] normalize: drop col={!r} (compile failed: {}: {})", col, type(e).__name__, e)
            continue
        out[col] = {**spec, "transform_py": code, "_fn": fn}
    return out


def _compile_transform(code: str):
    """编译 transform_py 字符串, 返回 callable transform. 失败 raise.

    注意: exec(code) 只**定义**函数, 不调用它, 因此不会触发 transform 体内的死循环 ——
    死循环只会在真正 fn(v) 执行时发生. 故父进程用本函数做"可编译性"校验是安全的, 真正的
    逐值执行一律下放到 _run_transform_on_distinct 的子进程里 (带墙钟超时).
    """
    g = {"__builtins__": _TRANSFORM_BUILTINS}
    exec(code, g)  # noqa: S102 — 沙箱已收 builtins
    fn = g.get("transform")
    if not callable(fn):
        raise ValueError("transform_py did not define a callable named 'transform'")
    return fn


# LLM 生成的 transform 可能含死循环 (实测: 某版 cn_to_int 对特定脏值 while 不推进指针,
# 整列 probe 把主线程钉死在 99% CPU 数十分钟). exec 沙箱只收 builtins, 拦不住死循环 ——
# 唯一能硬杀任意死循环 (含 C 扩展) 的办法是把执行放进子进程, 超时即 terminate.
# 阈值: 单值正常 transform 是微秒级, 整列 distinct (≤300) 正常总耗时 < 0.1s; 给 100x 余量
# 取 10s 作为**整列**墙钟预算, 足以区分"慢但正常"与"无限循环".
_PROBE_TIMEOUT_SEC = 10.0

# spawn (而非 fork): 父进程带 18 个 asyncio 工作线程, fork 会复制锁状态易死锁; spawn 干净.
_MP_CTX = mp.get_context("spawn")


def _transform_worker(code: str, values: list[str], q) -> None:
    """子进程入口 (spawn 要求模块级可 import): 编译 code 后逐值跑, 每个值即时回传一条结果.

    增量回传 (而非跑完一次性返回) 是为了在父进程超时杀进程后, 能用"已收到几条"精确定位
    到卡死在哪个值 (culprit = values[已收条数]), 把它作为 errored 反馈给模型重生成.
    """
    try:
        g = {"__builtins__": _TRANSFORM_BUILTINS}
        exec(code, g)  # noqa: S102
        fn = g.get("transform")
        if not callable(fn):
            q.put(("fatal", "transform_py did not define a callable named 'transform'"))
            return
    except Exception as e:  # 理论上父进程已校验过可编译, 防御性兜底
        q.put(("fatal", f"{type(e).__name__}: {e}"))
        return
    for v in values:
        try:
            out = fn(v)
            new_v, kind = _classify_transform_output(out, v)
            q.put(("ok", new_v, kind))
        except Exception as e:
            q.put(("err", f"{type(e).__name__}: {e}"))
    q.put(("end",))


def _run_transform_on_distinct(
    code: str, distinct: list[str], timeout: float = _PROBE_TIMEOUT_SEC,
) -> tuple[list[tuple], Optional[str]]:
    """在 spawn 子进程里编译 code 并逐值执行 distinct, 带整列墙钟超时.

    返回 (results, fatal):
      - fatal 非空 → 子进程编译失败, results 为空, 调用方按整列失败处理.
      - results 与 distinct 一一对齐, 每项是:
          ("ok", new_value, kind)  kind ∈ {"str","num","none","bad"} (见 _classify_transform_output)
          ("err", errmsg)          fn(v) 抛异常
          ("timeout",)             超时被杀时正卡在的那个值 (死循环嫌疑)
          ("skipped",)             超时后尚未执行到的值 (同版本已不可信)
    """
    if not distinct:
        return [], None
    q = _MP_CTX.Queue()
    p = _MP_CTX.Process(target=_transform_worker, args=(code, distinct, q), daemon=True)
    p.start()
    results: list[tuple] = []
    fatal: Optional[str] = None
    deadline = _time.monotonic() + timeout
    try:
        while len(results) < len(distinct):
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                break  # 整列超时
            try:
                item = q.get(timeout=remaining)
            except _queue.Empty:
                break  # 超时 (子进程卡在某个值上没再 put)
            if item[0] == "fatal":
                fatal = item[1]
                break
            if item[0] == "end":
                break
            results.append(item)
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():  # terminate 没生效 (极端情况), 强杀
                p.kill()
                p.join()
        else:
            p.join()
        try:
            q.close()
            q.cancel_join_thread()
        except Exception:
            pass
    if fatal is not None:
        return [], fatal
    # 超时: 用 ("timeout",) 标记卡死的那个值 (culprit = 已收条数对应的下一个),
    # 其余未执行值标 ("skipped",). 二者都会被 probe 计入 errored, 使这版 transform
    # 的 errored 数虚高 → 不会被 best_code 选中拿去 apply, 从而 apply 也不会再卡.
    if len(results) < len(distinct):
        results.append(("timeout",))
        while len(results) < len(distinct):
            results.append(("skipped",))
    return results, None


# ───────────────────── Phase 5.5: 按列全量数据生成 transform ─────────────────────

NORMALIZER_SYSTEM = """你是 doc 结构化的归一化代码生成器. 给你**某一列的全量去重原始值** + 一份目标格式描述,
你产出一段 Python 函数, 把该列每个原始值改写成目标格式.

输出唯一一个 ```python``` 代码块, 块内**定义且仅定义**一个函数:

    def transform(s):
        ...
        return ...

签名: `transform(s: str) -> str`, 输入是该列某一行的原始字符串值, 返回归一化后的字符串.

**思考纪律 (这是写代码任务, 不是辩论题, 控制思考开销)**:
  - 不要过度思考. 目标格式示例已给出标准答案, 看懂规律即可动笔, 别在"该听哪条规则"上反复横跳.
  - **不要逐个手算 distinct 值**: 它们只是让你识别格式种类 (几种单位/几种写法), 真正的计算交给代码.
    认出要覆盖的不同数据类型就够了, 不必在脑子里把每个值算一遍.
  - 同一条指令读一遍即可, 不要反复重读 / 反复自我确认同一个结论.
  - 不要揣测指令背后的"意图/是不是陷阱": 指令冲突时直接按下面「最高优先级规则」裁决, 不必解释为什么.
  - 想清楚"几类格式 + 各自怎么转 + 输出格式"三件事就足以写出代码, 到此为止.

**最高优先级规则 (冲突时唯一仲裁标准, 先读这条再读下面所有要点)**:
  user 给你的「目标格式」描述 + 其中的**转换示例**, 是这一列输出格式的**唯一真相**.
  下面"写法要点"是通用模板, 只在目标格式没明说时作参考; **一旦与目标格式的示例冲突, 无条件以示例为准**,
  不要在两者之间反复权衡 —— 示例怎么写, 你就怎么输出. 例如示例写 `'1.324 亿元' → '132400000'`,
  那 1.324e8 就输出整数字符串 '132400000'.

硬约束:
  - 只允许 `import re` 和 `import math`, 禁止其他任何 import; 禁止文件/网络/eval/exec, 不允许定义其他全局变量/类.
  - 可用 builtins (str/int/float/round/len/abs/min/max/sorted/enumerate 等) + 字符串方法 + re/math.
  - **严禁死循环**: 你的代码会被拿该列每个原始值实跑, 一旦某次调用陷入死循环, 整列归一会被强制
    终止并判失败. 写任何 `while`/`for` 循环都要保证**每次迭代都在推进** (指针自增 / 切片变短),
    且有明确的终止条件; 慎用 `while True`, 如确需用必须有可达的 break; 用正则边扫描边移动位置时,
    确保匹配位置严格前移, 不会卡在同一处反复匹配. 不确定能否终止时, 加一个迭代次数上限兜底.
  - **只做格式 / 单位换算, 不改动数值的有效内容**: 不许无谓舍入 (单位换算需要的除外), 不许改正负号, 不许重排数字.
  - 输入保证是非空字符串 (NULL/空值不会传入). 若某个输入不符合预期形态, `return s` 原样返回, 不要抛异常.
  - 函数要**覆盖给你的全部样例值**: 下面列出的 distinct 值就是这一列的真实全集 (或绝大部分), 你的逻辑必须
    能把其中**每一个**正确转成目标格式. 不要只处理头几个就了事, 也不要漏掉长尾形态.

写法要点 (通用模板, 与上面「目标格式示例」冲突时以示例为准):
  - 单位换算 (亿元/万元 → 个位元 等): **先匹配长单位再匹配短单位**, 严禁先 `replace('元','')`
    再处理 '亿元'/'万元' (会把 '亿元' 里的 '元' 先删掉导致换算失效). 推荐用 re 或 endswith 精确判断单位后,
    去掉单位字符、去掉千分位逗号、再乘以对应倍数 (亿=1e8, 万=1e4), 输出纯数字字符串.
  - 日期 → ISO 'YYYY-MM-DD': 用 re 抽年份, 再按 "第一/二/三/四季度末"、"年中"、"上半年"、"财年/日历年结束"、
    "最后一个交易日"、"X 月 Y 日" 等说法映射到对应月日. 已是 'YYYY-MM-DD' 的直接抽出返回.
  - 百分比: 看目标描述决定是"去百分号保留数值" (如 '98.3%' → '98.3') 还是"转 0~1 小数" (如 '98.3%' → '0.983').
  - 数值清洗: 去千分位逗号、去单位后缀、保留负号与小数位.
  - 数值输出格式 (除非目标格式另有明示, 按此默认, 别自行加减小数位):
      * 结果是整数 (无小数部分, 如换算后的 132400000.0) → 输出**纯整数字符串** '132400000', 不带 '.0' 不补小数;
      * 结果有小数 (如 '40755680.17') → **保留它原本的有效小数位**, 不四舍五入也不补尾零;
      * 一律禁止科学计数法 (1e8 / 1.324E8 这种); float 偏大时用 `repr`/格式化避免 Python 默认科学计数法.
      实现可参考: `v = float(num)*factor; return str(int(v)) if float(v).is_integer() else (...)`.

只输出一个 python 代码块, 块外不要任何解释文字.
"""

# 从 LLM 回复里抽第一个 python 代码块; 退化时取首个含 `def transform` 的 ``` 块.
_PY_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


def _extract_python(text: str) -> str | None:
    """从回复抽出含 `def transform` 的代码. 优先 ```python``` 块, 退化为裸文本."""
    for m in _PY_BLOCK_RE.finditer(text or ""):
        body = m.group(1).strip()
        if "def transform" in body:
            return body
    if text and "def transform" in text:
        return text.strip()
    return None


def _distinct_nonnull(rows: list[dict[str, str]], col: str, limit: int = 300) -> list[str]:
    """收集某列全部 distinct 非空原值 (保持首次出现顺序), 上限 limit 防 prompt 过长."""
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        v = r.get(col)
        if v is None or v == "" or v == "NULL":
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= limit:
            break
    return out


def _classify_transform_output(out: Any, original: str) -> tuple[str, str]:
    """把 transform 返回值归一到 (new_value, kind), 与 apply_column_normalize 的兜底口径一致.

    kind: 'str'(返回字符串) | 'num'(int/float/bool, str() 兜底) | 'none'(返回 None, 原样回退)
          | 'bad'(其他类型, 调用方按失败处理). new_value 对 none/bad 取 original.
    """
    if isinstance(out, str):
        return out, "str"
    if isinstance(out, (int, float)):  # bool 是 int 子类, str(True)='True'
        return str(out), "num"
    if out is None:
        return original, "none"
    return original, "bad"


def _probe_transform(
    code: str, distinct: list[str]
) -> tuple[list[tuple[str, str]], list[str], int, list[str]]:
    """拿全量 distinct 在**子进程**里实跑 code, 分类返回 (errored, unchanged, n_changed, timed_out).

    errored:   [(val, errmsg)] 抛异常 / 返回非法类型 —— 硬失败, 必须修.
    unchanged: [val] 转换后等于原值 (含 None 兜底) —— 可能是没覆盖到的格式特例 (软信号).
    n_changed: 成功改写条数.
    timed_out: [val] 死循环 culprit —— 子进程整列墙钟超时被杀, culprit 是卡住时正在跑的那个值
               (通常一个). 与 errored **分开返回**: 死循环不是"抛异常/返回非法类型", 修法完全不同
               (检查循环推进 / 加迭代上限, 而非 try/except), 上层据此给模型专属反馈.

    与旧版的区别: 不再在父进程直接 fn(v) (会被 LLM 生成的死循环钉死), 改为下放子进程带
    墙钟超时 (_run_transform_on_distinct). 超时时 culprit 进 timed_out, 其后未执行的值
    (skipped) 不进任何反馈列表 —— 它们只是"同版本没轮到", 不是独立问题, 报出来只会刷屏淹没
    真正的 culprit. 但 n_changed 不计入它们, 故该版仍会因 timed_out 非空而判失败、不被选中.
    """
    errored: list[tuple[str, str]] = []
    unchanged: list[str] = []
    timed_out: list[str] = []
    n_changed = 0
    results, fatal = _run_transform_on_distinct(code, distinct)
    if fatal is not None:
        # 子进程连编译都没过 (父进程已校验过, 一般到不了这): 整列判失败.
        return [(v, fatal) for v in distinct], [], 0, []
    for v, item in zip(distinct, results):
        tag = item[0]
        if tag == "ok":
            _, new_v, kind = item
            if kind == "bad":
                errored.append((v, "返回类型非法, 需要 str"))
            elif new_v != v:
                n_changed += 1
            else:
                unchanged.append(v)
        elif tag == "err":
            errored.append((v, item[1]))
        elif tag == "timeout":
            timed_out.append(v)  # 死循环 culprit, 单独反馈
        # else "skipped" — 超时后未执行到, 不报 (避免刷屏淹没 culprit)
    return errored, unchanged, n_changed, timed_out


def _dump_messages(messages) -> Any:
    """把 pydantic-ai 的 ModelMessage 列表转成可 JSON 落盘的结构 (用于 trace).

    优先用官方 ModelMessagesTypeAdapter 序列化; 失败则退化为各 part 文本的朴素抽取,
    保证 capture_run_messages 抓到的内容 (含模型报错前的半截输出) 总能进 fanout_allmsg.json.
    """
    if not messages:
        return []
    try:
        from pydantic_ai.messages import ModelMessagesTypeAdapter
        return json.loads(ModelMessagesTypeAdapter.dump_json(messages))
    except Exception:
        pass
    out: list[dict] = []
    for msg in messages:
        kind = type(msg).__name__
        parts_text: list[str] = []
        for p in getattr(msg, "parts", []) or []:
            content = getattr(p, "content", None)
            if content is not None:
                parts_text.append(str(content))
        out.append({"kind": kind, "parts": parts_text})
    return out


async def _gen_transform_for_col(
    col: str,
    target: str,
    doc_fmt: str,
    distinct: list[str],
    *,
    max_rounds: int = 5,
    trace: Optional[list[dict]] = None,
) -> str | None:
    """多轮对话生成单列 transform: 生成 → 全量 distinct 实跑 → 把失败/未覆盖回传 → 模型修订.

    解决单轮生成的盲区: doc 同一列常有多种格式, 单轮代码易漏掉长尾特例, 落进 `return s`
    兜底分支静默原样返回, 既不报错也无人察觉. 这里每轮都拿全量 distinct 实跑一遍, 把
    抛异常的值 (硬失败) 和原样返回的值 (疑似漏覆盖) 回传给模型迭代修订.

    用**同一个 Agent 维护多轮对话** (message_history): 模型自己上一轮的代码与首轮的完整任务
    都已在对话历史里, 因此反馈轮只需回传"实跑发现的问题", 不再重复粘贴上轮代码 / 全量样例.

    停止条件: errored (抛异常/非法类型) 为空且 unchanged 相比上一轮不再变化 (收敛), 或到
    max_rounds. unchanged 仅作软提示驱动继续修订, 不单独作硬停止 —— 本就是目标格式的值天然
    unchanged, 否则会无限循环. 返回 errored 最少的那版可编译代码; 始终拿不到则 None.
    """
    sample_block = "\n".join(f"  - {v!r}" for v in distinct)
    first_user = (
        f"列名: {col}\n"
        + (f"doc 原文形态 (planner 描述): {doc_fmt}\n" if doc_fmt else "")
        + f"目标格式 (planner 描述, 含示例):\n{target}\n\n"
        f"该列出现的全部 distinct 原始值 (共 {len(distinct)} 个, 你的 transform 必须覆盖其中所有情况):\n"
        f"{sample_block}\n\n"
        "请输出一个 python 代码块, 定义 transform(s)->str, 把上面每个原始值正确转成目标格式."
    )

    agent = Agent(model=_model_nothink(), output_type=str, instructions=NORMALIZER_SYSTEM)  # todo  think
    settings = OpenAIChatModelSettings(max_tokens=18192, temperature=0.4)  # todo param
    history: list = []  # pydantic-ai ModelMessage 列表, 跨轮累积对话上下文

    best_code: str | None = None
    best_errn: Optional[int] = None  # best_code 对应的 errored 条数
    prev_unchanged: Optional[set[str]] = None
    user = first_user

    for rnd in range(max_rounds):
        # capture_run_messages 只罩单次 run: run 抛异常时 result 拿不到, 但本次已交换的
        # messages (含模型报错前的半截输出) 仍能从 captured 取出, 落进 trace 供排查.
        with capture_run_messages() as captured:
            try:
                result = await agent.run(
                    user, message_history=history or None, model_settings=settings
                )
            except Exception as e:
                logger.warning("[fanout] gen_transform: col={!r} round {} chat failed: {}: {}", col, rnd, type(e).__name__, e)
                if trace is not None:
                    trace.append({
                        "role": "normalizer", "col": col, "round": rnd, "n_distinct": len(distinct),
                        "sent_user": user, "reply": "", "ok": False,
                        "err": f"chat failed: {type(e).__name__}: {e}",
                        "captured_messages": _dump_messages(captured),
                        "n_errored": 0, "n_unchanged": 0, "n_changed": 0,
                    })
                break
        reply = result.output or ""
        history = result.all_messages()  # 含本轮 user + assistant, 下一轮接着喂

        code = _extract_python(reply)
        fn = None
        err = ""
        if code:
            try:
                fn = _compile_transform(code)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        else:
            err = "no python block / no def transform"

        errored: list[tuple[str, str]] = []
        unchanged: list[str] = []
        n_changed = 0
        timed_out: list[str] = []
        if fn is not None:
            # fn 仅用于上面的可编译校验 (exec 只定义不执行, 安全); 真正逐值实跑由
            # _probe_transform 下放子进程并带超时, 故这里传 code 而非 fn.
            errored, unchanged, n_changed, timed_out = _probe_transform(code, distinct)
            # 留存"最干净"的版本: 含死循环的版本必须劣于任何不含死循环的版本 (即便它 errored 更少),
            # 因为死循环值在子进程里根本没跑出结果, 该版不可信. 用 (有无超时, errored 数) 元组排序,
            # 低者胜. 若每轮都超时, best_code 留 None → 整列跳过归一 (保留原值), 安全兜底.
            cur_key = (1 if timed_out else 0, len(errored))
            if best_errn is None or cur_key < best_errn:
                best_code, best_errn = code, cur_key

        if trace is not None:
            trace.append({
                "role": "normalizer", "col": col, "round": rnd, "n_distinct": len(distinct),
                "sent_user": user, "reply": reply, "ok": fn is not None, "err": err,
                "n_errored": len(errored), "n_unchanged": len(unchanged), "n_changed": n_changed,
                "n_timed_out": len(timed_out), "timed_out": timed_out,
            })

        if fn is None:
            logger.warning("[fanout] gen_transform: col={!r} round {} unusable ({})", col, rnd, err)
            user = f"你上一轮的代码无法使用 ({err}). 请重新输出一个可编译的 transform(s)->str, 只输出一个 python 代码块."
            continue

        logger.info(
            "[fanout] gen_transform: col={!r} round {}: changed={} unchanged={} errored={} timed_out={} ({} lines)",
            col, rnd, n_changed, len(unchanged), len(errored), len(timed_out), code.count("\n") + 1,
        )

        cur_unchanged = set(unchanged)
        if not errored and not timed_out and rnd == 0 and not unchanged:
            break  # 首轮即全部改写, 无失败无死循环无遗留, 直接收
        # 死循环必须修, 不能算收敛; 收敛仅在无 errored 且无 timed_out 时成立.
        converged = (
            not errored and not timed_out
            and prev_unchanged is not None and cur_unchanged == prev_unchanged
        )
        if converged or rnd == max_rounds - 1:
            break
        prev_unchanged = cur_unchanged

        # 反馈: 只回传实跑发现的问题, 上一轮代码与全量样例已在对话历史里, 不再重复粘贴.
        # 死循环 (timed_out) 与 抛异常/非法类型 (errored) 是两类不同的 bug, 分开给指令:
        # 死循环要查循环推进/加迭代上限, 而非 try/except —— 混在一起报会误导模型去加 try.
        err_block = "\n".join(f"  - {v!r} -> {msg}" for v, msg in errored[:40]) or "  (无)"
        unc_block = "\n".join(f"  - {v!r}" for v in unchanged[:40]) or "  (无)"
        loop_section = ""
        if timed_out:
            loop_block = "\n".join(f"  - {v!r}" for v in timed_out[:20])
            loop_section = (
                f"\n\n【⚠️ 死循环 —— 最高优先级, 必须修】:\n"
                f"你上一轮的 transform 在处理下面这些值时**陷入死循环** (整列执行超过 "
                f"{_PROBE_TIMEOUT_SEC:.0f}s 被强制终止), 导致整列无法完成:\n{loop_block}\n"
                "原因几乎一定是某个 `while`/`for` 循环的**终止/推进条件写错了** (例如指针没自增、"
                "`while True` 没有 break、正则反复匹配同一位置). 请定位并修正该循环: 确保每次迭代都有"
                "推进、有明确的终止条件; 对不确定能否终止的循环, 加一个迭代次数上限 (如最多 len(s) 次) 兜底. "
                "**不要用 try/except 去包死循环 —— 那拦不住死循环.**"
            )
        user = (
            "我把你上一轮的 transform 拿该列全部 distinct 原始值实跑了一遍, 发现:"
            + loop_section
            + "\n\n【抛异常 / 返回非法类型 —— 必须修复】:\n" + err_block + "\n\n"
            "【转换后原样返回 —— 可能是没覆盖到的格式特例; 若它本就是目标格式可保持不变, "
            "否则要补上对应处理分支】:\n" + unc_block + "\n\n"
            "请基于你上一轮的代码修正, 输出修正后的**完整** transform(s)->str, 覆盖上述所有应转换的值. "
            "只输出一个 python 代码块. (直接改, 不必逐个手算这些值, 也不必长篇解释为什么.)"
        )

    if best_code is not None:
        logger.info("[fanout] gen_transform: col={!r} done, residual={}", col, best_errn)
    else:
        logger.warning("[fanout] gen_transform: col={!r} no usable transform after {} rounds", col, max_rounds)
    return best_code


async def generate_transforms(
    plan: dict,
    rows: list[dict[str, str]],
    schema_cols: list[str],
    *,
    trace: Optional[list[dict]] = None,
) -> dict[str, str]:
    """Phase 5.5: 对每个 column_normalize 列, 喂全量 distinct 原值 + target_format 描述,
    让 LLM 针对真实数据生成 transform_py. 返回 {col: transform_py 源码}.

    与 planner 解耦: planner 只声明要做什么 (target_format 自然语言+示例), 转换逻辑在这里
    针对该列**全量数据**生成, 从而不会出现 planner 盲写规则覆盖不全的问题.
    每列走 _gen_transform_for_col 的多轮迭代 (生成→实跑→反馈未覆盖/失败值→修订), 以应对
    doc 中同列存在多种格式、单轮生成覆盖不全的情况. 始终拿不到可用代码的列跳过, 下游保留原值.

    模型: 用 contextvar 注入的 think 模型 (_model_think(); 写 transform 是推理任务: 单位换算/
        日期映射/边界处理, 建议 think 模型, 与 planner/worker 的 nothink 解耦). 输出纯文本 python 块.
    """
    raw = plan.get("column_normalize")
    if not isinstance(raw, dict) or not raw:
        return {}
    schema_set = set(schema_cols)
    out: dict[str, str] = {}
    for col, spec in raw.items():
        if col == "record_id" or col not in schema_set or not isinstance(spec, dict):
            continue
        target = spec.get("target_format")
        if not isinstance(target, str) or not target.strip():
            logger.warning("[fanout] gen_transform: skip col={!r} (target_format missing)", col)
            continue
        distinct = _distinct_nonnull(rows, col)
        if not distinct:
            continue  # 整列空, 无需归一
        doc_fmt = spec.get("doc_format", "") if isinstance(spec.get("doc_format"), str) else ""
        code = await _gen_transform_for_col(col, target, doc_fmt, distinct, trace=trace)
        if code:
            out[col] = code
    return out


def apply_column_normalize(
    plan: dict,
    rows: list[dict[str, str]],
    schema_cols: list[str],
    transforms: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, dict]]:
    """对 plan["column_normalize"] 中合规的列, 跑 Phase 5.5 生成的 transform_py 改写每行的值.

    transforms = generate_transforms 的返回 ({col: transform_py 源码}).
    输入 rows 不被原地修改, 返回新 rows. 对每列额外返回一份执行统计:
        {col: {"applied": int, "skipped_null": int, "failed": int, "coerced": int, "errors": [str, ...]}}
    便于上游打日志/落审计. coerced = applied 中因返回非 str 而被执行层兜底转换的条数.

    失败策略:
      - 整列 transform 编译失败 → 在 _sanitize_normalize_rules 已被丢弃, 不会到这里;
      - 行级运行报错 → 当前 cell 保留原值, failed 计数, 不影响其他行/其他列;
      - transform 返回 int/float/bool → 执行层 str() 兜底, 计入 applied + coerced (不算失败);
      - transform 返回 None → 视为"输入不符 doc_format 的原样返回", 保留原值, 计入 coerced;
      - transform 返回其他类型 (list/dict 等) → 判失败, 保留原值;
      - transform 死循环 → 整列墙钟超时被子进程强杀, 该列全部判 failed, 保留原值;
      - 空 / "NULL" 不送进 transform.

    执行模型: 不在父进程直接 _fn(v) (LLM 生成的 transform 可能死循环, 会把主线程钉死).
    改为按**全量 distinct 原值** (不设 probe 的 300 上限 —— 否则 distinct>300 时会撞到
    probe 没跑过的值而卡死) 在子进程里带超时实跑, 建 {原值: 结果} 映射, 再套用到每行.
    transform 契约是逐值纯函数, 映射与逐行调用等价, 且天然去重、更快.
    """
    rules = _sanitize_normalize_rules(plan, schema_cols, transforms)
    if not rules:
        return rows, {}

    stats: dict[str, dict] = {col: {"applied": 0, "skipped_null": 0, "failed": 0, "coerced": 0, "errors": [], "samples": []} for col in rules}
    new_rows = [dict(r) for r in rows]
    for col, rule in rules.items():
        st = stats[col]
        # 全量 distinct 非空原值 (limit 设为行数+1, 等效不截断).
        distinct = _distinct_nonnull(new_rows, col, limit=len(new_rows) + 1)
        results, fatal = _run_transform_on_distinct(rule["transform_py"], distinct)
        # 建 {原值: (kind, new_value, errmsg)}. kind ∈ {str,num,none} 为成功; bad/err 为失败.
        vmap: dict[str, tuple[str, str, str]] = {}
        if fatal is not None:  # 整列编译失败 (sanitize 已校验, 一般到不了这)
            for v in distinct:
                vmap[v] = ("err", v, fatal)
        else:
            for v, item in zip(distinct, results):
                tag = item[0]
                if tag == "ok":
                    _, new_v, kind = item
                    vmap[v] = ("err", v, "返回类型非法, 需要 str") if kind == "bad" else (kind, new_v, "")
                elif tag == "err":
                    vmap[v] = ("err", v, item[1])
                elif tag == "timeout":
                    vmap[v] = ("err", v, f"执行超时 (>{_PROBE_TIMEOUT_SEC:.0f}s, 疑似死循环)")
                else:  # "skipped" — 整列超时后未执行到的值
                    vmap[v] = ("err", v, "整列超时后未执行 (同一 transform 含死循环)")
        for r in new_rows:
            v = r.get(col)
            if v is None or v == "" or v == "NULL":
                st["skipped_null"] += 1
                continue
            kind, new_v, errmsg = vmap.get(v, ("err", v, "未在 distinct 映射中 (内部错误)"))
            if kind == "str":
                r[col] = new_v
                st["applied"] += 1
            elif kind == "num":
                # planner 常按 task_meaning "浮点数/整数" 把 transform 写成 return float()/int(),
                # 违反 transform(s)->str 硬契约. 子进程已 str() 兜底, 这里计 applied+coerced,
                # 不判失败 —— 否则带单位/千分位的脏值会原样进库, 下游 TRY_CAST 直接 NULL.
                r[col] = new_v
                st["applied"] += 1
                st["coerced"] += 1
            elif kind == "none":
                # transform 走了无 return 的兜底分支 (输入不符 doc_format). 回退原值, 不算失败.
                r[col] = v
                st["coerced"] += 1
            else:  # "bad" / "err" / 超时 / 内部错误
                st["failed"] += 1
                msg = f"{v!r} -> {errmsg}"
                errs = st["errors"]
                if msg not in errs and len(errs) < 200:
                    errs.append(msg)
                continue
            # 收成功改写的样例 (原值 != 新值), 去重, 上限 20, 供日志确认 transform 真实效果.
            samples = st["samples"]
            if r[col] != v and len(samples) < 20:
                pair = f"{v!r} -> {r[col]!r}"
                if pair not in samples:
                    samples.append(pair)
    return new_rows, stats


# ───────────────────── public entry ─────────────────────


@dataclass
class StructResult:
    """单 doc 结构化的轻量结果, 供包装层落库 / 统计 / 渲染 hint."""

    table_name: str
    primary_key: list[str]
    columns: list[str]
    rows: list[dict[str, str]] = field(default_factory=list)
    n_universe: int = 0
    n_sections: int = 0
    n_failed_sections: int = 0
    global_notes: str = ""
    # planner 产出的完整 plan (含 sections / record_universe / schema / column_normalize ...).
    # 留作审计: 评测时定位 schema 错位 / transform_py 写歪 / worker 抽错列等问题的源头.
    # 落盘到 logs/<task>/doc_extract/<stem>.result.json, 不进数据库.
    plan: dict[str, Any] = field(default_factory=dict)
    # column_normalize 后处理逐列统计: {col: {applied, skipped_null, failed, errors}}.
    normalize_stats: dict[str, dict] = field(default_factory=dict)
    # 逐 record 来源行号: {record_id: {section_id(str): src_line}}. worker 输出的 __src_line
    # 列被 assemble_wide 剥进这里, 不进 columns/rows/DB. 供后处理把可疑 cell 映射回 doc 原文.
    provenance: dict[str, dict[str, str]] = field(default_factory=dict)
    # planner / section worker 的逐次 LLM 原始返回 + 校验结果 (含失败重试前的版本).
    # 体积大, 不进 result.json 缓存, 由 doc_prepare 单独落 <stem>.fanout_allmsg.json.
    trace: list[dict] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "primary_key": self.primary_key,
            "columns": self.columns,
            "rows": self.rows,
            "stats": {
                "n_universe": self.n_universe,
                "n_sections": self.n_sections,
                "n_failed_sections": self.n_failed_sections,
            },
            "global_notes": self.global_notes,
            "plan": self.plan,
            "normalize_stats": self.normalize_stats,
            "provenance": self.provenance,
        }


async def structurize_doc(
    *,
    doc_path: Path,
    knowledge_md_path: Optional[Path] = None,
    concurrency: int = 6,
    table_name: Optional[str] = None,
) -> StructResult:
    """完整 fan-out 结构化: planner → 并发 section workers → 宽表.

    模型经 contextvar 注入 (调用方先在本 task 协程内 set_models()): planner / section worker
    用 nothink, Phase 5.5 transform 生成用 think. 不再逐层透传 model 参数, 且并发 task 间隔离.

    参数:
        doc_path: 目标 .md 文件.
        knowledge_md_path: task 的 knowledge.md; 若提供, 会按 doc stem grep 出
            相关片段作为 planner 的命名/口径锚点 (worker 仍只看 plan + doc).
        concurrency: section worker 并发上限.
        table_name: 输出表名, 默认取 doc stem.
    """
    doc_path = Path(doc_path)
    if not doc_path.is_file():
        raise FileNotFoundError(f"doc 文件不存在: {doc_path}")
    tbl = table_name or doc_path.stem

    raw_doc_text = doc_path.read_text(encoding="utf-8")
    # 喂给 planner / worker 的是带行号版本 (0001\t正文): 让 LLM 能锚定 section 区间与
    # 逐 record 来源行, 后处理也能按行号把可疑 cell 映射回 doc 原文. 原文与原始长度另存.
    doc_text = number_doc_lines(raw_doc_text)
    logger.info("[fanout] doc={} len={} chars (numbered)", doc_path.name, len(raw_doc_text))

    knowledge_excerpt = grep_knowledge_excerpt(knowledge_md_path, doc_path.stem)
    if knowledge_excerpt:
        logger.info(
            "[fanout] doc={} knowledge excerpt: {} chars",
            doc_path.name, len(knowledge_excerpt),
        )
    knowledge_global = read_knowledge_global(knowledge_md_path)
    if knowledge_global:
        logger.info(
            "[fanout] doc={} knowledge global head: {} chars",
            doc_path.name, len(knowledge_global),
        )

    # 中间步骤轨迹: planner / 每个 section worker 的原始 LLM 返回 + 校验失败原因 +
    # attempt 序号. 失败重试前的"坏"版本也收进来, 供 doc_prepare 落 fanout_allmsg.json.
    trace: list[dict] = []

    # Phase 1+2: planner
    plan = await run_planner(
        doc_text,
        doc_name=doc_path.name,
        knowledge_excerpt=knowledge_excerpt,
        knowledge_global=knowledge_global,
        trace=trace,
    )
    universe = {str(x) for x in plan["record_universe"]}
    logger.info(
        "[fanout] plan locked: |universe|={} sections={}",
        len(universe), len(plan["sections"]),
    )

    # Phase 3+4: fan-out workers
    sem = asyncio.Semaphore(concurrency)

    async def bounded(s):
        async with sem:
            return await run_section_worker(doc_text, plan, s, universe, trace)

    results = await asyncio.gather(*[bounded(s) for s in plan["sections"]])
    n_failed = sum(1 for *_x, fails in results if fails)

    # Phase 5: assemble wide
    cols, rows, provenance = assemble_wide(plan, results)
    logger.info(
        "[fanout] doc={} wide: {} cols × {} rows ({} section failed), provenance for {} records",
        doc_path.name, len(cols), len(rows), n_failed, len(provenance),
    )

    # Phase 5.5: 针对每个 column_normalize 列的全量 distinct 原值生成 transform_py
    # (planner 只声明 target_format, 转换逻辑在此处看到真实全量数据后生成, 避免覆盖不全).
    # 写 transform 是推理任务 → generate_transforms 内部用 contextvar 注入的 think 模型.
    transforms = await generate_transforms(plan, rows, cols, trace=trace)
    if transforms:
        logger.info("[fanout] doc={} generated transforms for cols={}", doc_path.name, list(transforms))

    # Phase 6: column_normalize 后处理 (worker 不可见, planner-only 规则)
    rows, norm_stats = apply_column_normalize(plan, rows, cols, transforms)
    if norm_stats:
        for col, st in norm_stats.items():
            logger.info(
                "[fanout] doc={} normalize col={}: applied={} skipped_null={} failed={}{}{}{}",
                doc_path.name, col, st["applied"], st["skipped_null"], st["failed"],
                f" coerced={st['coerced']}" if st.get("coerced") else "",
                f" samples={st['samples']}" if st.get("samples") else "",
                f" errors={st['errors']}" if st["errors"] else "",
            )

    return StructResult(
        table_name=tbl,
        primary_key=["record_id"],
        columns=cols,
        rows=rows,
        n_universe=len(universe),
        n_sections=len(plan["sections"]),
        n_failed_sections=n_failed,
        global_notes=str(plan.get("global_notes", "") or ""),
        plan=plan,
        normalize_stats=norm_stats,
        provenance=provenance,
        trace=trace,
    )
