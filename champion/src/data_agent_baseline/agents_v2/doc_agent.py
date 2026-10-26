"""非结构化 doc → 结构化记录的子 agent (doc_agent).

针对 ``context/doc/*.md`` 这类**散文化包含若干实体记录**的非结构化 doc.
与 ``split_subagent.py`` 的"维度段切分"不同, 本 agent 假设同一 uid 的
不同属性可能分散在文档的不同**属性段** (例如前 100 条数据行描述了
3 种属性, 后 100 条描述另外 2 种), 并不一定每段每个 uid 都出现.

流程
----
1. **read_doc_head + extract_uid_agent (Qwen #1)**: 复用与 split_subagent
   相同的"从 doc 开头几行抽 uid 字段名/正则/样例"步骤.

2. **grep_uids + filter_relevant_lines_agent (Qwen #2)**: 把每个样例 uid
   在全文 word-level 精确匹配, 然后让 LLM 判断每条命中行**是否真的
   描述当前 uid 这个实体**, 保留通过的 ``(line_no, content)``,
   并给每个 uid 编一个从 0 起的序号 (``uid_index``), 便于后续聚合时
   跨 uid 对齐.

3. **extract_attributes_agent (Qwen #3)**: 把所有 uid 通过过滤后的关联行
   一次性喂给 LLM, 让它**逐 uid** 列出该实体出现在 doc 中的所有
   可识别属性键值对 ``(attribute, value, line_no)``. line_no 必须落在
   该 uid 的关联行集合内, 用于追溯证据.

4. **locate_segments_agent (Qwen #4)**: 同一份 doc 通常按"属性段"
   划分行范围 — 每段是一张"虚拟表", 共享一组属性列, 但**列集合在
   段间不同** (例如段A: lines 1-216 = {codename, full_name};
   段B: lines 217-645 = {skin, hair, eye}; 段C: lines 646-862 =
   {gender, race}; 段D: lines 863-1087 = {publisher, alignment}).
   该步骤聚合第 3 步的 ``(attribute, line_no)`` 散点, 让 LLM:
   a) 切出 doc 的属性段 (start_line, end_line) 及每段内属于的属性列;
   b) 标记哪些段包含 question 真正需要的列 (``needed=True``).
   只有 needed 段会进入第 5 步, 不相关段直接丢弃.

5. **structurize_segment_agent (Qwen #5)**: **每段调用一次** LLM
   (而不是每列调一次), 截取段行范围内的原文 (含行号), 让模型一次性
   按行抽出 ``(uid, {col1: v1, col2: v2, ...})`` 行结构. 多段结果按
   uid pivot 成一张完整的结构化表, 写进 ``DocAgentResult.records``.

入口: ``run_doc_agent(doc_path, task_question, knowledge_md, model, ...)``,
返回 ``DocAgentResult``.

注意: 本子 agent 不直接落 CSV, 由下游 solver 决定如何与其他数据源
join 后再写 ``prediction.csv``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional

import json_repair
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents_v2.agent_util import save_history
from agents_v2.split_subagent import (
    UidExtraction,
    _parse_uid_extraction,
    _run_streaming,
    build_extract_uid_agent,
    grep_uids,
    read_doc_head,
)
from configs import pydantic_predefined_model


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class UidRelevantLines(BaseModel):
    """单个 uid 经 LLM 过滤后的真正关联行 (Qwen #2 输出子项)."""

    uid: str = Field(description="uid 字面量, 与输入对应")
    uid_index: int = Field(
        description="该 uid 在样例集合中的序号, 从 0 起"
    )
    relevant_line_nos: list[int] = Field(
        description="经过相关性判断后保留下来的行号 (1-based), 按行号升序"
    )


class FilterRelevantOutput(BaseModel):
    """Qwen #2 输出: 全部样例 uid 的过滤结果."""

    uid_lines: list[UidRelevantLines]


class AttributeKV(BaseModel):
    """单个属性键值对."""

    attribute: str = Field(
        description="原始属性键 (描述性英文短语, 例如 'date_of_birth' / 'occupation')"
    )
    value: str = Field(
        description="从原文中抽取的属性值, 字符串形式. 数字也用字符串保留, 由下游统一转型."
    )
    line_no: int = Field(
        description="该属性值出现的行号 (必须落在该 uid 的 relevant_line_nos 内)"
    )


class UidAttributeBundle(BaseModel):
    """单个 uid 经 Qwen #3 抽取出的所有属性键值对."""

    uid: str
    uid_index: int
    attributes: list[AttributeKV] = Field(default_factory=list)


class ExtractAttributesOutput(BaseModel):
    """Qwen #3 输出: 全部 uid 的属性抽取结果."""

    uid_attributes: list[UidAttributeBundle]


class SegmentColumn(BaseModel):
    """属性段内的一个目标列."""

    column_name: str = Field(
        description=(
            "结构化输出列名, 蛇形英文 (符合 SQL). 若该列有明确单位/格式,"
            " 列名末尾应带单位/格式后缀, 例如 'weight_kg' / 'height_cm'"
            " / 'birthday_iso' / 'temperature_c' / 'duration_min'."
            " 无单位/格式约束的标识列直接用原名 (例如 'publisher_id')."
        )
    )
    source_attributes: list[str] = Field(
        description="对应到 Qwen #3 中哪些原始属性键 (≥1, 支持同义合并)."
    )
    unit_hint: str = Field(
        default="",
        description=(
            "对该列单位/格式的自然语言说明, 喂给 step 5 做归一化."
            " 例如 'kilograms (numeric, 2 decimals)' / 'ISO date YYYY-MM-DD'"
            " / 'no conversion, keep raw string'. 无单位约束时填 ''."
        ),
    )


class AttributeSegment(BaseModel):
    """单个"属性段": doc 中的一段连续行 + 该段内的列集合.

    一个段就是一张"虚拟表" — 共享行范围, 列集合在段间不同.
    例如 superhero.md:
        - 段1: [1, 220] = {codename, full_name}
        - 段2: [221, 648] = {skin_color, hair_color, eye_color}
        - 段3: [649, 866] = {gender, race}
        - 段4: [867, 1087] = {publisher, alignment}

    **行号 (start_line, end_line) 由算法确定性推断**, LLM 不参与.
    LLM 只负责: ① 给段内列命名; ② 标记该段是否含 question 需要的列.
    """

    segment_index: int = Field(description="段序号, 从 1 起")
    start_line: int = Field(description="该段起始行号 (1-based, 含). 算法推断, 非 LLM 输出.")
    end_line: int = Field(description="该段结束行号 (1-based, 含). 算法推断, 非 LLM 输出.")
    columns: list[SegmentColumn] = Field(
        description="该段内的属性列集合 (≥1). 列在段间不重叠."
    )
    needed: bool = Field(
        description="该段是否包含 question 真正需要的列. 仅 needed=True 的段会进入第 5 步."
    )


class LocateSegmentsOutput(BaseModel):
    """Qwen #4 输出: doc 的属性段切分及每段是否需要."""

    segments: list[AttributeSegment]


class _LLMSegmentLabel(BaseModel):
    """LLM 仅给段命名列 + 标 needed (行号由算法注入, LLM 不输出)."""

    segment_index: int
    columns: list[SegmentColumn]
    needed: bool


class _LLMLocateOutput(BaseModel):
    """LLM step 4 的纯输出 (不含行号)."""

    segments: list[_LLMSegmentLabel]


class StructuredRecord(BaseModel):
    """最终输出的一行结构化记录 (uid + 选定列的值)."""

    uid: str
    values: dict[str, str] = Field(
        default_factory=dict,
        description="键为 AttributeSegment.column_name, 值为字符串. 缺失时取空字符串."
    )


class SegmentExtractionItem(BaseModel):
    """Qwen #5 在单个属性段内抽出的一行: uid + 该段所有列的值."""

    uid: str
    values: dict[str, str] = Field(
        description="键为 column_name, 值为该 uid 在该段对应列的字符串值. 缺列时填空字符串."
    )


class SegmentExtractionOutput(BaseModel):
    """Qwen #5 单段输出: 该段内全部 (uid, values) 数据行."""

    segment_index: int
    column_names: list[str]
    items: list[SegmentExtractionItem]


class DocAgentResult(BaseModel):
    """run_doc_agent 的最终返回值."""

    doc_path: str
    uid: UidExtraction
    grep_hits: dict[str, list[tuple[int, str]]] = Field(default_factory=dict)
    filter_output: Optional[FilterRelevantOutput] = None
    extract_output: Optional[ExtractAttributesOutput] = None
    locate_output: Optional[LocateSegmentsOutput] = None
    segment_extractions: list[SegmentExtractionOutput] = Field(default_factory=list)
    records: list[StructuredRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _format_grep_hits_block(
    grep_hits: dict[str, list[tuple[int, str]]],
    uid_index_map: dict[str, int],
) -> str:
    """格式化 grep 命中, 每个 uid 一块, 含 uid_index 标注."""
    blocks: list[str] = []
    for uid, hits in grep_hits.items():
        idx = uid_index_map.get(uid, -1)
        if not hits:
            blocks.append(f"### uid_index={idx}  uid={uid!r}\n(无命中)")
            continue
        body = "\n".join(f"  {ln:>5}: {content}" for ln, content in hits)
        blocks.append(
            f"### uid_index={idx}  uid={uid!r}  (共 {len(hits)} 次命中)\n{body}"
        )
    return "\n\n".join(blocks)


def _gather_relevant_lines(
    filter_output: FilterRelevantOutput,
    doc_lines: list[str],
) -> dict[str, dict[int, str]]:
    """按 uid 把过滤后保留的行 (line_no -> content) 整理出来."""
    out: dict[str, dict[int, str]] = {}
    for entry in filter_output.uid_lines:
        keep: dict[int, str] = {}
        for ln in sorted(set(entry.relevant_line_nos)):
            if 1 <= ln <= len(doc_lines):
                keep[ln] = doc_lines[ln - 1]
        out[entry.uid] = keep
    return out


def _format_relevant_for_extract(
    filter_output: FilterRelevantOutput,
    relevant_lines: dict[str, dict[int, str]],
) -> str:
    """把每个 uid 的关联原文块格式化, 喂给 Qwen #3."""
    blocks: list[str] = []
    for entry in filter_output.uid_lines:
        lines = relevant_lines.get(entry.uid, {})
        if not lines:
            blocks.append(
                f"### uid_index={entry.uid_index}  uid={entry.uid!r}\n(无关联行)"
            )
            continue
        body = "\n".join(f"  {ln:>5}: {txt}" for ln, txt in lines.items())
        blocks.append(
            f"### uid_index={entry.uid_index}  uid={entry.uid!r}\n{body}"
        )
    return "\n\n".join(blocks)


def _format_attr_scatter_for_locate(
    extract_output: ExtractAttributesOutput,
) -> str:
    """把 (attribute, line_no) 散点按 attribute 聚合, 让 LLM 直观看出每个属性的行段分布.

    输出形如:
        - sex: lines [7, 9, 12, ...]  (sample value: 'male')
        - date_of_birth: lines [7, 9, 12, ...]
        - occupation: lines [185, 198, ...]
    """
    by_attr: dict[str, list[int]] = {}
    samples: dict[str, str] = {}
    for bundle in extract_output.uid_attributes:
        for kv in bundle.attributes:
            by_attr.setdefault(kv.attribute, []).append(kv.line_no)
            samples.setdefault(kv.attribute, kv.value)
    lines: list[str] = []
    for attr, lns in by_attr.items():
        lns_sorted = sorted(lns)
        if len(lns_sorted) > 12:
            head = ", ".join(map(str, lns_sorted[:6]))
            tail = ", ".join(map(str, lns_sorted[-6:]))
            ln_str = f"[{head}, ..., {tail}] (共 {len(lns_sorted)})"
        else:
            ln_str = f"[{', '.join(map(str, lns_sorted))}]"
        lines.append(
            f"- {attr}: lines {ln_str}  (sample value: {samples[attr]!r})"
        )
    return "\n".join(lines)


def _infer_segment_boundaries(
    filter_output: FilterRelevantOutput,
    total_lines: int,
) -> list[tuple[int, int]]:
    """**算法确定性**地推断属性段的行范围, 不让 LLM 猜.

    依据: 在"维度段/属性段"模式下, 每个 uid 在每段恰好出现一次.
    设各 uid 命中数的众数为 K, 则有 K 个段; 第 i 段的 start_line 是
    "所有 uid 的第 i 次命中行号" 的最小值.

    Args:
        filter_output: Qwen #2 给出的 per-uid 真正关联行号.
        total_lines: doc 总行数, 用于段尾兜底.

    Returns:
        长度为 K 的 [(start_line, end_line), ...] 列表; 段间不重叠,
        覆盖 [1, total_lines]. 若推断失败 (例如 uid 命中太少), 返回
        单段 [(1, total_lines)] 兜底, 由下游 LLM 自己处理.
    """
    if not filter_output.uid_lines:
        return [(1, total_lines)]

    sorted_lns_per_uid: list[list[int]] = []
    for entry in filter_output.uid_lines:
        lns = sorted(set(entry.relevant_line_nos))
        if lns:
            sorted_lns_per_uid.append(lns)
    if not sorted_lns_per_uid:
        return [(1, total_lines)]

    # 取命中数的众数作为段数 K (排除离群).
    counts = [len(lns) for lns in sorted_lns_per_uid]
    k_seg, _ = Counter(counts).most_common(1)[0]
    if k_seg <= 0:
        return [(1, total_lines)]

    # 第 i 段 start = min over uids of lns[i] (仅取命中数 >= K 的 uid).
    starts: list[int] = []
    for i in range(k_seg):
        cands = [lns[i] for lns in sorted_lns_per_uid if len(lns) > i]
        if not cands:
            break
        starts.append(min(cands))
    if not starts:
        return [(1, total_lines)]
    # 第 1 段始终从 line 1 开始 (前面可能有标题/介绍).
    starts[0] = 1
    # 单调修正 (保险): 防止 starts 不严格递增.
    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            starts[i] = starts[i - 1] + 1

    # end_line = 下一段 start - 1; 最后一段到 total_lines.
    ranges: list[tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = (starts[i + 1] - 1) if i + 1 < len(starts) else total_lines
        ranges.append((s, e))
    return ranges


def _format_attr_scatter_per_segment(
    extract_output: ExtractAttributesOutput,
    seg_ranges: list[tuple[int, int]],
) -> str:
    """把第 3 步的属性散点**按段分组**展示, 让 LLM 直接看到每段内有哪些属性.

    一行形如:
        - segment_index=2  range=[217, 648]  total_attrs=3
            * skin_color: lines [221, 223, ...] (sample: '1')
            * hair_color: lines [221, 223, ...] (sample: '1')
            * eye_color:  lines [221, 223, ...] (sample: '7')
    """
    # attribute -> [line_no, ...]
    by_attr: dict[str, list[int]] = {}
    samples: dict[str, str] = {}
    for bundle in extract_output.uid_attributes:
        for kv in bundle.attributes:
            by_attr.setdefault(kv.attribute, []).append(kv.line_no)
            samples.setdefault(kv.attribute, kv.value)

    blocks: list[str] = []
    for i, (s, e) in enumerate(seg_ranges, start=1):
        # 找出 line_no 大多落在 [s, e] 的属性 (≥50% 在该段内).
        seg_attrs: list[tuple[str, list[int]]] = []
        for attr, lns in by_attr.items():
            in_seg = [ln for ln in lns if s <= ln <= e]
            if not in_seg:
                continue
            if len(in_seg) / max(1, len(lns)) >= 0.5:
                seg_attrs.append((attr, sorted(in_seg)))
        header = (
            f"- segment_index={i}  range=[{s}, {e}]  "
            f"total_attrs={len(seg_attrs)}"
        )
        if not seg_attrs:
            blocks.append(header + "\n    (无属性命中, 可能是空白段或未抽到)")
            continue
        body_lines = []
        for attr, lns in seg_attrs:
            if len(lns) > 8:
                lns_str = (
                    f"[{', '.join(map(str, lns[:4]))}, ..., "
                    f"{', '.join(map(str, lns[-4:]))}] (共 {len(lns)})"
                )
            else:
                lns_str = f"[{', '.join(map(str, lns))}]"
            body_lines.append(
                f"    * {attr}: lines {lns_str}  (sample: {samples[attr]!r})"
            )
        blocks.append(header + "\n" + "\n".join(body_lines))
    return "\n".join(blocks)


def _slice_doc_window(
    doc_lines: list[str], start_line: int, end_line: int, max_lines: int = 600
) -> str:
    """截取 doc 的 [start_line, end_line] 区间, 保留行号; 超长时截断头尾各 max_lines/2 行."""
    s = max(1, start_line)
    e = min(len(doc_lines), end_line)
    if s > e:
        return ""
    span = e - s + 1
    if span <= max_lines:
        return "\n".join(f"  {ln:>5}: {doc_lines[ln - 1]}" for ln in range(s, e + 1))
    half = max_lines // 2
    head = [f"  {ln:>5}: {doc_lines[ln - 1]}" for ln in range(s, s + half)]
    mid = [f"        ... 省略 {span - max_lines} 行 ..."]
    tail = [f"  {ln:>5}: {doc_lines[ln - 1]}" for ln in range(e - half + 1, e + 1)]
    return "\n".join(head + mid + tail)


# ---------------------------------------------------------------------------
# Qwen agents
# ---------------------------------------------------------------------------


FILTER_RELEVANT_INSTRUCTION = """
你将看到若干样例 uid 在一份散文化 doc 中的 word-level grep 命中
(每条形如 `<line_no>: <content>`).

你的任务是 **逐 uid** 判断每条命中行是否**真的在描述该 uid 这个实体**
(主语 / 直接定语), 还是只是叙述里碰巧出现了同样的 token (年份、楼号、
百分比、计数等).

判别启发:
- "Patient 43003 is a male born ..." → 行真正在描述 uid=43003.
- "...43003 patients were admitted in 2024..." → 行只是修饰语义, 应剔除.
- 数字型 uid (例如 7, 26) 误命中风险更高, 要特别保守.

输出要求:
- 对每个 uid 给出 `uid_index` (按输入中标注的 0-based 序号原样返回).
- `relevant_line_nos`: 保留下来的行号列表, 升序; 若全是误命中则给空列表.

**输出格式**: 严格 JSON, 不要其他说明:
```json
{
  "uid_lines": [
    {"uid": "43003", "uid_index": 0, "relevant_line_nos": [7, 12]},
    {"uid": "133382", "uid_index": 1, "relevant_line_nos": [9]}
  ]
}
```
"""


EXTRACT_ATTRIBUTES_INSTRUCTION = """
你将看到若干 uid 各自在 doc 中**经过相关性过滤后**的关联行 (line_no: 原文).

你的任务: **逐 uid** 把该实体能从这些行里读出的所有属性键值对都列出来.
属性键命名要语义化、英文蛇形 (例如 `date_of_birth`, `first_record_date`,
`sex`, `occupation`); 同一含义的属性, 不同 uid 之间请使用相同的键, 便于
后续对齐.

要求:
- 每个属性必须给出 `line_no`, 该行号必须**就是该 uid 的关联行之一**
  (输入里展示过的); 否则视为错误.
- 数值/日期一律以**原文字符串**保留, 不要自行格式化或推断.
- 若某行没有可抽属性 (纯铺垫描写), 直接跳过即可, 不要硬凑.
- 同一属性若被多次提及 (例如先记一个错误日期, 后被勘误), **优先取最终
  / 修正后的值**, 但请把对应行号填为给出该最终值的那一行.

**输出格式**: 严格 JSON:
```json
{
  "uid_attributes": [
    {
      "uid": "43003",
      "uid_index": 0,
      "attributes": [
        {"attribute": "sex", "value": "male", "line_no": 7},
        {"attribute": "date_of_birth", "value": "November 24th, 1937", "line_no": 7},
        {"attribute": "first_record_date", "value": "March 8th, 1994", "line_no": 7}
      ]
    }
  ]
}
```
"""


LOCATE_SEGMENTS_INSTRUCTION = """
你将看到:
1. 当前任务 question;
2. knowledge.md 全文 (描述目标 schema / 业务术语);
3. doc 的"属性段切分" — 行范围**已经由算法确定**, 不需要你猜;
4. 每段内已抽到的属性散点 (attribute → 该段内出现行号 + 样例值).

**核心背景: doc 的"属性段"模式**
一份散文化 doc 通常按"属性段"组织 — 每段是一张独立的"虚拟表",
共享同一组 uid, 但**列集合在段间不同**. 例如 superhero.md:
- 段1 [1, 220]:  列 = {codename, full_name}
- 段2 [221, 648]: 列 = {skin_color, hair_color, eye_color}
- 段3 [649, 866]: 列 = {gender, race}
- 段4 [867, 1087]: 列 = {publisher, alignment}

**你的任务 (只做两件事, 不要输出行号)**:
1. **为每段命名列**: 把段内的属性散点合并、命名成结构化列.
   - column_name: 蛇形英文 (符合 SQL), 优先与 knowledge.md 已定义字段名对齐.
   - **单位/格式后缀**: 若 knowledge.md 对该列指定了单位或格式
     (例如 weight 用 kg, height 用 cm, 日期用 ISO YYYY-MM-DD), **必须把
     单位/格式编码进列名后缀**, 例如 ``weight_kg`` / ``height_cm`` /
     ``birthday_iso`` / ``temperature_c`` / ``duration_min``. 这样下游 step 5
     看到列名就知道怎么归一化原文.
   - 若该列只是标识符 (例如 publisher_id / codename / full_name), 无单位
     约束, 直接用原名, 不加后缀.
   - source_attributes: 对应到第 3 步原始属性键 (≥1, 支持同义合并:
     dob/date_of_birth/birthday → birthday_iso).
   - **unit_hint**: 对该列单位/格式的自然语言说明, 给 step 5 做归一化用.
     - 例如 column_name=``weight_kg`` → unit_hint="kilograms, numeric string,
       up to 2 decimals; convert from lbs/g if needed".
     - 例如 column_name=``birthday_iso`` → unit_hint="ISO date YYYY-MM-DD".
     - 例如 column_name=``publisher_id`` → unit_hint="" (无单位约束).
   - 同段内多个原始属性若语义重复就合并, 否则各成一列.
2. **标记 needed**: 该段是否含 question 真正需要的列?
   只要段里有任何一列被 question 需要, needed 就标 True; 否则 False.

**注意**:
- 输入已给出每段的 segment_index 和行范围, 你**只输出 segment_index +
  columns + needed**, 不要再输出 start_line / end_line.
- 段数与输入完全一致, 不要合并/拆分段, 不要漏段.
- 只标 1-2 个真正相关的段为 needed=True; 其余即使语义沾边但不直接用也标 False.
- **务必通读 knowledge.md** 找到目标 schema 中每个字段的单位/格式约定,
  把它们正确编码进 column_name 和 unit_hint, 这是 step 5 归一化的唯一依据.

**输出格式**: 严格 JSON:
```json
{
  "segments": [
    {
      "segment_index": 2,
      "columns": [
        {
          "column_name": "height_cm",
          "source_attributes": ["height"],
          "unit_hint": "centimeters, numeric string with up to 2 decimals; convert from feet/inches if needed"
        },
        {
          "column_name": "weight_kg",
          "source_attributes": ["weight"],
          "unit_hint": "kilograms, numeric string with up to 2 decimals; convert from lbs if needed"
        }
      ],
      "needed": true
    },
    {
      "segment_index": 4,
      "columns": [
        {
          "column_name": "publisher_id",
          "source_attributes": ["publisher"],
          "unit_hint": ""
        },
        {
          "column_name": "alignment_id",
          "source_attributes": ["alignment", "moral_alignment"],
          "unit_hint": ""
        }
      ],
      "needed": true
    }
  ]
}
```
"""


STRUCTURIZE_SEGMENT_INSTRUCTION = """
你将看到 doc 的某一段连续原文 (含行号) — 该段是一张"虚拟表",
共享一组**目标列** ``column_names``, 段内大多数数据行形如
"...<uid>... <col1 value> ... <col2 value> ...".

你的任务: 从该段中**逐行扫描**出所有 (uid, {col1: v1, col2: v2, ...}) 行.

要求:
- `uid` 必须是 doc 中作为实体主键出现的字符串字面量 (与输入主键信息一致).
- `values` 是一个 dict, 键必须**严格等于**输入的 column_names, 一个不漏一个不增.
- **单位与格式归一化**: 输出值的单位/格式必须与 ``column_name`` 中标注的
  单位一致 (单位由第 4 步根据 knowledge.md 编码进列名).
  - 列名形如 ``weight_kg`` → 输出公斤数值字符串 (原文 "150 lbs" → "68.04").
  - 列名形如 ``height_cm`` → 输出厘米数值字符串 (原文 "5'9\\"" → "175.26").
  - 列名形如 ``birthday_iso`` → 输出 "YYYY-MM-DD" (原文"November 24th, 1937" → "1937-11-24").
  - 列名无单位/格式后缀 (例如 ``publisher_id`` / ``codename``) → 用**原文字符串**保留, 不做任何转换.
  - 输入会附带 ``## column_units`` 段, 给出每列期望单位/格式的明确说明, 务必严格按其执行.
- 数值统一不带单位符号 (输出 "175.26", 不要 "175.26 cm");原文无具体数字时填 "".
- 若某列在该 uid 对应行里未明确给出, 该列填空字符串 "".
- 若同一 uid 存在勘误信息 (例如先记一个错误值, 后被勘误), 取**最后一次/最终修正**的取值, 再做单位换算.
- 不要漏掉只出现一次的 uid; 不要把叙述里碰巧的同名 token 误当 uid.
- 段首/段尾的过渡段落 (无 uid) 直接跳过.

**输出格式**: 严格 JSON, segment_index / column_names 原样回填:
```json
{
  "segment_index": 2,
  "column_names": ["height_cm", "weight_kg"],
  "items": [
    {"uid": "7",  "values": {"height_cm": "188.0", "weight_kg": "95.25"}},
    {"uid": "26", "values": {"height_cm": "175.26", "weight_kg": "68.04"}}
  ]
}
```
"""


def build_filter_relevant_agent(model: Any) -> Agent:
    return Agent(model=model, instructions=FILTER_RELEVANT_INSTRUCTION, output_type=str)


def build_extract_attributes_agent(model: Any) -> Agent:
    return Agent(model=model, instructions=EXTRACT_ATTRIBUTES_INSTRUCTION, output_type=str)


def build_locate_segments_agent(model: Any) -> Agent:
    return Agent(model=model, instructions=LOCATE_SEGMENTS_INSTRUCTION, output_type=str)


def build_structurize_segment_agent(model: Any) -> Agent:
    return Agent(model=model, instructions=STRUCTURIZE_SEGMENT_INSTRUCTION, output_type=str)


def _parse_filter_output(text: str) -> FilterRelevantOutput:
    return FilterRelevantOutput(**json_repair.loads(text))


def _parse_extract_output(text: str) -> ExtractAttributesOutput:
    return ExtractAttributesOutput(**json_repair.loads(text))


def _parse_locate_output(text: str) -> _LLMLocateOutput:
    return _LLMLocateOutput(**json_repair.loads(text))


def _parse_segment_extraction(text: str) -> SegmentExtractionOutput:
    return SegmentExtractionOutput(**json_repair.loads(text))


# ---------------------------------------------------------------------------
# 第 5 步: 在每个属性段内提取 (uid, value), 然后 pivot 成 records
# ---------------------------------------------------------------------------


async def _structurize_all_segments(
    *,
    locate_output: LocateSegmentsOutput,
    doc_lines: list[str],
    uid_info: UidExtraction,
    model: Any,
    log_dir: Optional[Path],
    max_lines_per_segment: int,
) -> list[SegmentExtractionOutput]:
    """对每个 needed=True 段调一次 Qwen #5, 一次抽出该段所有列的 (uid, values) 行.

    不相关段 (needed=False) 直接跳过, 不浪费 token.
    """
    structurize_agent = build_structurize_segment_agent(model)
    out: list[SegmentExtractionOutput] = []
    for seg in locate_output.segments:
        if not seg.needed:
            logger.info(
                f"[doc-agent] skip segment {seg.segment_index} "
                f"({seg.start_line}-{seg.end_line}): needed=False"
            )
            continue
        if not seg.columns:
            logger.warning(
                f"[doc-agent] segment {seg.segment_index} has no columns, skip"
            )
            continue
        window = _slice_doc_window(
            doc_lines, seg.start_line, seg.end_line, max_lines=max_lines_per_segment
        )
        if not window:
            logger.warning(
                f"[doc-agent] segment {seg.segment_index} has empty window "
                f"({seg.start_line}-{seg.end_line}), skip"
            )
            continue
        column_names = [c.column_name for c in seg.columns]
        # 把每列附上 source_attributes 和单位说明, 帮助 LLM 在段内定位 + 归一化.
        col_hint = "\n".join(
            f"  - {c.column_name}  (源属性: {c.source_attributes})"
            for c in seg.columns
        )
        unit_block = "\n".join(
            f"  - {c.column_name}: "
            + (c.unit_hint if c.unit_hint else "no unit/format constraint, keep raw string")
            for c in seg.columns
        )
        prompt = (
            f"## 主键信息\n"
            f"- field_name: {uid_info.uid_field_name}\n"
            f"- regex: {uid_info.uid_regex}\n"
            f"- examples: {uid_info.uid_examples}\n\n"
            f"## segment_index = {seg.segment_index}\n"
            f"## column_names (必须严格按此列出, 不增不减): {column_names}\n"
            f"## 各列源属性提示:\n{col_hint}\n\n"
            f"## column_units (每列期望单位/格式, 务必严格归一化):\n{unit_block}\n\n"
            f"## doc 段原文 (line_no: content), 行范围 {seg.start_line}-{seg.end_line}\n"
            f"{window}\n"
        )
        try:
            res = await _run_streaming(
                structurize_agent,
                prompt,
                label=f"doc-structurize-seg{seg.segment_index}",
            )
        except Exception as e:
            logger.warning(
                f"[doc-agent] structurize seg{seg.segment_index} failed: {e}"
            )
            continue
        try:
            parsed = _parse_segment_extraction(res[0])
        except Exception as e:
            logger.warning(
                f"[doc-agent] parse structurize seg{seg.segment_index} failed: {e}"
            )
            continue
        # 强制 segment_index / column_names 与 locate_output 对齐, 防 LLM 自作主张.
        parsed.segment_index = seg.segment_index
        parsed.column_names = column_names
        # 同时清洗 items 里的 values: 只保留指定列, 缺则补空.
        for item in parsed.items:
            item.values = {col: item.values.get(col, "") for col in column_names}
        out.append(parsed)
        if log_dir is not None:
            try:
                save_history(
                    res[1],
                    Path(log_dir) / f"doc_structurize_seg{seg.segment_index}_allmsg.json",
                )
            except Exception as e:
                logger.warning(
                    f"[doc-agent] save structurize seg{seg.segment_index} history failed: {e}"
                )
    return out


def _pivot_records(
    locate_output: LocateSegmentsOutput,
    extractions: list[SegmentExtractionOutput],
) -> list[StructuredRecord]:
    """按 uid pivot: 跨段聚合到 {uid: {column_name: value}}.

    全列名 = 所有 needed=True 段的列并集 (按段顺序拼接), 缺列填空字符串.
    """
    all_columns: list[str] = []
    seen: set[str] = set()
    for seg in locate_output.segments:
        if not seg.needed:
            continue
        for c in seg.columns:
            if c.column_name not in seen:
                all_columns.append(c.column_name)
                seen.add(c.column_name)

    by_uid: dict[str, dict[str, str]] = {}
    for ext in extractions:
        for item in ext.items:
            slot = by_uid.setdefault(item.uid, {col: "" for col in all_columns})
            for col, val in item.values.items():
                if col in slot and val != "":
                    slot[col] = val
    return [
        StructuredRecord(
            uid=uid, values={col: by_uid[uid].get(col, "") for col in all_columns}
        )
        for uid in by_uid
    ]


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run_doc_agent(
    doc_path: Path,
    task_question: str,
    knowledge_md: str,
    model: Any,
    *,
    head_lines: int = 15,
    max_uid_examples: int = 5,
    max_lines_per_segment: int = 600,
    log_dir: Optional[Path] = None,
) -> DocAgentResult:
    """跑完整的 doc-agent 流程, 把非结构化 doc 变成结构化记录列表.

    Args:
        doc_path: 要分析的 doc 路径 (.md / .txt).
        task_question: 当前任务的 question 字段; 用于第 4 步选属性段时聚焦.
        knowledge_md: knowledge.md 全文; 用于第 4 步对齐目标 schema.
        model: pydantic_ai 模型 (建议 Qwen3.5-35B-A3B nothink 版).
        head_lines: 喂给 Qwen #1 抽 uid 的开头行数.
        max_uid_examples: 取前 N 个样例 uid 跑前 4 步.
        max_lines_per_segment: 第 5 步每个属性段最多展示的行数, 防 prompt 爆炸.
        log_dir: 若提供, 把每个 agent 的对话历史落到该目录.

    Returns:
        DocAgentResult: 含每一步中间产物及最终 records.
    """
    doc_path = Path(doc_path)
    doc_lines = doc_path.read_text(encoding="utf-8").splitlines()

    # 1. 抽 uid (复用 split_subagent 的实现)
    head_text = read_doc_head(doc_path, n_lines=head_lines)
    extract_agent = build_extract_uid_agent(model)
    extract_input = (
        f"## doc 开头 {head_lines} 行 (含行号)\n"
        + "\n".join(f"{i + 1:>4}: {l}" for i, l in enumerate(head_text.splitlines()))
    )
    extract_res = await _run_streaming(extract_agent, extract_input, label="doc-extract-uid")
    uid_out: UidExtraction = _parse_uid_extraction(extract_res[0])
    if log_dir is not None:
        try:
            save_history(extract_res[1], Path(log_dir) / "doc_extract_uid_allmsg.json")
        except Exception as e:
            logger.warning(f"[doc-agent] save extract-uid history failed: {e}")

    examples = uid_out.uid_examples[:max_uid_examples]
    uid_index_map = {uid: i for i, uid in enumerate(examples)}

    # 2. 全文 grep + LLM 过滤误命中
    grep_hits = grep_uids(doc_path, examples)
    filter_output: Optional[FilterRelevantOutput] = None
    if any(grep_hits.values()):
        filter_agent = build_filter_relevant_agent(model)
        filter_input = (
            f"## 主键信息\n"
            f"- field_name: {uid_out.uid_field_name}\n"
            f"- regex: {uid_out.uid_regex}\n"
            f"- examples (uid_index 见下方): {examples}\n\n"
            f"## 每个样例 uid 的全文 grep 命中\n"
            f"{_format_grep_hits_block(grep_hits, uid_index_map)}\n"
        )
        filter_res = await _run_streaming(filter_agent, filter_input, label="doc-filter-relevant")
        filter_output = _parse_filter_output(filter_res[0])
        if log_dir is not None:
            try:
                save_history(filter_res[1], Path(log_dir) / "doc_filter_relevant_allmsg.json")
            except Exception as e:
                logger.warning(f"[doc-agent] save filter-relevant history failed: {e}")

    # 3. 抽取每个 uid 的全部属性键值对
    extract_output: Optional[ExtractAttributesOutput] = None
    if filter_output is not None and filter_output.uid_lines:
        relevant_lines_map = _gather_relevant_lines(filter_output, doc_lines)
        attr_agent = build_extract_attributes_agent(model)
        attr_input = (
            "## 每个 uid 的关联原文 (line_no: content)\n"
            + _format_relevant_for_extract(filter_output, relevant_lines_map)
        )
        attr_res = await _run_streaming(attr_agent, attr_input, label="doc-extract-attrs")
        extract_output = _parse_extract_output(attr_res[0])
        if log_dir is not None:
            try:
                save_history(attr_res[1], Path(log_dir) / "doc_extract_attrs_allmsg.json")
            except Exception as e:
                logger.warning(f"[doc-agent] save extract-attrs history failed: {e}")

    # 4. 算法推断属性段边界 + LLM 给段命名列 + 标 needed
    locate_output: Optional[LocateSegmentsOutput] = None
    if extract_output is not None and extract_output.uid_attributes and filter_output is not None:
        # 4a. 算法确定性推断段边界 (不让 LLM 猜)
        seg_ranges = _infer_segment_boundaries(filter_output, len(doc_lines))
        logger.info(
            f"[doc-agent] inferred {len(seg_ranges)} segment(s): {seg_ranges}"
        )
        # 4b. 格式化 per-segment 属性散点, 让 LLM 直观看出每段有哪些属性
        scatter_block = _format_attr_scatter_per_segment(extract_output, seg_ranges)
        locate_agent = build_locate_segments_agent(model)
        locate_input = (
            f"## question\n{task_question}\n\n"
            f"## knowledge.md\n{knowledge_md}\n\n"
            f"## 已推断的属性段切分 (行范围已确定, 无需你修改)\n"
            f"{scatter_block}\n"
        )
        locate_res = await _run_streaming(locate_agent, locate_input, label="doc-locate-segments")
        # 4c. 解析 LLM 输出 (只含 segment_index + columns + needed, 无行号)
        llm_out: _LLMLocateOutput = _parse_locate_output(locate_res[0])
        # 4d. 合并算法行范围 + LLM 语义标签 → 完整 LocateSegmentsOutput
        llm_map = {seg.segment_index: seg for seg in llm_out.segments}
        merged_segs: list[AttributeSegment] = []
        for idx, (s, e) in enumerate(seg_ranges, start=1):
            label = llm_map.get(
                idx,
                _LLMSegmentLabel(segment_index=idx, columns=[], needed=False),
            )
            merged_segs.append(
                AttributeSegment(
                    segment_index=idx,
                    start_line=s,
                    end_line=e,
                    columns=label.columns,
                    needed=label.needed,
                )
            )
        locate_output = LocateSegmentsOutput(segments=merged_segs)
        logger.info(
            f"[doc-agent] locate_output: "
            + ", ".join(
                f"seg{s.segment_index}[{s.start_line}-{s.end_line}] needed={s.needed}"
                for s in locate_output.segments
            )
        )
        if log_dir is not None:
            try:
                save_history(locate_res[1], Path(log_dir) / "doc_locate_segments_allmsg.json")
            except Exception as e:
                logger.warning(f"[doc-agent] save locate-segments history failed: {e}")

    # 5. 对每个属性段, 在其行范围内逐行抽 (uid, value), 再 pivot 成结构化表
    segment_extractions: list[SegmentExtractionOutput] = []
    records: list[StructuredRecord] = []
    if locate_output is not None and locate_output.segments:
        segment_extractions = await _structurize_all_segments(
            locate_output=locate_output,
            doc_lines=doc_lines,
            uid_info=uid_out,
            model=model,
            log_dir=log_dir,
            max_lines_per_segment=max_lines_per_segment,
        )
        records = _pivot_records(locate_output, segment_extractions)

    return DocAgentResult(
        doc_path=str(doc_path),
        uid=uid_out,
        grep_hits=grep_hits,
        filter_output=filter_output,
        extract_output=extract_output,
        locate_output=locate_output,
        segment_extractions=segment_extractions,
        records=records,
    )


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import asyncio
    import json
    import sys

    if len(sys.argv) < 2:
        default_task = Path(
            "/Users/zhe/Documents/work/20260400-data-agent/data/input/task_396"  # 396 344
        )
    else:
        default_task = Path(sys.argv[1])

    async def _main():
        # 默认拿 task_344/Patient.md (患者人口学叙述) 跑一遍
        task_dir = default_task
        doc_files = sorted((task_dir / "context" / "doc").glob("*.md"))
        if not doc_files:
            raise FileNotFoundError(f"no .md doc found under {task_dir}/context/doc")
        doc_path = doc_files[0]

        task_json = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        question = task_json.get("question", "")
        knowledge_md = (task_dir / "context" / "knowledge.md").read_text(encoding="utf-8")

        model = pydantic_predefined_model.model_code_a3b_nothink
        result = await run_doc_agent(
            doc_path,
            task_question=question,
            knowledge_md=knowledge_md,
            model=model,
            log_dir=Path("/tmp/doc_agent_log"),
        )
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    asyncio.run(_main())
