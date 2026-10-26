"""维度段切分子 agent (split_subagent).

针对 "散文化结构化 doc 的维度段模式" (见
``workdir/gt_opus_0507/workdetail/hard_tasks_doc_dimension_analysis.md``)
设计的轻量子流程, 用 Qwen3.5-35B-A3B (model_ms_a3b) 完成两步:

流程
----
1. **read_doc_head**: 读 doc 开头 N 行 (默认 10).
2. **extract_uid_agent (Qwen #1)**: 从开头几行抽取主键 (uid)
   字段命名 / 正则 / 若干具体样例 uid.
3. **grep_uids**: 对每个样例 uid 在整篇 doc 中做 word-level 精确匹配
   (只看每行第一句, 避免数字 uid 的噪声), 收集 ``(line_no, content)``.
4. **analyze_segments_agent (Qwen #2)**: 把 grep 结果直接丢给 LLM, 让它
   - 过滤误命中 (例如纯数字 uid 在非主语位置出现);
   - 判断每个 uid 在 doc 中真正的"段标记行" (per-uid 段→行号映射);
   - 给出维度段数 S 和每段属性描述.
5. 后处理: 根据 LLM 给出的 per-uid segment_line_nos, 合并排序得到全局
   候选段边界, 写进 ``SplitResult.candidate_boundaries``.

入口: ``run_split_subagent(doc_path, model=model_ms_a3b, ...)``,
返回 ``SplitResult``.

注意: 本子 agent 只产出 "维度段结构描述", 真正按段落聚合还原 CSV
由下游 solver 负责.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import json_repair
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from loguru import logger

from agents_v2.agent_util import save_history
from configs import pydantic_predefined_model


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class UidExtraction(BaseModel):
    """Qwen #1 输出: 从 doc 开头识别出的主键信息."""

    uid_field_name: str = Field(
        description="主键字段名, 例如 'registry_id' / 'rec' / 'compound_id' / 'id'. "
                    "若 doc 没有显式字段名, 用一个简短的描述性名称."
    )
    uid_regex: str = Field(
        description="可在原文中匹配主键的 Python 正则 (不要含分组以外的 \\b 之类), "
                    "例如 'rec[0-9A-Za-z]{16}' 或 'TR\\d+' 或 '\\b\\d{1,5}\\b'."
    )
    uid_examples: list[str] = Field(
        description="从 doc 开头抽取出的若干具体主键样例 (3-6 个), "
                    "全文 grep 时会用到. 必须是字面量, 不是正则."
    )
    note: str = Field(
        default="",
        description="任何关于主键命名/出现位置的简短说明, 可空."
    )


class SegmentInfo(BaseModel):
    """单个维度段的结构化描述 (Qwen #2 输出的子项)."""

    index: int = Field(description="段序号, 从 1 开始")
    attributes: str = Field(
        description="该段所描述的属性, 例如 'codename + full_name' "
                    "或 'height/weight'."
    )
    sample_excerpt: str = Field(
        default="",
        description="该段的一行代表性原文 (帮助下游识别), 可空."
    )


class UidSegmentMapping(BaseModel):
    """单个 uid 在 doc 中真正属于"段标记"的行号列表 (Qwen #2 给出).

    LLM 需要从 grep 命中里剔除明显的误命中 (例如纯数字 uid 出现在年份/
    百分比/无关编号位置), 并按段顺序输出 ``segment_line_nos``;
    长度应等于 ``n_segments``.
    """

    uid: str = Field(description="uid 字面量, 与输入对应")
    segment_line_nos: list[int] = Field(
        description="该 uid 在每个维度段开头出现的行号, 按段顺序排列. "
                    "若某段确实没找到, 用 -1 占位."
    )


class SplitConfirmation(BaseModel):
    """Qwen #2 输出: 维度段结构 + per-uid 段→行号映射."""

    n_segments: int = Field(description="维度段数 (例如 2, 3, 4, 5)")
    segments: list[SegmentInfo] = Field(description="每段的属性描述, 长度等于 n_segments")
    uid_mappings: list[UidSegmentMapping] = Field(
        description="每个样例 uid 在文档中段→行号的映射 (剔除误命中后)"
    )
    separator: str = Field(
        default="",
        description="段间显式分隔符 (如 '***' / '*'); 若仅按主键自然循环, 填空字符串."
    )
    confidence: str = Field(
        default="medium",
        description="low / medium / high; 反映模型对自身判断的把握."
    )
    rationale: str = Field(
        default="",
        description="一句话简要说明依据; 可空."
    )


class CandidateBoundary(BaseModel):
    """从 LLM 给出的 uid_mappings 后处理出的全局候选段边界."""

    segment_index: int
    line_no: int
    line_text: str
    rationale: str = Field(default="", description="Qwen #3 选择该行作为分界的简短理由")


class BoundaryLocate(BaseModel):
    """Qwen #3 输出: 在 cluster_start 上方若干行里挑出真正的段分界行号."""

    boundary_line_no: int = Field(
        description="真正分界所在的 1-based 行号 (必须出现在输入展示的窗口内). "
                    "若窗口里没有任何明显分界 (即 cluster_start 自身就是段开头), "
                    "返回 cluster_start 本身."
    )
    rationale: str = Field(
        default="",
        description="一句话说明为什么挑这行 (例如: 是过渡段落首行 / 章节标题 / 显式分隔符)."
    )


class SplitResult(BaseModel):
    """run_split_subagent 的最终返回值."""

    doc_path: str
    uid: UidExtraction
    grep_hits: dict[str, list[tuple[int, str]]] = Field(
        default_factory=dict,
        description="每个样例 uid 的 word-level grep 原始命中 (line_no, content)",
    )
    confirmation: Optional[SplitConfirmation] = None
    candidate_boundaries: list[CandidateBoundary] = Field(
        default_factory=list,
        description="基于 confirmation.uid_mappings 后处理的全局段边界",
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def read_doc_head(doc_path: Path, n_lines: int = 10) -> str:
    """读 doc 开头 n_lines 行 (跳过纯空行不删, 保留原始行号关系)."""
    lines: list[str] = []
    with Path(doc_path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


# 分词正则: 一个 "token" = 字母数字 + 下划线/连字符的最长串.
# 这样 'rec2N69DMcrqN9PJC' / 'TR12' / '123' / 'patient_001' 都是单个 token,
# 不会被 'a 1 b' 里的子串 '1' 误命中.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _tokenize_line(line: str) -> set[str]:
    """把一行的 *第一句* 拆成 token 集合, 用于 word-level 精确匹配.

    为什么只取第一句:
        散文化 doc 通常每段开头第一句就点出 uid (例如 "Strategic Unit 7 ..."),
        后面几句是叙述性补充, 可能反复出现别的数字/编号 (年份、楼号、百分比),
        会污染匹配. 截到第一个 '.' 之前可大幅降噪.
    """
    first_sentence = line.split(".", 1)[0]
    return set(_TOKEN_RE.findall(first_sentence))


def grep_uids(
    doc_path: Path,
    uid_examples: list[str],
) -> dict[str, list[tuple[int, str]]]:
    """对每个 uid 做 *word-level* 精确匹配, 返回 ``{uid: [(line_no, content), ...]}``.

    为什么不用子串匹配:
        当 uid 是纯数字自增序号 (例如 '1', '2', '7') 时, 子串 ``in`` 操作
        会把所有含 '1'/'2'/'7' 的行 (无论是 '2024' 还是 '170' 这种)
        全部命中, 行数膨胀到几百, 后续众数推断完全失真.

    实现:
        - 先用 ``[A-Za-z0-9_\\-]+`` 把每行切成 token 集合;
        - 对每个 uid (作为整体 token) 做集合命中判断;
        - 行号从 1 开始计数, 与文本编辑器一致.
    """
    doc_path = Path(doc_path)
    text = doc_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 预先 tokenize 一遍, 多个 uid 共享, 避免重复扫描.
    line_tokens: list[set[str]] = [_tokenize_line(line) for line in lines]

    out: dict[str, list[tuple[int, str]]] = {}
    for uid in uid_examples:
        hits: list[tuple[int, str]] = []
        for idx, tokens in enumerate(line_tokens, start=1):
            if uid in tokens:
                hits.append((idx, lines[idx - 1]))
        out[uid] = hits
    return out


def _format_grep_hits_for_llm(
    grep_hits: dict[str, list[tuple[int, str]]],
    max_hits_per_uid: int = 80,
) -> str:
    """把 grep 结果格式化成给 Qwen 看的多块文本.

    每个 uid 一块, 块内按行号顺序列出 ``  <line_no>: <content>``;
    若命中数超过 max_hits_per_uid, 头尾各取一半 + 省略号, 控制 prompt 长度.
    """
    blocks: list[str] = []
    for uid, hits in grep_hits.items():
        if not hits:
            blocks.append(f"### uid = {uid!r}\n(无命中)")
            continue
        if len(hits) > max_hits_per_uid:
            half = max_hits_per_uid // 2
            shown = hits[:half] + [(-1, f"... 省略 {len(hits) - max_hits_per_uid} 行 ...")] + hits[-half:]
        else:
            shown = hits
        body = "\n".join(f"  {ln:>5}: {content}" for ln, content in shown)
        blocks.append(f"### uid = {uid!r}  (共 {len(hits)} 次命中)\n{body}")
    return "\n\n".join(blocks)


def _collect_segment_starts(confirmation: SplitConfirmation) -> list[int]:
    """跨 uid 取每段的最早 cluster_start 行号 (即段内第一条数据所在行).

    返回长度 == n_segments 的列表; 若某段无任何有效行号, 用 -1 占位.
    """
    n = confirmation.n_segments
    seg_starts: list[list[int]] = [[] for _ in range(n)]
    for m in confirmation.uid_mappings:
        if len(m.segment_line_nos) != n:
            continue
        for i, ln in enumerate(m.segment_line_nos):
            if ln > 0:
                seg_starts[i].append(ln)
    return [min(s) if s else -1 for s in seg_starts]


def _format_above_window_for_llm(
    doc_lines: list[str],
    cluster_start: int,
    above_lines: int,
    below_lines: int = 0,
) -> str:
    """格式化 cluster_start 周围若干行原文 (带行号), 用于 Qwen #3 判别.

    返回多行文本, 每行形如 ``  <line_no>: <content>``; 上方提供 above_lines
    行用于识别分界, 下方追加 below_lines 行让模型看到簇内更多数据 (便于
    与上一段尾部内容区分, 防止把上一段最后一条数据误判为分界).
    cluster_start 本身一定包含在窗口内.
    """
    start = max(1, cluster_start - above_lines)
    end = min(len(doc_lines), cluster_start + below_lines)
    return "\n".join(
        f"  {ln:>5}: {doc_lines[ln - 1]}" for ln in range(start, end + 1)
    )


# ---------------------------------------------------------------------------
# 三个 Qwen agent
# ---------------------------------------------------------------------------


EXTRACT_UID_INSTRUCTION = """
你的任务是从一段散文化的 doc 开头几行中, 找出"主键 (uid)"字段.

**主键的特征**:
- 在 doc 后续每条记录中都会出现, 用于标识同一实体.
- 形态可能是:
  * 数字 ID (如 `7`, `26`, `28` 这样的整数)
  * Airtable 风格的 `rec` + 16 位字母数字 (如 `rec2N69DMcrqN9PJC`)
  * 自定义前缀 + 数字 (如 `TR12`, `rec1`)
  * 自定义命名 (如 `compound_id`, `league_id`, `patient_id`)

**要求**:
- 给出可匹配的 Python 正则 `uid_regex`.
- 给出 3-6 个从开头文本中已经看到的具体样例 `uid_examples` (字面量).
- `uid_field_name` 用语义化字段名.

如果开头几行还看不出明确主键, 选择最可能的候选, 并在 note 里说明.

**输出格式**: 严格输出如下 JSON (不要加任何额外说明):
```json
{
  "uid_field_name": "registry_id",
  "uid_regex": "rec[0-9A-Za-z]{16}",
  "uid_examples": ["rec2N69DMcrqN9PJC", "recABCDEF12345678"],
  "note": ""
}
```
"""


CONFIRM_SEGMENTS_INSTRUCTION = """
你将看到一份 doc 的"维度段切分"原始信号:

1. **主键信息**: 字段名 / 正则 / 若干样例 uid (字面量).
2. **每个 uid 的全文 grep 命中**: 形如 `<line_no>: <content>` 的列表.
   注意: grep 已做 word-level 匹配并只看每行第一句, 但当 uid 是纯数字
   (如 '1', '7', '26') 时仍可能误命中无关编号. **你必须凭语义筛掉**
   这些误命中.

**散文化结构化 doc 的维度段模式**: 同一主键在每段各出现一次, 每段
描述该实体的某些维度属性. 段之间可能没有显式分隔符, 也可能有
`***` / `*` / 标题之类.

你的任务:
- 对每个 uid, 判断哪些命中行号是"该 uid 的真正段标记行" (即段开头),
  按段顺序输出 `uid_mappings[i].segment_line_nos`. 若某段确实缺失,
  用 `-1` 占位. 长度必须等于 `n_segments`.
- 给出 `n_segments` (一般 2-5).
- `segments[i].attributes` 描述第 i 段对应的属性
  (例如 'codename + full_name' / 'height/weight').
- `segments[i].sample_excerpt` 可填一行代表性原文.
- `separator`: 若识别到显式分隔符则填出, 否则空字符串.
- `confidence`: 'low' / 'medium' / 'high'.
- `rationale`: 一句话理由.

判断时的关键启发:
- 真正的段标记行通常以 uid 紧跟主语动词开头 (e.g. "Strategic Unit 7..." /
  "An entry has been recorded for ... 7"), 而误命中往往把 uid 嵌在叙述
  深处 (如年份、楼号、百分比、计数).
- 同一 uid 的段标记行在文档里应大致等距分布; 离群点可疑.
- 跨 uid 比较: 不同 uid 的段标记行应呈现循环模式 (uid_a 段1, uid_b 段1,
  ..., uid_a 段2, uid_b 段2, ...).

**输出格式**: 严格输出如下 JSON (不要加任何额外说明):
```json
{
  "n_segments": 2,
  "segments": [
    {"index": 1, "attributes": "codename + full_name", "sample_excerpt": ""},
    {"index": 2, "attributes": "height/weight", "sample_excerpt": ""}
  ],
  "uid_mappings": [
    {"uid": "rec2N69DMcrqN9PJC", "segment_line_nos": [10, 55]},
    {"uid": "recABCDEF12345678", "segment_line_nos": [12, 57]}
  ],
  "separator": "",
  "confidence": "high",
  "rationale": "每个uid在文档中出现两次, 分别对应两段不同属性"
}
```
"""


async def _run_streaming(agent: "Agent", prompt: str, label: str = ""):
    """用 run_stream 实时打印进度, 返回 (output, all_messages) 元组.

    output_type=str: stream_text(delta=True) 逐字符打印.
    必须在 async with 内读取 output / all_messages, 退出后即失效.
    """
    import sys
    if label:
        sys.stdout.write(f"\n=== [stream:{label}] ===\n")
        sys.stdout.flush()
    async with agent.run_stream(prompt) as s:
        async for text in s.stream_text(delta=True):
            sys.stdout.write(text)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        output = await s.get_output()
        messages = s.all_messages()
    return output, messages


def build_extract_uid_agent(model: Any) -> Agent:
    return Agent(
        model=model,
        instructions=EXTRACT_UID_INSTRUCTION,
        output_type=str,
    )


def _parse_uid_extraction(text: str) -> UidExtraction:
    data = json_repair.loads(text)
    return UidExtraction(**data)


def build_confirm_segments_agent(model: Any) -> Agent:
    return Agent(
        model=model,
        instructions=CONFIRM_SEGMENTS_INSTRUCTION,
        output_type=str,
    )


def _parse_split_confirmation(text: str) -> SplitConfirmation:
    data = json_repair.loads(text)
    return SplitConfirmation(**data)


LOCATE_BOUNDARY_INSTRUCTION = """
你将看到 doc 的一段连续行 (含行号), 中间某处是 cluster_start (该段第一条数据所在行).

特征:
- 窗口里的行可分两类: 含 uid 的"数据行" 和 不含 uid 的"分隔区域"(可能是空行 / 标题
  / 显式分隔符 `***` `---` `## xxx` / 一两行综述或承上启下的过渡文字).
- cluster_start 上方若存在数据行, 它们属于"上一段"; cluster_start 及其下方的数据行
  属于"当前段". 两段描述的属性集合通常不同 (例如上段讲 codename, 当前段讲 height/weight).
- 真正的"分界 boundary_line_no"就是 **分隔区域的第一行** (即上一段最后一条数据之后的
  第一行). 若 cluster_start 上方没有任何数据行 (只有空行甚至窗口截断), 说明它本身就是
  段首, 直接返回 cluster_start.

要求:
- boundary_line_no 必须是窗口里实际出现过的行号.
- rationale 一句话说明依据 (例如: "line N 是过渡段落首行" / "line N 是 ## 标题" /
  "上方仅有空行, 取 cluster_start 自身").

**输出格式**: 严格输出如下 JSON (不要加任何额外说明):
```json
{
  "boundary_line_no": 42,
  "rationale": "line 42 是显式分隔符 *** 所在行"
}
```
"""


def build_locate_boundary_agent(model: Any) -> Agent:
    return Agent(
        model=model,
        instructions=LOCATE_BOUNDARY_INSTRUCTION,
        output_type=str,
    )


def _parse_boundary_locate(text: str) -> BoundaryLocate:
    data = json_repair.loads(text)
    return BoundaryLocate(**data)


# ---------------------------------------------------------------------------
# Qwen #3: 真实段分界定位
# ---------------------------------------------------------------------------


async def _locate_real_boundaries(
    *,
    confirmation: SplitConfirmation,
    doc_path: Path,
    model: Any,
    above_lines: int,
    below_lines: int,
    log_dir: Optional[Path],
) -> list[CandidateBoundary]:
    """对每段 cluster_start 调 Qwen #3, 拿真实段分界 (不要用脚本启发, pattern 因 task 而异)."""
    doc_lines = Path(doc_path).read_text(encoding="utf-8").splitlines()
    starts = _collect_segment_starts(confirmation)
    locate_agent = build_locate_boundary_agent(model)
    boundaries: list[CandidateBoundary] = []
    for i, cluster_start in enumerate(starts, start=1):
        if cluster_start <= 0:
            continue
        window = _format_above_window_for_llm(doc_lines, cluster_start, above_lines, below_lines)
        prompt = (
            f"## segment_index = {i}\n"
            f"## cluster_start (该段第一条数据所在行) = {cluster_start}\n\n"
            f"## cluster_start 上方窗口 (含 cluster_start 本身, line_no: content)\n"
            f"{window}\n"
        )
        try:
            res = await _run_streaming(locate_agent, prompt, label=f"locate-boundary-seg{i}")
        except Exception as e:
            logger.warning(f"[split-subagent] locate-boundary seg{i} failed: {e}; fallback to cluster_start")
            boundaries.append(CandidateBoundary(
                segment_index=i,
                line_no=cluster_start,
                line_text=doc_lines[cluster_start - 1] if 1 <= cluster_start <= len(doc_lines) else "",
                rationale=f"locate-agent failed: {e}",
            ))
            continue
        out: BoundaryLocate = _parse_boundary_locate(res[0])
        ln = out.boundary_line_no if 1 <= out.boundary_line_no <= len(doc_lines) else cluster_start
        boundaries.append(CandidateBoundary(
            segment_index=i,
            line_no=ln,
            line_text=doc_lines[ln - 1] if 1 <= ln <= len(doc_lines) else "",
            rationale=out.rationale,
        ))
        if log_dir is not None:
            try:
                save_history(res[1], Path(log_dir) / f"split_locate_boundary_seg{i}_allmsg.json")
            except Exception as e:
                print(f"[split-subagent] save locate-boundary seg{i} history failed: {e}")
    return boundaries


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run_split_subagent(
    doc_path: Path,
    model,
    *,
    head_lines: int = 15,
    max_uid_examples: int = 5,
    max_hits_per_uid: int = 15,
    above_lines: int = 9,
    below_lines: int = 4,
    log_dir: Optional[Path] = None,
) -> SplitResult:
    """跑完整的 split-subagent 流程.

    Args:
        doc_path: 待分析的 .md / .txt doc 路径.
        model: pydantic_ai 模型; 默认 ``model_code_a3b`` (Qwen3.5-35B-A3B).
        head_lines: 喂给 Qwen #1 的开头行数.
        max_uid_examples: 最多拿前 N 个样例 uid 去 grep.
        max_hits_per_uid: 喂给 Qwen #2 时, 每个 uid 最多展示多少条命中
            (超出会头尾截取 + 省略号), 防止 prompt 过长.
        above_lines: 喂给 Qwen #3 时, 每段 cluster_start 上方展示多少行
            原文 (用于真实分界判别).
        log_dir: 若提供, 把 agent 的对话历史落到该目录.

    Returns:
        SplitResult: 包含 uid 抽取、grep 原始命中、Qwen 段结构确认及候选边界.
    """
    doc_path = Path(doc_path)

    # 1. 读开头
    head_text = read_doc_head(doc_path, n_lines=head_lines)

    # 2. Qwen #1 抽 uid
    extract_agent = build_extract_uid_agent(model)
    extract_input = (
        f"## doc 开头 {head_lines} 行 (含行号)\n"
        + "\n".join(f"{i + 1:>4}: {l}" for i, l in enumerate(head_text.splitlines()))
    )
    extract_res = await _run_streaming(extract_agent, extract_input, label="extract-uid")
    uid_out: UidExtraction = _parse_uid_extraction(extract_res[0])
    if log_dir is not None:
        try:
            save_history(extract_res[1], Path(log_dir) / "split_extract_uid_allmsg.json")
        except Exception as e:
            print(f"[split-subagent] save extract history failed: {e}")

    # 3. word-level grep
    examples = uid_out.uid_examples[:max_uid_examples]
    grep_hits = grep_uids(doc_path, examples)

    # 4. Qwen #2 直接基于 grep 结果分析: 过滤误命中 + 给段结构 + per-uid 段→行号
    confirmation: Optional[SplitConfirmation] = None
    candidate_boundaries: list[CandidateBoundary] = []
    if any(grep_hits.values()):
        confirm_agent = build_confirm_segments_agent(model)
        hits_block = _format_grep_hits_for_llm(grep_hits, max_hits_per_uid=max_hits_per_uid)
        confirm_input = (
            f"## 主键信息\n"
            f"- field_name: {uid_out.uid_field_name}\n"
            f"- regex: {uid_out.uid_regex}\n"
            f"- examples: {examples}\n"
            f"- note: {uid_out.note}\n\n"
            f"## 每个样例 uid 的全文 grep 命中 (line_no: content)\n"
            f"{hits_block}\n"
        )
        confirm_res = await _run_streaming(confirm_agent, confirm_input, label="confirm-segments")
        confirmation = _parse_split_confirmation(confirm_res[0])
        if log_dir is not None:
            try:
                save_history(
                    confirm_res[1],
                    Path(log_dir) / "split_confirm_segments_allmsg.json",
                )
            except Exception as e:
                print(f"[split-subagent] save confirm history failed: {e}")

        candidate_boundaries = await _locate_real_boundaries(
            confirmation=confirmation,
            doc_path=doc_path,
            model=model,
            above_lines=above_lines,
            below_lines=below_lines,
            log_dir=log_dir,
        )

    return SplitResult(
        doc_path=str(doc_path),
        uid=uid_out,
        grep_hits=grep_hits,
        confirmation=confirmation,
        candidate_boundaries=candidate_boundaries,
    )


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import asyncio
    import json
    import sys

    if len(sys.argv) < 2:
        # 默认拿 task_396/superhero.md 跑一遍
        default = Path(
            "/Users/zhe/Documents/work/20260400-data-agent/data/input/"
            "task_396/context/doc/superhero.md"
        )
        target = default
    else:
        target = Path(sys.argv[1])

    async def _main():
        model = pydantic_predefined_model.model_code_a3b_nothink  # 据说 think 和 structure output不兼容
        result = await run_split_subagent(target, model=model, log_dir=Path("/tmp/split_subagent_log"))
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    asyncio.run(_main())
