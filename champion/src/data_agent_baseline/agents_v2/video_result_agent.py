"""video_result_agent.py — 判断"视频题的最终答案是否已直接显示在视频里, 是否该直接采纳".

线上无 gold, agent 必须**独立**完成两件事:
  (1) 识别: 视频面板里"看起来像答案"的值/列表, 究竟是不是 question 要的最终答案
      (视频常有诱饵: 旧批次、邻表、单一产品高亮、"单位是产品数不是家数"之类陷阱);
  (2) 决策: 该**直接采纳**视频值, 还是必须**回结构化表重算**, 还是采纳但 SQL 复核.

为稳健, 不一步输出一大堆字段, 而是拆成 3 个窄步骤, 每步只回答一件事, 前一步喂后一步:

  Step 1  EXTRACT  (看视频, 多模态)   -> VideoClaim
      输入: question + 视频多模态 parts (帧图 + hiccup 版面树 + 旁白)
      输出: has_displayed_answer / claimed_values / source_frames / distractor_notes
  Step 2  ALIGN    (看结构化 schema)   -> AnswerShape   (仅当 step1 命中)
      输入: question + step1.claimed_values + 结构化 schema 预览 (不含 doc) + knowledge
      输出: target_columns / unit_consistent / normalized_values
  Step 3  DECIDE   (判去留)            -> Decision
      输入: question + step1 + step2 + 结构化 schema 预览
      输出: decision (ADOPT/RECOMPUTE/ADOPT_AND_VERIFY) / confidence / reason

不读取 gold, 不看 prediction, 不针对任何 task 写特例.

入口:
    run_video_result_async(task_dir=..., model=...) -> VideoResultResult
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent


DEFAULT_KNOWLEDGE_MAX_CHARS = 12000
DEFAULT_SCHEMA_MAX_CHARS = 14000
DEFAULT_TIMEOUT_S = 180.0


# ---------------------------------------------------------------------------
# 三步的结构化输出 (每步字段都很少, 模型聚焦单一判断)
# ---------------------------------------------------------------------------


class VideoClaim(BaseModel):
    """Step 1: 视频里是否直接显示了 question 的最终答案."""

    has_displayed_answer: bool = Field(
        description=(
            "视频面板是否针对 question **直接显示**了最终答案值/名单(而非只给筛选规则/阈值)。"
            "若视频只给'准入线/口径/批次配置', 答案行集仍需回表筛, 则为 false。"
        ),
    )
    claimed_values: List[List[str]] = Field(
        default_factory=list,
        description=(
            "从视频读出的候选最终答案, 行优先的二维数组(每行是一条答案记录, 列顺序与 question 要的字段一致)。"
            "has_displayed_answer=false 时留空。务必排除诱饵帧(旧批次/邻表/单一产品高亮/单位陷阱), 只填真正的目标值。"
        ),
    )
    source_frames: List[str] = Field(
        default_factory=list,
        description="支持上述答案的关键帧标识(如 '幻灯片 #7' 或时间戳), 便于审计。",
    )
    distractor_notes: str = Field(
        default="",
        description="简述识别到的诱饵/干扰(如'#2022-Q1 是旧批次不是当前''该卡是单产品高亮不是类型均值'), 没有就空。",
    )


class AnswerShape(BaseModel):
    """Step 2: 视频读出的答案该落到哪些结构化列, 单位/格式是否一致."""

    target_columns: List[str] = Field(
        default_factory=list,
        description="答案该对应的目标输出列名(尽量对齐结构化表/knowledge 里的真实列名)。",
    )
    unit_consistent: bool = Field(
        default=True,
        description=(
            "视频里数值的单位/格式是否与库内目标口径一致。"
            "若 knowledge/结构化表用不同单位(如库内'百万元'而视频显示'亿元'), 则 false。"
        ),
    )
    normalized_values: List[List[str]] = Field(
        default_factory=list,
        description=(
            "把 step1 候选值按目标口径归一后的二维数组(单位换算/去百分号等)。"
            "unit_consistent=true 时可与 step1 相同。"
        ),
    )
    note: str = Field(default="", description="对齐/归一的简要说明, 没有就空。")


class Decision(BaseModel):
    """Step 3: 最终决策."""

    decision: str = Field(
        description=(
            "三选一: "
            "'ADOPT' = 视频已给出完整明确答案, 直接输出视频值即可, 无需回表; "
            "'ADOPT_AND_VERIFY' = 采纳视频值, 但建议用结构化 SQL 低成本交叉复核一致后再输出; "
            "'RECOMPUTE' = 视频只给规则/不完整/单位不一致, 必须回结构化表重算。"
        ),
    )
    confidence: float = Field(
        description="对该决策的置信度 0.0~1.0。",
    )
    reason: str = Field(
        description="一句话说明为什么这么决策。",
    )


@dataclass
class VideoResultResult:
    """单 task 的端到端判定结果(三步串联)."""

    task_id: str
    has_video: bool = False
    has_displayed_answer: bool = False
    decision: str = "RECOMPUTE"
    confidence: float = 0.0
    claimed_values: list = field(default_factory=list)
    normalized_values: list = field(default_factory=list)
    target_columns: list = field(default_factory=list)
    source_frames: list = field(default_factory=list)
    distractor_notes: str = ""
    reason: str = ""
    error: str = ""

    def to_jsonable(self) -> dict:
        return {
            "task_id": self.task_id,
            "has_video": self.has_video,
            "has_displayed_answer": self.has_displayed_answer,
            "decision": self.decision,
            "confidence": self.confidence,
            "claimed_values": self.claimed_values,
            "normalized_values": self.normalized_values,
            "target_columns": self.target_columns,
            "source_frames": self.source_frames,
            "distractor_notes": self.distractor_notes,
            "reason": self.reason,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# prompts (每步独立, 各自聚焦)
# ---------------------------------------------------------------------------

STEP1_INSTRUCTION = """你是视频题的**答案抽取 agent**。你会看到一段任务讲解视频的内容(幻灯片帧图 + 版面文字 hiccup + 旁白)。

你的**唯一任务**: 判断视频面板是否针对 question **直接显示**了最终答案, 若有则原样读出。

## 推荐分析流程(7 步, 通用方法, 勿跳步)
这类讲解视频是"操作仪表盘"的录屏: 通常包含 配置/筛选面板、结果/汇总卡、排名/明细列表、
字典/参考页、以及**故意混入的干扰页**。答案常常不在单帧里直读, 而需跨帧关联。按以下流程读:

1. 先拆解 question 的目标形态: 要哪几个字段? 粒度是单值聚合 / 排名名单 / 分组计数?
   question 是否引用了"视频里配置的某范围/某口径/某方案"(若引用, 答案必须落在该配置生效的视图下)。
2. 通览所有帧, 先给每帧归类(配置面板 / 结果汇总 / 排名列表 / 字典参考 / 干扰), 建立索引;
   不要看到第一个数字就下结论。
3. 定位"作用域": 找出哪个筛选条件 / 单位 / 统计周期 / 分组维度被**锁定 / 选中 / 应用 / 高亮**。
   它决定了"在众多数字里, 哪个才是 question 要的那个答案"。
4. 在该作用域下定位结果: 同一指标可能在多个范围/视图下都出现过, 只取与 question + 锁定配置
   **一致**的那一处; 与作用域不符的同名数字不是答案。
5. 跨帧拼装: 若答案是名单, 名称与数值常分处不同页/帧, 按**排名 / 行号 / id 对齐**成完整行;
   若答案是单值聚合, 通常有一张汇总卡直接给出该值。允许跨帧拼, 但每一格都要能在某帧里指认到来源。
6. 显式排除干扰: 旧批次、邻近/相似指标、单一对象的高亮(不代表分组/全市场结果)、
   "这是计数不是金额 / 不是均值 / 单位是数量不是家数"、"仅作类别说明"、"未选中/未应用"的字段。
   逐条点名写进 distractor_notes, 不要当答案。
7. 完整性自检: 拼出的答案是否覆盖 question 要的**全部字段与全部行**? 只有当完整答案能从帧中复原
   时才判 has_displayed_answer=true; 若只看得到规则/阈值、或答案残缺、或拿不准作用域, 判 false
   (交给后续重算, 不要硬凑)。

判定要点:
- 只回答"视频里能不能直接看到答案", 不要去想怎么用 SQL 算。
- 区分两类视频:
  (A) 视频直接把最终结果摆在面板上(汇总卡数值、排行榜逐行、导出清单 R01..、字段卡的具体取值)
      -> has_displayed_answer=true, 把答案逐行填进 claimed_values。
  (B) 视频只给筛选规则/准入线/口径/批次配置, 没有摆出最终结果行集
      -> has_displayed_answer=false, claimed_values 留空。
- **严防诱饵**(视频故意放干扰): 旧批次、邻表/相邻指标、单一产品的高亮(不代表分组结果)、
  明确标注"单位是产品数量不是家数/不是管理人数"、"仅作类别说明"、"未选中字段"等。把这些写进 distractor_notes,
  不要当成答案。
- **旁白口头排除**: 若旁白(口播)明确口头排除某帧/某值(如"这只是参考 context/just context""不要当答案/do not use as the answer"
  "仅作备注/keep it as a memo""这是另一个指标/different metric"), 只把**该项**写进 distractor_notes 排除。
  口头排除针对的是某个诱饵, 不代表全视频结果都不可采纳。
- claimed_values 用行优先二维数组, 列顺序对齐 question 要的字段; 若 question 要多列(如名称+数值),
  每行就填多列, 并保证名称与数值来自**同一行/同一对象**(注意视频常把名称页与数值页拆成两帧, 要按排名对齐)。
- 名称/取值要逐字照抄视频, 不要自行改写、翻译、补单位。
- 宁缺毋滥: 拿不准作用域、或只能看到部分答案时, 判 has_displayed_answer=false, 不要凑一个不完整/不确定的答案。

严格按 schema 输出, 不要 markdown, 不要多余字段。"""


STEP2_INSTRUCTION = """你是视频题的**答案对齐 agent**。上一步已从视频读出候选答案。现在给你 question、候选答案、以及结构化数据源的 schema 预览(csv/json/db, 不含 doc)与 knowledge.md。

你的**唯一任务**: 判断这些候选答案该落到哪些**目标输出列**, 以及单位/格式是否与库内口径一致。

判定要点:
- target_columns: 对齐 question 要的字段, 尽量用结构化表/knowledge 里的真实列名。
- unit_consistent: 视频显示的数值单位是否与库内目标口径一致。重点查 knowledge.md 声明的默认单位
  (如"百万元""亿元""百分比数值如2.5表示2.5%")是否与视频显示的写法一致。不一致(如库内百万元、视频亿元)则 false。
- normalized_values: 若需要单位换算/去百分号/格式归一, 给出归一后的二维数组; 一致则可与输入相同。
- 不要在这步推翻"视频是否有答案", 只做列对齐与单位核对。

严格按 schema 输出, 不要 markdown。"""


STEP3_INSTRUCTION = """你是视频题的**决策 agent**。已知: question、视频抽取结果(是否有答案/候选值/诱饵说明)、答案对齐结果(目标列/单位是否一致)、结构化 schema 预览。

你的**唯一任务**: 在三者中选一个最终决策。

- ADOPT: 视频已给出**完整且明确**的最终答案, 且单位/格式一致(或已归一) -> 直接输出视频值, 无需回表。
- ADOPT_AND_VERIFY: 视频给出了答案, 但答案行数较多 / 有一定不确定 / 想稳妥 -> 采纳视频值, 但建议用结构化 SQL
  低成本复核一致后再输出。
- RECOMPUTE: 视频没有直接显示答案(只给规则/阈值/口径)、或答案不完整、或单位不一致无法可靠归一
  -> 必须回结构化表重算, 不要采纳视频里的数。

判定原则:
- 若 step1.has_displayed_answer=false, 必然 RECOMPUTE。
- 单值聚合(count/avg/max 等)且视频汇总卡直接给了该值 -> 倾向 ADOPT。
- 名单/多行且视频逐行摆出 -> ADOPT 或 ADOPT_AND_VERIFY(行多则后者)。
- 单位不一致且 step2 给不出可靠归一 -> RECOMPUTE。
- confidence 反映你对决策的把握。

严格按 schema 输出, 不要 markdown。"""


# ---------------------------------------------------------------------------
# 输入装配
# ---------------------------------------------------------------------------


def _read_question(task_dir: Path) -> str:
    try:
        return json.loads((task_dir / "task.json").read_text(encoding="utf-8")).get("question", "")
    except Exception as e:  # noqa: BLE001
        logger.warning("[video_result] read task.json failed for {}: {}", task_dir, e)
        return ""


def _read_knowledge(task_dir: Path, max_chars: int) -> str:
    kp = task_dir / "context" / "knowledge.md"
    if not kp.is_file():
        return ""
    text = kp.read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(knowledge 已截断)"
    return text


def _structured_schema_preview(task_dir: Path, max_chars: int) -> str:
    """结构化表 (csv/json/db) 的 schema 预览, **跳过 context/doc**.

    逐个 describe context 下 csv/json/db 子目录; doc 不输入(按需求)。
    """
    import tools_v2.describe_tool as dt

    base = str(task_dir)
    chunks = []
    for sub in ("context/csv", "context/json", "context/db"):
        d = task_dir / sub
        if not d.is_dir():
            continue
        try:
            chunks.append(dt.describe_context_dir(base, sub, skip_knowledge=True))
        except Exception as e:  # noqa: BLE001
            chunks.append(f"[ERROR] {sub}: {e!r}")
    text = "\n".join(chunks).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(schema 预览已截断)"
    return text or "(无结构化数据源)"


def _has_video(task_dir: Path) -> bool:
    vd = task_dir / "context" / "video"
    return vd.is_dir() and any(vd.iterdir())


def load_video_parts(task_dir: Path) -> list:
    """重建视频多模态 parts (帧图 + hiccup + 旁白) 供 Step1 用.

    优先从 ``workdir/_video_frames/video_input.json`` 重建(秒级, 不重抽帧),
    PNG 从同目录读取。若 json 里的图片路径是旧的绝对路径(跨 run 复制), 用 basename
    回退到当前 _video_frames 目录定位 PNG。无视频 / 无预处理 -> 返回 []。
    """
    vframes = task_dir / "workdir" / "_video_frames"
    vinput = vframes / "video_input.json"
    if not vinput.is_file():
        return []
    try:
        from video.build_video_input import SlideFrame, interleave, events_to_parts
    except Exception as e:  # pragma: no cover
        logger.debug("[video_result] import video helpers failed: {}", e)
        return []

    try:
        saved = json.loads(vinput.read_text(encoding="utf-8"))
        frames = []
        for f in saved.get("frames", []):
            p = Path(f["path"])
            if not p.is_file():
                # 跨 run 复制时 json 里残留旧绝对路径, 用 basename 回退到当前帧目录
                cand = vframes / p.name
                if cand.is_file():
                    p = cand
            frames.append(SlideFrame(
                index=f["index"], t=f["t"], path=str(p),
                burst_start=f["burst_start"], burst_end=f["burst_end"],
            ))
        events = interleave(frames, saved.get("transcript"), hiccup_map=saved.get("hiccup") or {})
        # image+hiccup: 帧图 + 版面树并列 + 旁白交错 (多模态)
        return events_to_parts(events, label=True, frames_mode="image+hiccup")
    except Exception as e:  # noqa: BLE001
        logger.debug("[video_result] rebuild parts failed: {}", e)
        return []


# ---------------------------------------------------------------------------
# 三步串联
# ---------------------------------------------------------------------------


async def run_video_result_async(
    *,
    task_dir: Path,
    model: Any = None,
    question: Optional[str] = None,
    knowledge_md: Optional[str] = None,
    schema_preview: Optional[str] = None,
    video_parts: Optional[list] = None,
    knowledge_max_chars: int = DEFAULT_KNOWLEDGE_MAX_CHARS,
    schema_max_chars: int = DEFAULT_SCHEMA_MAX_CHARS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    log_path: Optional[Path] = None,
) -> VideoResultResult:
    """对单个视频 task 判断"答案是否已在视频里 + 是否直接采纳"(三步串联).

    Args:
        task_dir: task 根目录(含 context/ 与 workdir/_video_frames/)。
        model: pydantic-ai 模型; None 时用 ``build_model_nothink()`` (线上同款 Qwen3.5)。
        question / knowledge_md / schema_preview / video_parts: 缺省时从 task_dir 装配。
        log_path: 若提供, 落盘三步的原始输出 JSON, 便于审计。

    Returns:
        VideoResultResult。无视频或预处理缺失时 has_video=False, decision=RECOMPUTE。
    """
    task_id = task_dir.name
    res = VideoResultResult(task_id=task_id)

    if not _has_video(task_dir):
        res.error = "no video"
        return res
    res.has_video = True

    if model is None:
        from tools_v2.general_tools import build_model
        model = build_model()  # think 模型: 识别+排诱饵+判去留是推理任务, think 是强项

    if question is None:
        question = _read_question(task_dir)
    if knowledge_md is None:
        knowledge_md = _read_knowledge(task_dir, knowledge_max_chars)
    if schema_preview is None:
        schema_preview = _structured_schema_preview(task_dir, schema_max_chars)
    if video_parts is None:
        video_parts = load_video_parts(task_dir)

    if not video_parts:
        res.error = "no video parts (未预处理?)"
        return res

    logs: dict = {}

    # ---- Step 1: EXTRACT (多模态) ----
    step1_agent = Agent(model=model, instructions=STEP1_INSTRUCTION, output_type=VideoClaim)
    step1_msg = [f"## question\n{question}\n\n## 视频内容(帧图+版面+旁白)如下:"] + list(video_parts)
    try:
        run1 = await asyncio.wait_for(step1_agent.run(step1_msg), timeout=timeout_s)
        claim: VideoClaim = run1.output
    except Exception as e:  # noqa: BLE001
        res.error = f"step1 failed: {type(e).__name__}: {e}"
        logs["step1_error"] = res.error
        _save_logs(log_path, logs)  # step1 失败轮也落盘, 便于排查是超时还是解析错
        return res
    logs["step1"] = claim.model_dump()
    logs["step1_trajectory"] = _capture_trajectory(run1)
    res.has_displayed_answer = bool(claim.has_displayed_answer)
    res.claimed_values = claim.claimed_values
    res.source_frames = claim.source_frames
    res.distractor_notes = claim.distractor_notes

    if not claim.has_displayed_answer:
        res.decision = "RECOMPUTE"
        res.confidence = 1.0
        res.reason = "视频未直接显示答案, 只给规则/口径, 需回表重算。"
        _save_logs(log_path, logs)
        return res

    # ---- Step 2: ALIGN (结构化 schema, 不含 doc) ----
    step2_agent = Agent(model=model, instructions=STEP2_INSTRUCTION, output_type=AnswerShape)
    step2_msg = (
        f"## question\n{question}\n\n"
        f"## 视频读出的候选答案(行优先)\n{json.dumps(claim.claimed_values, ensure_ascii=False)}\n\n"
        f"## knowledge.md\n{(knowledge_md or '(无)')[:knowledge_max_chars]}\n\n"
        f"## 结构化数据源 schema 预览(csv/json/db, 不含 doc)\n{schema_preview}"
    )
    try:
        run2 = await asyncio.wait_for(step2_agent.run(step2_msg), timeout=timeout_s)
        shape: AnswerShape = run2.output
    except Exception as e:  # noqa: BLE001
        # step2 失败不致命: 用 step1 原值兜底, 继续 step3
        logger.debug("[video_result] step2 failed: {}", e)
        logs["step2_error"] = f"{type(e).__name__}: {e}"
        shape = AnswerShape(target_columns=[], unit_consistent=True,
                            normalized_values=claim.claimed_values, note="step2 failed; fallback")
        run2 = None
    logs["step2"] = shape.model_dump()
    if run2 is not None:
        logs["step2_trajectory"] = _capture_trajectory(run2)
    res.target_columns = shape.target_columns
    res.normalized_values = shape.normalized_values or claim.claimed_values

    # ---- Step 3: DECIDE ----
    step3_agent = Agent(model=model, instructions=STEP3_INSTRUCTION, output_type=Decision)
    step3_msg = (
        f"## question\n{question}\n\n"
        f"## step1 视频抽取\n"
        f"has_displayed_answer=True\n"
        f"claimed_values={json.dumps(claim.claimed_values, ensure_ascii=False)}\n"
        f"distractor_notes={claim.distractor_notes}\n\n"
        f"## step2 对齐\n"
        f"target_columns={shape.target_columns}\n"
        f"unit_consistent={shape.unit_consistent}\n"
        f"normalized_values={json.dumps(shape.normalized_values, ensure_ascii=False)}\n\n"
        f"## 结构化 schema 预览(供你判断能否/是否需要重算)\n{schema_preview[:6000]}"
    )
    try:
        run3 = await asyncio.wait_for(step3_agent.run(step3_msg), timeout=timeout_s)
        dec: Decision = run3.output
    except Exception as e:  # noqa: BLE001
        res.error = f"step3 failed: {type(e).__name__}: {e}"
        logs["step3_error"] = res.error
        res.decision = "ADOPT_AND_VERIFY"  # 有答案但决策失败, 保守: 采纳+复核
        res.confidence = 0.3
        res.reason = "step3 失败, 保守采纳并建议复核。"
        _save_logs(log_path, logs)
        return res
    logs["step3"] = dec.model_dump()
    logs["step3_trajectory"] = _capture_trajectory(run3)

    decision = dec.decision.strip().upper()
    if decision not in ("ADOPT", "RECOMPUTE", "ADOPT_AND_VERIFY"):
        decision = "ADOPT_AND_VERIFY"
    res.decision = decision
    res.confidence = float(dec.confidence)
    res.reason = dec.reason
    _save_logs(log_path, logs)
    return res


def _capture_trajectory(run: Any) -> list:
    """把一次 agent.run 的完整对话历史序列化为可读 JSON.

    含 system / user / 模型 thinking 思维链 / 结构化输出工具调用. 多模态图片字节
    (BinaryContent) 会被替换成占位字符串, 避免日志里塞入 base64 把文件撑爆.
    """
    try:
        from agents_v2.agent_util import save_history  # noqa: F401  (确保依赖存在)
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        data = ModelMessagesTypeAdapter.dump_python(run.all_messages(), mode="json")
    except Exception as e:  # noqa: BLE001
        logger.debug("[video_result] dump trajectory failed: {}", e)
        return []

    def _scrub(obj):
        # 递归把 base64 图片数据替换成占位 (binary/image content 常见键: data / binary_content)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in ("data", "binary") and isinstance(v, str) and len(v) > 200:
                    out[k] = f"<binary {len(v)}B elided>"
                else:
                    out[k] = _scrub(v)
            return out
        if isinstance(obj, list):
            return [_scrub(x) for x in obj]
        if isinstance(obj, str) and len(obj) > 4000 and obj[:20].isalnum():
            # 兜底: 超长纯 base64 串 (无 key 包裹的图片) 也截断
            return f"<long-str {len(obj)}B elided>"
        return obj

    return _scrub(data)


def _save_logs(log_path: Optional[Path], logs: dict) -> None:
    if not log_path:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.debug("[video_result] save logs failed: {}", e)


async def run_video_result_voted(
    *,
    task_dir: Path,
    model: Any = None,
    vote_rounds: int = 5,
    log_dir: Optional[Path] = None,
    **kwargs,
) -> VideoResultResult:
    """多轮投票版: 并发跑 ``run_video_result_async`` N 次, 按多数票聚合.

    实测 (think 模型, 30 视频题): 单跑波动主要在边界题, 5 轮投票把 recall 稳到 ~0.8、
    且负类零误采纳。聚合规则:
      - has_displayed_answer: 多数为 true 才算 true (平票偏保守 false);
      - decision: 在 has_displayed_answer=true 的轮里取众数 (ADOPT/ADOPT_AND_VERIFY/RECOMPUTE);
        若多数轮判 false → 直接 RECOMPUTE;
      - claimed/normalized/target/source/distractor: 取"首个判 true 且 decision 与众数一致"的轮;
      - confidence: 取参与众数那批轮的均值。
    无视频时返回 has_video=False 的占位结果。
    """
    from collections import Counter

    n = max(1, int(vote_rounds))
    log_dir = Path(log_dir) if log_dir else None

    async def _one(i: int) -> VideoResultResult:
        lp = (log_dir / f"vote{i}.json") if log_dir else None
        return await run_video_result_async(task_dir=task_dir, model=model, log_path=lp, **kwargs)

    results = await asyncio.gather(*[_one(i) for i in range(n)], return_exceptions=True)
    oks = [r for r in results if isinstance(r, VideoResultResult)]

    def _write_result(agg: VideoResultResult, tally: dict) -> VideoResultResult:
        """把投票聚合后的最终结论 + 各轮票数落到 logs/<task>/video_result/result.json.

        这是排查时的"一眼看懂"文件: 不必翻 N 个 vote{i}.json, 直接看最终给 solver 的建议
        与投票分布。
        """
        if log_dir is not None:
            out = dict(agg.to_jsonable())
            out["vote"] = tally
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "result.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                logger.debug("[video_result] save result.json failed: {}", e)
        return agg

    agg = VideoResultResult(task_id=task_dir.name)
    if not oks:
        agg.error = "all vote rounds failed"
        return _write_result(agg, {"rounds_ok": 0, "rounds_total": n})
    agg.has_video = oks[0].has_video
    if not agg.has_video:
        agg.error = oks[0].error or "no video"
        return _write_result(agg, {"rounds_ok": len(oks), "rounds_total": n, "has_video": False})

    n_hda = sum(1 for r in oks if r.has_displayed_answer)
    vote_hda = n_hda > len(oks) / 2
    agg.has_displayed_answer = vote_hda

    # 各轮票数分布 (排查用)
    tally = {
        "rounds_ok": len(oks),
        "rounds_total": n,
        "has_displayed_answer_true": n_hda,
        "decision_counts": dict(Counter(r.decision for r in oks)),
        "per_round": [
            {"has_displayed_answer": r.has_displayed_answer,
             "decision": r.decision,
             "confidence": r.confidence,
             "claim_n_rows": len(r.claimed_values or []),
             "error": r.error}
            for r in oks
        ],
    }

    if not vote_hda:
        agg.decision = "RECOMPUTE"
        agg.confidence = round(sum(r.confidence for r in oks) / len(oks), 3)
        agg.reason = f"{n_hda}/{len(oks)} 轮认为视频直接显示了答案 (未过半), 判重算。"
        return _write_result(agg, tally)

    # 在判 true 的轮里取 decision 众数
    true_rounds = [r for r in oks if r.has_displayed_answer]
    dec_vote = Counter(r.decision for r in true_rounds).most_common(1)[0][0]
    picked = [r for r in true_rounds if r.decision == dec_vote]
    rep = picked[0]
    agg.decision = dec_vote
    agg.confidence = round(sum(r.confidence for r in picked) / len(picked), 3)
    agg.claimed_values = rep.claimed_values
    agg.normalized_values = rep.normalized_values
    agg.target_columns = rep.target_columns
    agg.source_frames = rep.source_frames
    agg.distractor_notes = rep.distractor_notes
    agg.reason = (f"{n_hda}/{len(oks)} 轮判定视频已显示答案; "
                  f"{len(picked)}/{len(true_rounds)} 轮决策为 {dec_vote}。{rep.reason}")
    return _write_result(agg, tally)
