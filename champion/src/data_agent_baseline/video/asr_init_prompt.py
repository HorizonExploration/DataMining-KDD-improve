#!/usr/bin/env python
"""为 faster-whisper 生成 ``initial_prompt``（按音频语言、用 qwen3.5 写）。

核心思路（对应 ASR 评测计划 §5「调用 qwen3.5 生成 init prompt」）：
  阶段1 检测语言：用 ``detect_audio_language`` 对【音频本身】现场检测中/英
         （限定 zh/en，取概率大者），作为写 prompt 的语言锚点。
         —— 不读 gold、不看题目语言：评测已证实有 task 音频中文但题目英文，
            prompt 必须按【音频语言】写，否则显著变差。
  阶段2 写 prompt：把 task 的 question + knowledge.md 喂给 qwen3.5，按检测语言
         产出一句【含领域术语】的描述（中文音频→简体；英文→英文），作为
         ``initial_prompt`` 软引导用词、纠正同音字。

合法性：只用推理时真实可得的 question / knowledge.md，绝不碰 gold。
可复现：写 prompt 用的 meta-prompt（system+user 模板）即下方常量。

库用法（单条，供 ``video.prepare_video`` 调用）::

    from video.asr_init_prompt import gen_init_prompt
    prompt = gen_init_prompt(
        video_path="context/video/briefing.mp4",
        question=task_question,
        knowledge=knowledge_md_text,
    )
"""
from __future__ import annotations

# 语言检测复用转写后端选择（容器→local CPU、本机→remote GPU），保持同口径。
from video.audio_transcribe import detect_audio_language

# qwen 服务地址/模型/key 不再硬编码：统一从 general_tools.get_model_config() 取
# （env 驱动，docker 走注入的环境变量、本机走默认值），与 baseline 其余调用同口径。

KNOWLEDGE_MAX_CHARS = 6000  # knowledge.md 截断上限，防止上下文过长
QWEN_TIMEOUT = 60.0         # 单条 qwen 调用超时（秒）；endpoint 偶发单条 hang，超时即重试
QWEN_RETRIES = 3            # 单条最多重试次数
QWEN_MAX_TOKENS = 300       # 输出上限：prompt 只需一句，封顶防 runaway 生成

__all__ = ["gen_init_prompt", "write_init_prompt", "detect_audio_language"]

# ============================================================
# 写 init prompt 的 meta-prompt（即可复现的模板）
# ============================================================
SYSTEM_PROMPT = """\
你在为语音识别（ASR）写一句 initial_prompt，用来软引导 faster-whisper 转写一段配音。

【这段配音是什么】一个操作员对着数据平台界面、边点边讲解，最后引向某个具体问题的答案。\
他念的是【口语化的操作旁白】：界面上看到的字段名/看板名、正在做的动作、要对比的数值口径，\
而【不是】数据库的主题综述。你的 prompt 要尽量贴近"他嘴里真会蹦出来的那些词"。

【最重要的原则】以下面给出的【任务问题】为锚来选词——配音是为了回答这个问题而讲的，\
所以紧扣问题的那几个名词（看板名、关键字段、指标口径、筛选条件）几乎一定会被念到。\
领域知识（knowledge.md）只用来【查这些词的规范中文写法】，\
绝不要把整库的表/字段都罗列进来——和本题问题无关的字段，配音不会念，写进去就是噪声。

判断一个词该不该写，问自己："操作员盯着屏幕讲这道题时，会把这个词念出声吗？"
- 会念的（要写）：看板/面板名、被筛选的字段、对比的指标、动作类词（如加载、展开、切换、核对、排序）、本题特有的口径词。
- 不会念的（别写）：整库主题概述式的大词、与本题无关的其它业务模块、英文字段名/表名/代码标识、\
  视频制作/文件/编号/格式等元信息。

其它要求：
1. 只输出那一句 prompt 本身：一句通顺的话，不要分点、不要引号、不要解释或前后缀。
2. 只写正确的标准中文词；不要编造或推测"容易听错的写法"，也不要重复同一个词。
3. 词数控制在 8~15 个，少而准，宁缺毋滥（贴题的具体词 > 多而泛的大词）。
4. 语言：{lang_rule}
5. 句式示例（仅示意语气，词要换成紧扣本题的）：\
"这是操作员在数据看板上的讲解，涉及历史分红看板、送股、转增股、送股比重、实施日期、证券代码的核对与排序。"\
"""

USER_TEMPLATE = """\
下面是一段【操作讲解配音】对应的任务问题与领域知识。这段配音是操作员对着界面边操作边讲、\
最终引向该问题答案的旁白。请预测他口播会念到的词，写成一句 ASR initial_prompt。

【任务问题】（选词的锚——紧扣它的名词几乎一定会被念到）
{question}

【领域知识 knowledge.md】（只用来查上述名词的规范中文写法，不要把整库字段都搬进来）
{knowledge}

请直接输出那一句 initial_prompt（{lang_name}），紧扣任务问题选词、贴近口播实际用语。"""

LANG_RULE = {
    "zh": "音频是【中文】，提示词必须用【简体中文】（顺带引导模型输出简体），不得夹带英文。",
    "en": "The audio is in ENGLISH; write the prompt in ENGLISH only. Anchor the wording to the "
          "task question — the operator narrates while clicking through the UI toward that answer, "
          "so name the dashboard/panel, the fields being filtered, the metrics compared, and the "
          "actions taken. Do NOT list the whole schema or use raw column/table identifiers "
          "like innercode or mf_xxx. Keep it to one natural sentence, 8-15 terms.",
}
LANG_NAME = {"zh": "简体中文", "en": "English"}

# 通用 UI/操作术语 heuristic 段（仅追加到【中文】prompt 末尾）。
# 数据驱动选出：在 ≥6 个中文 task 出现的高频通用词，且易被同音字替换。
# 实测把"字段→自断"等跨 task 系统性同音错治掉，中文 CER 0.0277→0.0191（-31%）。
# 详见 experiments/asr/RESULT_ui_terms.md。
UI_TERMS_ZH = "界面常见术语包括字段、批次、配置、加载、面板、看板、卡片、队列、筛选、核对、列、口径。"


def _clean_prompt(prompt: str, lang: str) -> str:
    """清洗模型输出，并对中文 prompt 追加通用 UI 术语段。

    - 去掉模型偶尔包的首尾引号 / 前缀。
    - 中文：末尾补句号后追加 ``UI_TERMS_ZH``（治"字段→自断"等跨 task 同音错，-31% CER）。
    """
    prompt = prompt.strip().strip('"').strip("“”").strip()
    if lang == "zh":
        if not prompt.endswith(("。", ".")):
            prompt += "。"
        prompt += UI_TERMS_ZH
    return prompt


def _one_prompt(client, qwen_model: str, system: str, user: str) -> str:
    """单条 qwen 调用，带超时 + 重试（endpoint 偶发单条 hang，见排查记录）。"""
    last_err = None
    for attempt in range(QWEN_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=qwen_model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=QWEN_MAX_TOKENS,  # 防 runaway 生成把单条拖到超长
                timeout=QWEN_TIMEOUT,        # 单条超时，超了重试而非永久阻塞
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # 超时/连接/5xx 都重试
            last_err = e
            print(f"      ⚠ 第 {attempt + 1}/{QWEN_RETRIES} 次失败: {str(e)[:120]}", flush=True)
    raise RuntimeError(f"qwen 调用连续 {QWEN_RETRIES} 次失败: {last_err}")


def _truncate_knowledge(knowledge: str | None) -> str:
    """knowledge.md 截断到 KNOWLEDGE_MAX_CHARS，防止上下文过长。"""
    if not knowledge:
        return "(无 knowledge.md)"
    if len(knowledge) > KNOWLEDGE_MAX_CHARS:
        return knowledge[:KNOWLEDGE_MAX_CHARS] + "\n…(已截断)"
    return knowledge


def write_init_prompt(
    question: str,
    knowledge: str | None,
    lang: str,
    *,
    qwen_endpoint: str | None = None,
    qwen_model: str | None = None,
    client=None,
) -> str:
    """阶段2：已知音频语言，用 qwen3.5 写一句 ASR ``initial_prompt``。

    Args:
        question: 任务问题（选词的锚）。
        knowledge: knowledge.md 文本（仅查名词规范写法，会被截断）；可为 None。
        lang: 音频语言 ``"zh"`` / ``"en"``（由 :func:`detect_audio_language` 得到）。
        qwen_endpoint / qwen_model: 远程 qwen 服务；``None`` 时从
            ``general_tools.get_model_config()`` 取（env 驱动，与 baseline 同口径）。
        client: 可选，复用的 ``openai.OpenAI`` 客户端（批量时传入省去重复建连）。

    Returns:
        清洗后的单句 prompt（中文音频已追加 ``UI_TERMS_ZH`` 段）。
    """
    from openai import OpenAI
    import httpx
    from tools_v2.speed_runtime import limit_sync_client

    from tools_v2.general_tools import get_model_config

    if lang not in LANG_RULE:
        lang = "zh"
    knowledge = _truncate_knowledge(knowledge)
    if client is None:
        cfg = get_model_config()
        client = OpenAI(
            base_url=qwen_endpoint or cfg["base_url"],
            api_key=cfg["api_key"],
            timeout=QWEN_TIMEOUT,
            http_client=limit_sync_client(httpx.Client(timeout=QWEN_TIMEOUT)),
        )
    if qwen_model is None:
        qwen_model = get_model_config()["model_name"]
    system = SYSTEM_PROMPT.format(lang_rule=LANG_RULE[lang])
    user = USER_TEMPLATE.format(question=question, knowledge=knowledge,
                                lang_name=LANG_NAME[lang])
    prompt = _one_prompt(client, qwen_model, system, user)
    return _clean_prompt(prompt, lang)


def gen_init_prompt(
    video_path,
    question: str,
    knowledge: str | None = None,
    *,
    qwen_endpoint: str | None = None,
    qwen_model: str | None = None,
    lang: str | None = None,
) -> str:
    """端到端为【一段音频/视频】生成 ASR ``initial_prompt``（检测语言 + 写 prompt）。

    这是 ``video`` 子包调用的主入口：先按【音频本身】检测中/英
    （除非显式传 ``lang`` 锚定），再用 qwen3.5 按该语言写一句紧扣 ``question``
    的软引导 prompt。

    Args:
        video_path: 视频/音频路径（``decode_audio`` 直接抽音频）。
        question: 任务问题。
        knowledge: knowledge.md 文本（可选）。
        qwen_endpoint / qwen_model: 远程 qwen 服务（阶段2 写 prompt）；``None`` 时从
            ``general_tools.get_model_config()`` 取（env 驱动）。
        lang: 显式锚定语言，跳过检测；None 表示按音频现场检测。

    Returns:
        单句 ``initial_prompt``，可直接传给 ``video.transcribe_video(initial_prompt=...)``。
    """
    if lang is None:
        lang = detect_audio_language(video_path)["lang"]
    return write_init_prompt(
        question, knowledge, lang,
        qwen_endpoint=qwen_endpoint, qwen_model=qwen_model,
    )
