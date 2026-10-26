"""Frame screenshot -> Hiccup layout tree, via the self-hosted Qwen3.5-VL endpoint.

Core, reusable pieces extracted from `codetest/t0608-qwen35-img2html.py`:
the layout prompt (`HICCUP_PROMPT`) and the single-frame request (`_call`).
Probe scripts and batch runners import these so prompt/endpoint/decoding live
in one place.

Endpoint + model match configs.pydantic_predefined_model (codelab_provider).
"""

from __future__ import annotations

import base64
import re
import os
import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
import httpx
from tools_v2.speed_runtime import limit_sync_client

from tools_v2.general_tools import get_model_config

_MODEL_CONFIG = get_model_config()
BASE_URL = _MODEL_CONFIG["base_url"]
API_KEY = _MODEL_CONFIG["api_key"]
MODEL = _MODEL_CONFIG["model_name"]

# Cookbook pixel budget (min/max are in pixels; 28*28 == one vision token block).
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

# One prompt, one recursive answer. The vocabulary is deliberately small so the
# model is nudged toward layout containers (:row/:col/:grid) instead of free-form
# div soup — that's what makes the nesting read as spatial relations.
HICCUP_PROMPT = (
    "把这张截图的版面布局描述成一棵 Hiccup 树（Clojure 的 "
    "[:tag {attrs} child ...] 标记字面量）。不要输出坐标——位置完全通过"
    "结构和子节点顺序来表达。\n"
    "\n"
    "容器（用来表达空间关系）：\n"
    "  [:page ...]            整个屏幕，根节点\n"
    "  [:row ...]             子节点从左到右并排\n"
    "  [:col ...]             子节点从上到下堆叠\n"
    "  [:card ...]            视觉上成组的面板 / 方框 / 区块\n"
    "叶子节点（实际内容，原文 OCR 文本作为最后一项）：\n"
    "  [:h1 \"..\"] [:h2 \"..\"] [:h3 \"..\"]   按显著程度区分的标题\n"
    "  [:p \"..\"]             段落 / 正文\n"
    "  [:label \"..\"] [:value \"..\"]        字段标签及其取值\n"
    "  [:button \"..\"] [:tag \"..\"]         按钮 / 标签片 / 徽章\n"
    "  [:icon \"..\"] [:other \"..\"]\n"
    "表格：[:table [:tr [:th \"..\"] ..] [:tr [:td \"..\"] ..] ..]（表头行用 "
    ":th）。各行列数保持一致；空单元格用 [:td \"\"]。柱状图、折线图、饼图也"
    "当作表格来转写：每个类别（柱子/折线点/扇形）占一行，类别标签和它旁边"
    "明文标注的数值放同一行；没有明文数值的单元格就留空 [:td \"\"]。\n"
    "\n"
    "任意节点都可加一个属性 map 来表示样式，例如 "
    "[:h1 {:color \"white\" :size \"large\"} \"标题\"]。color 是主要文字颜色；"
    "size 取 large|medium|small。无特别之处时省略属性。\n"
    "\n"
    "规则：嵌套 [:row]/[:col] 来还原真实排布（例如左侧边栏紧挨主面板 => "
    "[:row [:col ...侧边栏...] [:col ...主面板...]]）。只为确实可见的内容生成"
    "节点——不要用空的 [:p \"\"]/[:label \"\"] 节点凑数，也不要重复同一个节点或"
    "字符。所有可见文本一律按原文逐字转写——不要翻译或改写，也不要编造图上"
    "没有印出来的数值（尤其不要根据柱子长短、扇形大小去估算比例）。不要画"
    "图形：不用 :style 画柱子、不用方块字符，只转写文字和数值。只输出这棵 "
    "Hiccup 树，按层缩进以便阅读。不要加说明文字，不要 markdown code fence。"
)


def encode_image(path: Path) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _call(client: OpenAI, img_path: Path, prompt: str = HICCUP_PROMPT,
          nothink: bool = True, max_tokens: int = 3072, max_retries: int = 7):
    """One layout request: returns (raw_text, response).

    nothink=True (default) disables chain-of-thought so the whole budget goes to
    the tree, not <think>.

    A few frames (dense tables, step lists) tip greedy-ish decoding into a
    repetition loop that floods empty [:tr]/[:p ""] rows until the token budget
    runs out (finish_reason == "length"). temperature=0.3 makes each attempt
    differ, so simply re-sampling almost always escapes the loop — we retry up to
    max_retries times whenever a response comes back truncated, and keep the
    first clean ("stop") one. max_tokens is intentionally tight (real frames top
    out near ~1200 completion tokens): a smaller budget means a runaway loop hits
    "length" sooner, so we detect and retry faster instead of generating ~8k
    tokens of garbage first.
    """
    extra_body = {}
    if nothink:
        if os.environ.get('MODEL_THINKING_FORMAT') in {'ecnu', 'deepseek'}:
            extra_body = {'thinking': {'type': 'disabled'}}
        else:
            extra_body = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image(img_path)},
                    # "min_pixels": MIN_PIXELS,
                    # "max_pixels": MAX_PIXELS,
                },
            ],
        },
    ]
    resp = None
    for _ in range(max_retries + 1):
        resp = client.chat.completions.create(
            model=get_model_config()['model_name'],
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        if resp.choices[0].finish_reason != "length":
            break  # clean stop -> done; otherwise re-sample (temp>0 varies it)
    return resp.choices[0].message.content or "", resp


def clean_hiccup(raw: str) -> str:
    """清洗模型输出的 hiccup 文本(直接可喂给下游 LLM)。

    实测口径(对 experiments/video/slides_coarse_mq1.5 下 307 个输出审计):
      * 约 1/3 的输出被模型套了 markdown 代码围栏 ``` ```edn ... ``` ``` —— 去掉它;
      * 未见 CoT 泄漏、未见 token 截断(`_call` 已对截断重采样);
      * 偶发的多余尾括号(单个孤立 `]`/`)`)对 LLM 无害,且强行"截到第一棵树"有
        误删真实节点的风险(实测有节点被套在多余层里),故不做截断。
    因此只去围栏,不美化、不解析——LLM 读紧凑/半缩进的 hiccup 无障碍。
    """
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s).strip()
    return s


def frames_to_hiccup(
    frame_paths,
    *,
    client: OpenAI | None = None,
    workers: int = 8,
    prompt: str = HICCUP_PROMPT,
) -> dict[str, str]:
    """批量把帧 PNG 转成 hiccup 版面树,返回 ``{path: hiccup}`` 内存映射。

    所有帧抽完后统一调用(每帧一次 VL 请求,网络绑定,故并发)。结果留在内存供
    对齐/拼装等环节直接取用,不强制落盘。docker 提交时走 env 注入的同一 codelab
    端点(qwen3.5 30B-a3b),与 solver 主模型同源。

    Args:
        frame_paths: 帧 PNG 路径的可迭代(str 或 Path)。
        client: 复用的 OpenAI 客户端;``None`` 时按模块级 BASE_URL/API_KEY 现建。
        workers: 并发线程数(网络绑定,默认 8)。
        prompt: 版面 prompt(默认 :data:`HICCUP_PROMPT`)。

    Returns:
        ``{str(path): hiccup_text}``;单帧异常时该帧值为 ``""``(不中断整批)。
    """
    paths = [Path(p) for p in frame_paths]
    if client is None:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY,
                        http_client=limit_sync_client(httpx.Client(timeout=300)))

    def _one(p: Path) -> tuple[str, str]:
        try:
            raw, _ = _call(client, p, prompt)
            return str(p), clean_hiccup(raw)
        except Exception:
            return str(p), ""

    out: dict[str, str] = {}
    context = contextvars.copy_context()
    def _with_context(path):
        return context.copy().run(_one, path)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, hiccup in ex.map(_with_context, paths):
            out[path] = hiccup
    return out
