# ============================================================
# solver SQL 引擎统一使用 DuckDB (与 explore_data 同引擎).
# ============================================================


# 应用 hashline monkey patch
from agents_v2 import hashline_patch  # noqa: F401

from pydantic_deep import create_default_deps

from pydantic_deep import create_summarization_processor

from tqdm import tqdm
import asyncio
import json
import os
from pathlib import Path
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.usage import RunUsage, UsageLimits

from tools_v2 import general_tools
from tools_v2.timing import TaskTimer, format_timing_summary
from loguru import logger  # 若需要提交时不展示, 使用 debug

from pydantic_ai import RunContext
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models import ModelRequestContext


# ---- 轮数实时打印 ----
# 每次 agent 向模型发起请求 (即新的一轮 ReAct) 前自增并打印.
# 并行场景下每个 task 创建一组独立的 Hooks() + 计数器, 通过闭包绑定 task_id,
# 避免日志/计数互相干扰.


def _make_round_hooks(task_id: str):
    """为单个 task 构造一组独立的 round hooks (闭包持有计数器和 task_id)."""
    import time as _t
    counter = {"n": 0, "t_start": 0.0, "round_times": []}
    hooks = Hooks()

    @hooks.on.before_model_request
    async def _log_round(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext:
        counter["n"] += 1
        counter["t_start"] = _t.perf_counter()
        logger.info(
            "[{}] 🔄 agent round #{} (messages_in_context={})",
            task_id,
            counter["n"],
            len(request_context.messages),
        )
        return request_context

    @hooks.on.after_model_request
    async def _log_tool_calls(
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response,
    ) -> object:
        round_elapsed = _t.perf_counter() - counter["t_start"]
        counter["round_times"].append(round_elapsed)
        tool_names = _extract_tool_names_from_response(response)
        if tool_names:
            logger.info(
                "[{}] 🛠️ round #{} {:.1f}s tools: {}",
                task_id,
                counter["n"],
                round_elapsed,
                ",".join(tool_names),
            )
        else:
            logger.info(
                "[{}] 🛠️ round #{} {:.1f}s tools: -", task_id, counter["n"], round_elapsed
            )
        return response

    # 暴露统计(调用方可通过 hooks._round_stats 取)
    hooks._round_stats = counter  # type: ignore[attr-defined]
    return hooks


def _extract_tool_names_from_response(response) -> list[str]:
    """从模型响应中提取本轮实际产生的工具调用名."""
    tool_names: list[str] = []
    for part in getattr(response, "parts", None) or []:
        name = (
            getattr(part, "tool_name", None)
            or getattr(part, "name", None)
            or getattr(part, "tool", None)
        )
        if isinstance(name, str) and name:
            tool_names.append(name)
    seen: set[str] = set()
    unique_names: list[str] = []
    for n in tool_names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)
    return unique_names


import tools_v2
import tools_v2.describe_tool
from tools_v2.scaffold_tool import generate_solver_scaffold
from agents_v2.solver_agent import build_solver_agent
from agents_v2.pre_agent import run_pre_agent
from agents_v2.agent_util import save_history, build_workdir_local_backend
from doc_tools.doc_prepare import prepare_doc_tables_async, render_solver_hint
from agents_v2.doc_relevance_agent import run_doc_relevance_async
from agents_v2.table_relevance_agent import run_table_relevance_async
from agents_v2.video_result_agent import run_video_result_voted
from video.build_video_input import video_parts_for_task

## 常量定义

# 并行执行 task 的最大并发数, 通过 semaphore 限制
MAX_PARALLEL_TASKS = 16  # 只影响本地, docker由dockerfile环境变量控制

# pre-agent 多轮投票次数 (>=1, <=1 时不投票直接单轮)
PRE_AGENT_VOTE_ROUNDS = 3

# 视频结果判定 (video_result_agent): 判断"答案是否已直接显示在视频里, 是否建议直接采纳".
# 仅产出**建议**注入 solver prompt, 最终是否采纳由 solver agent 自行决定。<=0 时关闭。
VIDEO_RESULT_VOTE_ROUNDS = 5

# solver agent 最大请求轮数
SOLVER_REQUEST_LIMIT = 80

# solver 单 task 最大 attempt 次数 (只有 run_ok 且产出 prediction 才算真结束,
# 失败的 attempt 会继续重试, 全部失败才用残留脏 csv / solver.py 兜底)
MAX_ATTEMPTS = 5

# 结构化表相关性判定开关 (table_relevance): 默认开,默认折叠与任务无关的表的描述
ENABLE_TABLE_RELEVANCE = True


if general_tools.is_docker():
    logger.info('docker runtime, init model from os env')
else:
    logger.info('local runtime, init model from general_tools defaults/env')

# 注意: MODEL / MODEL_NOTHINK 不再在模块级构造, 而是每个 task 进入
# process_one_task 时各自构造一次. 这样配合 general_tools._next_base_url 的
# 多 endpoint round-robin (设 MODEL_API_URLS), 可把不同 task 摊到不同推理机;
# 单 endpoint (docker 提交端只注入 MODEL_API_URL) 时行为与原来等价.


def aggregate_run_stats(messages: list) -> dict:
    """从 captured_msgs 中统计 react 轮数和 token 使用量.

    与 ``result.usage()`` 路径一致, 但只依赖消息列表, 因此在 agent.run
    抛异常的兜底路径上同样可用. 一个 ModelResponse 视为一轮.
    """
    usage = RunUsage()
    turns = 0
    for msg in messages:
        if isinstance(msg, ModelResponse):
            turns += 1
            usage.incr(msg.usage)
    return {
        'turns': turns,
        'input_tokens': usage.input_tokens,
        'output_tokens': usage.output_tokens,
        'cache_read_tokens': usage.cache_read_tokens,
        'cache_write_tokens': usage.cache_write_tokens,
        'total_tokens': usage.total_tokens,
        'requests': usage.requests,
    }


async def process_one_task(
    task_id_p,
    *,
    semaphore: asyncio.Semaphore,
    input_base_dir: Path,
    output_base_dir: Path,
    temp_dir: Path,
    logs_base_dir: Path,
) -> dict | None:
    async with semaphore:
        cur_task_id = task_id_p.name
        general_tools.set_log_task_id(cur_task_id)
        timer = TaskTimer(cur_task_id)

        # 每个 task 各自构造 model: 配合多 endpoint round-robin 把不同 task 摊到
        # 不同推理机 (单 endpoint 时退化为与模块级单例等价).
        MODEL = general_tools.build_model()
        MODEL_NOTHINK = general_tools.build_model_nothink()

        # prepare temp dir
        task_temp_dir = temp_dir / cur_task_id
        task_out_dir = output_base_dir / cur_task_id
        task_log_dir = logs_base_dir / cur_task_id

        try:
            # general_tools.print_dir_permissions(task_temp_dir, task_out_dir, task_log_dir)
            # parents=True: 父目录(temp_dir/output_base_dir/logs_base_dir)理应已在 main
            # 里建好, 但带上 parents 可在父目录被中断/清理时自愈, 避免单个 task 因
            # FileNotFoundError 整体崩溃.
            task_temp_dir.mkdir(parents=True, exist_ok=True)
            task_out_dir.mkdir(parents=True, exist_ok=True)
            task_log_dir.mkdir(parents=True, exist_ok=True)
            general_tools.print_dir_permissions(task_temp_dir, task_out_dir, task_log_dir)
        except PermissionError as e:
            logger.error('ERROR: cannot create dirs for {}, skipping: {}', cur_task_id, e)
            return

        task_raw_input_dir = input_base_dir / cur_task_id

        # 将 task_raw_input_dir 中所有内容"复制"到 task_temp_dir.
        # copy_function 用 clone_or_copy: 本地 macOS(APFS) 走 clonefile 写时复制,
        # 既不占额外存储又与源数据物理隔离; docker 下回退普通 copy2, 行为不变.
        import shutil
        with timer.stage("copytree"):
            if task_temp_dir.exists():
                shutil.rmtree(task_temp_dir)
            shutil.copytree(
                task_raw_input_dir,
                task_temp_dir,
                copy_function=general_tools.clone_or_copy,
            )

        # 创建工作目录
        task_work_dir = task_temp_dir / 'workdir'
        task_work_dir.mkdir(exist_ok=True)

        task_dir = task_temp_dir
        task_json = task_dir / 'task.json'
        task_desc = json.loads(task_json.read_text())
        task_question = task_desc['question']
        logger.debug(f'Now process {cur_task_id} | {task_question}')

        # knowledge.md 提前读: 视频 ASR initial_prompt、doc 相关性判定、solver instruction
        # 都要用它. (read_knowledge 走 hashline 渲染, 与后续注入 solver 的口径一致.)
        knowledge_md = tools_v2.describe_tool.read_knowledge(task_dir, 'context')

        # ---- 视频预处理 (必须在 doc 相关性判定之前) ----
        # 抽幻灯片帧 + ASR 转写 + hiccup 版面, 按时间轴交错成多模态 parts.
        # 同时把 video_input.json / timeline.txt 落到 workdir/_video_frames/,
        # 供下面 doc_relevance_agent 读取视频文本(旁白+hiccup)判断该用哪些 doc.
        # 无视频时返回 [], 不影响后续. 任意失败均 warning 兜底, 不阻塞主流程.
        video_parts = []
        try:
            with timer.stage("video"):
                from tools_v2.speed_runtime import run_video
                video_parts = await run_video(video_parts_for_task,
                    task_dir,
                    question=task_question,
                    knowledge=knowledge_md,
                    log_dir=task_log_dir,
                )
            if video_parts:
                logger.info('[{}] video preprocessed: {} parts',
                            cur_task_id, len(video_parts))
        except Exception as e:
            logger.exception('task {} video preprocess failed (skip): {}', cur_task_id, e)

        # ---- 视频结果判定 (video_result_agent) ----
        # 在视频预处理后跑: 用 think 模型多轮投票判断"question 的最终答案是否已**直接显示**
        # 在视频面板里(可跨帧按排名/行号对齐拼装), 以及是否建议直接采纳。本步**只产出建议**,
        # 注入到 solver 的 prompt 里(见下方 video_result_hint), 最终是否采纳、怎么用,
        # 完全由 solver agent 自己决定。任何失败都不阻塞主流程(hint 留空)。
        video_result = None
        if video_parts and VIDEO_RESULT_VOTE_ROUNDS > 0:
            try:
                with timer.stage("video_result"):
                    video_result = await run_video_result_voted(
                        task_dir=task_dir,
                        vote_rounds=VIDEO_RESULT_VOTE_ROUNDS,
                        log_dir=task_log_dir / 'video_result',
                    )
                logger.info('[{}] video_result: decision={} hda={} conf={}',
                            cur_task_id, video_result.decision,
                            video_result.has_displayed_answer, video_result.confidence)
            except Exception as e:
                logger.exception('task {} video_result failed (skip): {}', cur_task_id, e)
                video_result = None

        # ---- doc 相关性判定 (在 doc_prepare 之前) ----
        # 用默认模型横向比较 question/knowledge/视频口径 与每个 doc 的内容预览, 判定
        # 哪些 doc 解题确需. 只把相关 doc 交给下面的 doc_prepare 抽取, 跳过无关 doc,
        # 以减少昂贵的 md→结构化 db 抽取量. 视频文本由 agent 内部从上一步落盘的
        # workdir/_video_frames/video_input.json 自动读取(含 hiccup).
        # 判定失败/超时时内部按召回优先全判相关兜底, 不会漏抽. relevant_stems=None
        # 表示不过滤(doc_relevance 整体失败时的安全降级 → 退回抽全部).
        relevant_stems = None
        try:
            with timer.stage("doc_relevance"):
                rel = await run_doc_relevance_async(
                    model=MODEL_NOTHINK,
                    task_dir=task_dir,
                    question=task_question,
                    knowledge_md=knowledge_md,
                    log_path=task_log_dir / 'doc_relevance_allmsg.json',
                )
            if rel.all_doc_stems:
                relevant_stems = set(rel.relevant_docs)
                logger.info(
                    'doc_relevance ok: {}/{} doc 相关, 跳过 {}',
                    len(relevant_stems), len(rel.all_doc_stems), rel.skipped_docs,
                )
        except Exception as e:
            logger.exception('doc_relevance failed for {}: {}', cur_task_id, e)

        # ---- doc 预抽取 (doc_pipeline) ----
        # 在 scaffold 之前: 把 context/doc/*.md 抽成结构化 sqlite,
        # 落盘到 context/db/_doc_extracted.db. scaffold_tool 后续扫描
        # context/db/*.db 时会自动把这些表加到 solver.py 的读数据 block.
        # relevant_stems 限定只抽取相关 doc (pdf reflow 仍对全部 pdf 执行).
        # 任意失败均 warning, 不阻塞主流程.
        doc_prep = None
        try:
            with timer.stage("doc_prepare"):
                doc_prep = await prepare_doc_tables_async(
                    task_dir=task_temp_dir,
                    log_dir=task_log_dir,
                    model_nothink=MODEL_NOTHINK,
                    model_think=MODEL,  # Phase 5.5 写 transform 是推理任务, 用 think 模型
                    relevant_stems=relevant_stems,
                )
            if doc_prep.db_path is not None:
                logger.info(
                    'doc_prepare ok: {} tables -> {}',
                    doc_prep.n_tables,
                    doc_prep.db_path,
                )
        except Exception as e:
            logger.exception('doc_prepare failed for {}: {}', cur_task_id, e)

        # 生成 solver.py 脚手架（固定路径、读数据逻辑、保存方式），
        # 让 agent 在已有骨架上填查询逻辑，避免路径/import 之类低级错误
        scaffold_pristine = None  # 脚手架原文快照, 供重试时把 solver.py 恢复到初始状态
        try:
            with timer.stage("scaffold"):
                scaffold_path = generate_solver_scaffold(task_temp_dir)
            logger.info('scaffold generated: {}', scaffold_path)
            # 快照原文: 每个 attempt 都会在上一次可能被改坏的 solver.py 上重跑,
            # 用它在重试前把 solver.py 恢复到"刚生成的脚手架"状态 (见 attempt 循环开头).
            try:
                scaffold_pristine = scaffold_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.warning('snapshot scaffold text failed for {}: {}', cur_task_id, e)
        except Exception as e:
            logger.error('scaffold generation failed for {}: {}', cur_task_id, e)
            scaffold_path = None

        # ---- 结构化表相关性判定 (在 describe_context_dir 之前) ----
        # 一个 task 的 context 往往挂十几~二十几张 csv/json/db 表, 但真正解题用到的中位仅
        # 1 张 (无关表占比中位 93%). 这些无关表的 schema 会被 describe_context_dir 全量拼进
        # solver prompt (实测中位 ~1.5万 token), 既费 token 又是噪声. 这里用默认模型横向比较
        # question/knowledge/视频口径 与每张表的 schema 预览, 判定哪些无关, 仅折叠其 describe.
        #
        # 软过滤 (关键): 判定结果**只用于折叠 prompt 描述** —— explore_data 仍注册全部表、
        # scaffold 仍 load 全部表, 故即便误折叠某张相关表, solver 仍能 SHOW TABLES / 查 schema
        # 找回 (代价从"答错"降为"多探一步"). _doc_extracted 的表归 doc_relevance 管, 不判定.
        # 判定失败/超时内部按召回优先全判相关兜底; collapse_keys=None 表示不折叠 (安全降级).
        collapse_keys = None
        if ENABLE_TABLE_RELEVANCE:
            try:
                with timer.stage("table_relevance"):
                    trel = await run_table_relevance_async(
                        model=MODEL_NOTHINK,
                        task_dir=task_dir,
                        question=task_question,
                        knowledge_md=knowledge_md,
                        log_path=task_log_dir / 'table_relevance_allmsg.json',
                    )
                if trel.all_tables:
                    collapse_keys = set(trel.skipped_describe_keys)
                    logger.info(
                        'table_relevance ok: {}/{} 表相关, 折叠 {} 张 (省 describe), skipped={}',
                        len(trel.relevant_tables), len(trel.all_tables),
                        len(trel.skipped_tables), trel.skipped_tables,
                    )
            except Exception as e:
                logger.exception('table_relevance failed for {}: {}', cur_task_id, e)

        with timer.stage("build_agent"):
            local_backend = build_workdir_local_backend(task_dir)

            # workdir + scaffold 已就绪, 此时再创建 solver agent,
            # 这样 build_solver_agent 内部能扫描到真实目录结构, 把
            # 真实目录树注入 system_instruction.
            agent = build_solver_agent(task_dir, model=MODEL)

            # 注入轮数打印 hook (每个 task 一组独立 hooks + 计数器, 闭包持有 task_id).
            agent._root_capability.capabilities.append(_make_round_hooks(cur_task_id))

            desc_str = tools_v2.describe_tool.describe_context_dir(
                task_dir, 'context', skip_knowledge=True, collapse_keys=collapse_keys,
            )

        # # pre agent: 从 knowledge.md 抽取与 question 相关的 use cases
        # pre_agent_out = await run_pre_agent(
        #     model=MODEL,
        #     task_question=task_question,
        #     knowledge_md=knowledge_md,
        #     # skill_info=skill_info,
        #     log_path=task_log_dir / 'pre_allmsg.json',
        #     vote_rounds=PRE_AGENT_VOTE_ROUNDS,
        # )


        video_rules = (
            "\n\n**视频题规则 (任务若附带 briefing.mp4 操作讲解视频时务必逐条遵守)**:\n"
            "user 消息会按时间轴交错给出【幻灯片帧】(图片 + hiccup 版面树) 与【对应旁白】"
            "(ASR 转写). 视频用于补充任务背景与口径, 但最终答案仍须用 solver.py 对结构化数据分析得出.\n"
            "1. **以图像帧 + hiccup 版面树为准, 旁白仅作参考**: ASR 转写由模型生成,"
            " 同音错字/漏字/口误很多 (如把字段名/批次号转错), 凡是与画面/hiccup 冲突, "
            "一律以图像帧和 hiccup 树为准, 不要被旁白文字带偏.\n"
            "2. **配置面板 = 筛选规则的权威来源**: 视频中 '配置面板 / 筛选配置 / "
            "SYSTEM CONFIGURATION / 准入线 / 门槛设置' 等帧上显示的 (字段, 运算符, 阈值) "
            "三元组就是 SQL WHERE 条件. 严格按面板上的运算符 `<` / `>` / `<=` / `>=` / `=` "
            "实现, **不要凭语义把 `<` 改成 `<=`, 也不要把 `>` 改成 `>=`**.\n"
            "3. **阈值被 mask 时去边界帧反推**: 若配置面板把阈值显示为 `...` / `***` / "
            "'已锁定', 必须在同一视频里找 '边界样例 / 边界样本核对 / 准入线附近记录' 等帧, "
            "通过'已纳入 / 未纳入'两侧样例的具体数值反推真实阈值 (如样例 A=91.54亿未过, "
            "样例 B=114.75亿已过 → 阈值落在 100亿附近, 同帧通常会显式标出 '门槛线 100.00亿'). "
            "**这种情况下边界帧不是干扰帧, 是阈值的唯一来源**.\n"
            "4. **旁白里的数字常是干扰**: 旁白通过ASR转写,噪音很多,其中的数字不可信, 不要轻易采信旁白里的数值信息 (如字段名/批次号等), 以免被干扰带偏.\n"
            "它们只是讲解过程中的过渡叙述, **不是答案也不是阈值**. "
            "只有出现在 '配置面板 / 筛选规则 / 导出队列 / 准入线' 等图像中的数值才是真正的规则.\n"
        )

        # ---- video_result 建议 (advisory, solver 拥有最终决定权) ----
        # 把上一步 video_result_agent 的判定结论, 作为**一条建议**写进 solver instruction。
        # 明确告诉 solver: 这是一个独立前置 agent 多轮投票得出的判断, 不是 ground truth;
        # solver 应结合自己对视频帧/结构化数据的理解, 自行决定是否采纳、是否复核。
        video_result_hint = ""
        if video_result is not None and video_result.has_video and not video_result.error:
            import json as _json
            dec = video_result.decision
            if dec in ("ADOPT", "ADOPT_AND_VERIFY") and video_result.has_displayed_answer:
                # 注入用 claimed_values (step1 原值) 优先: 实测 step2 的"对齐/归一"偶尔会把
                # step1 正确读出的多列答案塌成单列 (如 task_7 董秘姓名+电话被塌成只剩姓名),
                # 反而把对的答案改错。仅当 step2 明确判单位不一致时, normalized 才更可信 ——
                # 但那种情况此处暂不区分, 一律以 step1 原始 claimed 为准, 把"是否归一"留给 solver。
                claimed = _json.dumps(
                    video_result.claimed_values or video_result.normalized_values,
                    ensure_ascii=False,
                )
                act = ("该 agent 建议**直接采纳视频里的答案**(无需回表重算)。"
                       if dec == "ADOPT" else
                       "该 agent 建议**采纳视频里的答案, 但用结构化 SQL 低成本复核一致后再输出**。")
                video_result_hint = (
                    "\n\n**视频答案预判 (来自独立的 video_result 前置 agent, 多轮投票; 仅供参考, 你拥有最终决定权)**:\n"
                    f"- 推理路径: 该 agent 通览视频帧+版面+旁白, 解析 question 目标形态 → 定位被锁定/选中的作用域 "
                    f"→ 在该作用域下定位结果 → 跨帧按排名/行号对齐 → 排除诱饵帧 → 完整性自检。\n"
                    f"- 判定: 视频面板**已直接显示**本题最终答案 (置信度 {video_result.confidence})。{act}\n"
                    f"- 读出的候选答案(行优先, 列序对齐 question 要求): {claimed}\n"
                    f"- 目标输出列建议: {video_result.target_columns}\n"
                    f"- 关键帧: {video_result.source_frames}\n"
                    f"- 已排除的诱饵: {video_result.distractor_notes or '(无)'}\n"
                    "- **如何使用**: 若你核对视频帧后认同该判断, 可直接把上面的候选答案写进 prediction.csv "
                    "(ADOPT_AND_VERIFY 时再用结构化数据复核一次); 若你发现该判断与画面/口径不符(如作用域选错、"
                    "诱饵未排净、单位不一致、答案不完整), **以你自己的分析为准, 忽略本建议并照常用 solver.py 计算**。\n"
                )
            else:
                # 判 RECOMPUTE / 未显示答案: 明确告诉 solver 不要指望视频直接给答案
                video_result_hint = (
                    "\n\n**视频答案预判 (来自独立的 video_result 前置 agent; 仅供参考)**:\n"
                    f"- 判定: 视频**未直接显示**本题最终答案 (只给了规则/配置/分段/口径, 置信度 {video_result.confidence})。\n"
                    "- 含义: 不要试图从视频面板里直接抄答案; 请按视频里的规则/口径, 用 solver.py 对结构化数据计算得出最终结果。\n"
                )

        agent_instruction = "\n\n" \
         + f'`context/knowledge.md`描述任务涉及的字段和相关知识,内容如下:\n' + knowledge_md \
         + video_rules \
         + video_result_hint \
         # + f"\n\n{skill_info}\n"

        context_info = 'You have following context files:\n\n' + desc_str + '\n\n'

        # doc 抽取的表暴露给 agent (告诉它 context/db/_doc_extracted.db 里有什么,
        # 以及 __line_no/__line_nos 怎么用). 没 doc 时 doc_db_hint 为空字符串.
        doc_db_hint = ""
        if doc_prep is not None:
            doc_db_hint = render_solver_hint(doc_prep)
            if doc_db_hint:
                doc_db_hint = "\n\n" + doc_db_hint + "\n"

        scaffold_hint = ""
        if scaffold_path is not None:
            scaffold_hint = (
                "\n\n注意:`workdir/solver.py` 已经预生成了脚手架代码,"
                "包含正确的输入输出路径、所有数据文件的读取逻辑(已加载为 DataFrame)、"
                "以及最终保存到 `workdir/prediction.csv` 的样板. "
                "**请直接在该文件已有的 `## 进行查询` 段补全 SQL 逻辑并替换 `pred_df`**, "
                "不要修改路径定义、读数据 block 和最后的保存行,也不要新建 solver.py.\n"
            )
        user_input = (
            f"{context_info}"
            f"{doc_db_hint}"
            f"通过完成 `solver.py` 完成任务:\n{task_question}\n"
            # f"这是从knowledge.md中预抽取的与任务有关的use cases,仔细理解这些示例中的术语和方法,有助于完成任务.\n {pre_agent_out.model_dump()}"
            f"{scaffold_hint}"
        )

        # ---- 视频多模态输入拼装 ----
        # 视频已在前面预处理完毕 (video_parts). 这里把它按"帧在前、对应旁白紧随"
        # 交错前置到 user_input 之前 (图片+版面+旁白). 无视频时 video_parts 为 [],
        # final_input 保持纯文本不变.
        final_input = user_input
        if video_parts:
            video_intro = (
                "任务附带一段操作讲解视频(briefing.mp4). 下面按时间轴交错给出"
                "【幻灯片帧】(图片 + 其版面结构 hiccup 树)与【对应旁白】(ASR 转写),"
                "帧在前、讲解该帧的旁白紧随其后. 视频解读规则见 system instruction "
                "中的 '视频题规则'.\n"
            )
            final_input = [video_intro, *video_parts, "\n\n", user_input]
            logger.info('[{}] video input attached: {} parts',
                        cur_task_id, len(video_parts))

        deps = create_default_deps(local_backend)
        prediction_file = task_work_dir / 'prediction.csv'

        for attempt in range(1, MAX_ATTEMPTS + 1):
            # 重试前把 solver.py 恢复到脚手架初始状态: 上一次失败的 attempt 可能已把
            # solver.py 改成语法错/半成品, 若不恢复, scaffold_hint 里"solver.py 是刚
            # 预生成的脚手架"这句就成了假话, agent 会在坏文件上继续填. attempt 1 无需恢复
            # (刚生成过). scaffold_pristine 为 None (生成/快照失败) 时跳过, 不阻塞.
            if attempt > 1 and scaffold_pristine is not None and scaffold_path is not None:
                try:
                    scaffold_path.write_text(scaffold_pristine, encoding='utf-8')
                    # 同步清掉 _schema_shown.flag: 该 flag 由 solver.py 首次执行时写入,
                    # 用于后续执行跳过重复打印 schema. 恢复脚手架相当于"重新开始",
                    # 若不清掉, 本 attempt 首跑就不会打印 schema, 与初始状态不一致.
                    schema_flag = scaffold_path.parent / '_schema_shown.flag'
                    schema_flag.unlink(missing_ok=True)
                    logger.info('task {} attempt {}: solver.py restored to pristine scaffold (schema flag cleared)',
                                cur_task_id, attempt)
                except Exception as e:
                    logger.warning('task {} attempt {}: restore scaffold failed: {}',
                                   cur_task_id, attempt, e)
            # 删掉上一次残留的 prediction.csv: 这样本次 attempt run 之后的
            # prediction_file.exists() 只反映"本次真正产出了新预测", break 条件
            # (run_ok and exists) 才 sound —— 否则可能把上一次失败 attempt 的脏 csv
            # 误当成本次成功结果交付. 全失败兜底靠子进程重跑脚手架 solver.py, 不依赖残留.
            prediction_file.unlink(missing_ok=True)
            run_ok = False
            with capture_run_messages() as captured_msgs:
                try:
                    with timer.stage(f"agent_run.attempt{attempt}"):
                        result = await agent.run(
                            final_input,
                            instructions=agent_instruction,
                            deps=deps,
                            usage_limits=UsageLimits(request_limit=SOLVER_REQUEST_LIMIT),
                        )

                    result_output = result.output
                    run_ok = True

                    logger.debug('agent result output: {}', result_output)
                except Exception as e:
                    logger.exception('task {} attempt {} failed: {}', cur_task_id, attempt, e)
                    except_file = task_log_dir / 'exception.txt'
                    except_file.touch(exist_ok=True)
                    except_file.write_text(str(e))
                finally:
                    suffix = 'ok' if run_ok else 'failed'
                    log_file = task_log_dir / f'solver_allmsg.{suffix}.attempt{attempt}.json'
                    if captured_msgs:
                        msgs_list = list(captured_msgs)
                        save_history(msgs_list, log_file)
                        logger.info('saved {} messages to {}', len(msgs_list), log_file)
                        stats = aggregate_run_stats(msgs_list)
                        logger.info('task {} attempt {} stats: {}', cur_task_id, attempt, stats)
                    else:
                        logger.warning('task {} attempt {} no captured messages to save', cur_task_id, attempt)

            # 真结束条件: run 成功 (未抛异常) 且产出了 prediction.csv.
            # 仅有 prediction.csv 但 run_ok=False (如跑到 request_limit / 中途抛异常
            # 后残留半成品 csv) 不算成功, 继续重试.
            if run_ok and prediction_file.exists():
                logger.info('task {} prediction.csv generated on attempt {} (run_ok)', cur_task_id, attempt)
                break
            elif attempt < MAX_ATTEMPTS:
                logger.warning(
                    'task {} attempt {} not done (run_ok={}, pred_exists={}), retrying...',
                    cur_task_id, attempt, run_ok, prediction_file.exists(),
                )
            else:
                logger.error(
                    'task {} still not done after {} attempts (run_ok={}, pred_exists={})',
                    cur_task_id, MAX_ATTEMPTS, run_ok, prediction_file.exists(),
                )

        # 兜底: agent attempts 都失败时, 直接子进程跑一次 solver.py.
        # 哪怕 SQL 错也能产 prediction.csv (起码列名/scaffold 是对的, 比 0 分好).
        # 通用化: 不针对单 task, 仅在所有 attempt 都失败时启用.
        if not prediction_file.exists():
            solver_py = task_work_dir / 'solver.py'
            if solver_py.exists():
                logger.warning('task {} all attempts failed, running solver.py as fallback', cur_task_id)
                try:
                    import subprocess
                    import sys
                    # solver.py 依赖 tools_v2/agents_v2 等模块, 需把项目源码目录加到 PYTHONPATH
                    src_root = str(Path(__file__).resolve().parent)
                    env = dict(os.environ)
                    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
                    # 必须用当前解释器 sys.executable: 主进程跑在项目 venv (uv/docker) 里,
                    # pandas/duckdb/tools_v2 都装在这个 venv; 若写死 'python3' 会落到系统
                    # python, 依赖缺失直接 ModuleNotFoundError 静默失败, 兜底形同虚设.
                    subprocess.run(
                        [sys.executable, str(solver_py)],
                        cwd=str(task_dir),
                        timeout=120,
                        capture_output=True,
                        env=env,
                    )
                    if prediction_file.exists():
                        logger.info('task {} fallback solver.py produced prediction.csv', cur_task_id)
                except Exception as e:
                    logger.warning('task {} fallback solver.py failed: {}', cur_task_id, e)

        ## save prediction to output
        if prediction_file.exists():
            target_file = task_out_dir / 'prediction.csv'
            shutil.copy2(prediction_file, target_file)
            logger.info('prediction saved: {} -> {}', prediction_file, target_file)
        else:
            logger.error('ERROR: prediction.csv not found in {}', task_work_dir)

        logger.info('task {} completed, logs saved to {}, output saved to {}',
                    cur_task_id, task_log_dir, task_out_dir)

        # 落盘计时
        timer.dump(task_log_dir / 'timing.json')

        # 返回 token 统计供 main 汇总
        task_stats = None
        if captured_msgs:
            task_stats = aggregate_run_stats(list(captured_msgs))
            task_stats['task_id'] = cur_task_id
            task_stats['success'] = run_ok
            task_stats['timing'] = timer.summary()
        return task_stats



async def main(only_tasks: list[str] | None = None,
               run_group: str = '', max_parallel: int = MAX_PARALLEL_TASKS):
    from tools_v2.speed_runtime import install
    install()


    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if general_tools.is_docker():
        input_base_dir = Path('/input')
        output_base_dir = Path('/output')
        temp_dir = Path('/tmp/temp')
    else:
        base_dir = Path('/Users/zhe/Documents/work/20260400-data-agent/phase2/')
        input_base_dir = base_dir / 'input'
        # run_group: 批量跑 N 次时, run_agent_n_times.sh 传同一个 group 名,
        # 把本批所有 temp_{ts} 收拢到 phase2/<group>/ 下, 与零散单跑的 run 区分开,
        # 也便于和 run_logs/<同名时间戳>/ 互相对照. 默认空 => 仍直接落在 phase2/ 下.
        group = run_group.strip()
        out_root = base_dir / group if group else base_dir
        temp_dir = out_root / f'temp_{ts}'
        output_base_dir = temp_dir / 'output_final'  # 调试时,放到temp里便于管理

    # 环境变量 TEMP_DIR 优先; 未设置时统一落到 /tmp 下 (带时间戳, 避免多次运行互相覆盖).
    # 需在 output 默认值(基于 temp_dir)之前确定, 保证两者不脱节.
    env_temp = os.environ.get('TEMP_DIR', '').strip()
    if env_temp:
        temp_dir = Path(env_temp).expanduser()
    elif not general_tools.is_docker():
        temp_dir = Path(f'/tmp/temp_{ts}')
        output_base_dir = temp_dir / 'output_final'

    # 环境变量 INPUT_DIR / OUTPUT_DIR 优先级最高, 覆盖上面的默认路径.
    if os.environ.get('INPUT_DIR', '').strip():
        input_base_dir = Path(os.environ['INPUT_DIR'].strip()).expanduser()
    if os.environ.get('OUTPUT_DIR', '').strip():
        output_base_dir = Path(os.environ['OUTPUT_DIR'].strip()).expanduser()

    assert input_base_dir.exists(), f"输入目录 {input_base_dir} 不存在"

    general_tools.print_dir_permissions(input_base_dir, output_base_dir, temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)
    general_tools.add_loguru_file_sink(temp_dir / 'loguru_run.log')
    # 段错误兜底: native 库(whisper/duckdb/PyAV)在 4 路并发下若崩 SIGSEGV,
    # 进程直接死且无堆栈. faulthandler 把崩溃时所有线程的 C/Python 栈落到文件,
    # 据此定位是哪个 native 库. 必须在并发跑 task 之前启用.
    general_tools.enable_faulthandler(temp_dir / 'faulthandler.log')
    logger.info('temp dir maked. {} exist: {}', temp_dir, temp_dir.exists())
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 所有 task 的对话历史 / 异常信息统一落到 temp_dir/logs/<task_id>/,
    # 这个目录与每个 task_temp_dir 平级, 在 LocalBackend(root_dir=task_dir) 之外,
    # agent 通过 read_file 也访问不到, 不会把自己的历史当作输入再喂回 context.
    logs_base_dir = temp_dir / 'logs'
    logs_base_dir.mkdir(parents=True, exist_ok=True)

    all_task_ids = sorted([p.relative_to(input_base_dir) for p in input_base_dir.glob("task_*")])
    # Keep all tasks; prioritize text tasks while the provider's image path is
    # returning 500. Video tasks still run with their complete original inputs.
    all_task_ids.sort(key=lambda p: (
        (input_base_dir / p / 'context/video/briefing.mp4').exists(), p.name))

    # 仅本地 (非 docker) 支持 --only-tasks 子集筛选, 用于调试/验证少量 task.
    # docker 提交端必须跑全集, 故强制忽略此开关.
    if only_tasks and not general_tools.is_docker():
        wanted = set(only_tasks)
        kept = [p for p in all_task_ids if p.name in wanted]
        missing = wanted - {p.name for p in kept}
        if missing:
            logger.warning('only-tasks: 这些 task 在 input 目录下不存在, 忽略: {}', sorted(missing))
        if not kept:
            raise SystemExit(f'only-tasks: 没有任何 task 命中 {sorted(wanted)}, 退出')
        logger.info('only-tasks: 仅跑 {} 个 task: {}', len(kept), [p.name for p in kept])
        all_task_ids = kept

    semaphore = asyncio.Semaphore(max_parallel)
    logger.info('max parallel tasks: {}', max_parallel)

    # 记录批量执行总墙钟
    import time as _time
    _main_t0 = _time.perf_counter()

    # 并发调度: 用 semaphore 限制同时执行的 task 数量,
    # return_exceptions=True 防止单个 task 异常拖垮整批.
    coros = [
        process_one_task(
            p,
            semaphore=semaphore,
            input_base_dir=input_base_dir,
            output_base_dir=output_base_dir,
            temp_dir=temp_dir,
            logs_base_dir=logs_base_dir,
        )
        for p in all_task_ids
    ]
    pbar = tqdm(total=len(coros), desc='tasks')

    results: list[dict | None] = []

    async def _wrap(coro):
        try:
            result = await coro
            if result and result.get('success') and os.environ.get('KDD_RUN_IDENTITY'):
                from tools_v2.run_state import write_completion
                try:
                    write_completion(output_base_dir / result['task_id'], os.environ['KDD_RUN_IDENTITY'])
                except Exception as error:
                    logger.warning('任务完成标记未写入，后续将重新运行该题: {}', type(error).__name__)
            return result
        except Exception as e:
            logger.exception('unhandled task error: {}', e)
            return None
        finally:
            pbar.update(1)

    results = await asyncio.gather(*[_wrap(c) for c in coros])
    pbar.close()

    # ---- token 消耗汇总 ----
    all_stats = [r for r in results if r is not None]
    if all_stats:
        total = {
            'tasks': len(all_stats),
            'turns': sum(s['turns'] for s in all_stats),
            'input_tokens': sum(s['input_tokens'] for s in all_stats),
            'output_tokens': sum(s['output_tokens'] for s in all_stats),
            'cache_read_tokens': sum(s['cache_read_tokens'] for s in all_stats),
            'cache_write_tokens': sum(s['cache_write_tokens'] for s in all_stats),
            'total_tokens': sum(s['total_tokens'] for s in all_stats),
            'requests': sum(s['requests'] for s in all_stats),
        }
        logger.info('=' * 60)
        logger.info('TOKEN SUMMARY ({} tasks)', total['tasks'])
        logger.info('-' * 60)
        for s in sorted(all_stats, key=lambda x: x['task_id']):
            logger.info(
                '  {:<12} | turns={:<3} | in={:<8} out={:<8} cache_r={:<8} total={:<9} | {}',
                s['task_id'], s['turns'],
                s['input_tokens'], s['output_tokens'],
                s['cache_read_tokens'], s['total_tokens'],
                '✓' if s['success'] else '✗',
            )
        logger.info('-' * 60)
        logger.info(
            '  {:<12} | turns={:<3} | in={:<8} out={:<8} cache_r={:<8} total={:<9}',
            'TOTAL', total['turns'],
            total['input_tokens'], total['output_tokens'],
            total['cache_read_tokens'], total['total_tokens'],
        )
        logger.info('=' * 60)

        summary_file = logs_base_dir / 'token_summary.json'
        summary_file.write_text(json.dumps({'per_task': all_stats, 'total': total}, ensure_ascii=False, indent=2))
        logger.info('token summary saved to {}', summary_file)

    # ---- 耗时汇总 ----
    main_elapsed = _time.perf_counter() - _main_t0
    all_timings = [s['timing'] for s in all_stats if s.get('timing')] if all_stats else []
    if all_timings:
        timing_table = format_timing_summary(all_timings)
        logger.info('\n{}', timing_table)
        logger.info('Total wall clock: {:.1f}s', main_elapsed)

        timing_summary_file = logs_base_dir / 'timing_summary.json'
        timing_summary_file.write_text(json.dumps({
            'main_wall_s': round(main_elapsed, 3),
            'per_task': all_timings,
        }, ensure_ascii=False, indent=2))
        logger.info('timing summary saved to {}', timing_summary_file)



if __name__ == "__main__":
    import argparse

    # add_help=False: 不拦截 docker 镜像 CMD 里残留的 --help, 让它落入未知参数被忽略.
    # parse_known_args: 忽略未知参数, 行为与"没有 argparse"时一致, 不传则走默认值.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--only-tasks', type=str, default='',
        help='仅跑指定 task (逗号分隔, 如 "task_15,task_18"). 仅本地生效, docker 内忽略.',
    )
    parser.add_argument(
        '--run-group', type=str, default='',
        help='批量跑时的分组目录名, 本批所有 temp_{ts} 收拢到 phase2/<group>/ 下. '
             '默认空 => 直接落在 phase2/ 下. 仅本地生效, docker 内忽略.',
    )
    def _env_int(name: str, fallback: int) -> int:
        """读环境变量为 int; 未设置/空串/非法值均回退 fallback (不崩)."""
        v = os.environ.get(name, '').strip()
        if not v:
            return fallback
        try:
            return int(v)
        except ValueError:
            logger.warning('环境变量 {}={!r} 不是合法整数, 回退默认 {}', name, v, fallback)
            return fallback

    parser.add_argument(
        '--max-parallel', type=int,
        default=_env_int('MAX_PARALLEL', MAX_PARALLEL_TASKS),
        help=f'并行执行 task 的最大并发数 (默认取环境变量 MAX_PARALLEL, 否则 {MAX_PARALLEL_TASKS}). '
             f'命令行 --max-parallel 优先级最高.',
    )
    args, _ = parser.parse_known_args()
    only_tasks = [t.strip() for t in args.only_tasks.split(',') if t.strip()] or None
    asyncio.run(main(only_tasks=only_tasks,
                     run_group=args.run_group, max_parallel=args.max_parallel))
