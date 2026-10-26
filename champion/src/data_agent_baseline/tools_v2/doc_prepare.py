"""doc_prepare.py — 在 solver 启动前, 把 context/doc/*.md 抽成结构化 SQLite 表.

主流程定位
----------
zz_agent_v1.process_one_task 在拷贝 raw 输入到 task_temp_dir 之后,
**scaffold_tool.generate_solver_scaffold 之前** 调用本模块. 产物写入
``task_temp_dir/context/db/_doc_extracted.db`` —— 故意放在 ``context/db/``,
这样下游 ``scaffold_tool`` 扫描 ``context/db/*.db`` 时会自动把这些
新表加到 solver.py 的读数据 block, 无需改 scaffold 代码.

流程
----
1. 把 ``context/doc/*.pdf`` 用纯本地 reflow 转成同名 ``*.md`` (扫描件跳过),
   再列出 ``context/doc/*.md`` (原生 + reflow). 没有 doc 则直接返回, no-op.
2. 对每个 doc:
   a. 用内容 hash + 模型名作为 cache key, 命中 ``logs/<task>/doc_extract/<stem>.result.json``
      则直接 load, 跳过 LLM.
   b. 未命中: ``schema_inference.infer_doc_schema_async`` 推 schema,
      落盘到 ``logs/<task>/doc_extract/<stem>.schema.json``.
   c. ``doc_extractor.extract_doc_async`` 跑全量抽取,
      落盘到 ``logs/<task>/doc_extract/<stem>.result.json``.
3. 全部 doc 抽完后, 把所有 ExtractionResult 写入同一个 sqlite db
   (``context/db/_doc_extracted.db``), 每个 doc → 一张表.
4. 返回 ``DocPrepareResult`` (含 db 路径, 表列表, 每张表统计).

设计原则
--------
- **永不阻塞主流程**: 任何步骤失败都仅 warning, 返回 partial result.
- **可缓存**: 输入决定输出 (doc 内容 + knowledge.md + 模型), 直接哈希缓存.
- **对 solver 透明**: 走 ``context/db/`` 通道, scaffold 自动 pickup.
- **审计友好**: schema/result 全部落盘 JSON, 抽取失败的行单独可查.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tools_v2.doc_extractor import (
    ExtractionRecord,
    ExtractionResult,
    extract_doc_async,
)
from tools_v2.schema_inference import DocSchema, infer_doc_schema_async


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 抽取产物落到 context/db/ 下, 让 scaffold_tool 自动 pickup.
DOC_DB_FILENAME = "_doc_extracted.db"

# 单 doc 内并发 (LLM 调用). schema_inference 全局并发由外层 MAX_PARALLEL_TASKS
# + 本模块 per-doc 串行 + extractor 内部 6 并发共同决定.
DEFAULT_EXTRACT_CONCURRENCY = 6


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------


@dataclass
class DocTableInfo:
    """一张 doc → 一张 sqlite 表的统计."""

    doc_name: str
    table_name: str
    n_total: int
    n_ok: int
    n_failed: int
    n_rows_after_dedupe: int
    cache_hit: bool = False
    error: str = ""
    # 用于 render_solver_hint 注入 prompt 的元信息
    column_names: list[str] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    extras_top: list[str] = field(default_factory=list)  # schema 漏掉的高频字段, 仅供调试

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "table_name": self.table_name,
            "n_total": self.n_total,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "n_rows_after_dedupe": self.n_rows_after_dedupe,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "column_names": self.column_names,
            "primary_key": self.primary_key,
            "extras_top": self.extras_top,
        }


@dataclass
class DocPrepareResult:
    """整个 task 的 doc 预处理结果."""

    task_id: str
    db_path: Optional[Path] = None  # 没 doc 时为 None
    tables: list[DocTableInfo] = field(default_factory=list)

    @property
    def n_tables(self) -> int:
        return len([t for t in self.tables if not t.error])

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "db_path": str(self.db_path) if self.db_path else None,
            "n_tables": self.n_tables,
            "tables": [t.to_jsonable() for t in self.tables],
        }


# ---------------------------------------------------------------------------
# 缓存哈希
# ---------------------------------------------------------------------------


def _file_sha1(path: Path) -> str:
    """文件内容 sha1 (前 16 位)."""
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _cache_key(
    doc_path: Path, knowledge_md_path: Optional[Path], model_name: str
) -> str:
    """以 doc + knowledge + 模型名 联合哈希作为缓存 key."""
    parts = [_file_sha1(doc_path), model_name]
    if knowledge_md_path is not None and knowledge_md_path.is_file():
        parts.append(_file_sha1(knowledge_md_path))
    else:
        parts.append("no_knowledge")
    blob = "|".join(parts).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _model_name(model: Any) -> str:
    """尽力取 model 的可读名 (用于缓存 key)."""
    for attr in ("model_name", "name"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(model).__name__


# ---------------------------------------------------------------------------
# 单 doc 处理
# ---------------------------------------------------------------------------


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _fill_info_from_result(info: DocTableInfo, result: ExtractionResult) -> None:
    """把 ExtractionResult 的统计/schema 填进 info, 用于后续 prompt 渲染."""
    info.table_name = result.table_name
    info.n_total = result.n_total
    info.n_ok = result.n_ok
    info.n_failed = result.n_failed
    info.column_names = [c.name for c in result.schema.columns]
    info.primary_key = list(result.schema.primary_key)
    # extras 频次最高的前几个 key, 用于发现 schema 漏列 (调试线索)
    extras_freq = result.extras_frequency()
    info.extras_top = list(extras_freq.keys())[:5]


def _load_cached_result(
    cache_dir: Path, doc_stem: str, cache_key: str
) -> Optional[ExtractionResult]:
    """检查缓存, 命中则反序列化 ExtractionResult, 否则返回 None."""
    payload_path = cache_dir / f"{doc_stem}.result.json"
    key_path = cache_dir / f"{doc_stem}.cache_key"
    if not (payload_path.is_file() and key_path.is_file()):
        return None
    if key_path.read_text().strip() != cache_key:
        return None
    try:
        raw = json.loads(payload_path.read_text(encoding="utf-8"))
        schema = DocSchema.model_validate(raw["schema"])
        records = [
            ExtractionRecord(
                line_no=r["line_no"],
                values=r.get("values", {}),
                extras=r.get("extras", {}),
                error=r.get("error", ""),
            )
            for r in raw.get("records", [])
        ]
        return ExtractionResult(
            doc_path=raw["doc_path"],
            table_name=raw["table_name"],
            schema=schema,
            records=records,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[doc_prepare] cache load failed for {}: {}", doc_stem, e)
        return None


async def _prepare_one_doc(
    *,
    doc_path: Path,
    knowledge_md_path: Optional[Path],
    model: Any,
    cache_dir: Path,
    concurrency: int,
) -> tuple[Optional[ExtractionResult], DocTableInfo]:
    """处理单个 doc, 返回 (extraction_result, info).

    任意步骤失败时, 返回 (None, info_with_error). 不抛异常.
    """
    doc_stem = doc_path.stem
    info = DocTableInfo(
        doc_name=doc_path.name,
        table_name=doc_stem,
        n_total=0,
        n_ok=0,
        n_failed=0,
        n_rows_after_dedupe=0,
    )

    # 1) 缓存命中?
    try:
        ck = _cache_key(doc_path, knowledge_md_path, _model_name(model))
    except Exception as e:  # noqa: BLE001
        logger.warning("[doc_prepare] hash {} failed: {}", doc_path, e)
        ck = ""

    if ck:
        cached = _load_cached_result(cache_dir, doc_stem, ck)
        if cached is not None:
            logger.info(
                "[doc_prepare] cache hit: {} (table={}, ok={}/{})",
                doc_path.name,
                cached.table_name,
                cached.n_ok,
                cached.n_total,
            )
            _fill_info_from_result(info, cached)
            info.cache_hit = True
            return cached, info

    # 2) schema 推断
    try:
        _t_schema = time.perf_counter()
        schema = await infer_doc_schema_async(
            doc_path=doc_path,
            knowledge_md_path=knowledge_md_path,
            model=model,
        )
        logger.info('[doc_prepare] schema_inference {}: {:.2f}s', doc_path.name, time.perf_counter() - _t_schema)
        _save_json(
            schema.model_dump(),
            cache_dir / f"{doc_stem}.schema.json",
        )
        info.table_name = schema.table_name
    except Exception as e:  # noqa: BLE001
        logger.exception("[doc_prepare] schema_inference failed for {}", doc_path)
        info.error = f"schema_inference: {type(e).__name__}: {e}"
        return None, info

    # 3) 抽取
    try:
        _t_extract = time.perf_counter()
        result = await extract_doc_async(
            doc_path=doc_path,
            schema=schema,
            model=model,
            knowledge_md_path=knowledge_md_path,
            concurrency=concurrency,
        )
        logger.info('[doc_prepare] extract_doc {}: {:.2f}s', doc_path.name, time.perf_counter() - _t_extract)
    except Exception as e:  # noqa: BLE001
        logger.exception("[doc_prepare] extract_doc failed for {}", doc_path)
        info.error = f"extract_doc: {type(e).__name__}: {e}"
        return None, info

    _fill_info_from_result(info, result)

    # 4) 落盘 + 写 cache key
    try:
        _save_json(
            result.to_jsonable(),
            cache_dir / f"{doc_stem}.result.json",
        )
        if ck:
            (cache_dir / f"{doc_stem}.cache_key").write_text(ck)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[doc_prepare] save extraction result failed for {}: {}", doc_path, e
        )

    return result, info


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def _reflow_pdfs_to_md(doc_dir: Path) -> list[Path]:
    """把 ``doc_dir/*.pdf`` 用纯本地 reflow 转成同名 ``*.md``, 落回 ``doc_dir``.

    方案 B: 产物落回 ``context/doc/`` —— 既进入下游 ``.md`` 抽取链, 又被
    ``describe_tool`` 列出, 让 solver agent 能 ``read_file`` 读到原文做兜底校验.

    - 只处理 **数字原生文本 pdf** (``is_text_pdf`` 预检); 扫描件/图片型抽不出
      文字, warning 跳过, 不阻塞主流程.
    - 已存在同名 ``.md`` (原生 doc 或上次 reflow 产物) 时跳过, 避免覆盖原生
      doc; 全 60 task 实测 md/pdf 无同 stem 撞名, 该分支主要用于复跑幂等.
    - 任意单个 pdf 失败仅 warning, 返回已成功的部分.

    返回新生成的 ``.md`` 路径列表 (不含被跳过/已存在的).
    """
    # 延迟导入: 让本模块被 import 时不强依赖 PyMuPDF.
    from tools_v2.pdf_reflow import is_text_pdf, reflow_pdf_to_file

    pdf_files = sorted(doc_dir.glob("*.pdf"))
    if not pdf_files:
        return []

    produced: list[Path] = []
    for pdf in pdf_files:
        md_path = doc_dir / f"{pdf.stem}.md"
        if md_path.exists():
            logger.debug(
                "[doc_prepare] reflow skip {}: {} already exists",
                pdf.name,
                md_path.name,
            )
            continue
        try:
            if not is_text_pdf(pdf):
                logger.warning(
                    "[doc_prepare] {} 疑似扫描件/图片型 pdf, 跳过 reflow", pdf.name
                )
                continue
            reflow_pdf_to_file(pdf, md_path)
            produced.append(md_path)
            logger.info("[doc_prepare] reflow {} -> {}", pdf.name, md_path.name)
        except Exception as e:  # noqa: BLE001
            logger.warning("[doc_prepare] reflow {} failed: {}", pdf.name, e)
    return produced


async def prepare_doc_tables_async(
    *,
    task_dir: Path,
    log_dir: Path,
    model: Any,
    concurrency: int = DEFAULT_EXTRACT_CONCURRENCY,
    relevant_stems: Optional[set[str]] = None,
) -> DocPrepareResult:
    """把 ``task_dir/context/doc/*.md`` 全部结构化, 输出到 ``context/db/`` 下.

    参数:
        task_dir: task 的 temp 目录 (已拷贝过 raw 输入). 写产物用.
        log_dir: 该 task 的 logs 目录, 用于落 schema/result/cache_key.
        model: pydantic-ai 兼容的 model (推荐 ``MODEL_NOTHINK``).
        concurrency: 单 doc 内的抽取并发上限.
        relevant_stems: 可选的"相关 doc stem 集合"(由 ``doc_relevance_agent`` 判定).
            仅这些 stem 的 ``*.md`` 会被结构化抽取, 其余跳过, 以节省昂贵的 LLM 抽取.
            **注意: pdf reflow 不受此过滤影响, 仍对全部 pdf 执行** —— 被跳过抽取的 doc
            其 ``*.md`` 原文(含 pdf reflow 产物)仍留在 ``context/doc/``, solver 可
            ``read_file`` 兜底. ``None`` (默认) 表示不过滤, 抽取全部 doc.

    返回:
        ``DocPrepareResult``. 没 doc 时 ``db_path=None``, ``tables=[]``.
        即便部分 doc 抽取失败, 主流程仍可继续.
    """
    task_id = task_dir.name
    out = DocPrepareResult(task_id=task_id)

    doc_dir = task_dir / "context" / "doc"
    if not doc_dir.is_dir():
        logger.debug("[doc_prepare] no context/doc dir for {}, skip", task_id)
        return out

    # 先把 doc/*.pdf reflow 成同名 *.md (落回 context/doc/, 方案 B), 让 pdf
    # 走与原生 *.md 完全相同的抽取链, 同时暴露给 agent 兜底. 失败仅 warning.
    # 注意: reflow 对【全部】 pdf 执行, 不受 relevant_stems 过滤 —— 这样即便某 doc
    # 被判为无关而跳过结构化抽取, 其 reflow 出的 *.md 原文仍在, solver 能 read_file.
    try:
        _t_reflow = time.perf_counter()
        reflowed = _reflow_pdfs_to_md(doc_dir)
        if reflowed:
            logger.info(
                "[doc_prepare] task={} reflowed {} pdf(s) in {:.2f}s: {}",
                task_id,
                len(reflowed),
                time.perf_counter() - _t_reflow,
                [p.name for p in reflowed],
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("[doc_prepare] task={} pdf reflow stage failed: {}", task_id, e)

    doc_files = sorted(doc_dir.glob("*.md"))
    if not doc_files:
        logger.debug("[doc_prepare] no doc(*.md/*.pdf) in {}, skip", doc_dir)
        return out

    # 按 doc 相关性过滤要抽取的 md: 只抽取 relevant_stems 中的 doc, 跳过其余.
    # (pdf reflow 已在上面对全部 pdf 执行完毕, 故跳过抽取不影响 read_file 兜底.)
    if relevant_stems is not None:
        kept = [p for p in doc_files if p.stem in relevant_stems]
        skipped = [p.name for p in doc_files if p.stem not in relevant_stems]
        if skipped:
            logger.info(
                "[doc_prepare] task={} relevance filter: 抽取 {}/{} 个 doc, 跳过 {}: {}",
                task_id,
                len(kept),
                len(doc_files),
                len(skipped),
                skipped,
            )
        doc_files = kept
        if not doc_files:
            logger.info(
                "[doc_prepare] task={} 无相关 doc 需抽取 (全部跳过), 不生成 _doc_extracted.db",
                task_id,
            )
            return out

    knowledge_md_path = task_dir / "context" / "knowledge.md"
    if not knowledge_md_path.is_file():
        knowledge_md_path = None  # 没有也无妨

    cache_dir = log_dir / "doc_extract"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[doc_prepare] task={} found {} doc(s): {}",
        task_id,
        len(doc_files),
        [p.name for p in doc_files],
    )

    # doc 之间串行 (避免和外层 task 并发叠加打爆后端);
    # 单 doc 内的抽取仍然 concurrency 并发.
    extracted: list[ExtractionResult] = []
    for doc_path in doc_files:
        result, info = await _prepare_one_doc(
            doc_path=doc_path,
            knowledge_md_path=knowledge_md_path,
            model=model,
            cache_dir=cache_dir,
            concurrency=concurrency,
        )
        out.tables.append(info)
        if result is not None:
            extracted.append((result, info))  # type: ignore[arg-type]

    if not extracted:
        logger.warning(
            "[doc_prepare] task={} all docs failed or empty, no db generated",
            task_id,
        )
        _save_json(out.to_jsonable(), cache_dir / "summary.json")
        return out

    # 5) 写入 sqlite (放在 context/db/ 下, 让 scaffold pickup)
    db_dir = task_dir / "context" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / DOC_DB_FILENAME

    # 复跑前先删, 避免 to_sql replace 在并发场景下偶发的 schema 残留
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:  # noqa: BLE001
            pass

    # 单表隔离写入: 某张表 to_dataframe/to_sql 失败时, 只标记该表 error 并跳过,
    # **不影响其余表入库** (历史 bug: 一表抛异常会把 db_path 整体置 None,
    # 已抽好的其它表全被丢弃). 只要有 >=1 张表成功落库, db_path 即有效.
    n_written = 0
    for result, info in extracted:
        try:
            df = result.to_dataframe()
            info.n_rows_after_dedupe = len(df)
            with sqlite3.connect(str(db_path)) as conn:
                df.to_sql(
                    result.table_name, conn, if_exists="replace", index=False
                )
            n_written += 1
            logger.info(
                "[doc_prepare] task={} wrote table '{}' rows={} (ok={}/{})",
                task_id,
                result.table_name,
                len(df),
                result.n_ok,
                result.n_total,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "[doc_prepare] task={} write table '{}' failed: {}",
                task_id,
                result.table_name,
                e,
            )
            info.error = info.error or f"sqlite_write: {type(e).__name__}: {e}"

    # 只要写成功了至少一张表, db 就是可用的; 一张都没成才置 None.
    out.db_path = db_path if (n_written > 0 and db_path.exists()) else None

    _save_json(out.to_jsonable(), cache_dir / "summary.json")
    logger.info(
        "[doc_prepare] task={} done: {} table(s) → {}",
        task_id,
        out.n_tables,
        out.db_path,
    )
    return out


# ---------------------------------------------------------------------------
# Prompt 渲染: 把抽取结果暴露给 solver agent
# ---------------------------------------------------------------------------


# 抽取成功率低于该阈值的表会在 prompt 里打 ⚠ 标记, 提示 solver 用 read_file 兜底
_LOW_QUALITY_OK_RATIO = 0.95


def render_solver_hint(result: DocPrepareResult) -> str:
    """把 doc 抽取产物渲染成 solver agent prompt 的一段提示.

    返回值会拼到 ``user_input`` 的 ``context_info`` 之后, 让 agent 知道:
    - context/db/_doc_extracted.db 里多了几张表, 表名/列/对应 doc 是什么
    - 应当**直接 SQL 查这些表**, 不要回头去 read_file 读 doc/*.md 原文
    - ``__line_no`` / ``__line_nos`` 是审计列, 用于反查原文行号
    - 抽取不完美的表 (ok_ratio 低) 应当结合原文兜底校验

    没 doc / 没生成 db 时返回空字符串, 主流程拼接时不影响.
    """
    if result.db_path is None or not result.tables:
        return ""

    # 只展示成功生成的表
    valid = [t for t in result.tables if not t.error]
    if not valid:
        return ""

    lines: list[str] = []
    lines.append("## 已从 doc 自动抽取的结构化表")
    lines.append(
        f"位置: `context/db/{DOC_DB_FILENAME}` "
        f"(scaffold 已自动加载为 DataFrame, 可直接用 SQL / pandas 查)."
    )
    lines.append("")
    for t in valid:
        ok_ratio = (t.n_ok / t.n_total) if t.n_total else 0.0
        warn = " ⚠ 抽取不完美" if ok_ratio < _LOW_QUALITY_OK_RATIO else ""
        pk_str = (
            f", primary_key={t.primary_key}" if t.primary_key else ""
        )
        lines.append(
            f"- 表 `{t.table_name}` ← `doc/{t.doc_name}` "
            f"(rows={t.n_rows_after_dedupe}, ok={t.n_ok}/{t.n_total}){warn}"
        )
        # 列分两行显示, 业务列 + 审计列分开
        cols_str = ", ".join(t.column_names) if t.column_names else "(unknown)"
        lines.append(f"    业务列{pk_str}: {cols_str}")
        lines.append(
            "    审计列: __line_no (int, doc 中首次出现行号), "
            "__line_nos (str, 多段叙述时的全部行号, 逗号分隔)"
        )
        if t.extras_top:
            lines.append(
                f"    note: schema 漏掉的高频字段(可忽略, 调试线索): {t.extras_top}"
            )
    lines.append("")
    lines.append("使用须知:")
    lines.append(
        "1. **优先 SQL 查这些表**, 不要再 read_file 读 `doc/*.md` 原文 "
        "(doc 动辄上千行, 直接读会爆 token)."
    )
    lines.append(
        "2. 同主键的多段叙述已按 last-non-empty-wins 自动合并, "
        "**不要再自己去重**; 单段 record 的 ``__line_no == __line_nos``, "
        "多段合并 record 的 ``__line_nos`` 形如 ``'120,143,201'``."
    )
    lines.append(
        "3. 如需反查原文做证据校验, "
        "可用 ``WHERE __line_nos LIKE '%,143%'`` 之类查询定位行号, "
        "再用 read_file 精读对应行."
    )
    lines.append(
        "4. 标 ⚠ 的表抽取成功率低 (<95%), "
        "对该表的关键查询结果存疑时, 用 read_file 读原文兜底校验."
    )
    return "\n".join(lines)

