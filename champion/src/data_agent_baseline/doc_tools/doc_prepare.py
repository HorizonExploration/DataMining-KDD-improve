"""doc_prepare.py — fan-out doc 结构化的主流程接入接口.

把 ``context/doc/*.md`` 用 fan-out 引擎 (``doc_tools.fanout_struct``) 结构化成
**每 doc 一个同名 SQLite db**: ``context/db/<docstem>.db`` (表名 = doc stem).
产物落在 ``context/db/`` 下, scaffold 的 ``build_datasource_state`` 扫描该目录时
自动把这些表注册为 DuckDB 视图, 无需改 scaffold.

与 ``tools_v2.doc_prepare`` / ``doc_tools_deprecated.doc_pipeline`` 接口兼容:
    - ``prepare_doc_tables_async(*, task_dir, log_dir, model_nothink, model_think, concurrency, relevant_stems)``
    - ``render_solver_hint(result)``
故 ``zz_agent_v2`` 只需改一行 import 即可切换到本流程.

设计原则 (与旧版一致):
    - **永不阻塞主流程**: 任意步骤失败仅 warning, 返回 partial result.
    - **可缓存**: 输入、服务、两个模型、生成设置与处理代码共同约束缓存.
    - **对 solver 透明**: 走 ``context/db/`` 通道, scaffold 自动 pickup.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from doc_tools.fanout_struct import StructResult, set_models, structurize_doc
from tools_v2.artifact_cache import DocumentCache, document_identity, digest_json


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# pipeline 版本号, 并入 cache key: 引擎/prompt 改动后想强制失效旧缓存时 bump 它.
# v2: planner 不再写 transform_py (只声明 target_format) + 新增 Phase 5.5 think 模型生成 transform
#     + planner 同义异形名称分列规则 + knowledge 不全须补列说明.
PIPELINE_VERSION = "fanout_v2"

# 单 doc 内 section worker 并发上限.
DEFAULT_CONCURRENCY = 6


# ---------------------------------------------------------------------------
# 结果数据类 (字段与 tools_v2.doc_prepare 对齐, 保证 render_solver_hint 兼容)
# ---------------------------------------------------------------------------


@dataclass
class DocTableInfo:
    """一张 doc → 一张 sqlite 表的统计."""

    doc_name: str
    table_name: str
    db_name: str = ""  # 该表所在的 db 文件名 (与 doc 同名)
    n_total: int = 0
    n_ok: int = 0
    n_failed: int = 0
    n_rows_after_dedupe: int = 0
    cache_hit: bool = False
    cache_source: str = 'fresh'
    error: str = ""
    column_names: list[str] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "table_name": self.table_name,
            "db_name": self.db_name,
            "n_total": self.n_total,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "n_rows_after_dedupe": self.n_rows_after_dedupe,
            "cache_hit": self.cache_hit,
            "cache_source": self.cache_source,
            "error": self.error,
            "column_names": self.column_names,
            "primary_key": self.primary_key,
        }


@dataclass
class DocPrepareResult:
    """整个 task 的 doc 预处理结果."""

    task_id: str
    db_path: Optional[Path] = None  # 没 doc / 全失败时为 None; 否则为 context/db/ 目录
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


def _cache_identity(doc_path, knowledge_md_path, model, transform_model):
    return document_identity(doc_path, knowledge_md_path, model, transform_model, PIPELINE_VERSION)


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 缓存 load / save
# ---------------------------------------------------------------------------


def _load_cached(cache_dir: Path, doc_stem: str, cache_key: str) -> Optional[StructResult]:
    payload_path = cache_dir / f"{doc_stem}.result.json"
    key_path = cache_dir / f"{doc_stem}.cache_key"
    if not (payload_path.is_file() and key_path.is_file()):
        return None
    if key_path.read_text().strip() != cache_key:
        return None
    try:
        raw = json.loads(payload_path.read_text(encoding="utf-8"))
        return _restore_result(raw)
    except Exception as e:  # noqa: BLE001
        logger.warning("[doc_prepare] cache load failed for {}: {}", doc_stem, e)
        return None


def _restore_result(raw: dict) -> StructResult:
    stats = raw.get("stats", {})
    return StructResult(
        table_name=raw["table_name"],
        primary_key=raw.get("primary_key", ["record_id"]),
        columns=raw["columns"],
        rows=raw.get("rows", []),
        n_universe=stats.get("n_universe", 0),
        n_sections=stats.get("n_sections", 0),
        n_failed_sections=stats.get("n_failed_sections", 0),
        global_notes=raw.get("global_notes", ""),
        plan=raw.get("plan", {}) or {},
        normalize_stats=raw.get("normalize_stats", {}) or {},
        provenance=raw.get("provenance", {}) or {},
    )


def _fill_info(info: DocTableInfo, res: StructResult) -> None:
    """把 StructResult 的统计填进 info.

    本流程无"逐行 ok/fail"概念 (行由 section worker 整表产出), 这里把
    n_total/n_ok 映射成宽表行数, n_failed 映射成失败 section 数, 以便
    render_solver_hint 的 ok_ratio 阈值逻辑沿用 (section 全成功 → ratio=1).
    """
    info.table_name = res.table_name
    # column_names 用于 render_solver_hint, 必须与实际落库列一致 (剔除 noise/配套 raw),
    # 否则会向 solver 提示 DB 里不存在的列.
    _dropped = set(_db_excluded_cols(res.columns))
    info.column_names = [c for c in res.columns if c not in _dropped]
    info.primary_key = res.primary_key
    info.n_rows_after_dedupe = len(res.rows)
    info.n_total = res.n_sections
    info.n_ok = res.n_sections - res.n_failed_sections
    info.n_failed = res.n_failed_sections


# ---------------------------------------------------------------------------
# pdf reflow (复用 tools_v2.pdf_reflow)
# ---------------------------------------------------------------------------


def _reflow_pdfs_to_md(doc_dir: Path) -> list[Path]:
    """把 ``doc_dir/*.pdf`` 用纯本地 reflow 转成同名 ``*.md`` (扫描件跳过).

    已存在同名 ``.md`` 时跳过. 单 pdf 失败仅 warning. 返回新生成的 md 列表.
    """
    from tools_v2.pdf_reflow import is_text_pdf, reflow_pdf_to_file

    pdf_files = sorted(doc_dir.glob("*.pdf"))
    if not pdf_files:
        return []

    produced: list[Path] = []
    for pdf in pdf_files:
        md_path = doc_dir / f"{pdf.stem}.md"
        if md_path.exists():
            continue
        try:
            if not is_text_pdf(pdf):
                logger.warning("[doc_prepare] {} 疑似扫描件/图片型 pdf, 跳过 reflow", pdf.name)
                continue
            reflow_pdf_to_file(pdf, md_path)
            produced.append(md_path)
            logger.info("[doc_prepare] reflow {} -> {}", pdf.name, md_path.name)
        except Exception as e:  # noqa: BLE001
            logger.warning("[doc_prepare] reflow {} failed: {}", pdf.name, e)
    return produced


# ---------------------------------------------------------------------------
# 单 doc 落库
# ---------------------------------------------------------------------------


def _db_excluded_cols(cols: list[str]) -> list[str]:
    """落库时应剔除的列名列表 (noise / raw), 按输入顺序返回.

    planner 切出的这两类列前置流程会一直保留, 但它们不是真实业务数据列,
    进 DB 只会变成匹配不上 gold 的 extra 列倒扣 penalty. 故仅在最终落库时过滤,
    TSV (人工复盘用) 仍保留全列. record_id 等主键不受影响.

    - **noise 列** (``<section前缀>_noise``): 闲话垃圾桶, 一律剔除.
    - **raw 列** (``X_raw``): **仅当存在配套的非 raw 列** (``X`` 或 ``X_final``) 时才剔除
      —— 即确认它确实是"修正历史"双列里的原始值那一列, 真实数据在 ``X_final`` / ``X`` 中.
      若某个名字恰好以 ``_raw`` 结尾、却没有对应的非 raw 列, 说明它本身可能就是唯一的
      真实数据列, **保留**, 避免误删.
    """
    colset = set(cols)
    excluded: list[str] = []
    for c in cols:
        if c.endswith("_noise"):
            excluded.append(c)
        elif c.endswith("_raw"):
            base = c[: -len("_raw")]
            if base in colset or f"{base}_final" in colset:
                excluded.append(c)
            else:
                logger.info(
                    "[doc_prepare] 列 '{}' 以 _raw 结尾但无配套非 raw 列 (既无 '{}' 也无 '{}_final'), 保留落库",
                    c, base, base,
                )
    return excluded


def _write_struct_to_db(res: StructResult, db_path: Path) -> int:
    """把 StructResult 写成 db_path 里的一张表 (表名 = res.table_name). 返回行数.

    复跑前先删旧 db (每 doc 独占一个 db 文件, 直接 unlink 最干净).
    case-insensitive 重名列改名 _2/_3 (SQLite 列名 case-insensitive 唯一).
    落库前剔除 noise / 已配套的 raw 列 (见 ``_db_excluded_cols``), TSV 不受影响.
    """
    import pandas as pd

    # 落库列 = 全列剔除 noise / 配套 raw; TSV 走 _write_struct_to_tsv 仍存全列.
    dropped = set(_db_excluded_cols(res.columns))
    cols = [c for c in res.columns if c not in dropped]
    if dropped:
        logger.info(
            "[doc_prepare] table='{}' 落库剔除 noise/raw 列: {}",
            res.table_name, [c for c in res.columns if c in dropped],
        )
    if res.rows:
        df = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in res.rows], columns=cols)
    else:
        df = pd.DataFrame(columns=cols)

    seen_lower: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        lo = col.lower()
        if lo in seen_lower:
            seen_lower[lo] += 1
            new_cols.append(f"{col}_{seen_lower[lo]}")
        else:
            seen_lower[lo] = 1
            new_cols.append(col)
    if new_cols != list(df.columns):
        logger.info("[doc_prepare] table='{}' rename dup cols", res.table_name)
        df.columns = new_cols

    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:  # noqa: BLE001
            pass
    with sqlite3.connect(str(db_path)) as conn:
        df.to_sql(res.table_name, conn, if_exists="replace", index=False)
    return len(df)


def _write_struct_to_tsv(res: StructResult, tsv_path: Path) -> None:
    """把 StructResult 完整写成 tsv (方便人工 / 外部工具直接读取).

    与 db 不同, 这里不做列去重改名, 忠实保留原始 columns/rows. 仅供复盘读取.
    """
    import pandas as pd

    cols = res.columns
    df = pd.DataFrame(res.rows, columns=cols) if res.rows else pd.DataFrame(columns=cols)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tsv_path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


async def prepare_doc_tables_async(
    *,
    task_dir: Path,
    log_dir: Path,
    model_nothink: Any,
    model_think: Any = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    relevant_stems: Optional[set[str]] = None,
) -> DocPrepareResult:
    """把 ``task_dir/context/doc/*.md`` 用 fan-out 引擎结构化, 每 doc 落一个同名 db.

    参数:
        task_dir: task temp 目录 (已拷贝 raw 输入). 写产物用.
        log_dir: 该 task 的 logs 目录, 用于落 result/cache_key/summary.
        model_nothink: pydantic-ai 兼容 model, 用于 planner / section worker (纯抽取, 传 nothink).
        model_think: 可选, 专用于 Phase 5.5 生成 column_normalize 的 transform 代码 (推理任务,
            建议传 think 模型). None 时 fanout 内部退回用 ``model_nothink``.
        concurrency: 单 doc 内 section worker 并发上限.
        relevant_stems: 仅这些 stem 的 doc 会被结构化抽取; pdf reflow 不受此过滤.
            None 表示抽取全部 doc.

    返回:
        DocPrepareResult. 没 doc / 全失败时 db_path=None.

    模型经 ``fanout_struct.set_models()`` 注入 (contextvar), 各 agent 函数直接取用,
    不再逐层透传 (planner/worker/transform 嵌套较深). contextvar 保证并发 task 间隔离.
    """
    task_id = task_dir.name
    out = DocPrepareResult(task_id=task_id)

    doc_dir = task_dir / "context" / "doc"
    if not doc_dir.is_dir():
        logger.debug("[doc_prepare] no context/doc dir for {}, skip", task_id)
        return out

    # pdf reflow (对全部 pdf, 不受 relevant_stems 过滤)
    try:
        reflowed = _reflow_pdfs_to_md(doc_dir)
        if reflowed:
            logger.info(
                "[doc_prepare] task={} reflowed {} pdf(s): {}",
                task_id, len(reflowed), [p.name for p in reflowed],
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("[doc_prepare] task={} pdf reflow stage failed: {}", task_id, e)

    doc_files = sorted(doc_dir.glob("*.md"))
    if not doc_files:
        logger.debug("[doc_prepare] no doc(*.md) in {}, skip", doc_dir)
        return out

    # relevant_stems 过滤
    if relevant_stems is not None:
        kept = [p for p in doc_files if p.stem in relevant_stems]
        skipped = [p.name for p in doc_files if p.stem not in relevant_stems]
        if skipped:
            logger.info(
                "[doc_prepare] task={} relevance filter: 抽取 {}/{} doc, 跳过 {}: {}",
                task_id, len(kept), len(doc_files), len(skipped), skipped,
            )
        doc_files = kept
        if not doc_files:
            logger.info("[doc_prepare] task={} 无相关 doc 需抽取, 不生成 db", task_id)
            return out

    knowledge_md_path = task_dir / "context" / "knowledge.md"
    if not knowledge_md_path.is_file():
        knowledge_md_path = None

    cache_dir = log_dir / "doc_extract"
    cache_dir.mkdir(parents=True, exist_ok=True)
    shared_cache = DocumentCache()

    logger.info(
        "[doc_prepare] task={} found {} doc(s): {}",
        task_id, len(doc_files), [p.name for p in doc_files],
    )

    # 注入模块级模型供 fanout 各 agent 直接取用 (planner/worker=nothink, transform=think).
    set_models(model_nothink=model_nothink, model_think=model_think)
    # 输入、服务、两个模型、生成设置和处理代码共同约束缓存，避免跨 API 污染。

    # doc 之间串行 (单 doc 内已并发); 失败一张不影响其它.
    extracted: list[tuple[StructResult, DocTableInfo]] = []
    for doc_path in doc_files:
        info = DocTableInfo(doc_name=doc_path.name, table_name=doc_path.stem)
        out.tables.append(info)

        try:
            identity = _cache_identity(doc_path, knowledge_md_path, model_nothink, model_think)
            ck = digest_json(identity)
        except Exception as e:  # noqa: BLE001
            logger.warning("[doc_prepare] hash {} failed: {}", doc_path, e)
            ck = ""

        # 缓存命中?
        if ck:
            cached = _load_cached(cache_dir, doc_path.stem, ck)
            source = 'local'
            if cached is None:
                payload = shared_cache.load(identity)
                if payload is not None:
                    cached = _restore_result(payload)
                    source = 'shared'
            if cached is not None:
                _fill_info(info, cached)
                info.cache_hit = True
                info.cache_source = source
                extracted.append((cached, info))
                _save_json(cached.to_jsonable(), cache_dir / f'{doc_path.stem}.result.json')
                (cache_dir / f'{doc_path.stem}.cache_key').write_text(ck)
                _save_json({'key': ck, 'source': source, 'payload_sha256': digest_json(cached.to_jsonable())},
                           cache_dir / f'{doc_path.stem}.cache_source.json')
                logger.info(
                    "[doc_prepare] {} cache hit: {} ({} rows)",
                    source, doc_path.name, len(cached.rows),
                )
                continue

        # 结构化抽取
        try:
            _t = time.perf_counter()
            res = await structurize_doc(
                doc_path=doc_path,
                knowledge_md_path=knowledge_md_path,
                concurrency=concurrency,
                table_name=doc_path.stem,
            )
            logger.info(
                "[doc_prepare] structurize {}: {:.2f}s",
                doc_path.name, time.perf_counter() - _t,
            )
            _fill_info(info, res)
            extracted.append((res, info))
            # 落缓存
            try:
                _save_json(res.to_jsonable(), cache_dir / f"{doc_path.stem}.result.json")
                if ck:
                    (cache_dir / f"{doc_path.stem}.cache_key").write_text(ck)
                    stored = shared_cache.store(identity, res.to_jsonable())
                    _save_json({'key': ck, 'source': 'fresh', 'shared_stored': stored,
                                'payload_sha256': digest_json(res.to_jsonable())},
                               cache_dir / f'{doc_path.stem}.cache_source.json')
            except Exception as e:  # noqa: BLE001
                logger.warning("[doc_prepare] cache save failed for {}: {}", doc_path.name, e)
            # 落 fanout 中间轨迹 (planner/section worker 原始返回, 含失败重试版本).
            # 与 result.json 分开存: 体积大, 仅供失败复盘, 不参与缓存命中.
            try:
                if res.trace:
                    _save_json(
                        {"doc": doc_path.name, "n_steps": len(res.trace), "trace": res.trace},
                        cache_dir / f"{doc_path.stem}.fanout_allmsg.json",
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("[doc_prepare] trace save failed for {}: {}", doc_path.name, e)
        except Exception as e:  # noqa: BLE001
            logger.exception("[doc_prepare] structurize {} failed", doc_path)
            info.error = f"structurize: {type(e).__name__}: {e}"

    if not extracted:
        logger.warning("[doc_prepare] task={} all docs failed or empty, no db", task_id)
        _save_json(out.to_jsonable(), cache_dir / "summary.json")
        return out

    # 每 doc 写一个同名 db 到 context/db/
    db_dir = task_dir / "context" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for res, info in extracted:
        db_path = db_dir / f"{info.doc_name.rsplit('.', 1)[0]}.db"
        try:
            n_rows = _write_struct_to_db(res, db_path)
            info.n_rows_after_dedupe = n_rows
            info.db_name = db_path.name
            n_written += 1
            logger.info(
                "[doc_prepare] task={} wrote table '{}' rows={} -> {}",
                task_id, res.table_name, n_rows, db_path.name,
            )
            # 同时落一份完整 tsv 到 logs/doc_extract/ 方便人工读取 (失败不影响主流程)
            try:
                _write_struct_to_tsv(res, cache_dir / f"{info.doc_name.rsplit('.', 1)[0]}.tsv")
            except Exception as e:  # noqa: BLE001
                logger.warning("[doc_prepare] tsv dump failed for {}: {}", info.doc_name, e)
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "[doc_prepare] task={} write table '{}' failed: {}",
                task_id, res.table_name, e,
            )
            info.error = info.error or f"sqlite_write: {type(e).__name__}: {e}"

    # db_path 设为 context/db/ 目录 (非 None 即表示有产出); 一张都没写成才 None.
    out.db_path = db_dir if n_written > 0 else None

    _save_json(out.to_jsonable(), cache_dir / "summary.json")
    logger.info(
        "[doc_prepare] task={} done: {} table(s) → {}",
        task_id, out.n_tables, out.db_path,
    )
    return out


# ---------------------------------------------------------------------------
# Prompt 渲染: 把抽取结果暴露给 solver agent
# ---------------------------------------------------------------------------

# section 失败率高于该阈值的表会在 prompt 里打 ⚠ 标记 (这里 ok = 成功 section 占比).
_LOW_QUALITY_OK_RATIO = 0.95


def render_solver_hint(result: DocPrepareResult) -> str:
    """把 doc 抽取产物渲染成 solver agent prompt 的一段提示.

    每个 doc → 一个同名 db (``context/db/<docstem>.db``), 表名 = doc stem,
    scaffold 已自动注册为 DuckDB 视图. 没 doc / 没产出时返回空字符串.
    """
    if result.db_path is None or not result.tables:
        return ""

    valid = [t for t in result.tables if not t.error]
    if not valid:
        return ""

    lines: list[str] = []
    lines.append("## 已从 doc 自动抽取的结构化表")
    lines.append(
        "每个 doc 已结构化为一张同名表, 落在 `context/db/<doc名>.db` "
        "(scaffold 已自动加载为 DuckDB 视图, 可直接用 SQL / pandas 查)."
    )
    lines.append("")
    for t in valid:
        ok_ratio = (t.n_ok / t.n_total) if t.n_total else 1.0
        warn = " ⚠ 抽取不完美" if ok_ratio < _LOW_QUALITY_OK_RATIO else ""
        pk_str = f", primary_key={t.primary_key}" if t.primary_key else ""
        db_loc = f"context/db/{t.db_name}" if t.db_name else "context/db/"
        lines.append(
            f"- 表 `{t.table_name}` ← `doc/{t.doc_name}` "
            f"(rows={t.n_rows_after_dedupe}, {db_loc}){warn}"
        )
        cols_str = ", ".join(t.column_names) if t.column_names else "(unknown)"
        lines.append(f"    业务列{pk_str}: {cols_str}")
    lines.append("")
    lines.append("使用须知:")
    lines.append(
        "1. **优先 SQL 查这些表**, 不要再 read_file 读 `doc/*.md` 原文 "
        "(doc 动辄上千行, 直接读会爆 token)."
    )
    lines.append(
        "2. 主键 `record_id` 为每条 record 的稳定标识; 同 record 的多段叙述已自动合并到一行, "
        "**不要再自己去重**."
    )
    lines.append(
        "3. 标 ⚠ 的表抽取质量存疑, 对该表的关键查询结果存疑时, 用 read_file 读 doc 原文兜底校验."
    )
    return "\n".join(lines)
