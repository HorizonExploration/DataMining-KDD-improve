"""Compare prediction.csv against gold.csv per task using column-signature matching.

Scoring rules (per the official KDD Cup 2026 DataAgent-Bench spec):
  * Build a "column signature" by normalizing values then sorting them per column.
  * Ignore column names and row order; match purely on column content.
  * Duplicate column signatures must match the same number of times (multiset).
  * Score = max(0, Recall - lambda * (Extra / Predicted))
      Recall  = Matched / GoldCols
      Extra   = PredictedCols - Matched
  * Missing prediction.csv -> score 0.

Normalization (see spec 6.5):
  * Nulls ("", "null", "none", "nan", "nat", "<na>", case-insensitive) -> ""
  * Numeric -> Decimal rounded to 2 decimals, ROUND_HALF_UP
  * Date -> ISO YYYY-MM-DD
  * DateTime -> UTC ISO with Z if tz present, else raw ISO
  * String -> strip whitespace, case-sensitive
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import List, Optional

import pandas as pd


# Penalty weight for extra (unmatched) columns. Spec leaves this as a tunable
# constant; 0.1 is a reasonable default that lightly penalises redundancy.
LAMBDA_EXTRA = 0.1

NULL_TOKENS = {"", "null", "none", "nan", "nat", "<na>"}

# Pre-compiled patterns for date / datetime detection.
_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_DATETIME_HINT_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[ T]")


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

def _is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        try:
            import math
            if math.isnan(value):
                return True
        except Exception:
            pass
    s = str(value).strip().lower()
    return s in NULL_TOKENS


def _normalize_numeric(s: str) -> Optional[str]:
    try:
        d = Decimal(s.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    # 拒绝 NaN/Infinity 等特殊值: 它们能被 Decimal() 解析, 但 quantize 会抛 InvalidOperation.
    if not d.is_finite():
        return None
    try:
        q = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        # 超大数 quantize 会超出默认精度上限而失败, 视为非数值处理.
        return None
    if q == 0:
        # 避免 (-0.005, 0) 区间的极小负值产出 "-0.00", 与正侧的 "0.00" 失配.
        q = q.copy_abs()
    return format(q, "f")


def _normalize_date(s: str) -> Optional[str]:
    if not _DATE_RE.match(s):
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        # Try with single-digit month/day.
        parts = s.split("-")
        try:
            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            return None
    return dt.strftime("%Y-%m-%d")


def _normalize_datetime(s: str) -> Optional[str]:
    if not _DATETIME_HINT_RE.match(s):
        return None
    candidate = s.replace(" ", "T")
    # Handle trailing Z by converting to +00:00 for fromisoformat.
    iso = candidate
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def normalize_value(value) -> str:
    """Normalize a single cell to its canonical string form per spec 6.5."""
    if _is_null(value):
        return ""
    s = str(value).replace("\r", "").replace("\n", "").strip()
    if s == "":
        return ""

    # Numeric first (cheap, common).
    num = _normalize_numeric(s)
    if num is not None and re.match(r"^[\-+]?[\d,]*\.?\d+([eE][\-+]?\d+)?$", s):
        return num

    # Date / datetime
    d = _normalize_date(s)
    if d is not None:
        return d
    dt = _normalize_datetime(s)
    if dt is not None:
        return dt

    return s  # case-sensitive string


def column_signature(values: List) -> tuple:
    """Sorted tuple of normalized values for a column."""
    return tuple(sorted(normalize_value(v) for v in values))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_csv(
    gold_csv: Path,
    pred_csv: Path,
    lambda_extra: float = LAMBDA_EXTRA,
) -> dict:
    """Compare prediction vs gold using column-signature matching.

    Returns a dict with score and diagnostic fields.
    """
    out = {
        "gold_csv": str(gold_csv),
        "pred_csv": str(pred_csv),
        "score": 0.0,
        "recall": 0.0,
        "matched_cols": 0,
        "gold_cols": 0,
        "pred_cols": 0,
        "extra_cols": 0,
        "error": None,
    }

    if not gold_csv.exists():
        out["error"] = "gold missing"
        return out
    if not pred_csv.exists():
        out["error"] = "prediction missing"
        return out

    try:
        gold = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
        pred = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    except Exception as e:
        out["error"] = f"read error: {e}"
        return out

    gold_sigs = Counter(column_signature(gold[c].tolist()) for c in gold.columns)
    pred_sigs = Counter(column_signature(pred[c].tolist()) for c in pred.columns)

    # Multiset intersection size = matched columns
    matched = sum((gold_sigs & pred_sigs).values())
    n_gold = sum(gold_sigs.values())
    n_pred = sum(pred_sigs.values())
    extra = max(0, n_pred - matched)

    recall = matched / n_gold if n_gold else 0.0
    penalty = lambda_extra * (extra / n_pred) if n_pred else 0.0
    score = max(0.0, recall - penalty)

    out.update({
        "score": round(score, 4),
        "recall": round(recall, 4),
        "matched_cols": matched,
        "gold_cols": n_gold,
        "pred_cols": n_pred,
        "extra_cols": extra,
    })
    return out


# ---------------------------------------------------------------------------
# 运行统计 (轮数 / 耗时) — 从 pred_path/logs/<task>/ 读取
# ---------------------------------------------------------------------------

def _count_solver_turns(task_log_dir: Path) -> Optional[int]:
    """统计该 task 的 solver react 轮数.

    口径与 zz_agent_v2.aggregate_run_stats 一致: 一个 ModelResponse 视为一轮.
    solver_allmsg 落盘后, response 类型消息数 == 轮数. 多 attempt 时取各
    attempt 文件中的最大值 (captured_msgs 在 attempt 间会重置, 累计历史在最后
    一次 attempt 文件里最全; 取 max 避免把重试轮数叠加成虚高).
    """
    files = sorted(task_log_dir.glob("solver_allmsg.*.json"))
    if not files:
        return None
    best = None
    for f in files:
        try:
            msgs = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(msgs, list):
            continue
        turns = sum(1 for m in msgs if isinstance(m, dict) and m.get("kind") == "response")
        if best is None or turns > best:
            best = turns
    return best


def _read_task_elapsed(task_log_dir: Path) -> Optional[float]:
    """读取该 task 的总墙钟耗时 (秒), 来自 timing.json 的 total_wall_s."""
    tj = task_log_dir / "timing.json"
    if not tj.exists():
        return None
    try:
        d = json.loads(tj.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = d.get("total_wall_s")
    return float(v) if isinstance(v, (int, float)) else None


def read_run_stats(pred_path: Path) -> dict:
    """返回 {task_id: {"turns": int|None, "elapsed_s": float|None}}.

    数据源: pred_path/logs/<task_id>/ (solver_allmsg.*.json + timing.json).
    logs 目录不存在时返回空 dict (评分不受影响, 只是没有轮数/耗时列).
    """
    logs_dir = pred_path / "logs"
    if not logs_dir.exists():
        return {}
    out = {}
    for td in logs_dir.iterdir():
        if not td.is_dir() or not td.name.startswith("task_"):
            continue
        out[td.name] = {
            "turns": _count_solver_turns(td),
            "elapsed_s": _read_task_elapsed(td),
        }
    return out


_DOC_USAGE_CACHE: Optional[dict] = None


def load_doc_usage(gold_path: Path) -> dict:
    """读取 gold_path/_doc_usage.json, 返回 {task_id: verdict}.

    verdict 枚举: DOC_REQUIRED / DOC_OR_STRUCT / STRUCT_ONLY.
    文件不存在时返回 {}; 单个进程内缓存, 避免每个 task 反复读盘.
    """
    global _DOC_USAGE_CACHE
    if _DOC_USAGE_CACHE is not None:
        return _DOC_USAGE_CACHE
    f = gold_path / "_doc_usage.json"
    if not f.exists():
        _DOC_USAGE_CACHE = {}
        return _DOC_USAGE_CACHE
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        tasks = d.get("tasks", {}) if isinstance(d, dict) else {}
        _DOC_USAGE_CACHE = {
            tid: (v.get("verdict") if isinstance(v, dict) else None)
            for tid, v in tasks.items()
        }
    except Exception:
        _DOC_USAGE_CACHE = {}
    return _DOC_USAGE_CACHE


def task_has_video(pred_path: Path, task_id: str) -> bool:
    """判断 task 是否带视频. 优先看 pred_path/<task>/context/video, 回退看 input.

    视频文件夹存在且非空即视为视频题; 缺失/空目录视为无视频.
    """
    candidates = [
        pred_path / task_id / "context" / "video",
        Path("/Users/zhe/Documents/work/20260400-data-agent/phase2/input") / task_id / "context" / "video",
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            try:
                if any(d.iterdir()):
                    return True
            except Exception:
                pass
    return False


def _percentile(sorted_vals: List[float], p: float) -> float:
    """线性插值分位数 (p in [0,100]); sorted_vals 须已升序且非空."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def summarize_series(values: List) -> dict:
    """对一组数值算 count/mean/TP50/75/90/95 (忽略 None/NaN). 空集返回 0 占位."""
    import math
    nums = sorted(
        float(v) for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    )
    if not nums:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "count": len(nums),
        "mean": round(sum(nums) / len(nums), 2),
        "p50": round(_percentile(nums, 50), 2),
        "p75": round(_percentile(nums, 75), 2),
        "p90": round(_percentile(nums, 90), 2),
        "p95": round(_percentile(nums, 95), 2),
    }


# ---------------------------------------------------------------------------
# Per-task / batch helpers
# ---------------------------------------------------------------------------

def score_task(
    task_id: str,
    gold_path: Path,
    pred_path: Path,
    lambda_extra: float = LAMBDA_EXTRA,
    write_score: bool = True,
) -> dict:
    gold_csv = gold_path / task_id / "gold.csv"
    # Support both layouts:
    #   * legacy: <pred_path>/<task_id>/workdir/prediction.csv
    #   * new   : <pred_path>/<task_id>/prediction.csv  (artifacts/runs_0503/<run_id>/...)
    pred_csv_legacy = pred_path / task_id / "workdir" / "prediction.csv"
    pred_csv_flat = pred_path / task_id / "prediction.csv"
    if pred_csv_legacy.exists():
        pred_csv = pred_csv_legacy
    elif pred_csv_flat.exists():
        pred_csv = pred_csv_flat
    else:
        # Default to flat layout for the "missing" diagnostic; fall back to
        # legacy if its task dir exists but the file doesn't.
        pred_csv = pred_csv_flat if (pred_path / task_id).exists() and not (pred_path / task_id / "workdir").exists() else pred_csv_legacy
    res = compare_csv(gold_csv, pred_csv, lambda_extra=lambda_extra)
    res["task_id"] = task_id

    if write_score and res.get("error") is None:
        out_dir = pred_path / task_id
        if out_dir.exists():
            (out_dir / "score.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return res


def score_all(
    gold_path: Path,
    pred_path: Path,
    lambda_extra: float = LAMBDA_EXTRA,
    write_per_task: bool = True,
    write_summary: bool = True,
) -> pd.DataFrame:
    """Score every task whose gold.csv exists under gold_path.

    Tasks missing from pred_path (or missing prediction.csv) are still counted
    with score=0 so the average reflects coverage.
    """
    rows = []
    if not gold_path.exists():
        return pd.DataFrame(rows)
    gold_task_dirs = sorted(
        [p for p in gold_path.iterdir()
         if p.is_dir() and p.name.startswith("task_")
         and (p / "gold.csv").exists()],
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
    )
    for td in gold_task_dirs:
        rows.append(score_task(
            td.name, gold_path, pred_path,
            lambda_extra=lambda_extra,
            write_score=write_per_task,
        ))
    df = pd.DataFrame(rows)

    # 合并运行统计 (轮数 / 耗时), 数据源 pred_path/logs/<task>/. logs 缺失则跳过.
    run_stats = read_run_stats(pred_path)
    if run_stats and len(df):
        df["turns"] = df["task_id"].map(lambda t: (run_stats.get(t) or {}).get("turns"))
        df["elapsed_s"] = df["task_id"].map(lambda t: (run_stats.get(t) or {}).get("elapsed_s"))

    # 是否带视频 (用于汇总时分组对比)
    if len(df):
        df["has_video"] = df["task_id"].map(lambda t: task_has_video(pred_path, t))

    # doc 使用判定 (DOC_REQUIRED / DOC_OR_STRUCT / STRUCT_ONLY), 数据源
    # gold_path/_doc_usage.json. 文件缺失时该列为 None, 不影响其它统计.
    doc_usage = load_doc_usage(gold_path)
    if len(df) and doc_usage:
        df["doc_usage"] = df["task_id"].map(lambda t: doc_usage.get(t))

    cols = ["task_id", "score", "recall", "matched_cols", "gold_cols",
            "pred_cols", "extra_cols", "turns", "elapsed_s", "has_video",
            "doc_usage", "error"]
    df = df[[c for c in cols if c in df.columns]]

    if write_summary and pred_path.exists():
        df.to_csv(pred_path / "scores.csv", index=False)

        # 已产出 prediction.csv 的子集(error != "prediction missing"),
        # 用于观察"只看已提交任务"时的得分水平.
        has_pred = df[df["error"] != "prediction missing"] if len(df) else df

        def _group_stats(sub: pd.DataFrame) -> dict:
            if not len(sub):
                return {"tasks": 0, "mean_score": 0.0, "mean_recall": 0.0}
            return {
                "tasks": int(len(sub)),
                "mean_score": round(float(sub["score"].mean()), 4),
                "mean_recall": round(float(sub["recall"].mean()), 4),
            }

        video_df = df[df["has_video"] == True] if "has_video" in df.columns else df.iloc[0:0]
        novideo_df = df[df["has_video"] == False] if "has_video" in df.columns else df.iloc[0:0]

        # doc 使用分组 (DOC_REQUIRED / DOC_OR_STRUCT / STRUCT_ONLY) + 需 doc 合并组
        if "doc_usage" in df.columns:
            doc_req_df = df[df["doc_usage"] == "DOC_REQUIRED"]
            doc_or_df = df[df["doc_usage"] == "DOC_OR_STRUCT"]
            struct_df = df[df["doc_usage"] == "STRUCT_ONLY"]
            doc_any_df = df[df["doc_usage"].isin(["DOC_REQUIRED", "DOC_OR_STRUCT"])]
        else:
            empty = df.iloc[0:0]
            doc_req_df = doc_or_df = struct_df = doc_any_df = empty

        summary = {
            "tasks_scored": int(len(df)),
            "mean_score": float(df["score"].mean()) if len(df) else 0.0,
            "mean_recall": float(df["recall"].mean()) if len(df) else 0.0,
            "tasks_with_prediction": int(len(has_pred)),
            "mean_score_with_prediction": float(has_pred["score"].mean()) if len(has_pred) else 0.0,
            "mean_recall_with_prediction": float(has_pred["recall"].mean()) if len(has_pred) else 0.0,
            "by_video": {
                "with_video": _group_stats(video_df),
                "without_video": _group_stats(novideo_df),
                "note": "has_video = pred_path/<task>/context/video 非空 (回退到 phase2/input)",
            },
            "by_doc_usage": {
                "DOC_REQUIRED": _group_stats(doc_req_df),
                "DOC_OR_STRUCT": _group_stats(doc_or_df),
                "STRUCT_ONLY": _group_stats(struct_df),
                "DOC_NEEDED (REQUIRED+OR_STRUCT)": _group_stats(doc_any_df),
                "note": "doc_usage 来自 gold_path/_doc_usage.json; DOC_REQUIRED=gold 单元格仅在 doc-extracted 出现, DOC_OR_STRUCT=两路都覆盖, STRUCT_ONLY=结构化即可",
            },
            "lambda_extra": lambda_extra,
            "missing_predictions": df.loc[
                df["error"] == "prediction missing", "task_id"
            ].tolist(),
            "run_stats": {
                "turns": summarize_series(df["turns"].tolist()) if "turns" in df else {},
                "elapsed_s": summarize_series(df["elapsed_s"].tolist()) if "elapsed_s" in df else {},
                "note": "turns/elapsed_s 来自 pred_path/logs/<task>/; mean+TP50/75/90/95 分位",
            },
            "scoring_formula": {
                "per_task_recall": "matched_cols / gold_cols",
                "per_task_penalty": "lambda_extra * (extra_cols / pred_cols)",
                "per_task_score": "max(0, recall - lambda_extra * (extra_cols / pred_cols))",
                "final_score": "mean(per_task_score over all tasks)",
                "notes": [
                    "matched_cols = column-signature multiset intersection between prediction and gold",
                    "extra_cols = max(0, pred_cols - matched_cols)",
                    "missing prediction.csv -> per_task_score = 0",
                    "to recompute with a different lambda: re-read each <task>/score.json,",
                    "  take matched_cols/gold_cols/pred_cols, plug into the formula above,",
                    "  then average across tasks.",
                ],
            },
        }
        (pred_path / "scores_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return df


def recompute_with_lambda(pred_path: Path, lambda_extra: float) -> dict:
    """Recompute mean score from saved per-task score.json files using a new lambda.

    Reads pred_path/<task>/score.json (which contains matched_cols, gold_cols,
    pred_cols), applies the formula score = max(0, recall - lambda * extra/pred),
    and returns {task_id: score, ..., "_mean_score": float}.
    """
    out = {}
    for task_dir in sorted(pred_path.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        sj = task_dir / "score.json"
        if not sj.exists():
            continue
        d = json.loads(sj.read_text(encoding="utf-8"))
        gold_c = d.get("gold_cols", 0) or 0
        pred_c = d.get("pred_cols", 0) or 0
        matched = d.get("matched_cols", 0) or 0
        extra = max(0, pred_c - matched)
        recall = matched / gold_c if gold_c else 0.0
        penalty = lambda_extra * (extra / pred_c) if pred_c else 0.0
        score = max(0.0, recall - penalty)
        if d.get("error") == "prediction missing":
            score = 0.0
        out[task_dir.name] = round(score, 4)
    if out:
        out["_mean_score"] = round(sum(out.values()) / len(out), 4)
        out["_lambda_extra"] = lambda_extra
    return out


DEFAULT_PRED_PATH = "/Users/zhe/Documents/work/20260400-data-agent/workdir/temp_20260509_172524"
DEFAULT_PRED_PATH = Path(DEFAULT_PRED_PATH)

# 当只传入 run 名字 (如 temp_20260627_172740) 时, 在此目录下递归搜索定位 run 目录.
RUN_SEARCH_ROOT = Path("/Users/zhe/Documents/work/20260400-data-agent/phase2/")


def resolve_pred_path(raw: Path) -> Path:
    """把用户输入解析成实际的 run 目录.

    - 已是存在的路径 (完整或相对) -> 原样返回.
    - 否则当作 run 名字, 在 RUN_SEARCH_ROOT 下递归查找同名目录.
    """
    if raw.exists():
        return raw
    name = raw.name  # 容忍传入带斜杠的片段
    if RUN_SEARCH_ROOT.is_dir():
        # 先看根目录直接子目录 (最常见), 再递归兜底.
        direct = RUN_SEARCH_ROOT / name
        if direct.is_dir():
            return direct
        matches = sorted(p for p in RUN_SEARCH_ROOT.rglob(name) if p.is_dir())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"[警告] 在 {RUN_SEARCH_ROOT} 下找到多个名为 '{name}' 的目录, 使用第一个:")
            for m in matches:
                print(f"  - {m}")
            return matches[0]
    raise SystemExit(
        f"[错误] 无法定位 run: '{raw}'\n"
        f"  既不是已存在的路径, 也未在 {RUN_SEARCH_ROOT} 下找到名为 '{name}' 的目录."
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare prediction.csv against gold.csv per task using column-signature matching."
    )
    parser.add_argument(
        "pred_path",
        type=Path,
        nargs="?",
        default=DEFAULT_PRED_PATH,
        help="run 目录路径, 或仅 run 名字 (如 temp_20260627_172740, 会在 phase2/ 下自动搜索).",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=Path('/Users/zhe/Documents/work/20260400-data-agent/phase2/gold/'),
        help="Path to the gold directory (containing <task_id>/gold.csv).",
    )
    args = parser.parse_args()

    gold_path = args.gold_path
    pred_path = resolve_pred_path(args.pred_path)
    print(f"[run目录] {pred_path}")


    df = score_all(gold_path, pred_path)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print("\n=== Summary ===")
    print(f"tasks scored : {len(df)}")
    print(f"mean score   : {df['score'].mean():.4f}")
    print(f"mean recall  : {df['recall'].mean():.4f}")

    has_pred = df[df["error"] != "prediction missing"] if len(df) else df
    print(f"\n--- 已产出 prediction.csv 的子集 ---")
    print(f"tasks w/ pred: {len(has_pred)} / {len(df)}")
    if len(has_pred):
        print(f"mean score   : {has_pred['score'].mean():.4f}")
        print(f"mean recall  : {has_pred['recall'].mean():.4f}")

    # ---- 视频 / 非视频 分组统计 ----
    if "has_video" in df.columns and len(df):
        vdf = df[df["has_video"] == True]
        ndf = df[df["has_video"] == False]
        print(f"\n--- 视频 / 非视频 分组 ---")
        print(f"有视频 (n={len(vdf):>2}): mean_score={vdf['score'].mean():.4f} | mean_recall={vdf['recall'].mean():.4f}"
              if len(vdf) else "有视频 : (无)")
        print(f"无视频 (n={len(ndf):>2}): mean_score={ndf['score'].mean():.4f} | mean_recall={ndf['recall'].mean():.4f}"
              if len(ndf) else "无视频 : (无)")

    # ---- doc 使用情况分组 (DOC_REQUIRED / DOC_OR_STRUCT / STRUCT_ONLY + 合并需 doc) ----
    if "doc_usage" in df.columns and len(df):
        print(f"\n--- doc 使用情况分组 (来自 gold/_doc_usage.json) ---")
        for label in ["DOC_REQUIRED", "DOC_OR_STRUCT", "STRUCT_ONLY"]:
            sub = df[df["doc_usage"] == label]
            if len(sub):
                print(f"{label:<14} (n={len(sub):>2}): mean_score={sub['score'].mean():.4f} | "
                      f"mean_recall={sub['recall'].mean():.4f}")
        any_doc = df[df["doc_usage"].isin(["DOC_REQUIRED", "DOC_OR_STRUCT"])]
        if len(any_doc):
            print(f"{'DOC_NEEDED':<14} (n={len(any_doc):>2}): mean_score={any_doc['score'].mean():.4f} | "
                  f"mean_recall={any_doc['recall'].mean():.4f}   (REQUIRED+OR_STRUCT 合并)")

    # ---- 运行统计: 平均轮数 / 平均耗时 + 分位 (数据源 pred_path/logs/) ----
    if "turns" in df.columns or "elapsed_s" in df.columns:
        print(f"\n--- 运行统计 (来自 logs/) ---")
        if "turns" in df.columns:
            ts = summarize_series(df["turns"].tolist())
            if ts["count"]:
                print(f"轮数  (n={ts['count']:>2}): mean={ts['mean']:.1f} | "
                      f"TP50={ts['p50']:.0f} TP75={ts['p75']:.0f} "
                      f"TP90={ts['p90']:.0f} TP95={ts['p95']:.0f}")
            else:
                print("轮数  : (无数据, logs/ 中未找到 solver_allmsg)")
        if "elapsed_s" in df.columns:
            es = summarize_series(df["elapsed_s"].tolist())
            if es["count"]:
                print(f"耗时s (n={es['count']:>2}): mean={es['mean']:.1f} | "
                      f"TP50={es['p50']:.1f} TP75={es['p75']:.1f} "
                      f"TP90={es['p90']:.1f} TP95={es['p95']:.1f}")
            else:
                print("耗时s : (无数据, logs/ 中未找到 timing.json)")
    else:
        print(f"\n--- 运行统计 ---  (未找到 {pred_path / 'logs'}, 跳过轮数/耗时)")

    print(f"\nwritten      : {pred_path / 'scores.csv'}")
    print(f"               {pred_path / 'scores_summary.json'}")
    print(f"               <task>/score.json")
