#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DRY_RUN=0
FAST=0
RESUME_FROM=''
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --fast) FAST=1; shift ;;
        --resume)
            [[ "$#" -ge 2 ]] || { echo '--resume 需要实验目录' >&2; exit 2; }
            RESUME_FROM="$2"; shift 2 ;;
        *) break ;;
    esac
done
# 配置文件优先于终端变量；缺失的字段沿用终端设置或 env.sh 默认值。
export KDD_ENV_FILE="${KDD_ENV_FILE:-$KDD_ROOT/.env}"
if [[ -f "$KDD_ENV_FILE" ]]; then
    while IFS= read -r -d '' item; do
        export "$item"
    done < <(python - "$KDD_ENV_FILE" <<'PY'
import sys
from dotenv import dotenv_values
values = dotenv_values(sys.argv[1])
for key in ['MODEL_API_URL', 'MODEL_NAME', 'MODEL_VIDEO_NAME', 'MODEL_API_KEY', 'ASR_BACKEND',
            'MAX_PARALLEL', 'MODEL_MAX_CONCURRENCY', 'MODEL_RPM',
            'MODEL_HTTP_429_RETRIES', 'MODEL_HTTP_TIMEOUT',
            'MODEL_HTTP_CONNECT_TIMEOUT', 'MODEL_THINKING_FORMAT',
            'MODEL_API_PROFILE', 'MODEL_SOLVER_THINKING', 'MODEL_VERSION_LABEL',
            'KDD_DOC_CACHE_MODE', 'KDD_DOC_CACHE_DIR']:
    if values.get(key) or (key == 'MODEL_API_KEY' and key in values):
        print(f'{key}={values[key] or ""}', end='\0')
PY
)
fi
if [[ "$FAST" == 1 ]]; then
    export KDD_DOC_CACHE_MODE=readwrite
    export MODEL_MAX_CONCURRENCY="$(python - <<'PY'
import os
from urllib.parse import urlsplit
host = urlsplit(os.environ['MODEL_API_URL']).hostname
name = os.environ['MODEL_NAME']
limit = os.environ['MODEL_MAX_CONCURRENCY']
if host == 'chat.ecnu.edu.cn' and os.environ.get('MODEL_VIDEO_NAME') == 'ecnu-max':
    limit = str(min(int(limit), 3))
elif host == 'chat.ecnu.edu.cn' and name in {'ecnu-plus', 'ecnu-max'}:
    limit = '4' if name == 'ecnu-plus' else '3'
elif host == 'api.deepseek.com' and os.environ['MODEL_API_PROFILE'] == 'deepseek':
    limit = '6'
print(limit)
PY
)"
fi
python - <<'PY'
import math, os
for key in ['MAX_PARALLEL', 'MODEL_MAX_CONCURRENCY']:
    if int(os.environ[key]) < 1:
        raise SystemExit(f'{key} 必须是正整数')
rpm = float(os.environ['MODEL_RPM'])
if not math.isfinite(rpm) or rpm < 0:
    raise SystemExit('MODEL_RPM 必须是非负有限数，0 表示不额外限速')
if os.environ['MODEL_THINKING_FORMAT'] not in {'legacy', 'ecnu', 'deepseek', 'none'}:
    raise SystemExit('MODEL_THINKING_FORMAT 必须为 legacy/ecnu/deepseek/none')
if os.environ['MODEL_API_PROFILE'] not in {'openai', 'deepseek'}:
    raise SystemExit('MODEL_API_PROFILE 必须为 openai/deepseek')
if os.environ['MODEL_SOLVER_THINKING'] not in {'default', 'enabled', 'disabled'}:
    raise SystemExit('MODEL_SOLVER_THINKING 必须为 default/enabled/disabled')
if os.environ['KDD_DOC_CACHE_MODE'] not in {'off', 'readwrite'}:
    raise SystemExit('KDD_DOC_CACHE_MODE 必须为 off/readwrite')
PY
export KDD_DOC_CACHE_DIR="$(python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['KDD_DOC_CACHE_DIR']).expanduser()
root = Path(os.environ['KDD_ROOT'])
p = (p if p.is_absolute() else root / p).resolve()
if p.is_relative_to(root / 'data'):
    raise SystemExit('文档缓存不能放入原始数据或解题输入目录')
print(p)
PY
)"
if [[ -z "${MODEL_API_KEY:-}" && "$DRY_RUN" == 0 ]]; then
    echo '未配置 MODEL_API_KEY；请在本机 .env 或终端设置令牌。' >&2
    exit 1
fi
if [[ ! -f "$KDD_ROOT/data/input60.complete.json" ]]; then
    echo '固定 60 题输入尚未准备完成，请先运行 prepare_assets.py data。' >&2
    exit 1
fi
CHECK_TASK_ARGS=()
if [[ "${1:-}" == '--all' && "$#" == 1 ]]; then
    RUN_PREFIX=baseline
    DEFAULT_TIMEOUT=8h
elif [[ "$#" == 1 && "$1" =~ ^task_[0-9]+(,task_[0-9]+)*$ ]]; then
    IFS=',' read -r -a CHECK_TASK_ARGS <<< "$1"
    declare -A task_seen=()
    for task in "${CHECK_TASK_ARGS[@]}"; do
        [[ -d "$INPUT_DIR/$task" ]] || { echo "公开输入中不存在 $task" >&2; exit 2; }
        [[ -z "${task_seen[$task]:-}" ]] || { echo "重复题号 $task" >&2; exit 2; }
        task_seen[$task]=1
    done
    RUN_PREFIX="smoke_$1"
    DEFAULT_TIMEOUT=30m
    if [[ "${#CHECK_TASK_ARGS[@]}" -gt 1 ]]; then
        RUN_PREFIX=subset
        DEFAULT_TIMEOUT=2h
    fi
else
    echo '用法：bash scripts/run_tasks.sh [--dry-run] [--fast] [--resume 实验目录] task_题号[,task_题号...] 或 --all' >&2
    exit 2
fi
if [[ "$DRY_RUN" == 1 ]]; then
    KDD_FAST_PREVIEW="$FAST" KDD_RESUME_PREVIEW="$RESUME_FROM" python - "${CHECK_TASK_ARGS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
tasks = sys.argv[1:] or json.loads((Path(os.environ['KDD_ROOT']) / 'data/public_reference_tasks.json').read_text())['task_ids']
print(json.dumps({
    'dry_run': True, 'task_count': len(tasks), 'tasks': tasks,
    'model': os.environ['MODEL_NAME'], 'video_model': os.environ.get('MODEL_VIDEO_NAME', ''), 'api_url': os.environ['MODEL_API_URL'],
    'api_key_configured': bool(os.environ.get('MODEL_API_KEY')),
    'task_parallel': int(os.environ['MAX_PARALLEL']),
    'request_concurrency': int(os.environ['MODEL_MAX_CONCURRENCY']),
    'rpm': float(os.environ['MODEL_RPM']),
    'profile': os.environ['MODEL_API_PROFILE'],
    'nothink_format': os.environ['MODEL_THINKING_FORMAT'],
    'solver_thinking': os.environ['MODEL_SOLVER_THINKING'],
    'fast': os.environ['KDD_FAST_PREVIEW'] == '1',
    'document_cache': os.environ['KDD_DOC_CACHE_MODE'],
    'cache_dir': os.environ['KDD_DOC_CACHE_DIR'],
    'resume_from': os.environ['KDD_RESUME_PREVIEW'] or None,
}, ensure_ascii=False, indent=2))
PY
    exit 0
fi
if [[ -n "$RESUME_FROM" ]]; then
    [[ "$RESUME_FROM" == /* ]] || RESUME_FROM="$KDD_ROOT/$RESUME_FROM"
    [[ -f "$RESUME_FROM/run_manifest.json" ]] || {
        echo '来源没有本版 run_manifest.json，无法严格续跑；不会调用模型。' >&2
        exit 2
    }
fi
export RUN_ID="${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$"
export RUN_DIR="$KDD_ROOT/runs/$RUN_ID"
export OUTPUT_DIR="$RUN_DIR/predictions"
export TEMP_DIR="$RUN_DIR/temp"
mkdir -p "$OUTPUT_DIR" "$TEMP_DIR"
cd "$KDD_ROOT/champion"
{
    date -Is
    git rev-parse HEAD
    python --version
    uv --version
    sha256sum uv.lock
    printf 'MODEL_NAME=%s\nMODEL_API_URL=%s\nINPUT_DIR=%s\n' "$MODEL_NAME" "$MODEL_API_URL" "$INPUT_DIR"
    printf 'MAX_PARALLEL=%s\nMODEL_MAX_CONCURRENCY=%s\nMODEL_RPM=%s\nMODEL_THINKING_FORMAT=%s\nASR_BACKEND=%s\n' \
        "$MAX_PARALLEL" "$MODEL_MAX_CONCURRENCY" "$MODEL_RPM" "$MODEL_THINKING_FORMAT" "$ASR_BACKEND"
    printf 'MODEL_API_PROFILE=%s\nMODEL_SOLVER_THINKING=%s\nMODEL_VERSION_LABEL=%s\n' \
        "$MODEL_API_PROFILE" "$MODEL_SOLVER_THINKING" "${MODEL_VERSION_LABEL:-unspecified}"
    printf 'MODEL_HTTP_TIMEOUT=%s\nMODEL_HTTP_CONNECT_TIMEOUT=%s\nMODEL_HTTP_429_RETRIES=%s\n' \
        "${MODEL_HTTP_TIMEOUT:-300}" "${MODEL_HTTP_CONNECT_TIMEOUT:-5}" "${MODEL_HTTP_429_RETRIES:-5}"
    printf 'KDD_DOC_CACHE_MODE=%s\nKDD_DOC_CACHE_DIR=%s\nFAST=%s\nRESUME_FROM=%s\n' \
        "$KDD_DOC_CACHE_MODE" "$KDD_DOC_CACHE_DIR" "$FAST" "$RESUME_FROM"
    printf 'RUN_TIMEOUT=%s\n' "${KDD_RUN_TIMEOUT:-$DEFAULT_TIMEOUT}"
} > "$RUN_DIR/versions.txt"
tar --exclude='__pycache__' -czf "$RUN_DIR/source.tar.gz" -C "$KDD_ROOT" \
    champion/src champion/pyproject.toml champion/uv.lock champion/uv.toml scripts
RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_ARGS=(--resume "$RESUME_FROM")
fi
PENDING_TASKS="$(python "$KDD_ROOT/scripts/prepare_run.py" "${RESUME_ARGS[@]}" "${CHECK_TASK_ARGS[@]}")"
export KDD_RUN_IDENTITY="$(python -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["RUN_DIR"])/"run_manifest.json").read_text())["identity_sha256"])')"
echo "实验目录：$RUN_DIR"
set +e
if [[ -n "$PENDING_TASKS" ]]; then
    timeout "${KDD_RUN_TIMEOUT:-$DEFAULT_TIMEOUT}" python src/data_agent_baseline/zz_agent_v2.py \
        --only-tasks "$PENDING_TASKS" --max-parallel "$MAX_PARALLEL" 2>&1 | tee "$RUN_DIR/console.log"
    STATUS=${PIPESTATUS[0]}
else
    echo '所选任务均有匹配且校验通过的成功预测，无需再次调用模型。' | tee "$RUN_DIR/console.log"
    STATUS=0
fi
python "$KDD_ROOT/scripts/check_run_outputs.py" "${CHECK_TASK_ARGS[@]}"
OUTPUT_STATUS=$?
if [[ "$STATUS" == 0 && "$OUTPUT_STATUS" != 0 ]]; then
    STATUS=$OUTPUT_STATUS
fi
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit-code.txt"
exit "$STATUS"
