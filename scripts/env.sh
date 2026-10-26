# 使用方式：source /home/yosoro126/DM-KDD/KDD/scripts/env.sh
export KDD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$KDD_ROOT/champion/.venv/bin:$KDD_ROOT/tools/bin:$HOME/.local/bin:$PATH"
export HF_HOME="$KDD_ROOT/downloads/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export MODEL_API_URL="${MODEL_API_URL:-https://chat.ecnu.edu.cn/open/api/v1}"
export MODEL_NAME="${MODEL_NAME:-ecnu-plus}"
export ASR_BACKEND="${ASR_BACKEND:-local}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export MODEL_MAX_CONCURRENCY="${MODEL_MAX_CONCURRENCY:-3}"
export MODEL_RPM="${MODEL_RPM:-0}"
export MODEL_THINKING_FORMAT="${MODEL_THINKING_FORMAT:-legacy}"
export MODEL_API_PROFILE="${MODEL_API_PROFILE:-openai}"
export MODEL_SOLVER_THINKING="${MODEL_SOLVER_THINKING:-default}"
export KDD_DOC_CACHE_MODE="${KDD_DOC_CACHE_MODE:-off}"
export KDD_DOC_CACHE_DIR="${KDD_DOC_CACHE_DIR:-$KDD_ROOT/cache/doc_extract}"
export INPUT_DIR="$KDD_ROOT/data/input60"
# 禁用参考代码的多地址默认值，使用明确的单一模型接口。
export MODEL_API_URLS=''
export DEFAULT_MODEL_API_URLS=''
