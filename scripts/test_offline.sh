#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh
python scripts/test_model_http.py
python scripts/test_shared_budget.py
python scripts/test_ecnu_routing.py
python scripts/test_speed_features.py
