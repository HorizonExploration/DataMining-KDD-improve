"""Strict resume receipts; only verified successful predictions can be reused."""
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tools_v2.artifact_cache import atomic_json, digest_json, file_digest


def valid_csv(path: Path) -> bool:
    try:
        with path.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.reader(source, strict=True)
            header = next(reader)
            return bool(header) and all(len(row) == len(header) for row in reader)
    except (OSError, UnicodeError, StopIteration, csv.Error):
        return False


def run_identity(root: Path) -> dict:
    source = root / 'champion/src'
    files = sorted(source.rglob('*.py'))
    files += [root / 'champion' / name for name in ('uv.lock', 'uv.toml', 'pyproject.toml')]
    files += [root / 'scripts' / name for name in ('run_tasks.sh', 'env.sh', 'prepare_run.py', 'check_run_outputs.py')]
    task_ids = json.loads((root / 'data/public_reference_tasks.json').read_text())['task_ids']
    inputs = {}
    for task in task_ids:
        directory = root / 'data/input60' / task
        inputs[task] = digest_json({str(p.relative_to(directory)): file_digest(p)
                                   for p in sorted(directory.rglob('*')) if p.is_file()})
    asr = {}
    if os.environ.get('ASR_BACKEND', 'local') == 'local':
        for name in ('tiny', 'medium'):
            for filename in ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt'):
                path = root / 'champion/asr_models' / name / filename
                asr[f'{name}/{filename}'] = file_digest(path)
    keys = ('MODEL_NAME', 'MODEL_VIDEO_NAME', 'MODEL_VERSION_LABEL', 'MODEL_API_PROFILE', 'MODEL_THINKING_FORMAT',
            'MODEL_SOLVER_THINKING', 'ASR_BACKEND', 'KDD_DOC_CACHE_MODE', 'MAX_PARALLEL',
            'MODEL_MAX_CONCURRENCY', 'MODEL_RPM', 'MODEL_HTTP_TIMEOUT',
            'MODEL_HTTP_CONNECT_TIMEOUT', 'MODEL_HTTP_429_RETRIES')
    return {
        'schema': 1, 'source': {str(p.relative_to(root)): file_digest(p) for p in files},
        'inputs': inputs, 'asr_weights': asr,
        'configuration': {k: os.environ.get(k, '') for k in keys},
        'endpoint_sha256': digest_json(os.environ.get('MODEL_API_URL', '')),
    }


def write_completion(task_output: Path, identity: str) -> bool:
    prediction = task_output / 'prediction.csv'
    if not identity or not valid_csv(prediction):
        return False
    atomic_json(task_output / 'completion.json', {
        'success': True, 'run_identity': identity, 'prediction_sha256': file_digest(prediction),
        'completed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    })
    return True


def reuse_predictions(run_dir: Path, source: Path, identity: dict, tasks: list[str]) -> list[str]:
    manifest = json.loads((source / 'run_manifest.json').read_text())
    fingerprint = digest_json(identity)
    if not isinstance(manifest, dict) or manifest.get('identity') != identity or manifest.get('identity_sha256') != fingerprint:
        raise ValueError('续跑来源的源码、输入、模型或实验设置不同；请创建独立实验')
    reused = []
    for task in tasks:
        parent = source / 'predictions' / task
        prediction = parent / 'prediction.csv'
        try:
            receipt = json.loads((parent / 'completion.json').read_text())
            if not isinstance(receipt, dict):
                continue
            if receipt.get('success') is not True or receipt.get('run_identity') != fingerprint:
                continue
            if not valid_csv(prediction) or file_digest(prediction) != receipt['prediction_sha256']:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        target = run_dir / 'predictions' / task
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prediction, target / 'prediction.csv')
        shutil.copy2(parent / 'completion.json', target / 'completion.json')
        reused.append(task)
    return reused
