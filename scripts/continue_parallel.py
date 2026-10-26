"""Continue an interrupted run after changing only task concurrency.

The strict resume implementation remains unchanged. This explicit migration
validates the old receipts and records their provenance under the new identity.
No gold answers are read here.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'champion/src/data_agent_baseline'))
from dotenv import dotenv_values
from tools_v2.artifact_cache import atomic_json, digest_json
from tools_v2.run_state import reuse_predictions, run_identity


def validate_scheduling_change(old, current, reviewed_changes=None, allow_request_limit_change=False):
    expected = copy.deepcopy(old)
    expected['configuration']['MAX_PARALLEL'] = current['configuration']['MAX_PARALLEL']
    if allow_request_limit_change:
        expected['configuration']['MODEL_MAX_CONCURRENCY'] = current['configuration']['MODEL_MAX_CONCURRENCY']
    for path, change in (reviewed_changes or {}).items():
        if old['source'].get(path) != change['before'] or current['source'].get(path) != change['after']:
            raise ValueError('已审查源码变化的旧/新哈希不匹配')
        expected['source'][path] = change['after']
    if expected != current:
        raise ValueError('除 MAX_PARALLEL 外还有源码、输入、模型或配置变化，拒绝迁移')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--http-concurrency', type=int, default=4, choices=range(1, 6))
    parser.add_argument('--reviewed-changes', type=Path,
                        help='Explicit before/after source hash audit for a recorded speed variant')
    args = parser.parse_args()
    if args.parallel < 1:
        raise SystemExit('parallel must be positive')
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / 'runs'):
        raise SystemExit('Source must be under runs')
    if not (source / 'exit-code.txt').is_file():
        raise SystemExit('Previous launcher has not exited; wait for the task boundary')
    # Match the launcher configuration, without printing secrets or changing .env.
    allowed = {
        'MODEL_API_URL', 'MODEL_NAME', 'MODEL_API_KEY', 'ASR_BACKEND',
        'MAX_PARALLEL', 'MODEL_MAX_CONCURRENCY', 'MODEL_RPM',
        'MODEL_HTTP_429_RETRIES', 'MODEL_HTTP_TIMEOUT', 'MODEL_HTTP_CONNECT_TIMEOUT',
        'MODEL_THINKING_FORMAT', 'MODEL_API_PROFILE', 'MODEL_SOLVER_THINKING',
        'MODEL_VERSION_LABEL', 'KDD_DOC_CACHE_MODE', 'KDD_DOC_CACHE_DIR',
    }
    config = dotenv_values(os.environ.get('KDD_ENV_FILE', ROOT / '.env'))
    for key in allowed:
        if config.get(key) or (key == 'MODEL_API_KEY' and key in config):
            os.environ[key] = config[key] or ''
    os.environ.update(MAX_PARALLEL=str(args.parallel), MODEL_MAX_CONCURRENCY=str(args.http_concurrency),
                      KDD_DOC_CACHE_MODE='readwrite')
    if os.environ.get('MODEL_NAME') != 'ecnu-plus':
        raise SystemExit('This continuation is restricted to the current ecnu-plus experiment')
    parent = json.loads((source / 'run_manifest.json').read_text())
    old = parent['identity']
    if parent['identity_sha256'] != digest_json(old):
        raise SystemExit('Source manifest integrity check failed')
    current = run_identity(ROOT)
    reviewed = json.loads(args.reviewed_changes.read_text()) if args.reviewed_changes else {}
    validate_scheduling_change(old, current, reviewed, allow_request_limit_change=True)
    scheduling_changes = {key: [old['configuration'][key], current['configuration'][key]]
                          for key in ('MAX_PARALLEL', 'MODEL_MAX_CONCURRENCY')
                          if old['configuration'][key] != current['configuration'][key]}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    bridge = ROOT / 'runs' / f'scheduling_resume_{stamp}_{os.getpid()}'
    bridge.mkdir()
    tasks = json.loads((ROOT / 'data/public_reference_tasks.json').read_text())['task_ids']
    reused = reuse_predictions(bridge, source, old, tasks)
    for task in reused:
        receipt_path = bridge / 'predictions' / task / 'completion.json'
        receipt = json.loads(receipt_path.read_text())
        original = copy.deepcopy(receipt)
        receipt.update(run_identity=digest_json(current), scheduling_migration={
            'source_run': str(source), 'original_receipt': original,
            'change': scheduling_changes,
            'reviewed_source_changes': reviewed,
        })
        atomic_json(receipt_path, receipt)
    atomic_json(bridge / 'run_manifest.json', {
        'identity': current, 'identity_sha256': digest_json(current), 'tasks': tasks,
    })
    atomic_json(bridge / 'scheduling-migration.json', {
        'source_run': str(source), 'source_identity': parent['identity_sha256'],
        'target_identity': digest_json(current), 'reused_tasks': reused,
        'only_configuration_change': scheduling_changes,
        'reviewed_source_changes': reviewed,
        'score_scope': 'public60_with_recorded_speed_variant' if reviewed else 'public60_with_recorded_scheduling_change',
    })
    # Persist the requested concurrency in the existing private configuration.
    # Never copy credentials into experiment artifacts.
    env_file = Path(os.environ.get('KDD_ENV_FILE', ROOT / '.env')).resolve()
    from dotenv import set_key
    set_key(str(env_file), 'MAX_PARALLEL', str(args.parallel))
    set_key(str(env_file), 'MODEL_MAX_CONCURRENCY', str(args.http_concurrency))
    set_key(str(env_file), 'KDD_DOC_CACHE_MODE', 'readwrite')
    os.chmod(env_file, 0o600)
    env = os.environ.copy()
    env.update(KDD_ENV_FILE=str(env_file), KDD_RUN_TIMEOUT='24h')
    with (bridge / 'launch.log').open('w') as log:
        process = subprocess.Popen(
            ['bash', str(ROOT / 'scripts/run_tasks.sh'), '--resume', str(bridge), '--all'],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    for _ in range(240):
        matches = list((ROOT / 'runs').glob(f'baseline_*_{process.pid}'))
        if matches and (matches[0] / 'resume.json').exists():
            run = matches[0]
            atomic_json(source / 'continued-by.json', {'run_dir': str(run), 'bridge': str(bridge)})
            atomic_json(bridge / 'launched-run.json', {'run_dir': str(run), 'launcher_pid': process.pid})
            print(json.dumps({'run_dir': str(run), 'bridge': str(bridge),
                              'reused_tasks': reused, 'parallel': args.parallel,
                              'http_concurrency': args.http_concurrency}, ensure_ascii=False))
            return
        if process.poll() is not None:
            raise SystemExit(f'Continuation launch failed; inspect {bridge / "launch.log"}')
        time.sleep(0.5)
    raise SystemExit(f'Continuation still initializing; inspect {bridge / "launch.log"}')


if __name__ == '__main__':
    main()
