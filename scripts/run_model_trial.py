"""Isolated, explicitly labelled ECNU model trial; preserve the full baseline."""
import argparse
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'champion/src/data_agent_baseline'))
from tools_v2.artifact_cache import atomic_json, digest_json
from tools_v2.run_state import run_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ecnu-max', 'ecnu-plus'], required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    args = parser.parse_args()
    tasks = json.loads((ROOT / 'data/public_reference_tasks.json').read_text())['task_ids']
    if len(args.tasks) != len(set(args.tasks)) or not set(args.tasks).issubset(tasks):
        raise SystemExit('Choose unique public60 task IDs')
    for key, value in dotenv_values(ROOT / '.env').items():
        if value is not None and key.isupper():
            os.environ[key] = value
    if not os.environ.get('MODEL_API_KEY'):
        raise SystemExit('MODEL_API_KEY is missing')
    os.environ.update(MODEL_NAME=args.model, MAX_PARALLEL='1', MODEL_MAX_CONCURRENCY='2',
                      MODEL_API_PROFILE='openai', MODEL_THINKING_FORMAT='ecnu',
                      MODEL_SOLVER_THINKING='disabled', KDD_DOC_CACHE_MODE='readwrite',
                      MODEL_VERSION_LABEL=f'{args.model}@{datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()}')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run = ROOT / 'runs/baseline_full_20261025/model_trials' / f'{args.model.replace("-", "_")}_{stamp}_{os.getpid()}'
    run.mkdir(parents=True)
    os.environ.update(RUN_DIR=str(run), OUTPUT_DIR=str(run / 'predictions'),
                      TEMP_DIR=str(run / 'temp'), INPUT_DIR=str(ROOT / 'data/input60'),
                      PYTHONUNBUFFERED='1')
    (run / 'predictions').mkdir()
    (run / 'temp').mkdir()
    identity = run_identity(ROOT)
    fingerprint = digest_json(identity)
    os.environ['KDD_RUN_IDENTITY'] = fingerprint
    atomic_json(run / 'run_manifest.json', {'identity': identity, 'identity_sha256': fingerprint,
                                          'tasks': args.tasks, 'scope': 'selected_model_trial'})
    started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    (run / 'versions.txt').write_text(started + '\n' + '\n'.join(
        f'{key}={value}' for key, value in identity['configuration'].items()) + '\n')
    atomic_json(run / 'protocol.json', {'scope': 'selected_tasks_only', 'tasks': args.tasks,
                                       'model': args.model, 'full_baseline_modified': False,
                                       'gold_policy': 'read only after Agent process exits'})
    with tarfile.open(run / 'source.tar.gz', 'w:gz') as archive:
        for name in ['champion/src', 'champion/uv.lock', 'scripts']:
            archive.add(ROOT / name, arcname=name,
                        filter=lambda item: None if '__pycache__' in item.name else item)
    state = {'run_dir': str(run), 'started_at': started, 'launcher_pid': os.getpid(),
             'status': 'RUNNING', 'tasks': args.tasks, 'model': args.model}
    atomic_json(run / 'trial-state.json', state)
    atomic_json(ROOT / 'docs/latest-model-trial.json', state)
    print('MODEL_TRIAL_DIR=' + str(run), flush=True)
    with (run / 'console.log').open('wb') as log:
        process = subprocess.Popen([
            'timeout', '90m', sys.executable, 'src/data_agent_baseline/zz_agent_v2.py',
            '--only-tasks', ','.join(args.tasks), '--max-parallel', '1',
        ], cwd=ROOT / 'champion', env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        state['agent_timeout_pid'] = process.pid
        atomic_json(run / 'trial-state.json', state)
        code = process.wait()
    check = subprocess.run([sys.executable, str(ROOT / 'scripts/check_run_outputs.py'), *args.tasks],
                           cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True)
    (run / 'output-check.log').write_text(check.stdout + check.stderr)
    if code == 0 and check.returncode:
        code = check.returncode
    (run / 'exit-code.txt').write_text(str(code) + '\n')
    with (run / 'evaluation.log').open('w') as log:
        score = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_run.py'), str(run),
                                '--tasks', *args.tasks], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    state.update(status='FINISHED', finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 exit_code=code, evaluation_exit_code=score.returncode)
    atomic_json(run / 'trial-state.json', state)
    atomic_json(ROOT / 'docs/latest-model-trial.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
