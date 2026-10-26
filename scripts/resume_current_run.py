"""Resume verified public60 outputs in the same directory, with live console logs."""
import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'champion/src/data_agent_baseline'))
from tools_v2.artifact_cache import atomic_json, digest_json, file_digest
from tools_v2.run_state import run_identity, valid_csv


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def validate_reviewed_change(old, current, review):
    expected = copy.deepcopy(old)
    for path, change in review.get('source_changes', {}).items():
        if old['source'].get(path) != change['before'] or current['source'].get(path) != change['after']:
            raise ValueError('Reviewed source hash does not match')
        expected['source'][path] = change['after']
    allowed = {'MODEL_NAME', 'MODEL_VIDEO_NAME', 'MODEL_MAX_CONCURRENCY',
               'MODEL_THINKING_FORMAT', 'MODEL_SOLVER_THINKING', 'MAX_PARALLEL'}
    for key, change in review.get('configuration_changes', {}).items():
        if key not in allowed or old['configuration'].get(key) != change['before'] or current['configuration'].get(key) != change['after']:
            raise ValueError('Reviewed routing configuration does not match')
        expected['configuration'][key] = change['after']
    if expected != current:
        raise ValueError('Unreviewed source/input/weights/endpoint/config change')


def import_equivalent_trial(trial, run, identity, fingerprint, reused, history):
    """Import an independently completed pilot, with model and data equivalence checks."""
    trial = trial.resolve()
    if not trial.is_relative_to(run / 'model_trials'):
        raise ValueError('Pilot must be inside this experiment model_trials directory')
    manifest = json.loads((trial / 'run_manifest.json').read_text())
    old = manifest['identity']
    if manifest['identity_sha256'] != digest_json(old) or old['source'] != identity['source']:
        raise ValueError('Pilot source fingerprint differs')
    if old['endpoint_sha256'] != identity['endpoint_sha256'] or old['asr_weights'] != identity['asr_weights']:
        raise ValueError('Pilot endpoint/ASR differs')
    for key in ('MODEL_API_PROFILE', 'MODEL_THINKING_FORMAT', 'MODEL_SOLVER_THINKING', 'ASR_BACKEND'):
        if old['configuration'].get(key) != identity['configuration'].get(key):
            raise ValueError('Pilot inference settings differ')
    for task in manifest['tasks']:
        if task in reused:
            continue
        video = (ROOT / 'data/input60' / task / 'context/video/briefing.mp4').exists()
        def resolved(config):
            return config.get('MODEL_VIDEO_NAME') if video and config.get('MODEL_VIDEO_NAME') else config['MODEL_NAME']
        if resolved(old['configuration']) != resolved(identity['configuration']) or old['inputs'][task] != identity['inputs'][task]:
            raise ValueError('Pilot resolved model or task input differs')
        path = trial / 'predictions' / task / 'prediction.csv'
        receipt = json.loads((path.parent / 'completion.json').read_text())
        if (receipt.get('success') is not True or receipt.get('run_identity') != manifest['identity_sha256']
                or not valid_csv(path) or file_digest(path) != receipt.get('prediction_sha256')):
            raise ValueError('Pilot has no valid successful prediction')
        target = run / 'predictions' / task
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / 'prediction.csv')
        original = copy.deepcopy(receipt)
        receipt.update(run_identity=fingerprint, imported_model_trial={
            'run_dir': str(trial), 'original_receipt': original,
            'resolved_model': resolved(identity['configuration']),
            'note': 'Same code, inputs, endpoint, ASR and effective inference settings; scheduling/version labels recorded separately',
        })
        atomic_json(target / 'completion.json', receipt)
        reused.append(task)
    atomic_json(history / 'imported-model-trial.json', {'source': str(trial), 'tasks': manifest['tasks']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--allow-task-parallel-change', action='store_true',
                        help='Record an explicit change of MAX_PARALLEL only; preserve verified predictions')
    parser.add_argument('--reviewed-routing-change', type=Path)
    parser.add_argument('--import-trial', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if run.parent != ROOT / 'runs' or not (run / 'run_manifest.json').is_file():
        raise SystemExit('Choose an existing experiment directly under runs')
    lock = (run / '.experiment.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This experiment already has an active launcher')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(a.endswith(b'zz_agent_v2.py') for a in argv):
            raise SystemExit('An Agent process is already running; refusing duplicate requests')
    for key, value in dotenv_values(os.environ.get('KDD_ENV_FILE', ROOT / '.env')).items():
        if value is not None and key.isupper():
            os.environ[key] = value
    if not os.environ.get('MODEL_API_KEY'):
        raise SystemExit('MODEL_API_KEY is not configured')
    identity = run_identity(ROOT)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    fingerprint = digest_json(identity)
    old_fingerprint = manifest['identity_sha256']
    if old_fingerprint != digest_json(manifest['identity']):
        raise SystemExit('Original manifest fingerprint is invalid')
    changed = manifest['identity'] != identity
    review = json.loads(args.reviewed_routing_change.read_text()) if args.reviewed_routing_change else None
    if changed:
        if review:
            validate_reviewed_change(manifest['identity'], identity, review)
        else:
            expected = copy.deepcopy(manifest['identity'])
            expected['configuration']['MAX_PARALLEL'] = identity['configuration']['MAX_PARALLEL']
            if not args.allow_task_parallel_change or expected != identity:
                raise SystemExit('Only an explicitly authorized MAX_PARALLEL change may be migrated; source/input/model/other settings must match')
    tasks = json.loads((ROOT / 'data/public_reference_tasks.json').read_text())['task_ids']
    reused = []
    for task in tasks:
        prediction = run / 'predictions' / task / 'prediction.csv'
        try:
            receipt = json.loads((prediction.parent / 'completion.json').read_text())
            if (receipt.get('success') is True and receipt.get('run_identity') == old_fingerprint
                    and valid_csv(prediction) and receipt['prediction_sha256'] == file_digest(prediction)):
                reused.append(task)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    pending = [task for task in tasks if task not in reused]
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    history = run / 'sessions' / f'{stamp}_{os.getpid()}'
    history.mkdir(parents=True)
    for name in ('run_manifest.json', 'resume.json', 'supervisor-state.json', 'progress.json', 'exit-code.txt', 'current_session.json',
                 'evaluation.json', 'evaluation.log', 'output-check.json', 'output-check.log', 'protocol.json'):
        path = run / name
        if path.exists():
            shutil.copy2(path, history / name)
    if changed:
        migration = {
            'changed_at': now(), 'from_identity': old_fingerprint, 'to_identity': fingerprint,
            'MAX_PARALLEL': [manifest['identity']['configuration']['MAX_PARALLEL'], identity['configuration']['MAX_PARALLEL']],
            'source_manifest': str(history / 'run_manifest.json'),
        }
        if review:
            migration['reviewed_routing_change'] = review
        # Archive every original receipt before changing any of them.
        originals = history / 'original_receipts'
        originals.mkdir()
        for task in reused:
            shutil.copy2(run / 'predictions' / task / 'completion.json', originals / f'{task}.json')
        for task in reused:
            path = run / 'predictions' / task / 'completion.json'
            receipt = json.loads(path.read_text())
            receipt.update(run_identity=fingerprint)
            receipt.setdefault('experiment_migrations' if review else 'scheduling_migrations', []).append(dict(migration, original_receipt=str(originals / f'{task}.json')))
            atomic_json(path, receipt)
        manifest.update(identity=identity, identity_sha256=fingerprint)
        manifest.setdefault('experiment_history' if review else 'scheduling_history', []).append(migration)
        atomic_json(run / 'run_manifest.json', manifest)
        atomic_json(history / 'scheduling-change.json', migration)
        if review:
            atomic_json(run / 'protocol.json', {
                'scope': 'public60_routing_repair_with_historical_predictions',
                'policy': review.get('policy'), 'configuration': identity['configuration'],
                'prior_predictions_preserved': reused.copy(),
                'original_baseline_evaluation': str(history / 'evaluation.json'),
                'original_manifest': str(history / 'run_manifest.json'),
                'reviewed_change': review, 'models_changed': True,
                'score_policy': 'All 60 tasks; describe as mixed-model repair, not a pure ecnu-max rerun',
            })
    if args.import_trial:
        import_equivalent_trial(args.import_trial, run, identity, fingerprint, reused, history)
        pending = [task for task in tasks if task not in reused]
    for name in ('evaluation.json', 'evaluation.log', 'output-check.json', 'output-check.log'):
        (run / name).unlink(missing_ok=True)
    # Keep prior attempt logs and small solver files; raw inputs are re-copied by Agent.
    for task in pending:
        logs = run / 'temp/logs' / task
        if logs.exists():
            target = history / 'task_logs' / task
            target.parent.mkdir(parents=True, exist_ok=True)
            logs.rename(target)
        solver = run / 'temp' / task / 'workdir/solver.py'
        if solver.exists():
            target = history / 'solvers' / task / 'solver.py'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(solver, target)
    (run / 'exit-code.txt').unlink(missing_ok=True)
    atomic_json(run / 'resume.json', {
        'source_run': str(run), 'resume_mode': 'in_place_verified_success_receipts',
        'reused_tasks': reused, 'pending_tasks': pending,
        'prior_resume_metadata': str(history / 'resume.json'),
        'metric_scope': 'all_public60_with_preserved_history',
    })
    session = {
        'started_at': now(), 'launcher_pid': os.getpid(), 'launcher_marker': 'resume_current_run.py',
        'run_dir': str(run), 'history_dir': str(history),
        'console_start_byte': (run / 'console.log').stat().st_size if (run / 'console.log').exists() else 0,
        'reused_tasks': reused, 'pending_tasks': pending, 'status': 'STARTING',
    }
    atomic_json(run / 'current_session.json', session)
    os.environ.update(RUN_DIR=str(run), OUTPUT_DIR=str(run / 'predictions'),
                      TEMP_DIR=str(run / 'temp'), INPUT_DIR=str(ROOT / 'data/input60'),
                      KDD_RUN_IDENTITY=fingerprint, PYTHONUNBUFFERED='1')
    status_path = ROOT / 'docs/deployment-status.json'
    deployment = json.loads(status_path.read_text())
    deployment.update(active_full_run=str(run.relative_to(ROOT)), full60_run='RUNNING',
                      full60_progress=str((run / 'progress.json').relative_to(ROOT)),
                      full60_supervisor=str((run / 'supervisor-state.json').relative_to(ROOT)),
                      full60_last_checked_at=now())
    atomic_json(status_path, deployment)
    print(f'实验目录：{run}\n已校验复用 {len(reused)} 题，待执行 {len(pending)} 题。', flush=True)
    result_code = 0
    if pending:
        child = subprocess.Popen([
            'timeout', os.environ.get('KDD_RUN_TIMEOUT', '24h'), sys.executable,
            'src/data_agent_baseline/zz_agent_v2.py', '--only-tasks', ','.join(pending),
            '--max-parallel', os.environ.get('MAX_PARALLEL', '6'),
        ], cwd=ROOT / 'champion', env=os.environ.copy(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0)
        session.update(status='RUNNING', agent_timeout_pid=child.pid)
        atomic_json(run / 'current_session.json', session)
    with (run / 'supervisor.log').open('a') as log:
        watcher = subprocess.Popen([sys.executable, str(ROOT / 'scripts/watch_run.py'), str(run)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=log, start_new_session=True)
    (run / 'supervisor.pid').write_text(str(watcher.pid) + '\n')
    if pending:
        with (run / 'console.log').open('ab', buffering=0) as console:
            while True:
                data = os.read(child.stdout.fileno(), 65536)
                if not data:
                    break
                console.write(data)
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
        result_code = child.wait()
    check = subprocess.run([sys.executable, str(ROOT / 'scripts/check_run_outputs.py')],
                           cwd=ROOT, env=os.environ.copy())
    if result_code == 0 and check.returncode:
        result_code = check.returncode
    session.update(status='FINISHED', finished_at=now(), exit_code=result_code)
    atomic_json(run / 'current_session.json', session)
    (run / 'exit-code.txt').write_text(str(result_code) + '\n')
    return result_code


if __name__ == '__main__':
    raise SystemExit(main())
