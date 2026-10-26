"""Supervise the already running public60 experiment; never call the model."""
import json
import sys
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1]).resolve()
SESSION = json.loads((RUN / 'current_session.json').read_text()) if (RUN / 'current_session.json').exists() else {}
LAUNCHER_PID = int(SESSION.get('launcher_pid', RUN.name.rsplit('_', 1)[-1]))
LAUNCHER_MARKER = SESSION.get('launcher_marker', 'run_tasks.sh').encode()
STATE = RUN / 'supervisor-state.json'

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def save(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def update_progress():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/monitor_run.py'), str(RUN)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return json.loads((RUN / 'progress.json').read_text())

state = {'started_at': now(), 'pid': os.getpid(), 'run_dir': str(RUN),
         'status': 'WATCHING', 'evaluation_policy': 'after_exit_code_only'}
save(STATE, state)
missing_since = None
while not (RUN / 'exit-code.txt').exists():
    try:
        progress = update_progress()
        state.update(updated_at=now(), successful=progress['successful'],
                     attempted=progress['attempted'], current_task=progress['current_task'], active_tasks=progress.get('active_tasks', []))
        state.pop('monitor_error', None)
        # PID + command verification avoids mistaking a reused PID for the job.
        cmdline = Path(f'/proc/{LAUNCHER_PID}/cmdline')
        alive = cmdline.exists() and LAUNCHER_MARKER in cmdline.read_bytes()
        if alive:
            missing_since = None
        else:
            missing_since = missing_since or time.monotonic()
            if time.monotonic() - missing_since > 120:
                state.update(status='PROCESS_LOST', updated_at=now(),
                             reason='Agent disappeared without launcher exit record; predictions retained; no grading.')
                save(STATE, state)
                break
        save(STATE, state)
    except Exception as exc:
        state.update(updated_at=now(), monitor_error=str(exc))
        save(STATE, state)
    time.sleep(30)
else:
    progress = update_progress()
    state.update(attempted=progress['attempted'], active_tasks=progress.get('active_tasks', []),
                 current_task=progress.get('current_task'))
    state.update(status='EVALUATING', updated_at=now(),
                 run_exit_code=(RUN / 'exit-code.txt').read_text().strip())
    save(STATE, state)
    env = os.environ.copy()
    env['PATH'] = str(ROOT / 'tools/bin') + os.pathsep + env.get('PATH', '')
    with (RUN / 'evaluation.log').open('w') as log:
        result = subprocess.run(
            [str(ROOT / 'champion/.venv/bin/python'),
             str(ROOT / 'scripts/evaluate_run.py'), str(RUN)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=1800,
        )
    complete = progress['successful'] == 60 and state['run_exit_code'] == '0'
    state.update(status=('COMPLETE' if complete else 'PARTIAL')
                 if result.returncode == 0 else 'EVALUATION_FAILED',
                 updated_at=now(), successful=progress['successful'],
                 evaluation_exit_code=result.returncode)
    if result.returncode == 0:
        report = json.loads((RUN / 'evaluation.json').read_text())
        state['scores'] = {k: report[k] for k in
                           ['scope', 'task_count', 'course_score_100', 'official_task_accuracy']}
    save(STATE, state)

doc = ROOT / 'docs/deployment-status.json'
try:
    record = json.loads(doc.read_text())
    if record.get('active_full_run') == str(RUN.relative_to(ROOT)):
        record.update(full60_run=state['status'], full60_finished_at=now(),
                      full60_successful=state.get('successful'),
                      full60_supervisor=str(STATE.relative_to(ROOT)))
        if 'scores' in state:
            record['full60_evaluation'] = state['scores']
        save(doc, record)
except Exception as exc:
    print('Could not update deployment status:', exc, flush=True)
print(json.dumps(state, ensure_ascii=False), flush=True)
