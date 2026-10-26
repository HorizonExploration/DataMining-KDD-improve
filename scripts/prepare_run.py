"""Create a run fingerprint and optionally reuse independently verified outputs."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'champion/src/data_agent_baseline'))
from tools_v2.artifact_cache import atomic_json, digest_json
from tools_v2.run_state import run_identity, reuse_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=Path)
parser.add_argument('tasks', nargs='*')
args = parser.parse_args()
run = Path(os.environ['RUN_DIR']).resolve()
public = json.loads((ROOT / 'data/public_reference_tasks.json').read_text())['task_ids']
tasks = args.tasks or public
if len(tasks) != len(set(tasks)) or not set(tasks).issubset(public):
    raise SystemExit('续跑题集必须来自公开清单，且不能重复')
if args.resume:
    source = args.resume.resolve()
    if not source.is_relative_to(ROOT / 'runs') or source == run:
        raise SystemExit('续跑来源必须是项目 runs 下的另一个实验目录')
    if not (source / 'run_manifest.json').is_file():
        raise SystemExit('来源没有本版 run_manifest.json，无法严格验证旧结果；请新建实验')
identity = run_identity(ROOT)
reused = []
if args.resume:
    try:
        reused = reuse_predictions(run, source, identity, tasks)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
atomic_json(run / 'run_manifest.json', {'identity': identity, 'identity_sha256': digest_json(identity), 'tasks': tasks})
atomic_json(run / 'resume.json', {
    'source_run': str(args.resume.resolve()) if args.resume else None,
    'reused_tasks': reused, 'pending_tasks': [t for t in tasks if t not in reused],
    'metric_scope': 'resumed_predictions_plus_new_tasks' if reused else 'new_tasks_only',
})
print(','.join(t for t in tasks if t not in reused))
