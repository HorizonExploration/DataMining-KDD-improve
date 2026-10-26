"""Summarize completed public60 results and measured times; no model requests."""
import collections
import csv
import hashlib
import json
import re
import statistics
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/baseline_full_20261025'
evaluation = json.loads((RUN / 'evaluation.json').read_text())
session = json.loads((RUN / 'current_session.json').read_text())
protocol = json.loads((RUN / 'protocol.json').read_text())
supervisor = json.loads((RUN / 'supervisor-state.json').read_text())
history = Path(session['history_dir'])
previous = json.loads((history / 'evaluation.json').read_text())
pilot_dir = Path(protocol['reviewed_change']['pilot'])
pilot_state = json.loads((pilot_dir / 'trial-state.json').read_text())
course = {row['task_id']: row for row in evaluation['course_results']}
official = {row['task_id']: row for row in evaluation['official_results']}
assert len(course) == 60 and evaluation['scope'] == 'public60'
old_tasks = set(protocol['prior_predictions_preserved'])
new_tasks = set(course) - old_tasks
video_tasks = {task for task in course if (ROOT / 'data/input60' / task / 'context/video/briefing.mp4').exists()}


def score_counts(tasks):
    values = [course[t]['score'] for t in tasks]
    return {'tasks': len(values), 'mean_course_score_100': sum(values) / len(values) * 100,
            'course_full': sum(v == 1 for v in values), 'course_partial': sum(0 < v < 1 for v in values),
            'course_zero': sum(v == 0 for v in values),
            'official_passed': sum(official[t]['score'] == 1 for t in tasks)}


def requests_between(start, end):
    return [row for row in requests if start <= datetime.fromisoformat(row['timestamp']) <= end]


requests = [json.loads(line) for line in (RUN / 'temp/logs/model_requests.jsonl').read_text().splitlines()]
start, end = (datetime.fromisoformat(session[k]) for k in ('started_at', 'finished_at'))
repair_requests = requests_between(start, end)
old_session = json.loads((history / 'current_session.json').read_text())
old_start, old_end = (datetime.fromisoformat(old_session[k]) for k in ('started_at', 'finished_at'))
old_requests = requests_between(old_start, old_end)
usage = collections.Counter()
for row in repair_requests:
    for key, value in row.get('usage', {}).items():
        if isinstance(value, (int, float)):
            usage[key] += value

stages = collections.Counter()
repair_timings = {}
for task in session['pending_tasks']:
    data = json.loads((RUN / 'temp/logs' / task / 'timing.json').read_text())
    repair_timings[task] = data
    stages.update(data['stages'])
raw = (RUN / 'console.log').read_bytes()[session['console_start_byte']:].decode(errors='replace')
raw = re.sub(r'\x1b\[[0-9;]*m', '', raw)
inner_stages = {}
for label, pattern in [('asr', r'ASR transcribe: ([0-9.]+)s'),
                       ('frame_layout', r'hiccup generation: ([0-9.]+)s'),
                       ('frame_extraction', r'extract_frames: ([0-9.]+)s')]:
    values = [float(v) for v in re.findall(pattern, raw)]
    inner_stages[label] = {'samples': len(values), 'sum_seconds': sum(values),
                           'mean_seconds': statistics.mean(values) if values else None}

with zipfile.ZipFile(RUN / 'historical-provenance.zip') as archive:
    starts = [datetime.fromisoformat(archive.read(name).decode().splitlines()[0])
              for name in archive.namelist() if name.endswith('/versions.txt')]
    earliest = min(starts)
    old_timings = {}
    for name in sorted(archive.namelist()):
        if re.search(r'/temp/logs/task_\d+/timing.json$', name):
            old_timings[Path(name).parent.name] = (json.loads(archive.read(name)), 'historical-provenance.zip:' + name)

unchanged = True
for task in old_tasks:
    receipt = json.loads((history / 'original_receipts' / f'{task}.json').read_text())
    checksum = hashlib.sha256((RUN / 'predictions' / task / 'prediction.csv').read_bytes()).hexdigest()
    unchanged = unchanged and receipt['prediction_sha256'] == checksum
assert unchanged

rows = []
for task in sorted(course):
    if task == 'task_102':
        timing_path = pilot_dir / 'temp/logs/task_102/timing.json'
        timing = json.loads(timing_path.read_text())
        timing_source = str(timing_path.relative_to(ROOT))
    elif task in repair_timings:
        timing = repair_timings[task]
        timing_source = str((RUN / 'temp/logs' / task / 'timing.json').relative_to(ROOT))
    elif (RUN / 'temp/logs' / task / 'timing.json').exists():
        timing_path = RUN / 'temp/logs' / task / 'timing.json'
        timing, timing_source = json.loads(timing_path.read_text()), str(timing_path.relative_to(ROOT))
    else:
        timing, timing_source = old_timings.get(task, ({}, 'not_available'))
    result = course[task]
    rows.append({'task_id': task, 'has_video': task in video_tasks,
                 'model_phase': 'retained_ecnu_plus' if task in old_tasks else 'ecnu_max_repair',
                 'course_score_100': round(result['score'] * 100, 4),
                 'official_passed': official[task]['passed'],
                 'official_first_failure': official[task].get('error_code'),
                 'gold_columns': official[task].get('gold_column_count'),
                 'prediction_columns': official[task].get('prediction_column_count'),
                 'gold_rows': official[task].get('gold_row_count'),
                 'prediction_rows': official[task].get('prediction_row_count'),
                 'last_available_attempt_wall_seconds': timing.get('total_wall_s'),
                 'timing_source': timing_source})
csv_path = RUN / 'analysis_task_results.csv'
with csv_path.open('w', encoding='utf-8-sig', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

wall = (end - start).total_seconds()
task_walls = [data['total_wall_s'] for data in repair_timings.values()]
scorer_differences = [task for task in course if (course[task]['score'] == 1) != official[task]['passed']]
baseline = 75.6387
assessment = {
    'scope': 'public60_mixed_model_repair_with_retained_predictions',
    'course_lambda': 0.1, 'valid_predictions': len(list((RUN / 'predictions').glob('*/prediction.csv'))),
    'final_course_score_100': evaluation['course_score_100'], 'reference_baseline_100': baseline,
    'gain_over_reference_points': evaluation['course_score_100'] - baseline,
    'relative_gain_over_reference_percent': (evaluation['course_score_100'] / baseline - 1) * 100,
    'prior_course_score_100': previous['course_score_100'],
    'gain_over_incomplete_run_points': evaluation['course_score_100'] - previous['course_score_100'],
    'official_task_accuracy': evaluation['official_task_accuracy'], 'overall': score_counts(set(course)),
    'groups': {'retained_plus_33': score_counts(old_tasks), 'new_max_27': score_counts(new_tasks),
               'text_32': score_counts(set(course) - video_tasks), 'video_28': score_counts(video_tasks)},
    'original_33_prediction_hashes_unchanged': unchanged,
    'course_zero_tasks': [task for task in course if course[task]['score'] == 0],
    'official_first_failure_counts': dict(collections.Counter(
        row.get('error_code') for row in official.values() if not row['passed'])),
    'scorer_disagreement_tasks': scorer_differences,
    'date_format_theoretical_gain_points': sum(1 - course[t]['score'] for t in scorer_differences) / 60 * 100,
    'time': {
        'first_full_attempt_started_at': earliest.isoformat(),
        'final_evaluation_finished_at': supervisor.get('finished_at', supervisor['updated_at']),
        'overall_calendar_seconds_including_interruptions': (datetime.fromisoformat(supervisor.get('finished_at', supervisor['updated_at'])) - earliest).total_seconds(),
        'prior_plus_failed_27_wall_seconds': (old_end - old_start).total_seconds(),
        'max_pilot_wall_seconds': (datetime.fromisoformat(pilot_state['finished_at']) - datetime.fromisoformat(pilot_state['started_at'])).total_seconds(),
        'max_repair_26_wall_seconds': wall, 'max_repair_tasks_per_hour': len(task_walls) / wall * 3600,
        'max_repair_wall_seconds_per_completed_task': wall / len(task_walls),
        'max_repair_mean_task_latency_seconds': statistics.mean(task_walls),
        'max_repair_median_task_latency_seconds': statistics.median(task_walls),
        'mean_active_task_slots_including_wait': sum(task_walls) / wall,
        'stage_task_seconds_including_queue_and_overlap': dict(stages), 'inner_video_stage_seconds': inner_stages,
    },
    'api': {'prior_failed_plus_27_response_status': dict(collections.Counter(x['status'] for x in old_requests)),
            'max_repair_26_response_status': dict(collections.Counter(x['status'] for x in repair_requests)),
            'max_repair_26_usage': dict(usage),
            'max_repair_http_success_percent': sum(x['status'] == 200 for x in repair_requests) / len(repair_requests) * 100},
    'slowest_repair_tasks': [{'task_id': task, 'wall_seconds': data['total_wall_s'], 'course_score_100': course[task]['score'] * 100}
                             for task, data in sorted(repair_timings.items(), key=lambda item: item[1]['total_wall_s'], reverse=True)[:8]],
    'limitations': ['Single public60 run; no held-out set or repeated seeds',
                    'Different tasks and settings across model phases; no causal model or concurrency speedup claim',
                    'Calendar span includes debugging, failed calls and WSL interruptions',
                    'Task/stage durations overlap and include queue waits; they are not additive wall-clock fractions',
                    'Only the latest 26-task repair time is directly measured; not a cold full60 runtime',
                    'Per-task CSV timing is the last available attempt, not total lifetime retries'],
}
(RUN / 'analysis_summary.json').write_text(json.dumps(assessment, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: assessment[k] for k in ('final_course_score_100', 'gain_over_reference_points', 'overall', 'time')}, ensure_ascii=False, indent=2))
