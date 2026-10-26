"""预测运行结束后独立评测；默认固定 60 题，可显式选单题用于冒烟测试。"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
parser.add_argument('--tasks', nargs='+', help='只评测指定题号；此结果不能作为 60 题基线')
args = parser.parse_args()
run = args.run_dir.resolve()
if not (run / 'exit-code.txt').is_file():
    raise SystemExit('运行尚未结束或缺少退出记录；不在解题期间读取 gold。')
ids = json.loads((ROOT / 'data/public_reference_tasks.json').read_text())['task_ids']
tasks = args.tasks or ids
if not tasks or len(tasks) != len(set(tasks)) or not set(tasks).issubset(ids):
    raise SystemExit('题号必须属于公开题单且不能重复。')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


course = load('kdd_course_evaluator', ROOT / 'champion/src/data_agent_baseline/run_compare_to_gt.py')
official = load('kdd_official_evaluator', ROOT / 'DataSpace/evaluation/evaluate.py')
bench = ROOT / 'data/DataSpace-Benchmark'
course_results = []
official_results = []
for task in tasks:
    course_results.append(course.score_task(
        task, bench / 'output', run / 'predictions', lambda_extra=0.1,
    ))
    config = official.load_task_config(bench / 'evaluation/configs' / f'{task}.json')
    official_results.append(official.evaluate_task(
        config=config, prediction_path=run / 'predictions' / task / 'prediction.csv',
        gold_path=bench / 'output' / task / 'gold.csv',
    ).to_dict())
report = {
    'scope': 'public60' if set(tasks) == set(ids) else 'selected_tasks_only',
    'task_count': len(tasks), 'lambda': 0.1,
    'course_score_100': sum(r['score'] for r in course_results) / len(tasks) * 100,
    'official_task_accuracy': sum(r['score'] for r in official_results) / len(tasks),
    'course_results': course_results, 'official_results': official_results,
}
(run / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in report.items() if not k.endswith('_results')}, ensure_ascii=False, indent=2))
