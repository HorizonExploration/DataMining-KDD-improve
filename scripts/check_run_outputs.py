"""Check selected task outputs without reading gold answers."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('tasks', nargs='*')
args = parser.parse_args()
root = Path(os.environ['KDD_ROOT'])
run = Path(os.environ['RUN_DIR'])
manifest = json.loads((root / 'data/public_reference_tasks.json').read_text())['task_ids']
tasks = args.tasks or manifest
results = []
for task in tasks:
    result = {'task_id': task, 'valid': False}
    path = run / 'predictions' / task / 'prediction.csv'
    try:
        frame = pd.read_csv(path)
        identity = os.environ.get('KDD_RUN_IDENTITY')
        if identity:
            receipt = json.loads((path.parent / 'completion.json').read_text())
            with path.open('rb') as source:
                checksum = hashlib.file_digest(source, 'sha256').hexdigest()
            if receipt.get('success') is not True or receipt.get('run_identity') != identity or receipt.get('prediction_sha256') != checksum:
                raise ValueError('任务成功标记不匹配')
        result.update(valid=True, rows=len(frame), columns=frame.columns.tolist())
    except Exception as error:
        result['error'] = type(error).__name__
    results.append(result)
report = {'expected': len(tasks), 'valid_predictions': sum(r['valid'] for r in results), 'tasks': results}
(run / 'output-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f"预测检查：{report['valid_predictions']}/{report['expected']} 个可读 CSV。")
raise SystemExit(0 if all(r['valid'] for r in results) else 1)
