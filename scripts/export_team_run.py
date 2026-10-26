"""Export a finished public60 run for team review, without copied input assets."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/baseline_full_20261025')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/shared_public60_20261026')
    args = parser.parse_args()
    source, output = args.run.resolve(), args.output.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise SystemExit('Export must be separate from the original run')
    if output.exists():
        raise SystemExit('Output already exists; choose a new directory')
    report = json.loads((source / 'evaluation.json').read_text())
    tasks = [row['task_id'] for row in report['course_results']]
    if report.get('scope') != 'public60' or len(tasks) != 60 or len(set(tasks)) != 60:
        raise SystemExit('Expected a completed public60 evaluation')
    if (source / 'exit-code.txt').read_text().strip() != '0':
        raise SystemExit('Source run must have exited successfully')
    # Validate all predictions before creating the archive. CSV bytes stay unchanged.
    for task in tasks:
        receipt = json.loads((source / 'predictions' / task / 'completion.json').read_text())
        prediction = source / 'predictions' / task / 'prediction.csv'
        if not receipt.get('success') or hashlib.sha256(prediction.read_bytes()).hexdigest() != receipt['prediction_sha256']:
            raise SystemExit(f'Invalid prediction receipt: {task}')
    output.mkdir(parents=True)

    def copy(path):
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            # Preserve receipts and hashes, but remove this workstation's root path.
            value = json.loads(path.read_text())
            text = json.dumps(value, ensure_ascii=False, indent=2)
            target.write_text(text.replace(str(ROOT) + '/', '') + '\n')
        else:
            shutil.copyfile(path, target)

    for name in ('evaluation.json', 'analysis_summary.json', 'analysis_task_results.csv',
                 'protocol.json', 'run_manifest.json', 'output-check.json', 'exit-code.txt'):
        path = source / name
        if path.is_file():
            copy(path)
    for task in tasks:
        for name in ('prediction.csv', 'completion.json', 'score.json'):
            path = source / 'predictions' / task / name
            if path.is_file():
                copy(path)
    # Keep original relative locations: archived failed solvers must not be
    # mistaken for the generator of the final prediction (notably task_102).
    for pattern in ('**/solver.py', '**/timing.json', 'sessions/*/original_receipts/*.json',
                    'sessions/*/run_manifest.json', 'sessions/*/protocol.json'):
        for path in sorted(source.glob(pattern)):
            copy(path)
    index = {}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            index[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'archive_manifest.json').write_text(json.dumps({
        'kind': 'review_archive_not_resumable_run',
        'source_run': source.relative_to(ROOT).as_posix(),
        'prediction_count': len(tasks),
        'input_assets_included': False,
        'full_conversations_included': False,
        'files_sha256': index,
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported {len(tasks)} predictions and {len(index)} files to {output}')


if __name__ == '__main__':
    main()
