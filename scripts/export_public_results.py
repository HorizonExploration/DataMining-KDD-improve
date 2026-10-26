"""Export scores/configuration without credentials, raw data or model conversations."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/baseline_full_20261025')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/public60_20261026')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    evaluation = json.loads((run / 'evaluation.json').read_text())
    if evaluation.get('scope') != 'public60' or evaluation.get('task_count') != 60:
        raise SystemExit('Only an explicitly evaluated public60 run can be exported here')
    output.mkdir(parents=True, exist_ok=True)
    for source, destination in [('evaluation.json', 'evaluation.json'), ('analysis_summary.json', 'summary.json')]:
        data = sanitize(json.loads((run / source).read_text()))
        (output / destination).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    protocol = json.loads((run / 'protocol.json').read_text())
    previous = Path(protocol.get('original_baseline_evaluation', ''))
    if previous.is_file():
        data = sanitize(json.loads(previous.read_text()))
        (output / 'evaluation_before_routing.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    with (run / 'analysis_task_results.csv').open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        rows = [{key: sanitize(value) for key, value in row.items()} for row in reader]
        with (output / 'task_results.csv').open('w', encoding='utf-8-sig', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    identity = manifest['identity']
    public_manifest = {
        'scope': 'public60_mixed_model_repair_with_historical_predictions',
        'task_ids': manifest['tasks'], 'run_identity_sha256': manifest['identity_sha256'],
        'source_hashes_at_experiment_time': identity['source'],
        'model_configuration': identity['configuration'],
        'input_hashes': identity['inputs'], 'asr_weight_hashes': identity['asr_weights'],
        'endpoint_sha256': identity['endpoint_sha256'],
        'credentials_included': False, 'raw_data_included': False,
        'raw_predictions_included': False,
    }
    (output / 'inference_manifest.json').write_text(json.dumps(public_manifest, ensure_ascii=False, indent=2) + '\n')
    diagnostic = json.loads((ROOT / 'docs/ecnu-models-multimodal-probe.json').read_text())
    public_diagnostic = {'listed_models': diagnostic['listed_models'], 'tests': [
        {key: row[key] for key in ('model', 'input', 'status', 'seconds', 'response_model') if key in row}
        for row in diagnostic['tests']]}
    (output / 'api_diagnostics.json').write_text(json.dumps(public_diagnostic, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported public60 scores and configuration to {output}; no predictions or credentials copied.')


if __name__ == '__main__':
    main()
