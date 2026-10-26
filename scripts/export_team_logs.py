"""Archive retained experiment logs with credentials and local root paths removed."""
import argparse
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.log', '.json', '.jsonl', '.txt', '.tsv', '.cache_key', '.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/baseline_full_20261025')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/shared_public60_20261026/logs')
    args = parser.parse_args()
    source, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit('Output already exists; choose a new directory')
    if (source / 'exit-code.txt').read_text().strip() != '0':
        raise SystemExit('Only export a finished successful run')
    secrets = set()
    env = ROOT / '.env'
    if env.is_file():
        for line in env.read_text().splitlines():
            match = re.match(r'(?:export\s+)?(\w+)\s*=\s*(.*)', line)
            if match and any(word in match[1].upper() for word in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET')):
                value = match[2].strip().strip('\"\'')
                if len(value) > 8:
                    secrets.add(value)
    token = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    bearer = re.compile(r'(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]{12,}')
    entries = []
    redactions = 0
    output.mkdir(parents=True)
    archive = output / 'experiment-logs.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=9) as bundle:
        def add(name, raw):
            nonlocal redactions
            text = raw.decode('utf-8')
            for secret in sorted(secrets, key=len, reverse=True):
                redactions += text.count(secret)
                text = text.replace(secret, '[REDACTED]')
            text, count = token.subn('[REDACTED]', text)
            redactions += count
            text, count = bearer.subn(r'\1[REDACTED]', text)
            redactions += count
            text = text.replace(str(ROOT) + '/', '')
            data = text.encode('utf-8')
            if any(secret in text for secret in secrets) or token.search(text) or bearer.search(text):
                raise ValueError(f'Credential check failed for {name}')
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            bundle.addfile(info, io.BytesIO(data))
            entries.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})

        paths = {p for p in source.iterdir() if p.is_file() and p.suffix in TEXT_SUFFIXES}
        paths.update(p for p in (source / 'temp/logs').rglob('*') if p.is_file())
        paths.update(p for p in (source / 'sessions').rglob('*') if p.is_file() and p.suffix in TEXT_SUFFIXES)
        for trial in (source / 'model_trials').iterdir():
            if trial.is_dir():
                paths.update(p for p in trial.iterdir() if p.is_file() and p.suffix in TEXT_SUFFIXES)
                paths.update(p for p in (trial / 'temp/logs').rglob('*') if p.is_file())
        for path in sorted(paths):
            add(path.relative_to(source).as_posix(), path.read_bytes())
        # Earlier cleaned runs retain only a provenance subset. Export every
        # surviving text record, without embedding old source tarballs.
        with zipfile.ZipFile(source / 'historical-provenance.zip') as history:
            for name in sorted(history.namelist()):
                if Path(name).suffix in TEXT_SUFFIXES:
                    add('historical-provenance/' + name, history.read(name))
    manifest = {
        'source_run': source.relative_to(ROOT).as_posix(),
        'file_count': len(entries),
        'uncompressed_bytes': sum(row['bytes'] for row in entries),
        'archive_bytes': archive.stat().st_size,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'credential_redaction_count': redactions,
        'project_root_paths_replaced': True,
        'scope': 'all_retained_text_logs_and_session_metadata; older_cleaned_runs_provenance_only',
        'files': entries,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
