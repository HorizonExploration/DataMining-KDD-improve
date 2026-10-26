"""下载固定版本公开数据/ASR 权重；不读取 gold 答案内容，不调用模型 API。"""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_REVISION = '96be19257d7aa251dbce4771d46a26fc36f6dc52'
ASR_REVISIONS = {
    'tiny': 'd90ca5fe260221311c53c58e660288d3deb8d356',
    'medium': '08e178d48790749d25932bbc082711ddcfdfbc4f',
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download_file(repo, revision, name, destination, dataset=False, source_url=None):
    """使用 requests 流式下载，避开本机 HF SDK 大文件连接停滞。"""
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + '.part')
    prefix = 'datasets/' if dataset else ''
    url = source_url or f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}'
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            # Hugging Face 主站需代理；实际文件 CDN 已实测可直连，避免代理流量限速。
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as probe:
                probe.raise_for_status()
                resolved = probe.url
            session = requests.Session()
            session.trust_env = not urlsplit(resolved).hostname.endswith(('.hf.co', '.githubusercontent.com'))
            with session.get(resolved, headers=headers, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                written = offset if mode == 'ab' else 0
                total = written + int(response.headers.get('Content-Length', 0))
                last_report = time.monotonic()
                print(f'下载 {repo}/{name}: {written}/{total} bytes', flush=True)
                with part.open(mode) as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report >= 20:
                            print(f'下载 {name}: {written / 2**20:.1f}/{total / 2**20:.1f} MiB', flush=True)
                            last_report = time.monotonic()
                if total and written != total:
                    raise OSError(f'下载长度不符: {written}/{total}')
            part.rename(destination)
            return destination
        except (requests.RequestException, OSError) as error:
            print(f'{name} 第 {attempt + 1} 次下载失败: {type(error).__name__}', flush=True)
            if attempt == 2:
                raise RuntimeError(f'{repo}/{name} 下载失败，保留 .part 供重试') from None


def prepare_data():
    def download(name):
        return download_file('HKUSTDial/DataSpace', DATA_REVISION, name,
                             ROOT / 'downloads' / name, dataset=True)

    data = ROOT / 'data'
    benchmark = data / 'DataSpace-Benchmark'
    marker = benchmark / '.archive_sha256'
    integrity_path = ROOT / 'docs/data-integrity.json'
    local_manifest = data / 'public_reference_tasks.json'
    if marker.is_file() and local_manifest.is_file() and integrity_path.is_file():
        previous = json.loads(integrity_path.read_text())
        if previous.get('dataset_revision') != DATA_REVISION:
            raise ValueError('已有数据版本与准备脚本不同，请单独迁移')
        actual = marker.read_text().strip()
        if actual != previous['archive_sha256'] or digest(local_manifest) != previous['manifest_sha256']:
            raise ValueError('已有数据完成标记或题单与校验记录不同')
        manifest_path = local_manifest
        print('复用已解压数据和校验记录；不重新下载压缩包。', flush=True)
    else:
        manifest_path = download('data/public_reference_tasks.json')
        checksum_path = download('release/DataSpace-Benchmark.zip.sha256')
        archive = download('release/DataSpace-Benchmark.zip')
        expected = checksum_path.read_text().split()[0].lower()
        actual = digest(archive)
        if actual != expected:
            raise ValueError(f'压缩包校验失败: {actual} != {expected}')
        print('压缩包 SHA-256 校验通过:', actual, flush=True)
    manifest = json.loads(manifest_path.read_text())
    ids = manifest['task_ids']
    if not manifest['task_count'] == len(ids) == len(set(ids)) == 60:
        raise ValueError('公开题单不是无重复的 60 题')
    data.mkdir(exist_ok=True)
    if not marker.exists():
        if benchmark.exists():
            raise FileExistsError(f'{benchmark} 已存在但无完成标记，请检查后重试')
        staging = data / '.extracting'
        staging.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (staging / name).resolve().is_relative_to(staging.resolve()):
                    raise ValueError('压缩包包含越界路径')
            z.extractall(staging)
        extracted = staging / 'DataSpace-Benchmark'
        if not (extracted / 'input').is_dir():
            raise ValueError(f'解压层级不同于指南，请检查 {staging}')
        extracted.rename(benchmark)
        marker.write_text(actual + '\n')
        if not any(staging.iterdir()):
            staging.rmdir()
    if marker.read_text().strip() != actual:
        raise ValueError('已解压数据的压缩包哈希不同')
    gold_ids = {p.parent.name for p in (benchmark / 'output').glob('task_*/gold.csv')}
    config_ids = {p.stem for p in (benchmark / 'evaluation/configs').glob('task_*.json')}
    if set(ids) != gold_ids or set(ids) != config_ids:
        raise ValueError('题单、gold 路径、评测配置的 ID 不一致')
    for task in ids:
        source = benchmark / 'input' / task
        if not (source / 'task.json').is_file() or not (source / 'context').is_dir():
            raise ValueError(f'任务输入缺失: {task}')
    target = data / 'input60'
    complete = data / 'input60.complete.json'
    if target.exists() and not complete.exists():
        raise FileExistsError(f'{target} 已存在但无完成标记，请检查后重试')
    if not target.exists():
        staging = data / '.input60-building'
        if staging.exists():
            raise FileExistsError(f'{staging} 残留，请检查后重试')
        staging.mkdir()
        for task in ids:
            source = benchmark / 'input' / task
            dest = staging / task
            dest.mkdir()
            shutil.copy2(source / 'task.json', dest / 'task.json')
            shutil.copytree(source / 'context', dest / 'context')
        if list(staging.rglob('gold.csv')):
            raise ValueError('输入中发现 gold.csv')
        staging.rename(target)
        complete.write_text(json.dumps({'revision': DATA_REVISION, 'task_ids': ids}, indent=2))
    for task in ids:
        if not (target / task / 'task.json').is_file() or not (target / task / 'context').is_dir():
            raise ValueError(f'input60 不完整: {task}')
    if {p.name for p in target.iterdir()} != set(ids) or list(target.rglob('gold.csv')):
        raise ValueError('input60 题目集合不符或包含答案')
    if manifest_path.resolve() != local_manifest.resolve():
        shutil.copy2(manifest_path, local_manifest)
    (ROOT / 'docs/data-integrity.json').write_text(json.dumps({
        'dataset_revision': DATA_REVISION, 'archive_sha256': actual,
        'manifest_sha256': digest(manifest_path), 'task_count': 60,
        'task_ids': ids, 'gold_content_read': False,
    }, ensure_ascii=False, indent=2))
    print('固定 60 题输入准备完成:', target, flush=True)


def prepare_asr():
    records = {}
    for name, revision in ASR_REVISIONS.items():
        path = ROOT / 'champion/asr_models' / name
        files = {}
        for f in ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt']:
            download_file(f'Systran/faster-whisper-{name}', revision, f, path / f)
            p = path / f
            if not p.is_file():
                raise FileNotFoundError(p)
            files[f] = {'size': p.stat().st_size, 'sha256': digest(p)}
        records[name] = {'revision': revision, 'files': files}
        print('ASR 权重就绪:', name, flush=True)
    (ROOT / 'docs/asr-integrity.json').write_text(json.dumps(records, indent=2))


def prepare_ffmpeg():
    from static_ffmpeg import run
    binary_dir = Path(run.get_platform_dir())
    if (binary_dir / 'installed.crumb').is_file() and all((binary_dir / name).is_file() for name in ('ffmpeg', 'ffprobe')):
        record = ROOT / 'docs/ffmpeg-integrity.json'
        if record.is_file():
            expected = json.loads(record.read_text()).get('executables', {})
            for name, checksum in expected.items():
                if digest(binary_dir / name) != checksum:
                    raise ValueError(f'现有 {name} 与校验记录不一致')
        run.get_or_fetch_platform_executables_else_raise()
        print('复用已安装的静态 FFmpeg/FFprobe；不重新下载压缩包。', flush=True)
        return
    archive = ROOT / 'downloads/ffmpeg-linux-v8.zip'
    package_archive = Path(run.get_platform_dir() + '.zip')
    if package_archive.exists() and not archive.exists() and not archive.with_suffix('.zip.part').exists():
        package_archive.rename(archive.with_suffix('.zip.part'))
    url = run.get_platform_http_zip()
    download_file('static-ffmpeg', 'v8.0', 'linux.zip', archive, source_url=url)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f'FFmpeg 压缩包 CRC 校验失败: {bad}')
        for name in z.namelist():
            if not (binary_dir.parent / name).resolve().is_relative_to(binary_dir.parent.resolve()):
                raise ValueError('FFmpeg 压缩包包含越界路径')
        z.extractall(binary_dir.parent)
    (binary_dir / 'installed.crumb').write_text(f'installed from {url}\n')
    executables = run.get_or_fetch_platform_executables_else_raise()
    (ROOT / 'docs/ffmpeg-integrity.json').write_text(json.dumps({
        'source': url, 'archive_sha256': digest(archive),
        'executables': {Path(p).name: digest(Path(p)) for p in executables},
    }, indent=2))
    print('静态 FFmpeg/FFprobe 解压与校验完成', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['data', 'asr', 'ffmpeg'])
    args = parser.parse_args()
    (ROOT / 'docs').mkdir(exist_ok=True)
    {'data': prepare_data, 'asr': prepare_asr, 'ffmpeg': prepare_ffmpeg}[args.kind]()
