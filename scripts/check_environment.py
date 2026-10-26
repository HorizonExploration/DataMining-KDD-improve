"""不调用 LLM：实际验证表格/SQL、PDF、视频和 Agent 入口。"""
import importlib
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import av
import duckdb
import pandas as pd
import polars as pl
import pyarrow
import pymupdf
from static_ffmpeg import run

ROOT = Path(__file__).resolve().parents[1]
report = {'checks': {}, 'versions': {}}
for package in ['pandas', 'duckdb', 'polars', 'pyarrow', 'pymupdf', 'av',
                'faster-whisper', 'pydantic-ai', 'ctranslate2']:
    report['versions'][package] = version(package)

with tempfile.TemporaryDirectory(prefix='kdd-env-') as directory:
    work = Path(directory)
    frame = pd.DataFrame({'region': ['A', 'B', 'A'], 'amount': [10, 20, 30]})
    frame.to_csv(work / 'orders.csv', index=False)
    result = duckdb.sql(f"SELECT region, SUM(amount) total FROM read_csv('{work / 'orders.csv'}') GROUP BY region ORDER BY region").fetchall()
    assert result == [('A', 40.0), ('B', 20.0)], result
    assert pl.read_csv(work / 'orders.csv').height == 3
    frame.to_parquet(work / 'orders.parquet', index=False)
    assert pd.read_parquet(work / 'orders.parquet').equals(frame)
    import sqlite3
    with sqlite3.connect(work / 'orders.db') as db:
        frame.to_sql('orders', db, index=False)
        assert db.execute('SELECT SUM(amount) FROM orders').fetchone()[0] == 60
    report['checks']['csv_sql_parquet'] = 'PASS'
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), 'KDD environment OK')
    doc.save(work / 'example.pdf')
    doc.close()
    with pymupdf.open(work / 'example.pdf') as doc:
        assert 'KDD environment OK' in doc[0].get_text()
    report['checks']['pdf_roundtrip'] = 'PASS'
    ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()
    binary_dir = ROOT / 'tools/bin'
    binary_dir.mkdir(exist_ok=True, parents=True)
    for name, path in [('ffmpeg', ffmpeg), ('ffprobe', ffprobe)]:
        link = binary_dir / name
        if not link.exists():
            link.symlink_to(path)
        report['versions'][name] = subprocess.check_output([path, '-version'], text=True).splitlines()[0]
    video = work / 'example.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=blue:s=160x120:r=2:d=1', '-c:v', 'mpeg4',
                    str(video)], check=True, timeout=30)
    with av.open(str(video)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 2
        assert frames[0].to_image().size == (160, 120)
    subprocess.run([ffprobe, '-v', 'error', str(video)], check=True, timeout=30)
    report['checks']['ffmpeg_encode_pyav_decode'] = 'PASS'

sys.path.insert(0, str(ROOT / 'champion/src/data_agent_baseline'))
os.chdir(ROOT / 'champion')
importlib.import_module('zz_agent_v2')
report['checks']['agent_entry_import_no_api_call'] = 'PASS'
(ROOT / 'docs/environment-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
