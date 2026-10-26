"""Content-addressed, opt-in cache for input-derived document extraction."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, sort_keys=True)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def document_identity(doc: Path, knowledge: Path | None, model, transform_model, version: str) -> dict:
    package = Path(__file__).resolve().parents[1]
    champion = package.parents[1]
    code = [package / 'doc_tools/doc_prepare.py', package / 'doc_tools/fanout_struct.py',
            package / 'tools_v2/artifact_cache.py', package / 'tools_v2/pdf_reflow.py',
            package / 'tools_v2/general_tools.py', package / 'tools_v2/model_http.py',
            champion / 'uv.lock']

    def model_identity(item):
        return {
            'name': str(getattr(item, 'model_name', type(item).__name__)),
            'provider': str(getattr(item, 'system', 'unknown')),
            'endpoint_sha256': hashlib.sha256(str(getattr(item, 'base_url', '')).encode()).hexdigest(),
            'settings_sha256': digest_json(getattr(item, 'settings', None)),
        }

    label = os.environ.get('MODEL_VERSION_LABEL') or 'unversioned@' + datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    return {
        'schema': 1, 'doc_name': doc.name, 'doc_sha256': file_digest(doc),
        'knowledge_sha256': file_digest(knowledge) if knowledge and knowledge.is_file() else None,
        'extract_model': model_identity(model),
        'transform_model': model_identity(transform_model or model),
        'model_version_label': label, 'pipeline_version': version,
        'api_profile': os.environ.get('MODEL_API_PROFILE', 'openai'),
        'thinking_format': os.environ.get('MODEL_THINKING_FORMAT', 'legacy'),
        'solver_thinking': os.environ.get('MODEL_SOLVER_THINKING', 'default'),
        'code': {str(p.relative_to(champion)): file_digest(p) for p in code},
    }


def complete_document(payload: dict) -> bool:
    """Partial extraction and failed normalization are never shared."""
    try:
        stats = payload['stats']
        columns = payload['columns']
        rows = payload['rows']
        if not isinstance(payload['table_name'], str) or not payload['table_name']:
            return False
        if not isinstance(columns, list) or not columns or not isinstance(rows, list):
            return False
        if stats['n_sections'] < 1 or stats['n_failed_sections'] != 0 or stats['n_universe'] != len(rows):
            return False
        if any(not isinstance(row, dict) for row in rows):
            return False
        if any(not set(columns).issubset(row) for row in rows):
            return False
        if any(item.get('failed', 0) for item in payload.get('normalize_stats', {}).values()):
            return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


class DocumentCache:
    def __init__(self):
        mode = os.environ.get('KDD_DOC_CACHE_MODE', 'off')
        if mode not in {'off', 'readwrite'}:
            raise ValueError('KDD_DOC_CACHE_MODE must be off/readwrite')
        self.enabled = mode == 'readwrite'
        root = Path(os.environ.get('KDD_ROOT', Path(__file__).resolve().parents[4]))
        self.directory = Path(os.environ.get('KDD_DOC_CACHE_DIR', root / 'cache/doc_extract'))

    def load(self, identity: dict) -> dict | None:
        if not self.enabled:
            return None
        key = digest_json(identity)
        try:
            envelope = json.loads((self.directory / f'{key}.json').read_text())
            payload = envelope['payload']
            if envelope['identity'] != identity or envelope['payload_sha256'] != digest_json(payload):
                return None
            return payload if complete_document(payload) else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def store(self, identity: dict, payload: dict) -> bool:
        if not self.enabled or not complete_document(payload):
            return False
        atomic_json(self.directory / f'{digest_json(identity)}.json', {
            'identity': identity, 'payload': payload, 'payload_sha256': digest_json(payload),
            'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        })
        return True
