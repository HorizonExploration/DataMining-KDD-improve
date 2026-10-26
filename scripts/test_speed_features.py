"""Offline integration checks for warm preprocessing and strict resume."""
import copy
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'champion/src/data_agent_baseline'))
from tools_v2.artifact_cache import DocumentCache, atomic_json, digest_json, document_identity
from tools_v2.run_state import reuse_predictions, write_completion
from doc_tools.fanout_struct import StructResult
from doc_tools import doc_prepare


@dataclass
class Model:
    model_name: str = 'offline-model'
    base_url: str = 'https://one.example.test/v1'
    system: str = 'openai'
    settings: dict = field(default_factory=lambda: {'temperature': 0})


def result():
    return StructResult(table_name='records', primary_key=['record_id'],
                        columns=['record_id', 'value'],
                        rows=[{'record_id': '1', 'value': 'A'}],
                        n_universe=1, n_sections=1, n_failed_sections=0)


class SpeedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = patch.dict(os.environ, KDD_DOC_CACHE_DIR=str(self.root / 'cache'),
                                 KDD_DOC_CACHE_MODE='readwrite', MODEL_VERSION_LABEL='fixture-v1')
        environment.start()
        self.addCleanup(environment.stop)

    def task(self, name):
        p = self.root / name
        (p / 'context/doc').mkdir(parents=True)
        (p / 'context/doc/records.md').write_text('record 1 has value A')
        (p / 'context/knowledge.md').write_text('record_id is the key')
        return p

    async def prepare(self, task, model=None):
        return await doc_prepare.prepare_doc_tables_async(
            task_dir=task, log_dir=self.root / 'logs' / task.name,
            model_nothink=model or Model(), model_think=model or Model())

    async def test_warm_cross_run_reuses_exact_rows_without_model_call(self):
        a = self.task('run_a')
        b = self.task('run_b')
        extractor = AsyncMock(return_value=result())
        with patch.object(doc_prepare, 'structurize_doc', extractor):
            first = await self.prepare(a)
            second = await self.prepare(b)
        extractor.assert_awaited_once()
        self.assertFalse(first.tables[0].cache_hit)
        self.assertEqual(second.tables[0].cache_source, 'shared')
        rows = []
        for task in (a, b):
            with sqlite3.connect(task / 'context/db/records.db') as db:
                rows.append(db.execute('SELECT * FROM records').fetchall())
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[0], [('1', 'A')])
        self.assertTrue((self.root / 'logs/run_b/doc_extract/records.result.json').is_file())

    async def test_cold_mode_and_different_provider_force_extraction(self):
        extractor = AsyncMock(return_value=result())
        with patch.object(doc_prepare, 'structurize_doc', extractor):
            await self.prepare(self.task('one'))
            other = await self.prepare(self.task('two'), Model(base_url='https://two.example.test/v1'))
            with patch.dict(os.environ, KDD_DOC_CACHE_MODE='off'):
                cold = await self.prepare(self.task('three'))
        self.assertEqual(extractor.await_count, 3)
        self.assertFalse(other.tables[0].cache_hit)
        self.assertFalse(cold.tables[0].cache_hit)

    async def test_partial_extraction_is_not_shared(self):
        partial = result()
        partial.n_failed_sections = 1
        extractor = AsyncMock(return_value=partial)
        with patch.object(doc_prepare, 'structurize_doc', extractor):
            await self.prepare(self.task('one'))
            second = await self.prepare(self.task('two'))
        self.assertEqual(extractor.await_count, 2)
        self.assertFalse(second.tables[0].cache_hit)
        self.assertFalse((self.root / 'cache').exists())

    async def test_identity_includes_input_knowledge_both_models_and_settings(self):
        task = self.task('task')
        doc = task / 'context/doc/records.md'
        knowledge = task / 'context/knowledge.md'
        base = document_identity(doc, knowledge, Model(), Model(), 'v1')
        variants = [
            document_identity(doc, knowledge, Model(model_name='different'), Model(), 'v1'),
            document_identity(doc, knowledge, Model(), Model(model_name='different'), 'v1'),
            document_identity(doc, knowledge, Model(settings={'temperature': 1}), Model(), 'v1'),
            document_identity(doc, knowledge, Model(), Model(), 'v2'),
        ]
        knowledge.write_text('changed rule')
        variants.append(document_identity(doc, knowledge, Model(), Model(), 'v1'))
        knowledge.write_text('record_id is the key')
        doc.write_text('changed document')
        variants.append(document_identity(doc, knowledge, Model(), Model(), 'v1'))
        doc.write_text('record 1 has value A')
        with patch.dict(os.environ, MODEL_VERSION_LABEL='fixture-v2'):
            variants.append(document_identity(doc, knowledge, Model(), Model(), 'v1'))
        self.assertTrue(all(digest_json(v) != digest_json(base) for v in variants))

    async def test_corrupt_or_failed_normalization_cache_is_rejected(self):
        cache = DocumentCache()
        identity = {'fixture': 1}
        payload = result().to_jsonable()
        bad = copy.deepcopy(payload)
        bad['normalize_stats'] = {'value': {'failed': 1}}
        self.assertFalse(cache.store(identity, bad))
        self.assertTrue(cache.store(identity, payload))
        file = cache.directory / f'{digest_json(identity)}.json'
        value = json.loads(file.read_text())
        value['payload']['rows'][0]['value'] = 'corrupt'
        atomic_json(file, value)
        self.assertIsNone(cache.load(identity))

    def parent(self):
        source = self.root / 'parent'
        identity = {'source': 'same-code', 'inputs': 'same-input', 'model': 'same-model'}
        atomic_json(source / 'run_manifest.json', {'identity': identity, 'identity_sha256': digest_json(identity)})
        for task in ('task_1', 'task_2'):
            output = source / 'predictions' / task
            output.mkdir(parents=True)
            (output / 'prediction.csv').write_text('value\nA\n')
        self.assertTrue(write_completion(source / 'predictions/task_1', digest_json(identity)))
        return source, identity

    async def test_resume_requires_success_receipt_and_preserves_parent(self):
        source, identity = self.parent()
        target = self.root / 'next'
        reused = reuse_predictions(target, source, identity, ['task_1', 'task_2'])
        self.assertEqual(reused, ['task_1'])
        self.assertTrue((source / 'predictions/task_1/prediction.csv').is_file())
        self.assertFalse((target / 'predictions/task_2/prediction.csv').exists())

    async def test_resume_rejects_source_model_or_input_change(self):
        source, identity = self.parent()
        for key in ('source', 'inputs', 'model'):
            changed = dict(identity, **{key: 'changed'})
            with self.assertRaises(ValueError):
                reuse_predictions(self.root / key, source, changed, ['task_1'])

    async def test_resume_rejects_tampered_prediction(self):
        source, identity = self.parent()
        (source / 'predictions/task_1/prediction.csv').write_text('value\nchanged\n')
        self.assertEqual(reuse_predictions(self.root / 'next', source, identity, ['task_1']), [])

    async def test_output_guard_requires_matching_completion_for_new_runs(self):
        run = self.root / 'guard'
        output = run / 'predictions/task_44'
        output.mkdir(parents=True)
        prediction = output / 'prediction.csv'
        prediction.write_text('value\nA\n')
        assets = self.root / 'guard_assets'
        (assets / 'data').mkdir(parents=True)
        (assets / 'data/public_reference_tasks.json').write_text(json.dumps({'task_ids': ['task_44']}))
        environment = dict(os.environ, KDD_ROOT=str(assets), RUN_DIR=str(run), KDD_RUN_IDENTITY='fixture')

        def check():
            return subprocess.run([sys.executable, str(ROOT / 'scripts/check_run_outputs.py'), 'task_44'],
                                  env=environment, capture_output=True).returncode

        self.assertEqual(check(), 1)
        self.assertTrue(write_completion(output, 'fixture'))
        self.assertEqual(check(), 0)
        prediction.write_text('value\nchanged\n')
        self.assertEqual(check(), 1)

    async def test_main_persists_receipt_before_whole_batch_summary(self):
        import zz_agent_v2
        inputs = self.root / 'input'
        (inputs / 'task_1').mkdir(parents=True)
        output = self.root / 'output'

        async def fake(task, **kwargs):
            directory = kwargs['output_base_dir'] / task.name
            directory.mkdir(parents=True)
            (directory / 'prediction.csv').write_text('value\nA\n')
            stats = {k: 0 for k in ['turns', 'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'total_tokens', 'requests']}
            return dict(stats, task_id=task.name, success=True)

        with patch.dict(os.environ, INPUT_DIR=str(inputs), OUTPUT_DIR=str(output), TEMP_DIR=str(self.root / 'temp'), KDD_RUN_IDENTITY='fixture'), patch.object(zz_agent_v2, 'process_one_task', fake), patch.object(zz_agent_v2.general_tools, 'is_docker', return_value=False):
            await zz_agent_v2.main(max_parallel=1)
        receipt = json.loads((output / 'task_1/completion.json').read_text())
        self.assertTrue(receipt['success'])
        self.assertEqual(receipt['run_identity'], 'fixture')


if __name__ == '__main__':
    unittest.main()
