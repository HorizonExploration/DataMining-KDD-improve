"""Offline mixed-thread/async transport tests; no model API calls."""
import asyncio
import concurrent.futures
import copy
import importlib.util
import sys
import threading
import time
import unittest
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'champion/src/data_agent_baseline'))
from tools_v2 import speed_runtime as runtime
if os.environ.get('KDD_SPEED_RUNTIME_TEST_FILE'):
    spec = importlib.util.spec_from_file_location('next_runtime', os.environ['KDD_SPEED_RUNTIME_TEST_FILE'])
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        runtime._budget = runtime.RequestBudget(2)

    async def test_mixed_threads_and_async_share_actual_inflight_limit(self):
        lock = threading.Lock()
        active = maximum = 0
        def enter():
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
        def leave():
            nonlocal active
            with lock:
                active -= 1
        def sync_handler(request):
            enter()
            try:
                time.sleep(0.02)
                return httpx.Response(200, json={'ok': True})
            finally:
                leave()
        async def async_handler(request):
            enter()
            try:
                await asyncio.sleep(0.02)
                return httpx.Response(200, json={'ok': True})
            finally:
                leave()
        with runtime.limit_sync_client(httpx.Client(transport=httpx.MockTransport(sync_handler))) as sync:
            async with runtime.limit_async_client(httpx.AsyncClient(transport=httpx.MockTransport(async_handler))) as client:
                await asyncio.gather(*[client.get('https://offline.test') for _ in range(8)],
                                     *[asyncio.to_thread(sync.get, 'https://offline.test') for _ in range(8)])
        self.assertEqual(maximum, 2)
        self.assertEqual(runtime._budget.active, 0)

    async def test_cancel_waiter_never_leaks_late_permit(self):
        runtime._budget = runtime.RequestBudget(1)
        runtime._budget.acquire()
        async with runtime.limit_async_client(httpx.AsyncClient()) as client:
            task = asyncio.create_task(client.get('https://offline.test'))
            await asyncio.sleep(0.03)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        runtime._budget.release()
        await asyncio.sleep(0.03)
        self.assertEqual(runtime._budget.active, 0)

    async def test_cpu_video_is_serial_without_blocking_event_loop(self):
        active = maximum = ticks = 0
        def cpu_job():
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            time.sleep(0.04)
            active -= 1
        async def ticker():
            nonlocal ticks
            for _ in range(5):
                await asyncio.sleep(0.01)
                ticks += 1
        await asyncio.gather(runtime.run_video(cpu_job), runtime.run_video(cpu_job), ticker())
        self.assertEqual(maximum, 1)
        self.assertEqual(ticks, 5)

    async def test_transport_errors_release_both_kinds_of_permits(self):
        def fail(request):
            raise RuntimeError('offline error')
        async with runtime.limit_async_client(httpx.AsyncClient(transport=httpx.MockTransport(fail))) as client:
            with self.assertRaises(RuntimeError):
                await client.get('https://offline.test')
        with runtime.limit_sync_client(httpx.Client(transport=httpx.MockTransport(fail))) as client:
            with self.assertRaises(RuntimeError):
                client.get('https://offline.test')
        self.assertEqual(runtime._budget.active, 0)

    async def test_stream_close_releases_exactly_once(self):
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'ok'
        async with runtime.limit_async_client(httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream())))) as client:
            response = await client.send(client.build_request('GET', 'https://offline.test'), stream=True)
            self.assertEqual(runtime._budget.active, 1)
            await response.aclose()
            await response.aclose()
            self.assertEqual(runtime._budget.active, 0)

    async def test_reviewed_source_migration_is_hash_exact(self):
        spec = importlib.util.spec_from_file_location('continuation', ROOT / 'scripts/continue_parallel.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old = {'source': {'agent.py': 'before'}, 'configuration': {'MAX_PARALLEL': '2'}, 'inputs': 'unchanged'}
        current = copy.deepcopy(old)
        current['source']['agent.py'] = 'after'
        current['source']['new.py'] = 'added'
        current['configuration']['MAX_PARALLEL'] = '4'
        audit = {'agent.py': {'before': 'before', 'after': 'after'},
                 'new.py': {'before': None, 'after': 'added'}}
        module.validate_scheduling_change(old, current, audit)
        with self.assertRaises(ValueError):
            module.validate_scheduling_change(old, current)
        current['inputs'] = 'changed'
        with self.assertRaises(ValueError):
            module.validate_scheduling_change(old, current, audit)


if __name__ == '__main__':
    unittest.main()
