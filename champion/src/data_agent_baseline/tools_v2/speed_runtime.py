"""One process-wide request budget for asyncio clients and video worker threads.

Transport scheduling only: prompts, images, model settings and outputs are intact.
"""
from __future__ import annotations

import asyncio
import os
import json
from pathlib import Path
import threading
import time

import httpx


class RequestBudget:
    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError('request limit must be positive')
        self.limit = limit
        self.ceiling = limit
        self.blocked_until = 0.0
        self.recover_after = 0.0
        self.rejections = 0
        self.active = 0
        self.condition = threading.Condition()

    def try_acquire(self):
        with self.condition:
            if self.active >= self.limit or time.monotonic() < self.blocked_until:
                return False
            self.active += 1
            return True

    async def acquire_async(self):
        # No background acquire thread: cancellation cannot leak a late permit.
        while not self.try_acquire():
            await asyncio.sleep(0.02)

    def acquire(self):
        with self.condition:
            while self.active >= self.limit or time.monotonic() < self.blocked_until:
                delay = self.blocked_until - time.monotonic()
                self.condition.wait(timeout=max(0.02, delay) if delay > 0 else None)
            self.active += 1

    def note_response(self, status):
        with self.condition:
            now = time.monotonic()
            changed = False
            if status == 429:
                self.rejections += 1
                self.limit = max(1, self.limit - 1)
                self.blocked_until = max(self.blocked_until, now + min(30, 2 ** min(self.rejections, 4)))
                self.recover_after = now + 60
                changed = True
            elif status < 400 and now >= self.recover_after and self.limit < self.ceiling:
                self.limit += 1
                self.rejections = 0
                self.recover_after = now + 60
                changed = True
            self.condition.notify_all()
            if changed:
                directory = os.environ.get('TEMP_DIR')
                if directory:
                    try:
                        from tools_v2.artifact_cache import atomic_json
                        atomic_json(Path(directory) / 'logs/request_budget.json', {
                            'configured_ceiling': self.ceiling, 'effective_limit': self.limit,
                            'active': self.active, 'last_status': status,
                            'cooldown_seconds': round(max(0, self.blocked_until - now), 2),
                            'recovery_policy': 'one slot per 60s of successful traffic',
                        })
                    except OSError:
                        pass

    def release(self):
        with self.condition:
            if self.active <= 0:
                raise RuntimeError('unbalanced request budget release')
            self.active -= 1
            self.condition.notify_all()


_budget = None
_budget_lock = threading.Lock()
_video_cpu_lock = threading.Lock()


async def run_video(function, *args, **kwargs):
    """Keep CPU video jobs serial while other task coroutines make progress."""
    def work():
        with _video_cpu_lock:
            return function(*args, **kwargs)
    return await asyncio.to_thread(work)


def request_budget():
    global _budget
    with _budget_lock:
        if _budget is None:
            _budget = RequestBudget(int(os.environ.get('MODEL_MAX_CONCURRENCY', '4')))
        return _budget


class AsyncStream(httpx.AsyncByteStream):
    def __init__(self, stream, budget):
        self.stream, self.budget, self.closed = stream, budget, False

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self):
        if not self.closed:
            self.closed = True
            try:
                await self.stream.aclose()
            finally:
                self.budget.release()


class SyncStream(httpx.SyncByteStream):
    def __init__(self, stream, budget):
        self.stream, self.budget, self.closed = stream, budget, False

    def __iter__(self):
        try:
            yield from self.stream
        finally:
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.stream.close()
            finally:
                self.budget.release()


def limit_async_client(client):
    original = client.send

    async def send(request, *, stream=False, **kwargs):
        budget = request_budget()
        await budget.acquire_async()
        handed_off = False
        try:
            response = await original(request, stream=stream, **kwargs)
            budget.note_response(response.status_code)
            if stream and not response.is_closed:
                response.stream = AsyncStream(response.stream, budget)
                handed_off = True
            return response
        finally:
            if not handed_off:
                budget.release()

    client.send = send
    return client


def limit_sync_client(client):
    from tools_v2.model_http import _record_request
    original = client.send

    def send(request, *, stream=False, **kwargs):
        budget = request_budget()
        queued_at = time.monotonic()
        budget.acquire()
        started = time.monotonic()
        handed_off = False
        try:
            response = original(request, stream=stream, **kwargs)
            budget.note_response(response.status_code)
            _record_request(response, attempt=0, elapsed=time.monotonic() - started,
                            queue_wait=started - queued_at, stream=stream)
            if stream and not response.is_closed:
                response.stream = SyncStream(response.stream, budget)
                handed_off = True
            return response
        finally:
            if not handed_off:
                budget.release()

    client.send = send
    return client


def install():
    """Install before creating per-task model clients; preserve existing retries."""
    from tools_v2 import general_tools
    if getattr(general_tools, '_speed_budget_installed', False):
        return
    previous = general_tools.limit_http_client
    general_tools.limit_http_client = lambda client: previous(limit_async_client(client))
    general_tools._speed_budget_installed = True
