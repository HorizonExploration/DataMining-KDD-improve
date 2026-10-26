"""Shared HTTP request limit across every model client in one event loop."""
from __future__ import annotations

import asyncio
import os
import math
import json
import time
import weakref
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from loguru import logger

_limits: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_pacers: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_now = time.monotonic
_sleep = asyncio.sleep


async def _pace_request() -> None:
    """Smooth request starts across clients; RPM=0 leaves throughput unchanged."""
    rpm = float(os.environ.get('MODEL_RPM', '0'))
    if not math.isfinite(rpm) or rpm < 0:
        raise ValueError('MODEL_RPM must be finite and nonnegative')
    if rpm == 0:
        return
    loop = asyncio.get_running_loop()
    if loop not in _pacers:
        _pacers[loop] = (asyncio.Lock(), [0.0])
    lock, next_start = _pacers[loop]
    async with lock:
        delay = next_start[0] - _now()
        if delay > 0:
            await _sleep(delay)
        next_start[0] = _now() + 60.0 / rpm


def _retry_delay(value: str, attempt: int) -> float:
    try:
        delay = float(value)
        if math.isfinite(delay):
            return max(0.0, delay)
    except ValueError:
        pass
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return min(5.0 * 2**attempt, 60.0)


def _record_request(response, *, attempt, elapsed, queue_wait, stream):
    """Record timing and provider usage for all stages, never prompts or headers."""
    directory = os.environ.get('TEMP_DIR')
    if not directory:
        return
    try:
        try:
            payload = {} if stream else response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        usage = payload.get('usage') or {}
        if not isinstance(usage, dict):
            usage = {}
        safe_usage = {k: usage[k] for k in (
            'prompt_tokens', 'completion_tokens', 'total_tokens',
            'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens',
        ) if isinstance(usage.get(k), (int, float))}
        details = usage.get('prompt_tokens_details') or {}
        if isinstance(details, dict) and isinstance(details.get('cached_tokens'), (int, float)):
            safe_usage['cached_prompt_tokens'] = details['cached_tokens']
        task_context = getattr(logger, '_zz_task_id_ctx', None)
        try:
            requested_model = json.loads(response.request.content).get('model')
        except (ValueError, TypeError, AttributeError, httpx.RequestNotRead):
            requested_model = os.environ.get('MODEL_NAME')
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task_id': task_context.get() if task_context else '-',
            'requested_model': requested_model,
            'response_model': str(payload.get('model', ''))[:128],
            'host': response.request.url.host,
            'status': response.status_code, 'retry_attempt': attempt,
            'service_seconds': round(elapsed, 6),
            'queue_seconds': round(queue_wait, 6), 'stream': stream,
            'usage': safe_usage,
        }
        destination = Path(directory) / 'logs/model_requests.jsonl'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
    except (OSError, ValueError, TypeError, AttributeError):
        logger.warning('模型请求统计未能写入；不影响求解。')


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in _limits:
        limit = int(os.environ.get('MODEL_MAX_CONCURRENCY', '3'))
        if limit < 1:
            raise ValueError('MODEL_MAX_CONCURRENCY must be positive')
        _limits[loop] = asyncio.Semaphore(limit)
    return _limits[loop]


class _LimitedStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, semaphore: asyncio.Semaphore):
        self.stream = stream
        self.semaphore = semaphore
        self.closed = False

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
                self.semaphore.release()


def limit_http_client(client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Limit all requests, retain permits through response bodies, back off on 429."""
    original_send = client.send

    async def send(request, *, stream=False, **kwargs):
        queued_at = _now()
        semaphore = _semaphore()
        await semaphore.acquire()
        handed_to_stream = False
        try:
            retries = max(0, int(os.environ.get('MODEL_HTTP_429_RETRIES', '5')))
            for attempt in range(retries + 1):
                await _pace_request()
                started_at = _now()
                response = await original_send(request, stream=stream, **kwargs)
                _record_request(response, attempt=attempt, elapsed=_now() - started_at,
                                queue_wait=started_at - queued_at, stream=stream)
                if response.status_code != 429 or attempt == retries:
                    if stream and not response.is_closed:
                        response.stream = _LimitedStream(response.stream, semaphore)
                        handed_to_stream = True
                    return response
                retry_after = response.headers.get('Retry-After', '')
                await response.aclose()
                delay = _retry_delay(retry_after, attempt)
                logger.warning('模型接口 HTTP 429，等待 {} 秒后重试（{}/{}）', delay, attempt + 1, retries)
                await _sleep(delay)
                queued_at = _now()
        finally:
            if not handed_to_stream:
                semaphore.release()

    client.send = send
    return client
