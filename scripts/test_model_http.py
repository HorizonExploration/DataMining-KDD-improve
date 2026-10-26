"""Offline tests for the request limit; no keys or model requests required."""
import asyncio
import json
import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'champion/src/data_agent_baseline'))
from tools_v2.model_http import limit_http_client
from tools_v2 import model_http


class RequestLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.defaults = patch.dict(os.environ, MODEL_RPM='0')
        self.defaults.start()
        self.addCleanup(self.defaults.stop)

    async def test_rpm_spacing_shared_by_clients_and_retries(self):
        clock = [0.0]
        starts = []

        async def sleep(delay):
            clock[0] += delay

        async def handle(request):
            starts.append(clock[0])
            if len(starts) == 1:
                return httpx.Response(429, headers={'Retry-After': '0'})
            return httpx.Response(200, text='OK')

        with patch.dict(os.environ, MODEL_RPM='120'), patch.object(model_http, '_now', lambda: clock[0]), patch.object(model_http, '_sleep', sleep):
            async with limit_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handle))) as a, limit_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handle))) as b:
                self.assertEqual((await a.get('https://example.test')).status_code, 200)
                self.assertEqual((await b.get('https://example.test')).status_code, 200)
        self.assertEqual(starts, [0.0, 0.5, 1.0])

    async def test_retry_after_seconds_and_http_date_are_respected(self):
        self.assertEqual(model_http._retry_delay('300', 0), 300)
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=300))
        self.assertGreater(model_http._retry_delay(future, 0), 298)
        self.assertEqual(model_http._retry_delay('invalid', 2), 20)

    async def test_deepseek_profile_and_thinking_controls(self):
        from tools_v2 import general_tools
        from pydantic_ai.models import create_async_http_client
        clients = []

        def client():
            c = create_async_http_client()
            clients.append(c)
            return c

        config = {'base_url': 'https://example.test', 'api_key': 'offline-test', 'model_name': 'deepseek-flash'}
        try:
            with patch.dict(os.environ, MODEL_API_PROFILE='deepseek', MODEL_THINKING_FORMAT='deepseek', MODEL_SOLVER_THINKING='disabled'), patch.object(general_tools, 'get_model_config', return_value=config), patch.object(general_tools, '_build_http_client', client):
                model = general_tools.build_model_nothink()
                self.assertEqual(model.settings['extra_body'], {'thinking': {'type': 'disabled'}})
                self.assertTrue(model.profile.supports_thinking)
                self.assertEqual(model.profile.openai_chat_thinking_field, 'reasoning_content')
                self.assertEqual(model.profile.openai_chat_send_back_thinking_parts, 'field')
                solver = general_tools.build_model()
                self.assertEqual(solver.settings['extra_body'], {'thinking': {'type': 'disabled'}})
        finally:
            await asyncio.gather(*(c.aclose() for c in clients))

    async def test_deepseek_tool_roundtrip_retains_reasoning_field(self):
        from pydantic_ai import Agent
        from tools_v2 import general_tools
        bodies = []

        async def handle(request):
            body = json.loads(request.content)
            bodies.append(body)
            message = {'role': 'assistant', 'reasoning_content': 'synthetic-test-data'}
            if len(bodies) == 1:
                message.update(content=None, tool_calls=[{'id': 'call_test', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{"text":"OK"}'}}])
                finish = 'tool_calls'
            else:
                assistant = next(x for x in body['messages'] if x['role'] == 'assistant')
                self.assertEqual(assistant['reasoning_content'], 'synthetic-test-data')
                self.assertTrue(any(x['role'] == 'tool' for x in body['messages']))
                message['content'] = 'OK'
                finish = 'stop'
            return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 0, 'model': 'deepseek-flash', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}]})

        def echo(text: str) -> str:
            return text

        config = {'base_url': 'https://example.test', 'api_key': 'offline-test', 'model_name': 'deepseek-flash'}
        async with limit_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            with patch.dict(os.environ, MODEL_API_PROFILE='deepseek', MODEL_SOLVER_THINKING='enabled'), patch.object(general_tools, 'get_model_config', return_value=config), patch.object(general_tools, '_build_http_client', return_value=client):
                model = general_tools.build_model()
                result = await Agent(model, tools=[echo]).run('Call echo with OK')
                self.assertEqual(result.output, 'OK')
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0]['thinking'], {'type': 'enabled'})

    async def test_metrics_include_usage_but_not_prompts_or_credentials(self):
        calls = 0

        async def handle(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, text='rate limited', headers={'Retry-After': '0'})
            return httpx.Response(200, json={'model': 'returned-model', 'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15, 'prompt_tokens_details': {'cached_tokens': 8}}})

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, TEMP_DIR=directory):
            async with limit_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
                await client.post('https://example.test/chat/completions', headers={'Authorization': 'Bearer test-secret'}, json={'messages': [{'role': 'user', 'content': 'private-test-prompt'}]})
            text = (Path(directory) / 'logs/model_requests.jsonl').read_text()
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual([row['status'] for row in rows], [429, 200])
            row = rows[-1]
            self.assertEqual(row['usage']['total_tokens'], 15)
            self.assertEqual(row['usage']['cached_prompt_tokens'], 8)
            self.assertEqual(row['response_model'], 'returned-model')
            self.assertNotIn('test-secret', text)
            self.assertNotIn('private-test-prompt', text)

    async def test_limit_shared_by_separate_clients_until_body_complete(self):
        active = peak = 0

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                nonlocal active
                try:
                    await asyncio.sleep(0.01)
                    yield b'OK'
                finally:
                    active -= 1

        async def handle(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            return httpx.Response(200, stream=Body())

        clients = [limit_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handle))) for _ in range(4)]
        try:
            with patch.dict(os.environ, MODEL_MAX_CONCURRENCY='2'):
                results = await asyncio.gather(*[clients[i % 4].get('https://example.test') for i in range(12)])
            self.assertEqual(peak, 2)
            self.assertEqual(active, 0)
            self.assertTrue(all(r.text == 'OK' for r in results))
        finally:
            await asyncio.gather(*(c.aclose() for c in clients))

    async def test_stream_close_releases_permit(self):
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'OK'

        async def handle(request):
            return httpx.Response(200, stream=Body())

        with patch.dict(os.environ, MODEL_MAX_CONCURRENCY='1'):
            async with limit_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
                first = await client.send(client.build_request('GET', 'https://example.test'), stream=True)
                second = asyncio.create_task(client.get('https://example.test'))
                await asyncio.sleep(0.01)
                self.assertFalse(second.done())
                await first.aclose()
                self.assertEqual((await asyncio.wait_for(second, 1)).status_code, 200)

    async def test_already_consumed_stream_does_not_leak_permit(self):
        async def handle(request):
            return httpx.Response(200, text='OK')

        with patch.dict(os.environ, MODEL_MAX_CONCURRENCY='1'):
            async with limit_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
                response = await client.send(client.build_request('GET', 'https://example.test'), stream=True)
                await response.aclose()
                self.assertEqual((await asyncio.wait_for(client.get('https://example.test'), 1)).text, 'OK')

    async def test_429_retries_and_cancelled_request_releases_permit(self):
        calls = 0
        started = asyncio.Event()

        async def handle(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={'Retry-After': '0'})
            if request.url.path == '/cancel':
                started.set()
                await asyncio.Event().wait()
            return httpx.Response(200, text='OK')

        with patch.dict(os.environ, MODEL_MAX_CONCURRENCY='1'):
            async with limit_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
                self.assertEqual((await client.get('https://example.test')).text, 'OK')
                self.assertEqual(calls, 2)
                pending = asyncio.create_task(client.get('https://example.test/cancel'))
                await started.wait()
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertEqual((await asyncio.wait_for(client.get('https://example.test'), 1)).text, 'OK')


if __name__ == '__main__':
    unittest.main()
