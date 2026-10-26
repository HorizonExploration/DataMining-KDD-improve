"""Offline checks for generic input-based model routing and video worker context."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'champion/src/data_agent_baseline'))
from tools_v2 import general_tools
from video import frame_html_tool


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / 'task_video/context/video').mkdir(parents=True)
        (root / 'task_video/context/video/briefing.mp4').touch()
        env = patch.dict(os.environ, MODEL_NAME='ecnu-plus', MODEL_VIDEO_NAME='ecnu-max',
                         MODEL_API_URL='https://offline.test/v1', MODEL_API_KEY='offline',
                         MODEL_API_URLS='', INPUT_DIR=str(root), MODEL_THINKING_FORMAT='ecnu')
        env.start()
        self.addCleanup(env.stop)
        prices = patch.object(general_tools, '_register_self_hosted_model_price')
        prices.start()
        self.addCleanup(prices.stop)

    async def test_concurrent_tasks_and_background_threads_keep_their_model(self):
        async def route(task):
            general_tools.set_log_task_id(task)
            await asyncio.sleep(0.01)
            return (general_tools.get_model_config()['model_name'],
                    await asyncio.to_thread(lambda: general_tools.get_model_config()['model_name']))
        self.assertEqual(await asyncio.gather(route('task_video'), route('task_text')),
                         [('ecnu-max', 'ecnu-max'), ('ecnu-plus', 'ecnu-plus')])

    async def test_video_layout_threads_use_max_and_ecnu_thinking_parameter(self):
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                                                            message=SimpleNamespace(content='[:page]'))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        general_tools.set_log_task_id('task_video')
        with patch.object(frame_html_tool, 'encode_image', return_value='data:image/png;base64,fixture'):
            result = frame_html_tool.frames_to_hiccup(['one.png', 'two.png'], client=client, workers=2)
        self.assertTrue(all(value == '[:page]' for value in result.values()))
        self.assertEqual([x['model'] for x in requests], ['ecnu-max', 'ecnu-max'])
        self.assertTrue(all(x['extra_body'] == {'thinking': {'type': 'disabled'}} for x in requests))

    async def test_unset_video_model_preserves_legacy_single_model_mode(self):
        general_tools.set_log_task_id('task_video')
        with patch.dict(os.environ, MODEL_VIDEO_NAME=''):
            self.assertEqual(general_tools.get_model_config()['model_name'], 'ecnu-plus')


if __name__ == '__main__':
    unittest.main()
