"""验证模型工具调用；需要用户令牌，会产生一次模型请求。"""
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
from dotenv import dotenv_values

config_path = Path(os.environ.get('KDD_ENV_FILE', Path(__file__).resolve().parents[1] / '.env'))
values = dotenv_values(config_path)
for name in ['MODEL_API_URL', 'MODEL_NAME', 'MODEL_API_KEY', 'ASR_BACKEND', 'MODEL_SOLVER_THINKING']:
    if values.get(name) or (name == 'MODEL_API_KEY' and name in values):
        os.environ[name] = values[name] or ''
key = os.environ.get('MODEL_API_KEY', '').strip()
if not key:
    raise SystemExit('未配置 MODEL_API_KEY。请在本机 .env 或终端设置真实令牌后重试。')
body = {
    'model': os.environ['MODEL_NAME'],
    'messages': [{'role': 'user', 'content': '请调用 echo 工具，参数 text 为 OK。'}],
    'tools': [{'type': 'function', 'function': {
        'name': 'echo', 'description': '返回文本',
        'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}},
                       'required': ['text']},
    }}],
    'stream': False,
}
thinking = os.environ.get('MODEL_SOLVER_THINKING', 'default')
if thinking in {'enabled', 'disabled'}:
    body['thinking'] = {'type': thinking}
elif thinking != 'default':
    raise SystemExit('MODEL_SOLVER_THINKING 必须为 default/enabled/disabled')
response = requests.post(
    os.environ['MODEL_API_URL'].rstrip('/') + '/chat/completions', json=body,
    headers={'Authorization': f'Bearer {key}'}, timeout=(15, 60),
)
if not response.ok:
    raise SystemExit(f'接口返回 HTTP {response.status_code}，请核对地址、模型、令牌权限和额度。')
message = response.json()['choices'][0]['message']
calls = message.get('tool_calls') or []
for call in calls:
    function = call.get('function', {})
    if function.get('name') == 'echo':
        arguments = json.loads(function['arguments'])
        if arguments.get('text') == 'OK':
            report = {
                'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'status': 'PASS', 'check': 'echo_tool_call',
                'model': os.environ['MODEL_NAME'],
                'api_url': os.environ['MODEL_API_URL'],
                'solver_thinking': thinking,
            }
            output = Path(__file__).resolve().parents[1] / 'docs/model-check.json'
            output.parent.mkdir(exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print('PASS：模型返回 echo(text="OK") 工具调用。')
            break
else:
    raise SystemExit('接口有响应，但没有返回预期的 echo(text="OK") 工具调用。')
