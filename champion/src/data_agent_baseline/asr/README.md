# 远程 GPU faster-whisper 服务（`data_agent_baseline.asr`）

极薄的无状态 HTTP 层：只做纯计算（在 GPU 上跑 `transcribe` / `detect_language`），
语言检测 / prompt 选择 / 两遍策略 / CER 计算等业务逻辑全部留在 client。

- `server.py`         —— 部署在 GPU host 的 FastAPI 服务。
- `remote_whisper.py` —— 本机用的 drop-in 替身 `RemoteWhisperModel`。

设计文档：`analysis/video_audio/20260603_远程GPU_ASR服务方案.md`

## 端点

| 方法 | 路径 | 入参（multipart） | 返回 |
|---|---|---|---|
| POST | `/transcribe` | `audio`(.npy 二进制) + `model` + `params`(JSON) | `{segments:[{text,start,end,...}], info:{...}}` |
| POST | `/detect_language` | 同上 | `{language, language_probability, all_language_probs}` |
| GET | `/health` | — | `{status, device, compute_type, loaded}` |

- `audio`：client 用 `np.save` 序列化的 float32 一维音频（sample_rate=16000）。
- `params`：JSON，原样透传给 `WhisperModel.transcribe` / `detect_language`
  （`language`、`beam_size`、`vad_filter`、`initial_prompt`、`condition_on_previous_text` …）。
- 模型按名缓存常驻显存，首次用时懒加载，tiny / medium 可同时驻留。

## 依赖

仓库内（client 侧）：`faster-whisper`、`av`、`numpy` 已在本项目 `pyproject.toml`，
client 仅额外需要 `requests`。

GPU host 独立部署时用本目录的 `requirements.txt`（含 `fastapi`/`uvicorn`/`python-multipart`）：

```bash
pip install -r requirements.txt
```

GPU 机器前置：NVIDIA 驱动 + cuDNN（CTranslate2 float16 推理依赖）。

## 启动

包已 editable 安装，可在任意目录用模块路径启动（推荐）。
**设备自动探测**：有 CUDA 用 `cuda`、否则回退 `cpu`；**量化默认 `int8`**
（CPU/GPU 一致——最终在 CPU 部署，GPU 上也用 int8 保证结果一致）。
**模型默认从 `~/.cache/huggingface/hub` 加载**。故下面这一行在 GPU host 与本机都能直接跑：

```bash
uvicorn data_agent_baseline.asr.server:app --host 0.0.0.0 --port 8900
```

- GPU host → 自动 `cuda` / `int8`；本机无 GPU → 自动 `cpu` / `int8`。
- 模型查 `~/.cache/huggingface/hub`（HF 默认缓存），无需额外配置。
- 用 `GET /health` 确认实际生效的 `device` / `compute_type` / `model_dir`。

如需强制覆盖，设环境变量（如 GPU 上想用 float16）：

```bash
ASR_COMPUTE_TYPE=float16 \
  uvicorn data_agent_baseline.asr.server:app --host 0.0.0.0 --port 8900
```

环境变量（均可不设）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `ASR_DEVICE` | 自动探测 | 设备；有 CUDA→`cuda`，否则→`cpu` |
| `ASR_COMPUTE_TYPE` | `int8` | 量化；CPU/GPU 一致。GPU 想更快可设 `float16` |
| `ASR_TOKEN` | （空） | 设置后所有请求需带 `?token=<它>` 才放行 |
| `ASR_MODEL_DIR` | `~/.cache/huggingface/hub` | HF 模型缓存根目录（`download_root`）；见下「离线模型」 |

### 离线模型（GPU host 无外网时上传本地模型）

模型走标准 HF 缓存布局：`<ASR_MODEL_DIR>/models--Systran--faster-whisper-<m>/...`。
默认 `ASR_MODEL_DIR=~/.cache/huggingface/hub`，所以把本机已下载的模型整目录 rsync
到远端同一缓存即可，server 收到 `model="tiny"` 会自动在那里找到：

```bash
for m in tiny medium; do
  rsync -av ~/.cache/huggingface/hub/models--Systran--faster-whisper-$m \
    <user>@<gpu-host>:~/.cache/huggingface/hub/
done
```

远端无外网时，加 `HF_HUB_OFFLINE=1` 启动，避免去 HF 校验/下载：

```bash
HF_HUB_OFFLINE=1 \
  uvicorn data_agent_baseline.asr.server:app --host 0.0.0.0 --port 8900
```

> 若想把模型放到非默认位置，设 `ASR_MODEL_DIR=/your/cache`（同样是 HF 缓存布局，
> 即里面放 `models--Systran--faster-whisper-*`），server 以它作 `download_root`。

### 常驻（GPU host 落定后三选一）

裸 uvicorn（nohup）：

```bash
nohup uvicorn data_agent_baseline.asr.server:app --host 0.0.0.0 --port 8900 \
  > asr_server.log 2>&1 &
```

systemd（`/etc/systemd/system/asr-whisper.service`）：

```ini
[Unit]
Description=faster-whisper remote ASR
After=network.target

[Service]
WorkingDirectory=/opt/data-agent
Environment=ASR_DEVICE=cuda
Environment=ASR_COMPUTE_TYPE=float16
ExecStart=/opt/data-agent/.venv/bin/uvicorn data_agent_baseline.asr.server:app --host 0.0.0.0 --port 8900
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now asr-whisper
```

nvidia-docker（容器）：

```bash
docker run --gpus all -p 8900:8900 -v $PWD:/app -w /app \
  python:3.11 bash -c "pip install -r src/data_agent_baseline/asr/requirements.txt && \
    uvicorn data_agent_baseline.asr.server:app --host 0.0.0.0 --port 8900"
```

## 连通与安全

默认监听内网，不裸暴露公网。二选一：

- **SSH 隧道**（推荐）：`ssh -L 8900:localhost:8900 <gpu-host>`，client 连 `http://localhost:8900`。
- **token 鉴权**：服务端设 `ASR_TOKEN=xxx`，client 构造时传 `token="xxx"`（或设环境变量 `ASR_TOKEN`）。

## client 用法

```python
from data_agent_baseline.asr import RemoteWhisperModel as WhisperModel
model = WhisperModel("medium", endpoint="http://localhost:8900")

# 下游与原生 faster_whisper 完全一致
lang, prob, all_probs = model.detect_language(audio=audio)          # audio: float32 numpy (16k)
segs, info = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True)
text = "".join(s.text for s in segs)
```
