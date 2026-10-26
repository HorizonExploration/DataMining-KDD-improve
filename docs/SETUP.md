# 安装、数据与运行准备

仓库根目录以 `scripts/env.sh` 的位置动态确定，不依赖开发者本机的绝对路径。安装环境与下载资源均在本机生成，不包含在 Git 仓库中。

## 依赖

- Linux / WSL2，Python 3.12，uv。
- 推荐 4 核 CPU、16 GiB 内存、8 GiB swap，预留 40–60 GiB 磁盘。
- LLM 使用远程 API；本地 faster-whisper 使用 CPU + int8，不要求显卡。

```bash
cd champion
uv sync --frozen --python 3.12
cd ..
source scripts/env.sh
uv pip check --python champion/.venv/bin/python
```

`champion/uv.lock` 是依赖版本依据。不要先重新克隆上游覆盖 `champion/` 或 `DataSpace/`，当前源码已经包含本项目改进。

## 数据与工具

```bash
python scripts/prepare_assets.py data
python scripts/prepare_assets.py asr
python scripts/prepare_assets.py ffmpeg
python scripts/check_environment.py
```

数据来自 HKUSTDial/DataSpace，固定版本为 `96be19257d7aa251dbce4771d46a26fc36f6dc52`。脚本校验归档 SHA-256，从完整 410 题输入中构建固定公开 60 题的 `data/input60/`，只复制题目及 context，不复制 gold。

ASR 使用固定版本的 tiny、medium 权重，存入 `champion/asr_models/`。FFmpeg/FFprobe 由 static-ffmpeg 提供并链接到 `tools/bin/`。下载较大，脚本支持保留部分下载与完整资产复用。

`docs/data-integrity.json` 等本机完整性记录由脚本生成。它们不提交到 Git，但不要在整理本地环境时随意删除，否则准备脚本可能重新检查或下载资产。

## API 配置

```bash
cp -n .env.example .env
chmod 600 .env
```

在本机编辑 `.env` 填入 `MODEL_API_KEY`，不要将令牌写入 README、日志、命令历史或提交。默认模板为文本 plus、视频 max，四题并行、请求合计上限二。它们是不同层级的并发参数。

先执行非计费预览：

```bash
bash scripts/run_tasks.sh --dry-run --all
```

再执行会调用模型的验证：

```bash
python scripts/check_model.py
bash scripts/run_tasks.sh task_44
python scripts/run_model_trial.py --model ecnu-max --tasks task_102
```

文本工具验证不等于视频端到端验证。本轮故障正是文本成功而 plus 图片请求失败；确认视频帧、工具调用与最终 CSV 都可用后再运行全量。

## 全量、日志与评分

```bash
PYTHONUNBUFFERED=1 KDD_RUN_TIMEOUT=24h bash scripts/run_tasks.sh --all
```

保存启动时打印的实验目录。`console.log` 实时追加进度，`predictions/` 保存归档输出，`temp/logs/` 保存工具反馈、耗时及脱敏用量。普通新建实验结束后手动执行：

```bash
python scripts/evaluate_run.py runs/<实验目录>
```

不要让 Agent 读取 gold 或把答案路径加入输入。评分脚本在进程退出记录存在后才读取 gold。具体续跑方式见[运行说明](加速运行与断点续跑.md)。

## 无数据、无令牌的离线检查

```bash
bash scripts/test_offline.sh
```

这组检查使用临时数据与模拟 HTTP，不下载基准数据或调用模型。`check_environment.py` 则实际验证本地表格、SQL、PDF、FFmpeg/PyAV 和 Agent 导入能力。
