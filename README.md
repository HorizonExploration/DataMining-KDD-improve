# DataMining-KDD-improve

基于 KDD Cup 2026 Data Agents 的数据挖掘课程项目：让 LLM 自动读取表格、数据库、文档和视频，调用工具完成分析，并输出每题的 `prediction.csv`。

本仓库在冠军参考方案上实现了 **ECNU 混合模型路由、同步/异步统一限流、文档缓存、可校验断点续跑和独立评测**。这是参考方案的课程改进项目，不是上游官方仓库。来源与版本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 实验结果

2026-10-26 完成固定公开 **60 题**，课程评分使用 `λ=0.1`，缺失预测计零。

| 指标 | 结果 |
|---|---:|
| 预测文件覆盖 | 60/60 |
| 课程得分 | **78.9027 / 100** |
| 课程参考基线 | 75.6387 / 100 |
| 相对参考基线提高 | **3.2640 分** |
| 课程满分 / 部分分 / 零分 | 36 / 15 / 9 题 |
| DataSpace 官方整题通过 | **40/60，66.67%** |

结果来自 **33 份历史 `ecnu-plus` 输出 + 27 份 `ecnu-max` 补跑输出**。原先图片请求持续返回 HTTP 500，切换视频题模型后恢复了结果覆盖；生成 CSV 不代表答案正确。

最后 26 题补跑耗时 **71 分 29 秒**，不是从零跑完 60 题的时间。首次启动到最终评分历时约 **10 小时 51 分钟**，其中包含调试、失败重试、配置切换和 WSL 中断。单次公开题结果不能证明模型泛化能力或并发加速倍数。

公开结果：[评测 JSON](results/public60_20261026/evaluation.json) · [逐题成绩与耗时](results/public60_20261026/task_results.csv) · [分析摘要](results/public60_20261026/summary.json) · [详细评价](docs/本次全量实验评价与耗时分析.md)。

## 目录

```text
champion/       改进后的 Agent、工具、多模态处理和锁定依赖
DataSpace/      上游 DataSpace 参考代码与官方整表评测器
scripts/        数据准备、运行、续跑、监控、评测和离线检查
config/         上游源码版本与来源
docs/           安装、架构、运行和实验分析
results/        可公开的脱敏评分、逐题统计和实验配置
.env.example    ECNU 混用配置模板，不含令牌
```

运行后会在本机生成 `data/`、`cache/`、`runs/`、`tools/`、`champion/.venv/` 和 `champion/asr_models/`，这些不提交到 Git。原始课程 PPT、数据、答案、模型权重、预测内容、完整对话和本机配置也不随仓库发布。

## 安装与准备

推荐 Linux / WSL2、Python 3.12、4 核 CPU、16 GiB 内存和至少 40–60 GiB 可用磁盘。LLM 通过 API 调用，本地使用 CPU + int8 进行语音转写，不要求 CUDA。

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后：

```bash
git clone https://github.com/HorizonExploration/DataMining-KDD-improve.git
cd DataMining-KDD-improve
cd champion
uv sync --frozen --python 3.12
cd ..
source scripts/env.sh
python scripts/prepare_assets.py data
python scripts/prepare_assets.py asr
python scripts/prepare_assets.py ffmpeg
python scripts/check_environment.py
```

下载脚本固定数据和 ASR 版本，并校验数据归档哈希。首次下载较大；已有完整资源时会复用。详细说明见 [安装与数据准备](docs/SETUP.md)。

## 配置模型

```bash
cp -n .env.example .env
chmod 600 .env
# 在本机编辑 .env，填写 MODEL_API_KEY
```

默认模板采用以下策略：

```dotenv
MODEL_NAME=ecnu-plus
MODEL_VIDEO_NAME=ecnu-max
MAX_PARALLEL=4
MODEL_MAX_CONCURRENCY=2
MODEL_THINKING_FORMAT=ecnu
MODEL_SOLVER_THINKING=disabled
KDD_DOC_CACHE_MODE=readwrite
```

包含 `context/video/briefing.mp4` 的任务使用 `ecnu-max`，无视频任务使用 `ecnu-plus`。同步截图请求和异步 Agent 请求共用进程内预算，遇到 429 自动降速。视频预处理在后台单路执行，降低 CPU 与内存争抢。将 `MODEL_VIDEO_NAME` 留空可使用单一模型。

ECNU 接口的模型可用性、额度和费用以账户及[官方文档](https://developer.ecnu.edu.cn/vitepress/llm/model.html)为准。本次实测中 plus 文本成功但图片返回 500，max 图片和图像工具闭环通过；不能将这一服务现象理解为模型本体永久没有视觉能力。

## 运行与实时进度

```bash
source scripts/env.sh
# 仅预览，不调用模型
bash scripts/run_tasks.sh --dry-run --all
# 单题端到端验证；下面的命令会调用模型 API
bash scripts/run_tasks.sh task_44
# 全量公开 60 题，实时输出并保存 console.log
PYTHONUNBUFFERED=1 KDD_RUN_TIMEOUT=24h bash scripts/run_tasks.sh --all
```

启动时会打印实际实验目录。另开终端，先进入项目并执行 `source scripts/env.sh`，再查看：

```bash
tail -n 80 -F runs/<实验目录>/console.log
watch -n 5 'python scripts/monitor_run.py runs/<实验目录> | python -m json.tool'
```

实验中断后，在同一目录继续，已完成结果必须通过指纹、完成凭据和 CSV 哈希检查：

```bash
python -u scripts/resume_current_run.py runs/<实验目录>
```

原地续跑会保存历史尝试、拒绝重复启动，并在结束后自动评分。新建实验的 `run_tasks.sh` 在结束后执行输出检查，评分需要单独执行：

```bash
python scripts/evaluate_run.py runs/<实验目录>
```

`--tasks task_44` 仅用于单题评分，不能作为完整 60 题结果。评分脚本要求实验已经退出；gold 不作为 Agent 输入。目录分离不是操作系统级沙箱，使用任意代码工具时仍需检查实际访问边界。

缓存、配置变更和严格续跑见 [加速运行与断点续跑](docs/加速运行与断点续跑.md)。不要直接覆盖已有实验的模型配置或修改预测来提高成绩。

## 离线验证

安装依赖后，不需要令牌或数据集即可运行：

```bash
bash scripts/test_offline.sh
```

检查 HTTP 并发与退避、线程与协程共享预算、模型路由、缓存完整性和结果复用，不调用付费模型。需要数据、ASR 和 FFmpeg 的本地能力检查使用 `python scripts/check_environment.py`。

## 实现与后续方向

- 通用多模态路由与逐次请求模型记录，避免将 max 请求误记为 plus。
- 同步/异步统一限流、429 冷却与渐进恢复，降低无效重试。
- 只缓存完整成功的文档抽取，使用输入、模型、参数与代码指纹隔离结果。
- 可审计的完成凭据、原地续跑和配置迁移，保留旧结果与评分。
- 同时报告课程列匹配分与官方整表准确率，区分输出有效性与业务正确性。

后续优先解决输出列数、日期格式与去重统计错误，再优化 ASR 复用和视频流水线。架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，本轮故障定位见 [ECNU 混合模型修复说明](docs/ECNU混合模型修复说明.md)。

## 参考与许可

- [zhezh/kddcup2026_champion](https://github.com/zhezh/kddcup2026_champion)：Agent 参考实现。
- [HKUSTDial/DataSpace](https://github.com/HKUSTDial/DataSpace)：任务与官方评测参考。
- [HKUSTDial/DataSpace 数据集](https://huggingface.co/datasets/HKUSTDial/DataSpace)：数据下载来源。
- [ECNU 模型平台](https://developer.ecnu.edu.cn/vitepress/llm/)：模型服务。

上游固定版本见 [config/upstream_versions.json](config/upstream_versions.json)。新增独立部分的许可范围与第三方权益说明见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
