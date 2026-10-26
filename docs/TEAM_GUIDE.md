# 团队成员使用指南

仓库：https://github.com/HorizonExploration/DataMining-KDD-improve

本项目是在公开参考方案上做的课程改进。先阅读根目录README了解结果和限制，再按自己的工作范围准备环境。不能把生成了CSV等同于回答正确。

## 已共享的资料

| 资料 | 位置 | 用途 |
|---|---|---|
| 项目介绍与基本命令 | [README](../README.md) | 快速了解目标、目录、模型和运行方式 |
| 实际Agent和工具源码 | `champion/src/data_agent_baseline/` | 模型路由、数据处理、视频理解、求解循环 |
| 官方评测及参考实现 | `DataSpace/` | 理解评测规则，保留上游许可和引用 |
| 锁定依赖与配置模板 | `champion/uv.lock`、`.env.example` | 重建Python环境、自行配置令牌 |
| 安装与数据准备 | [SETUP](SETUP.md) | 下载固定版本数据、ASR和FFmpeg |
| 系统结构 | [ARCHITECTURE](ARCHITECTURE.md) | 了解模块职责、改进范围和实验边界 |
| 运行与续跑 | [运行说明](加速运行与断点续跑.md) | 新实验、实时进度、恢复和缓存 |
| 原始与修复后的评测 | [结果索引](../results/public60_20261026/README.md) | 查看60题成绩、前后差异与配置指纹 |
| 每题成绩和耗时 | [task_results.csv](../results/public60_20261026/task_results.csv) | 筛选零分题、部分分题和慢题 |
| 总体评价 | [实验分析](本次全量实验评价与耗时分析.md) | 报告写作、错误分析和后续优化 |
| API故障定位 | [混合模型修复](ECNU混合模型修复说明.md) | 理解plus图片500与max验证过程 |
| 来源和贡献边界 | [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) | 区分上游实现与本项目改进 |

## 哪些资料需要另行准备

- **API令牌**：各成员在本机配置获准使用的令牌；仓库只有空白模板，不包含可用密钥。
- **数据、gold和ASR权重**：由准备脚本按固定版本下载。不要将gold复制进Agent的输入目录。
- **课程原始PPT与教师要求**：仓库没有上传原课件，由项目负责人通过团队内部渠道另行提供。
- **原始预测、完整日志和模型对话**：未上传到GitHub，仍保留在负责人本机的 `runs/baseline_full_20261025/`。错误复盘需要时，只内部提供相关题目的输出与对应来源记录。

只做报告和成绩分析的成员，可以直接使用公开JSON、CSV和文档，不需要下载数十GiB数据或申请API。做代码开发的成员可先运行离线检查；做真实实验的成员再准备完整资源。

## 开发成员的最短开始流程

先安装uv和Python 3.12，然后：

```bash
git clone https://github.com/HorizonExploration/DataMining-KDD-improve.git
cd DataMining-KDD-improve
cd champion
uv sync --frozen --python 3.12
cd ..
source scripts/env.sh
bash scripts/test_offline.sh
```

离线检查不需要 `.env`、真实令牌或数据集。发布时已在无本机数据的导出副本中通过28项检查。

要执行真实实验，再按SETUP下载资源并填写本机 `.env`：

```bash
cp -n .env.example .env
chmod 600 .env
# 在本机编辑器中填写MODEL_API_KEY
python scripts/prepare_assets.py data
python scripts/prepare_assets.py asr
python scripts/prepare_assets.py ffmpeg
python scripts/check_environment.py
bash scripts/run_tasks.sh --dry-run --all
```

确认后先做单题验证，再全量运行。后两条会调用模型：

```bash
bash scripts/run_tasks.sh task_44
PYTHONUNBUFFERED=1 KDD_RUN_TIMEOUT=24h bash scripts/run_tasks.sh --all
```

新方案应创建新实验目录；原地续跑只用于恢复原实验。不要用复用了全部旧结果的目录，声称完成了新方案的独立全量实验。

## 团队协作建议

| 工作方向 | 主要位置 | 推荐交付 |
|---|---|---|
| 数据与输出规范 | `tools_v2/`、`agents_v2/solver_agent.py` | 列数、日期、单位、排序和去重规则的通用检查 |
| 视频证据与推理 | `video/`、`agents_v2/` | 画面与转写对齐、证据提取、计数一致性验证 |
| 模型及性能工程 | `general_tools.py`、`model_http.py`、`speed_runtime.py` | 兼容性验证、缓存和内存/耗时对照 |
| 评测与报告 | `scripts/evaluate_run.py`、`results/`、`docs/` | 逐题分析、对照实验表和报告案例 |

以上代码位置除顶层目录外均相对 `champion/src/data_agent_baseline/`。分工按模块和通用问题划分，不按题号硬编码答案。

各成员从main新建分支，例如 `git switch -c feat/output-validation`；提交前运行离线检查，并说明是否改变提示词、模型、缓存或评分规则。令牌、数据、权重和完整运行目录不要加入提交。

并发上限只约束单个Python进程。若多人共用同一服务令牌，多进程的调用会在服务端合并统计，应协调总预算；不要同时写同一实验目录。当前推荐配置是任务并发4、请求总上限2、视频预处理单路。

## 讨论成绩时应保持的口径

- 课程评分78.9027/100，超过75.6387参考基线3.2640分；官方整题通过40/60。两者不是同一指标。
- 课程分布为36题满分、15题部分分、9题零分。原始33份plus输出加27份max补跑，属于混合模型恢复实验。
- 最后26题补跑用时71分29秒；完整调试日历跨度约10小时51分钟。不能将71分钟当成从零60题的运行时间。
- 9道零分题和列数/日期问题是当前重点。API成功与答案正确必须分开讨论，不能按gold手改CSV后当成Agent结果。
- 每份 `completion.json` 记录生成来源。历史复用或试跑导入的输出，其真实solver和日志可能在 `sessions/` 或 `model_trials/`，不要把根目录遗留的临时solver误认为该结果的生成代码。

## 负责人可以发给团队的说明

> 代码、环境模板、运行文档、上游来源和本轮60题评分已经放在GitHub。请先看README和TEAM_GUIDE；做代码开发可先运行离线检查，做实验再按SETUP下载数据与权重，并自行填写API令牌。原始课件和需要复盘的本机预测/日志由负责人内部补充，不在公开仓库共享。当前成绩78.90分，后续优先优化输出规范、去重逻辑和视频流水线，并保留独立实验记录。
