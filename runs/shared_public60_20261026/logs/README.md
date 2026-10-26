# 全量实验日志归档

日志包约296MiB，存放在GitHub Release附件中，不随每次克隆代码下载：

**[下载 experiment-logs.tar.gz](https://github.com/HorizonExploration/DataMining-KDD-improve/releases/download/logs-public60-20261026/experiment-logs.tar.gz)** · [Release说明](https://github.com/HorizonExploration/DataMining-KDD-improve/releases/tag/logs-public60-20261026)

本目录的 `manifest.json` 列出归档中每个文件的路径、大小与SHA-256。GitHub网页不直接展开压缩包内的对话。可以从上方链接手动下载，也可以使用以下命令：

```bash
mkdir -p runs/local_log_review
gh release download logs-public60-20261026 --repo HorizonExploration/DataMining-KDD-improve --pattern experiment-logs.tar.gz --dir runs/shared_public60_20261026/logs
tar -xzf runs/shared_public60_20261026/logs/experiment-logs.tar.gz -C runs/local_log_review
```

| 解压后的路径 | 内容 |
|---|---|
| `console.log`、`session-launcher.log`、`supervisor.log` | 保留的主运行输出、启动器和调度记录 |
| `temp/logs/task_*/` | 当前题目日志、成功/失败求解对话、工具反馈、视频处理与计时 |
| `temp/logs/model_requests.jsonl` | 当前保留的请求状态、耗时、用量记录 |
| `sessions/*/task_logs/` | 历次续跑前归档的题目日志与失败尝试 |
| `sessions/*/` 下其他文本 | 历史配置、完成凭据、求解代码和会话状态 |
| `model_trials/*/` | ecnu-max单题验证的控制台、模型对话、请求记录和评测日志 |
| `historical-provenance/` | 早期已清理目录留下的请求指标、计时与来源记录 |

这是**本机目前保留日志的完整文本归档**。早期已清理实验只剩来源记录，不能恢复当时已删除的完整对话。本轮60题含历史复用和试跑导入，不能将所有日志时长相加当成一次连续运行耗时。

task_102最终CSV来自max试跑，优先看 `model_trials/ecnu_max_20261026_031112_40106/temp/logs/task_102/`；`temp/logs/task_102/` 和历史会话还包含旧plus失败记录。其他题目也应结合完成凭据与会话时间判断生成来源。

导出脚本为 `scripts/export_team_logs.py`。导出时检查本机配置中的凭据值和常见令牌格式，并移除命中的凭据；本机项目根路径改为项目相对路径。原始本地日志未改动。模型对话含实际任务内容、工具输出、数据片段及可能的编码图片，因此此日志包比纯评分资料范围更大，但没有额外打包完整数据集、权重或输入文件目录。

该日志归档用于复盘，不是断点续跑目录；文件索引与校验清单见本目录 `manifest.json`。最终预测及其哈希仍以父目录 `predictions/` 和完成凭据为准。
