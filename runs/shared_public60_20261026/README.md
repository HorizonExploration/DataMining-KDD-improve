# 本轮实验团队共享归档

来源：`runs/baseline_full_20261025`；60题课程评分78.9027/100，官方通过40/60。原始33份plus输出与27份max补跑组成同一轮恢复实验。

| 路径 | 内容 |
|---|---|
| `predictions/task_*/prediction.csv` | 60份实际预测，字节与原实验完全一致 |
| `predictions/task_*/completion.json` | 成功凭据、预测哈希、复用及模型试跑导入来源 |
| `evaluation.json` | 完整60题的两种评分结果 |
| `analysis_*` | 汇总和逐题耗时、得分 |
| `protocol.json`、`run_manifest.json` | 实验配置、输入/源码指纹与模型切换记录 |
| `temp/` | 保留下来的求解代码和计时，不含输入数据副本 |
| `sessions/` | 历史尝试的求解代码、计时、原始凭据及部分配置记录 |
| `model_trials/` | max单题验证的求解代码与计时 |
| `archive_manifest.json` | 归档文件SHA-256及范围说明 |

本归档由 `python scripts/export_team_run.py` 从已完成实验导出；该脚本不会覆盖已有目标目录。原始CSV未改动，JSON中的本机项目根路径替换为项目相对路径。没有复制数据集、gold、数据库、视频、PDF、抽帧、缓存、权重、完整模型对话或原运行日志。PPT在 `intro-plan/`。

**日志补充**：本机目前保留的完整文本日志及模型对话已另行打包至 [logs/](logs/README.md)，该压缩包由 `export_team_logs.py` 独立生成，使用自己的 `logs/manifest.json` 校验；不在最初的 `archive_manifest.json` 文件清单内。对话可能含工具返回的数据片段及编码图片。上段描述的是最初预测/计时导出的范围。

**这是复盘归档，不能直接作为断点续跑目录。** 历史路径是来源标识，归档并不包含它们所指的全部文件；JSON路径处理也不能保证原运行身份校验可在异机重建。新方案应独立运行并生成新目录。

求解代码有历史失败尝试，不能一律认定为最终预测的生成代码。例如task_102最终预测来自 `model_trials/ecnu_max_20261026_031112_40106/temp/task_102/workdir/solver.py`；根目录 `temp/task_102/` 是旧尝试。请结合完成凭据、模型阶段和历史记录判断来源。CSV是最终输出的依据，仓库未保证每道历史题都保留完整生成过程。

只分析结果无需下载数据或调用API。要独立重算评分，先按SETUP下载数据及gold、安装依赖，再将本归档复制到忽略的工作目录后评分，避免改动已提交的归档：

```bash
source scripts/env.sh
cp -a runs/shared_public60_20261026 runs/local_rescore_20261026
python scripts/evaluate_run.py runs/local_rescore_20261026
```

评分只读取本地预测与gold，不调用LLM；这复核已保存输出的成绩，不是重新生成60题。完整实验分析见 [评价文档](../../docs/本次全量实验评价与耗时分析.md)。
