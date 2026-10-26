# 公开60题实验结果（2026-10-26）

课程评分78.9027/100，λ=0.1；官方整题准确率40/60。结果来自33份历史plus输出与27份max补跑输出。26题最终补跑用时71分29秒，不是从零60题的耗时。

| 文件 | 内容 |
|---|---|
| [evaluation.json](evaluation.json) | 最终两种评分器的60题记录 |
| [evaluation_before_routing.json](evaluation_before_routing.json) | 路由修复前的60题评测；缺失输出计零 |
| [task_results.csv](task_results.csv) | 每题得分、模态、模型阶段、行列数、最后可取得的单次耗时 |
| [summary.json](summary.json) | 分组成绩、时间、请求状态、瓶颈与局限 |
| [inference_manifest.json](inference_manifest.json) | 运行时源码、输入与ASR哈希及非敏感模型设置 |
| [api_diagnostics.json](api_diagnostics.json) | 同一截图在ECNU两模型上的最小对照诊断 |

绝对本机路径已改为项目相对路径。这里不包含令牌、原始数据、gold内容、实际预测表格、图片或完整模型对话。CSV中的历史日志位置是证据来源标识，不代表这些大体积本机日志也随仓库发布。

原33份预测哈希保持不变。完整数据由准备脚本下载，预测需要重新运行Agent生成；结果文件不能替代独立复现实验。详细分析见[实验评价](../../docs/本次全量实验评价与耗时分析.md)。
