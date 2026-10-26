# 上游来源与改进说明

本项目基于公开参考实现进行课程实验。`champion/` 和 `DataSpace/` 以普通文件纳入仓库，便于克隆后获得实际源码；不是需要额外递归拉取的 Git submodule。

| 目录 / 资源 | 上游 | 固定版本 |
|---|---|---|
| `champion/` | https://github.com/zhezh/kddcup2026_champion | `bdc874fc4260e3565ae0dce041728fdf5b376709` |
| `DataSpace/` | https://github.com/HKUSTDial/DataSpace | `6491caa4c70cc06cacb6103ba73cefb00746abfe` |
| 数据下载 | https://huggingface.co/datasets/HKUSTDial/DataSpace | `96be19257d7aa251dbce4771d46a26fc36f6dc52` |

DataSpace 的 MIT License 与引用信息保留在 `DataSpace/LICENSE`、`DataSpace/CITATION.cff`。所用 champion 快照没有独立 LICENSE 文件；其原有代码归原作者，本项目不将其重新声明为 MIT。数据集和模型权重不随本仓库分发。

本项目新增的独立模块包括 `artifact_cache.py`、`model_http.py`、`run_state.py`、`speed_runtime.py`，以及顶层部署、检查、续跑、路由测试和分析脚本。对上游主循环、模型配置、文档准备和视频处理的修改在源码中保留；架构与实验贡献见 `docs/ARCHITECTURE.md`。

根目录 LICENSE 的授权范围是本项目新增的独立文件与文档，不覆盖第三方原有代码或数据。公开报告中的78.90分是本课程公开60题实验结果，不是上游原仓库或国际比赛全题集的成绩。
