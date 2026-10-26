# ECNU 多模态故障与混合模型补跑

> 公开仓库说明：本页记述实际修复过程，提到的 `runs/`、`sessions/` 和 `docs/*.json` 是本机证据位置，不随仓库发布。脱敏诊断与前后评分见 [公开结果目录](../results/public60_20261026/README.md)。

课件第2、3页要求视频结合语音与画面；第8页课程基线75.6387使用ecnu-max 32题、ecnu-plus 28题，但没有给出逐题分配。此处的路由依据实际输入是否有视频，未按标准答案选择模型。

## 故障定位

纯文本ecnu-plus返回200；同一真实视频截图PNG/JPEG及默认、新旧思考参数均500。最新串行对照中，同URL、同令牌、同PNG，ecnu-plus返回500，ecnu-max返回200，且识别出的标题及178家公司与图片一致。ecnu-max的图片+工具调用及工具结果回传也返回200。官方文档称两者均支持图像理解，因此现有证据只能定位到当前ecnu-plus图片请求服务/兼容链路，不能据500断言模型本体没有多模态能力。

证据：ecnu-models-multimodal-probe.json、ecnu-max-image-tool-probe.json。
官方模型说明：https://developer.ecnu.edu.cn/vitepress/llm/model.html
多模态API：https://developer.ecnu.edu.cn/vitepress/llm/api/completions.html

## 试跑与答案正确性

ecnu-max独立运行task_102，约177秒，38个API响应全部200，成功生成prediction.csv。随后独立评分为0分；这表明API兼容性和工具链修复，不代表推理正确。日志中的solver把跨类别重叠的公司数相加得到37，尽管视频文字已说明跨类别重叠、原始获奖记录为36条。这是去重/计数推理错误，不能把有效CSV称为正确答案。未手工填写正确答案，也未把gold提供给模型。

## 当前.env与运行设置

```dotenv
MODEL_NAME=ecnu-plus
MODEL_VIDEO_NAME=ecnu-max
MAX_PARALLEL=4
MODEL_MAX_CONCURRENCY=2
MODEL_THINKING_FORMAT=ecnu
MODEL_SOLVER_THINKING=disabled
```

有context/video/briefing.mp4的任务统一走ecnu-max，文本任务走ecnu-plus；上下文变量将模型选择传入视频后台线程和截图识别线程。实际请求模型从请求体读取到脱敏指标，避免把max错误标成全局plus。ecnu-max官方并发上限3，混用进程保守限制全部请求合计2；本地视频预处理仍1路。

## 补跑过程与最终结果

原33份结果与42.0971666667分/官方35%的原评测保存于sessions/20261026_031530_41205。已核验源码、输入、端点、ASR与有效推理参数，将独立max试跑输出导入当前实验；这一步形成34份有效CSV，随后补跑剩余26题。最终已在2026-10-26 04:27完成，60题均有预测，课程得分78.9027，官方整题通过40/60。导入的task_102依旧记为0分，未掩盖试跑错误。

新配置属于保留历史plus结果的混合模型修复实验，不是纯max的60题独立重跑。原始完成凭据和manifest已归档，当前凭据记录模型配置变化与试跑来源。最终全部60题均进入独立评分，旧42.0972分保留为不完整实验的历史结果，不与新成绩混淆。

运行目录：runs/baseline_full_20261025。只读进度：python scripts/monitor_run.py runs/baseline_full_20261025。未来原地续跑：source scripts/env.sh后运行python scripts/resume_current_run.py runs/baseline_full_20261025。

离线验证：3项模型路由/线程上下文检查、9项HTTP检查、10项缓存与续跑检查通过；额外核验迁移拒绝未登记的输入、权重和端点变更。
