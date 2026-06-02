# Report Evidence and Claims Boundary Guide

中文标题：报告证据与结论边界指南

## 1. 本文目的

本文帮助你在报告、答辩、GitHub README、简历、作品集或面试中正确描述 VisionGuard，避免把项目证据夸大成没有做过的验证。

核心目标：把“有什么证据”与“能主张什么结论”对应起来。

## 2. 主结论边界

推荐主表述：

> VisionGuard is a modular driver drowsiness monitoring system that combines specialist visual evidence and rule-based temporal fusion to produce warning-candidate states.

中文可写为：

> VisionGuard 是一个模块化驾驶员疲劳相关视觉证据监测系统，它结合眼部/嘴部 specialist evidence、信号质量检查和规则式时间融合，生成 warning-candidate 状态。

不要写成：

- fully validated drowsiness diagnosis system；
- certified driver safety system；
- end-to-end drowsy/not-drowsy classifier；
- system detects drowsiness with final accuracy。

除非未来有带人工真值标注的视频级 full-system evaluation，否则不能主张 final full-system drowsiness accuracy。

## 3. 项目证据类型

| Evidence type | 支持什么 | 不支持什么 | 位置 |
|---|---|---|---|
| dataset/preprocessing evidence | 数据来源、label mapping、manifest、split、crop 可追踪 | 不证明 runtime accuracy | `artifacts/mappings/`, preprocessing docs |
| specialist model metrics | eye open/closed 或 no-yawn/yawn classifier performance | 不等于 full-system drowsiness accuracy | `outputs/mrl_eye/results/`, `report_assets/mouth_yawn_evaluation_refresh/` |
| confusion matrices | class-level correct/incorrect patterns | 不说明 temporal fusion 是否正确 | `outputs/mrl_eye/figures/`, `report_assets/.../figures/` |
| ROC/AUC / PR curves | probability ranking 能力 | 不代表每个 runtime alert 都正确 | `report_assets/mouth_yawn_evaluation_refresh/figures/`, MRL figures if available |
| training curves | training/validation behavior | 不证明真实驾驶泛化 | `outputs/mrl_eye/figures/` |
| runtime probability traces | `p_eye_closed`, `p_yawn` 随时间变化 | 不是真值 drowsiness label | `outputs/system_video_upload_runs/` |
| upload warning intervals | rule-based warning-candidate timeline | 不等于人工标注疲劳段 | upload summaries/timelines |
| keyframes | interval 的可视化证据 | 不证明模型一定正确 | upload keyframe artifacts |
| evidence figures | runtime evidence visualization | 不是 accuracy/ROC/PR 图 | upload figure artifacts |
| History/Insights charts | Live Monitor product summaries | 不是 model evaluation report | `SystemUI/src/components/history-48h/`, `SystemUI/src/components/insights/` |
| archive records | compact runtime summaries | 不保存 raw frame/video，也不是 ground truth | `src/backend/local_archive.py` |
| frontend screenshots | product UI demonstration | 不证明算法性能 | `report_assets/all_figures/` |
| deployment screenshots | remote demo reachability | 不证明 production readiness | deployment docs |

## 4. Specialist Metrics vs Full-System Accuracy

眼部模型指标评估的是 MRL Eye open/closed classification。它说明模型在 eye ROI 图像上区分 closed/open 的能力。

嘴部模型指标评估的是 YawDD/YAWDD+ mouth crop no-yawn/yawn classification。它说明模型在 mouth ROI 图像上区分 no-yawn/yawn 的能力。

两者都不是 full drowsiness accuracy，因为完整系统还包括：

- face/landmark 检测；
- ROI crop 质量；
- runtime webcam domain shift；
- signal quality；
- temporal fusion；
- debounce/cooldown；
- UI alert policy；
- History/Insights product summaries。

full-system accuracy 需要：

1. 明确的视频级或时间段级 ground-truth drowsiness labels；
2. 明确 evaluation protocol；
3. 与 runtime pipeline 一致的 inference setting；
4. 对 false positive、false negative、latency、signal failure 的定义；
5. 独立 test set。

Source: `docs/tech_learning/MODEL_EVALUATION_AND_SELECTION_GUIDE_ZH.md`, `docs/tech_learning/RUNTIME_INFERENCE_AND_TEMPORAL_FUSION_GUIDE_ZH.md`

## 5. Runtime Evidence 和 Warning-Candidate Intervals

正确解释：

- `p_eye_closed` 是 eye-closure evidence；
- `p_yawn` 是 yawn evidence；
- signal quality 影响证据可靠性；
- rule-based temporal fusion 生成 warning-candidate；
- warning interval 表示 runtime rule 在一段 timeline 上持续触发。

不应解释为：

- `p_eye_closed` 是 fatigue probability；
- `p_yawn` 是 fatigue diagnosis；
- warning interval 是 ground-truth drowsiness segment；
- high-confidence candidate 证明驾驶员疲劳。

Source: `src/runtime/realtime_temporal_state.py`, `src/runtime/system_video_upload_pipeline.py`

## 6. Video Upload Evidence

Video Upload 可用于展示 runtime pipeline 如何处理一段视频：

- summary；
- timeline；
- alert intervals；
- evidence figures；
- keyframes；
- HTML/Markdown report artifacts；
- technical details。

正确写法示例：

> The uploaded-video analysis demonstrates how the runtime evidence pipeline produces warning-candidate intervals and supporting figures for selected videos.

错误写法：

> The uploaded video proves the model detects drowsiness accurately.

Source: `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx`, `src/runtime/system_video_upload_pipeline.py`

## 7. Live Monitor、History 和 Insights Evidence

Live Monitor 展示 realtime product behavior。History 汇总 Live Monitor records。Insights 总结最近 Live Monitor alert patterns。

这些都不是模型评估报告：

- History counts 是 runtime records；
- Insights bullets 是产品 analytics；
- Recent Drives 是 session/drive summaries；
- signal interruptions 是 camera/ROI reliability 问题；
- charts 不应被当作 precision/recall/F1。

Source: `SystemUI/src/components/history-48h/History48hPage.tsx`, `SystemUI/src/components/insights/InsightsPage.tsx`

## 8. 正确措辞示例

- “The system produces warning-candidate intervals based on sustained eye-closure and yawn evidence.”
- “The eye specialist estimates `p_eye_closed`, which is later used by the temporal fusion layer.”
- “The mouth/yawn specialist estimates `p_yawn` from mouth ROI crops.”
- “The upload analysis demonstrates how the runtime pipeline behaves on selected videos.”
- “History and Insights summarize recent Live Monitor warning-candidate records.”
- “Specialist model metrics evaluate image-level classification tasks, not final system-level drowsiness detection.”

## 9. 错误措辞示例

| 错误说法 | 问题 |
|---|---|
| “The system detects drowsiness with 99% accuracy.” | 把 specialist metric 夸大成 full-system accuracy |
| “The model proves the driver is tired.” | 模型输出不是医学或安全真值 |
| “Every closed-eye frame means drowsiness.” | 正常眨眼也会闭眼 |
| “History shows model accuracy over 48 hours.” | History 是 runtime records，不是 evaluation |
| “Video Upload results are ground-truth labels.” | upload intervals 是 rule-based output |
| “Vercel deployment means the backend is in the cloud.” | 当前 backend 通过 tunnel 访问本地 Mac |

## 10. 报告中如何呈现

推荐章节安排：

| Report section | 可以写什么 |
|---|---|
| Methodology | 模块化系统、specialist models、MediaPipe ROI、temporal fusion |
| Data preprocessing | manifest、label mapping、subject-level split、crop generation |
| Model evaluation | eye/mouth specialist metrics、confusion matrix、PR/ROC if available |
| System implementation | FastAPI backend、Next.js frontend、runtime pipeline |
| Runtime demonstration | upload evidence figures、keyframes、Live Monitor behavior |
| Limitations | no final video-level drowsiness accuracy、domain shift、signal quality |

## 11. 简历或 PhD Portfolio 中如何呈现

可信、简洁的描述：

- built a modular driver-monitoring prototype；
- trained/evaluated specialist CNNs for eye-state and mouth/yawn evidence；
- used MediaPipe landmarks for ROI extraction；
- integrated rule-based temporal fusion for warning-candidate states；
- built FastAPI backend and Next.js frontend；
- implemented local archive, History, Insights, and upload evidence reports；
- deployed frontend to Vercel with remote testing through Cloudflare tunnel.

避免写成 commercial safety-certified product 或 medical diagnosis tool。

## 12. Red-Line Claims

除非未来有新证据，否则不要主张：

- certified driver safety system；
- medical fatigue diagnosis；
- production-ready autonomous safety product；
- final full-system drowsiness accuracy；
- real-world safety validation；
- generalized performance across all drivers/cameras/lighting；
- clinical fatigue measurement；
- cloud-native production backend。

## 13. 初学者检查清单

- 我写的是 specialist metric，还是 full-system claim？
- 我是否说明了 `p_eye_closed` / `p_yawn` 的 evidence 性质？
- 我是否把 warning-candidate 和 ground truth 区分开？
- 我是否标注了 figure/table 的来源路径？
- 我是否说明了 runtime demonstration 的限制？
- 我是否避免把 History/Insights 写成 accuracy？

## 14. 常见错误

- 使用最高单项 metric 当作系统准确率；
- 只展示好结果，不写限制；
- 混合 demo evidence 和 evaluation evidence；
- 说 “detects fatigue” 但没有说明 warning-candidate；
- 不写 dataset boundary 和 runtime boundary；
- 把 frontend screenshot 当成模型性能证据；
- 把 upload keyframe 当成真值标签。
