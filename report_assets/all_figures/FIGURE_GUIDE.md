# VisionGuard Final Figure Guide

生成时间：2026-05-24T23:00:29

本目录汇总了两批最终 report figures。第一批中已经 QA 修复的 Figure 1、5、10、11、12 使用 `rev1_visual_fixes` 版本，并复制为正式文件名；第一批其他图使用原 pass1 输出；第二批使用 pass2 输出图，并以 `pass2_revised_figure_captions.md` 作为最终 caption 语义来源。

重要语义边界：VisionGuard 应描述为“specialist CNN visual evidence extraction + temporal rule-based fusion”的系统。Eye 与 Mouth/Yawn 结果是 specialist model metrics，不是 final system-level drowsiness accuracy。runtime/demo 图展示 warning-candidate evidence，不是 labelled ground-truth drowsiness。remote demo 架构不是 production deployment。

## 文件索引

| 编号 | 图名 | 文件 | 推荐位置 |
|---|---|---|---|
| Figure 1 | Overall System Architecture Diagram | `fig01_system_architecture.png`<br>`fig01_system_architecture.svg` | Main report |
| Figure 2 | Runtime Inference Pipeline Diagram | `fig02_runtime_inference_pipeline.png`<br>`fig02_runtime_inference_pipeline.svg` | Main report |
| Figure 3 | Dual Expert Model Evidence Flow Diagram | `fig03_dual_expert_model_evidence_flow.png`<br>`fig03_dual_expert_model_evidence_flow.svg` | Main report |
| Figure 4 | Fusion Logic / Warning State Diagram | `fig04_fusion_warning_state_diagram.png`<br>`fig04_fusion_warning_state_diagram.svg` | Main report |
| Figure 5 | Archive / History / Insights Data Flow Diagram | `fig05_archive_history_insights_data_flow.png`<br>`fig05_archive_history_insights_data_flow.svg` | Main report |
| Figure 6 | Remote Demo Deployment Architecture Diagram | `fig_remote_demo_deployment_architecture.png`<br>`fig_remote_demo_deployment_architecture.svg` | Main report with caveat |
| Figure 7 | Combined Specialist Model Performance Summary Chart | `fig07_combined_specialist_performance_summary.png`<br>`fig07_combined_specialist_performance_summary.svg` | Main report |
| Figure 8 | Eye Model Performance Summary Chart | `fig08_eye_model_performance_summary.png`<br>`fig08_eye_model_performance_summary.svg` | Main report |
| Figure 9 | Mouth/Yawn Model Performance Summary Chart | `fig09_mouth_yawn_model_performance_summary.png`<br>`fig09_mouth_yawn_model_performance_summary.svg` | Main report |
| Figure 10 | Eye Model Confusion Matrix | `fig10_eye_model_confusion_matrix.png`<br>`fig10_eye_model_confusion_matrix.svg` | Main report |
| Figure 11 | Mouth/Yawn Model Confusion Matrix | `fig11_mouth_yawn_model_confusion_matrix.png`<br>`fig11_mouth_yawn_model_confusion_matrix.svg` | Main report |
| Figure 12 | Eye Training Curve | `fig12_eye_training_curve.png`<br>`fig12_eye_training_curve.svg` | Main report |
| Figure 13 | Mouth/Yawn PR Curve | `fig15_mouth_yawn_pr_curve.png`<br>`fig15_mouth_yawn_pr_curve.svg` | Main report |
| Figure 14 | Mouth/Yawn ROC Curve | `fig16_mouth_yawn_roc_curve.png`<br>`fig16_mouth_yawn_roc_curve.svg` | Main report |
| Figure 15 | Mouth/Yawn Threshold Sweep | `fig17_mouth_yawn_threshold_sweep.png`<br>`fig17_mouth_yawn_threshold_sweep.svg` | Main report |
| Appendix Figure A1 | Mouth/Yawn False Positive Error Gallery | `fig18_mouth_yawn_false_positive_error_gallery.png` | Appendix |
| Appendix Figure A2 | Mouth/Yawn False Negative Error Gallery | `fig19_mouth_yawn_false_negative_error_gallery.png` | Appendix |
| Appendix Figure A3 | Mouth/Yawn Sample Grid | `fig20_mouth_yawn_sample_grid.png` | Appendix |
| Figure 16 | Example Temporal Evidence Timeline | `fig_runtime_temporal_evidence_timeline.png`<br>`fig_runtime_temporal_evidence_timeline.svg` | Main report with saved-demo caveat |
| Figure 17 | Warning-candidate Interval Timeline | `fig_warning_candidate_interval_timeline.png`<br>`fig_warning_candidate_interval_timeline.svg` | Main report or appendix after caption fix |
| Appendix Figure A4 | Keyframe Contact Sheet | `fig_keyframe_contact_sheet.png` | Appendix |
| Appendix Figure A5 | Live Monitor Interface Screenshot | `fig_live_monitor_interface.png` | Appendix |
| Appendix Figure A6 | Video Upload Analysis Interface Screenshot | `fig_video_upload_interface.png` | Appendix |
| Appendix Figure A7 | History 48h Interface Screenshot | `fig_history_48h_interface.png` | Appendix |
| Appendix Figure A8 | Insights Interface Screenshot | `fig_insights_interface.png` | Appendix |

## 每张图的含义、技术细节与结论

### Figure 1. Overall System Architecture Diagram

- 文件：`fig01_system_architecture.png`, `fig01_system_architecture.svg`
- 来源批次：Pass1 - architecture/methodology
- 推荐使用：Main report
- 图代表什么：VisionGuard 的总体架构：输入视频/实时帧进入两个专门 CNN evidence model，输出 p_eye_closed 与 p_yawn，再由 temporal rule-based fusion 产生 warning-candidate 记录，并进入本地 UI 与 archive。
- 必要技术细节：强调系统不是单一 end-to-end drowsiness classifier，而是 Eye MobileNetV2、Mouth/Yawn ResNet18 与规则融合的模块化链路。archive 明确为 local compact SQLite archive。rev1 版本修复了底部注释与箭头拥挤问题。
- 图能支持的结论：这张图适合放在方法/系统设计开头，用来说明论文中的系统边界、模型分工与 warning-candidate 输出语义。
- 使用注意：不能解读为生产部署图或最终系统级准确率图。

### Figure 2. Runtime Inference Pipeline Diagram

- 文件：`fig02_runtime_inference_pipeline.png`, `fig02_runtime_inference_pipeline.svg`
- 来源批次：Pass1 - architecture/methodology
- 推荐使用：Main report
- 图代表什么：运行时从帧/上传视频采样到 ROI 预处理、专家模型 evidence extraction、signal-quality checks、temporal fusion、UI/archive 输出的完整流程。
- 必要技术细节：图中把视觉 evidence 与 temporal rule fusion 分开，避免把 VisionGuard 描述为单阶段分类器。它也表达了 warning-candidate 是规则融合后的运行时状态，而不是 labelled ground truth。
- 图能支持的结论：可以支撑方法章节里对 runtime inference flow 的解释，尤其是上传视频与实时监控共享的处理逻辑。
- 使用注意：不包含具体阈值或 debounce/cooldown 公式，不能替代代码级算法说明。

### Figure 3. Dual Expert Model Evidence Flow Diagram

- 文件：`fig03_dual_expert_model_evidence_flow.png`, `fig03_dual_expert_model_evidence_flow.svg`
- 来源批次：Pass1 - architecture/methodology
- 推荐使用：Main report
- 图代表什么：Eye expert 与 Mouth/Yawn expert 两条并行 evidence lane。Eye 输出 p_eye_closed，Mouth/Yawn 输出 p_yawn。
- 必要技术细节：Eye model 是 MobileNetV2 on MRL Eye；Mouth/Yawn model 是 ResNet18 on YawDD/YawDD+ Dash，p_yawn = softmax(logits)[1]，类别 0=no_yawn、1=yawn。
- 图能支持的结论：这张图说明系统的可解释性来自两个专门证据模型，而不是一个黑盒最终 drowsiness 分类器。
- 使用注意：两个专家模型的指标来自不同数据集，不能直接合成为系统级驾驶疲劳准确率。

### Figure 4. Fusion Logic / Warning State Diagram

- 文件：`fig04_fusion_warning_state_diagram.png`, `fig04_fusion_warning_state_diagram.svg`
- 来源批次：Pass1 - architecture/methodology
- 推荐使用：Main report
- 图代表什么：temporal fusion 如何结合 signal quality、eye temporal evidence、recent mouth/yawn evidence，形成 warning-candidate state。
- 必要技术细节：图中保留了 rule-based fusion、warning-candidate、debounce/cooldown、archiveable record 等概念，但没有改变或暴露具体实现阈值。
- 图能支持的结论：适合解释为什么运行时输出是 candidate warning，而不是人工标注的真实疲劳标签。
- 使用注意：这是概念化状态图；具体状态更新逻辑仍以 runtime code 为准。

### Figure 5. Archive / History / Insights Data Flow Diagram

- 文件：`fig05_archive_history_insights_data_flow.png`, `fig05_archive_history_insights_data_flow.svg`
- 来源批次：Pass1 - architecture/methodology
- 推荐使用：Main report
- 图代表什么：FastAPI archive/run endpoints、本地 SQLite archive、History 48h、Insights、Video Upload 之间的数据流。
- 必要技术细节：rev1 中确认 /api/runs/{session_id}/summary、/timeline、/keyframes、/files/{relative_path:path} 在 src/backend/app.py 中注册。archive 是 local compact SQLite archive，存 summary records 和 run references。
- 图能支持的结论：可以支撑产品/系统章节对历史记录、洞察页面、上传分析结果如何复用 archive 数据的说明。
- 使用注意：不能描述为 cloud production database，也不能暗示云端保存 raw webcam frames 或 raw uploaded videos。

### Figure 6. Remote Demo Deployment Architecture Diagram

- 文件：`fig_remote_demo_deployment_architecture.png`, `fig_remote_demo_deployment_architecture.svg`
- 来源批次：Pass2 - remote demo
- 推荐使用：Main report with caveat
- 图代表什么：远程演示架构：Vercel frontend 通过 Cloudflare Quick Tunnel 访问本地 FastAPI backend，本地 backend 使用本地模型 checkpoints 与 local compact SQLite archive。
- 必要技术细节：图中有 hosted demonstration entry point、temporary tunnel bridge、local machine components 三个区域。active API groups 来自 src/backend/app.py。
- 图能支持的结论：适合说明当前可远程演示，但系统依赖本地后端和临时 tunnel。
- 使用注意：必须写明 remote demonstration architecture，不是 production deployment；不包含 managed auth、生产数据库、认证安全部署或云端 raw media storage。

### Figure 7. Combined Specialist Model Performance Summary Chart

- 文件：`fig07_combined_specialist_performance_summary.png`, `fig07_combined_specialist_performance_summary.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：Eye MobileNetV2 与 Mouth/Yawn ResNet18 的 specialist model performance 摘要对比。
- 必要技术细节：两组结果分别来自各自 held-out test dataset；Eye 使用默认/argmax test metrics，Mouth/Yawn 使用 inference-only refresh 后的 held-out YawDD/YawDD+ Dash test metrics。
- 图能支持的结论：展示两个 evidence specialist 都具备较高的单任务识别性能，可作为 runtime fusion 的输入证据来源。
- 使用注意：不能把这张图解读为 final system-level drowsiness detection accuracy。

### Figure 8. Eye Model Performance Summary Chart

- 文件：`fig08_eye_model_performance_summary.png`, `fig08_eye_model_performance_summary.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：Eye MobileNetV2 在 MRL Eye held-out split 上的默认/argmax 测试表现。
- 必要技术细节：使用同一种 evaluation mode，避免混入 validation-selected-threshold metrics。重点指标约为 test accuracy 98.63%、macro F1 98.63%、closed-eye recall 98.52%。
- 图能支持的结论：Eye specialist 对 closed/open 眼部状态具有稳定识别能力，可作为 p_eye_closed evidence。
- 使用注意：这是 eye-state specialist metric，不是驾驶员疲劳最终判断。

### Figure 9. Mouth/Yawn Model Performance Summary Chart

- 文件：`fig09_mouth_yawn_model_performance_summary.png`, `fig09_mouth_yawn_model_performance_summary.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：Mouth/Yawn ResNet18 在 held-out YawDD/YawDD+ Dash test split 上的 refresh 后性能。
- 必要技术细节：测试样本 11,154；no_yawn 9,924，yawn 1,230。accuracy 0.9937242245，yawn precision 0.9647435897，yawn recall 0.9788617886，yawn F1 0.9717514124。
- 图能支持的结论：Mouth/Yawn specialist 对 yawn evidence 的识别性能强，适合作为 temporal fusion 的 p_yawn 输入。
- 使用注意：这是 mouth/yawn specialist result，不是 end-to-end drowsiness result。

### Figure 10. Eye Model Confusion Matrix

- 文件：`fig10_eye_model_confusion_matrix.png`, `fig10_eye_model_confusion_matrix.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：Eye MobileNetV2 default/argmax confusion matrix，标签为 closed 与 open。
- 必要技术细节：rev1 修复了标题/副标题重叠问题，并保持原始 matrix counts 与 default/argmax evaluation wording。
- 图能支持的结论：用于展示 eye specialist 在两个眼部状态类别上的错误分布。
- 使用注意：不要引入 validation-selected-threshold metrics，也不要描述为最终 drowsiness classification。

### Figure 11. Mouth/Yawn Model Confusion Matrix

- 文件：`fig11_mouth_yawn_model_confusion_matrix.png`, `fig11_mouth_yawn_model_confusion_matrix.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：Mouth/Yawn ResNet18 refresh 后的 held-out test confusion matrix，标签为 no_yawn 与 yawn。
- 必要技术细节：矩阵为 [[9880, 44], [26, 1204]]，类别顺序 [no_yawn, yawn]。该矩阵来自真实 per-sample inference refresh。
- 图能支持的结论：模型只有 44 个 no_yawn 被判为 yawn、26 个 yawn 被判为 no_yawn，说明 yawn evidence 的分类质量较高。
- 使用注意：这是 specialist confusion matrix，不能称为系统级 drowsiness confusion matrix。

### Figure 12. Eye Training Curve

- 文件：`fig12_eye_training_curve.png`, `fig12_eye_training_curve.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：Eye MobileNetV2 的真实 epoch-level training history，可视化 loss 与 validation metrics。
- 必要技术细节：rev1 把 validation accuracy、validation macro F1、closed-eye recall 分成独立 panel，避免曲线遮挡。只使用 standalone Eye history JSON。
- 图能支持的结论：用于说明 Eye specialist 训练收敛过程和验证表现趋势。
- 使用注意：没有生成或伪造 Mouth/Yawn training curve，因为缺少真实 standalone Mouth/Yawn history。

### Figure 13. Mouth/Yawn PR Curve

- 文件：`fig15_mouth_yawn_pr_curve.png`, `fig15_mouth_yawn_pr_curve.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：Mouth/Yawn ResNet18 的 precision-recall curve。
- 必要技术细节：曲线由 held-out test-set 的真实 p_yawn 概率生成，不使用 runtime video p_yawn。PR AUC 约 0.9945387626，Average Precision 约 0.9945400300。
- 图能支持的结论：模型在 yawn positive class 上具有强 precision/recall trade-off。
- 使用注意：仅代表 specialist test split，不代表系统级疲劳检测 PR curve。

### Figure 14. Mouth/Yawn ROC Curve

- 文件：`fig16_mouth_yawn_roc_curve.png`, `fig16_mouth_yawn_roc_curve.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：Mouth/Yawn ResNet18 的 ROC curve。
- 必要技术细节：由 held-out test predictions 的 p_yawn 生成，ROC AUC 约 0.9983857807。
- 图能支持的结论：模型区分 yawn 与 no_yawn 的概率排序能力很强。
- 使用注意：不是 end-to-end drowsy/not-drowsy ROC。

### Figure 15. Mouth/Yawn Threshold Sweep

- 文件：`fig17_mouth_yawn_threshold_sweep.png`, `fig17_mouth_yawn_threshold_sweep.svg`
- 来源批次：Pass1 - specialist model evidence
- 推荐使用：Main report
- 图代表什么：对 p_yawn threshold 从低到高变化时 accuracy、precision_yawn、recall_yawn、F1_yawn 等指标的影响。
- 必要技术细节：来自 refresh 生成的 threshold sweep table，基于 held-out test p_yawn probability，不使用 runtime demo timeline。
- 图能支持的结论：可用于讨论 Mouth/Yawn specialist 的阈值敏感性和 precision/recall trade-off。
- 使用注意：阈值 sweep 是 specialist 分析，不应直接替代 temporal fusion rule tuning。

### Appendix Figure A1. Mouth/Yawn False Positive Error Gallery

- 文件：`fig18_mouth_yawn_false_positive_error_gallery.png`
- 来源批次：Pass1 - specialist model appendix
- 推荐使用：Appendix
- 图代表什么：Mouth/Yawn held-out test split 中被模型误判为 yawn 的 no_yawn 样本。
- 必要技术细节：图片来自真实 per-sample refresh output 对应的 dataset sample，不是合成 failure case。
- 图能支持的结论：可用于错误分析，观察 false positive 可能来自嘴部姿态、画面质量或相似外观。
- 使用注意：标签较小，适合 appendix；不能当作 runtime failure gallery。

### Appendix Figure A2. Mouth/Yawn False Negative Error Gallery

- 文件：`fig19_mouth_yawn_false_negative_error_gallery.png`
- 来源批次：Pass1 - specialist model appendix
- 推荐使用：Appendix
- 图代表什么：Mouth/Yawn held-out test split 中真实 yawn 但被模型误判为 no_yawn 的样本。
- 必要技术细节：来自真实 held-out sample 与 refresh predictions。用于分析 yawn evidence miss cases。
- 图能支持的结论：帮助说明 yawn specialist 的剩余错误类型。
- 使用注意：不是系统运行时漏报案例，也不是 ground-truth drowsiness failure gallery。

### Appendix Figure A3. Mouth/Yawn Sample Grid

- 文件：`fig20_mouth_yawn_sample_grid.png`
- 来源批次：Pass1 - specialist model appendix
- 推荐使用：Appendix
- 图代表什么：Mouth/Yawn held-out test split 的代表性样本和模型输出。
- 必要技术细节：样本来自真实 refresh prediction file，用于展示 specialist dataset 的视觉输入类型。
- 图能支持的结论：为读者提供 mouth/yawn specialist 评估数据的直观背景。
- 使用注意：不是 runtime warning timeline。

### Figure 16. Example Temporal Evidence Timeline

- 文件：`fig_runtime_temporal_evidence_timeline.png`, `fig_runtime_temporal_evidence_timeline.svg`
- 来源批次：Pass2 - runtime demonstration evidence
- 推荐使用：Main report with saved-demo caveat
- 图代表什么：一个 saved video-upload demonstration run 中，p_eye_closed、p_yawn 与 warning-candidate fusion state 随时间变化的时间线。
- 必要技术细节：源数据为 stage17_test_B_realistic_drowsy_simulation 的 fusion_timeline.csv 和 fusion_summary.json；总采样 103 帧，约 21.26 秒。图中展示 eye/mouth evidence probability 与 normal、eye warning-candidate、mouth warning-candidate、high-confidence warning-candidate state。
- 图能支持的结论：这张图说明 temporal fusion 如何把瞬时 evidence 转化为运行时 candidate states。
- 使用注意：这是 example saved run，不是 labelled ground-truth drowsiness，也不支持 detection accuracy claim。

### Figure 17. Warning-candidate Interval Timeline

- 文件：`fig_warning_candidate_interval_timeline.png`, `fig_warning_candidate_interval_timeline.svg`
- 来源批次：Pass2 - runtime demonstration evidence
- 推荐使用：Main report or appendix after caption fix
- 图代表什么：同一个 saved video-upload demonstration run 中各类 warning-candidate interval 的开始/结束时间和帧范围。
- 必要技术细节：源数据来自 fusion_summary.json。eye warning intervals 包括 frames 75-95 与 200-260；mouth warning intervals 包括 frames 335-400 与 435-510；high-confidence warning-candidate interval 为 frames 405-430。
- 图能支持的结论：可以直观看出不同 evidence source 触发的 candidate intervals 在时间上的先后与重叠关系。
- 使用注意：必须使用 rev1 caption fix 中的措辞：warning-candidate interval timeline from a saved video-upload demonstration run；不是 ground-truth drowsiness labels。

### Appendix Figure A4. Keyframe Contact Sheet

- 文件：`fig_keyframe_contact_sheet.png`
- 来源批次：Pass2 - runtime demonstration appendix
- 推荐使用：Appendix
- 图代表什么：saved analysis run 的 high-confidence warning-candidate segment 中抽取的代表性 keyframes。
- 必要技术细节：三张 keyframe 对应 frames 405、420、430，时间约 16.88s、17.51s、17.92s，并标出 p_eye_closed 与 p_yawn。
- 图能支持的结论：为 runtime warning-candidate evidence 提供视觉上下文。
- 使用注意：这些不是 labelled ground-truth drowsiness examples，也不是伪造 failure case。

### Appendix Figure A5. Live Monitor Interface Screenshot

- 文件：`fig_live_monitor_interface.png`
- 来源批次：Pass2 - UI/product evidence
- 推荐使用：Appendix
- 图代表什么：VisionGuard live monitor route 的界面视图，包括摄像头区域、risk gauge、recent events 和 event counters。
- 必要技术细节：源图为已有真实 Playwright route capture，不是本轮新拍摄。截图展示的是 interface state，不是带有真实 runtime warning session 的证据。
- 图能支持的结论：用于说明实时监控 UI 的信息结构和 warning-candidate 事件展示位置。
- 使用注意：不要把静态界面截图当作运行时检测结果。

### Appendix Figure A6. Video Upload Analysis Interface Screenshot

- 文件：`fig_video_upload_interface.png`
- 来源批次：Pass2 - UI/product evidence
- 推荐使用：Appendix
- 图代表什么：Video Upload Analysis route 的上传、backend status、preview 与 processing status 区域。
- 必要技术细节：截图显示 backend URL/status、上传控件和静态未分析状态。它是 route capture，不是伪造分析结果。
- 图能支持的结论：用于说明上传视频分析 workflow 的前端入口和状态结构。
- 使用注意：不能声称截图中展示了真实分析完成结果，除非另有对应 run evidence。

### Appendix Figure A7. History 48h Interface Screenshot

- 文件：`fig_history_48h_interface.png`
- 来源批次：Pass2 - UI/product evidence
- 推荐使用：Appendix
- 图代表什么：History 48h route 的 warning-candidate history、filters、summary stats 和 review queue/timeline 区域。
- 必要技术细节：数据语义为 local/demo 或 current local archive records。截图展示产品如何过滤、汇总和审阅 warning-candidate records。
- 图能支持的结论：用于说明历史记录与人工复核 UX。
- 使用注意：不要描述为 production user history 或系统级准确率证据。

### Appendix Figure A8. Insights Interface Screenshot

- 文件：`fig_insights_interface.png`
- 来源批次：Pass2 - UI/product evidence
- 推荐使用：Appendix
- 图代表什么：Insights route 的 local warning-candidate analytics，包括 dominant pattern、high-priority share、signal-quality burden、review completion 和 trend chart。
- 必要技术细节：截图基于 local archive/fallback/local records 的界面展示。趋势图是产品层 summary，不是模型训练或系统级评估图。
- 图能支持的结论：用于说明系统如何把 warning-candidate records 转化为面向用户的趋势洞察。
- 使用注意：不要描述为 production analytics 或 ground-truth safety report。

## 不应从这些图中得出的结论

- 不要声称 VisionGuard 已有 final end-to-end drowsiness detection accuracy。
- 不要把 Eye 或 Mouth/Yawn specialist metrics 合并解释成系统级疲劳检测指标。
- 不要把 saved video-upload demonstration run 解释为 labelled ground-truth validation。
- 不要把 remote demo architecture 描述为 production deployment。
- 不要暗示当前 archive 是 cloud production database；它是 local compact SQLite archive。

## 复制清单

机器可读复制记录见 `copy_manifest.json`。
