# 模型评估与模型选择技术学习指南

本文档解释 VisionGuard 如何理解模型评估结果，以及为什么最终选择 MobileNetV2 作为眼部 specialist、ResNet18 作为嘴部 specialist。

建议先阅读：

- `docs/tech_learning/MODEL_TRAINING_TECHNICAL_GUIDE_ZH.md`
- `docs/tech_learning/DATA_PREPROCESSING_TECHNICAL_GUIDE_ZH.md`
- `docs/AI_PROJECT_CONTEXT.md`
- `docs/PROJECT_CURRENT_STATUS.md`

---

## 1. 本文档的目的

本文档关注两个问题：

1. 如何读懂本项目中的 accuracy、precision、recall、F1、confusion matrix、ROC/AUC 等指标？
2. 为什么模型选择不能只看最高 accuracy？

核心边界：

- 本项目评估的是 specialist image-level classifiers。
- 这些指标不等于最终 full-system drowsiness accuracy。
- Runtime system 会把 `p_eye_closed`、`p_yawn`、signal quality 和 temporal fusion rules 合并后才产生 alert 状态。

---

## 2. VisionGuard 的评估范围

VisionGuard 主要有两类 offline specialist evaluation：

| Specialist | 输入 | 输出 | 评估对象 |
|---|---|---|---|
| Eye open/closed | Eye ROI image | `p_eye_closed` | 单张眼部 ROI 是否 closed/open |
| Mouth/yawn | Mouth ROI image | `p_yawn` | 单张嘴部 ROI 是否 no_yawn/yawn |

这些评估回答的是：

- 模型能否在 held-out test set 上正确分类 ROI 图像？
- 对关键类别是否有足够 recall / precision？
- 模型是否适合 runtime inference？

它们不回答：

- 驾驶者是否真实疲劳？
- 系统对 drowsiness 的最终准确率是多少？
- 每个 alert interval 是否有 ground-truth drowsiness label？

---

## 3. 到底在评估什么？

本项目的模型评估至少包含五个层面：

1. **Image-level classification performance**：单张 ROI 图像分类是否正确。
2. **Class-level performance**：关键类别是否容易漏检或误报。
3. **Validation/test performance**：验证集用于模型选择，测试集用于最终报告。
4. **Runtime usefulness**：模型大小、速度、checkpoint 可用性、推理输出是否稳定。
5. **Qualitative error patterns**：错误样例是否集中在眼镜、光照、模糊、说话、嘴部 crop 偏移等情况。

因此，模型选择不是“哪个数字最大就选哪个”，而是一个综合工程判断。

---

## 4. Confusion Matrix

Confusion matrix 显示真实类别和预测类别之间的关系。二分类常见四个概念：

| 概念 | 含义 |
|---|---|
| True Positive | 真实为正类，预测为正类 |
| True Negative | 真实为负类，预测为负类 |
| False Positive | 真实为负类，预测为正类 |
| False Negative | 真实为正类，预测为负类 |

### 4.1 眼部任务中的 confusion matrix

MRL Eye 标签映射：

- `0 = closed`
- `1 = open`

如果把 `closed` 看作关注类别：

- false open：真实 closed，但预测 open。这类错误可能漏掉闭眼证据。
- false closed：真实 open，但预测 closed。这类错误可能增加误报闭眼证据。

Source: `src/training/train_mrl_eye_baselines.py`

### 4.2 嘴部任务中的 confusion matrix

YawDD/YawDD+ mouth/yawn 标签映射：

- `0 = no_yawn`
- `1 = yawn`

如果把 `yawn` 看作关注类别：

- false yawn：真实 no_yawn，但预测 yawn。这会增加错误打哈欠证据。
- missed yawn：真实 yawn，但预测 no_yawn。这会漏掉打哈欠证据。

Source:

- `src/training/train_classifier.py`
- `report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json`

---

## 5. 核心指标

### 5.1 Accuracy

Accuracy 是所有样本中预测正确的比例。

优点：直观。

缺点：当类别不平衡时容易误导。例如 no_yawn 样本远多于 yawn 样本时，一个模型即使偏向 no_yawn，也可能有较高 accuracy。

### 5.2 Precision

Precision 回答：“被模型预测为某类的样本中，有多少是真的？”

对于 yawn：

```text
yawn precision = true_yawn_predictions / all_yawn_predictions
```

如果 yawn precision 低，说明 false yawn 多，系统可能产生误导性的打哈欠证据。

### 5.3 Recall

Recall 回答：“真实属于某类的样本中，有多少被找出来？”

对于 closed eye：

```text
closed recall = correctly_predicted_closed / all_true_closed
```

如果 closed recall 低，模型会漏掉闭眼证据。

### 5.4 F1-score

F1 是 precision 和 recall 的调和平均。它适合在 false positive 和 false negative 都重要时使用。

```text
F1 = 2 * precision * recall / (precision + recall)
```

### 5.5 Macro F1 和 weighted F1

Macro F1 对每个类别的 F1 平均，不按类别样本数加权。因此它更能反映少数类表现。

Weighted F1 会按类别样本数加权，适合看整体样本分布下的平均表现，但可能掩盖少数类问题。

### 5.6 ROC/AUC 和 PR/AUC

ROC curve 画的是不同阈值下 true positive rate 与 false positive rate 的关系。AUC 越高，说明模型给正类排序的能力越强。

PR curve 更关注 precision 和 recall。对于类别不平衡任务，PR curve 往往比 ROC curve 更直观。

本项目中：

- Mouth/yawn evaluation refresh 有 PR curve、ROC curve 和 threshold sweep。
- MRL Eye 输出中已检查到 PR curve、confusion matrix、training curve 和 threshold sweep；本地 `outputs/mrl_eye/figures/` 与云端 MRL Eye figures folder 中都没有找到 eye ROC curve artifact。
- MRL Eye 完整 ROC curve 需要 per-sample `p_eye_closed` scores。当前本地结果保存了 threshold sweep CSV，但没有保存逐样本 prediction CSV；因此只能直接画 9 个阈值点的粗略 threshold ROC。若要生成标准 ROC/AUC，应使用现有 checkpoint、manifest 和 `src/training/train_mrl_eye_baselines.py` 的 prediction logic 重新跑一次 inference-only prediction pass，保存 `y_true` 和 `p_eye_closed` 后再计算。这个过程不需要 retraining。

Source:

- `report_assets/mouth_yawn_evaluation_refresh/figures/`
- `outputs/mrl_eye/figures/`
- `outputs/mrl_eye/results/*_test_threshold_sweep.csv`
- `src/training/train_mrl_eye_baselines.py`

---

## 6. 评估 artifacts 和 source of truth

| 路径 | 内容 | 用途 |
|---|---|---|
| `outputs/mrl_eye/results/mrl_eye_initial_results.csv` | MRL Eye 三个候选模型结果 | 眼部候选模型比较 |
| `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` | 眼部最终模型选择 | MobileNetV2 runtime 选择依据 |
| `outputs/mrl_eye/results/resnet18_metrics.json` | 眼部 ResNet18 详细指标 | 候选模型评估 |
| `outputs/mrl_eye/results/mobilenet_v2_metrics.json` | 眼部 MobileNetV2 详细指标 | 最终模型评估 |
| `outputs/mrl_eye/results/efficientnet_b0_metrics.json` | 眼部 EfficientNet-B0 详细指标 | 候选模型评估 |
| `outputs/mrl_eye/results/*_test_threshold_sweep.csv` | 眼部阈值扫描结果 | 可支持粗略 threshold ROC 点；不是完整 probability-ranked ROC |
| `outputs/mrl_eye/figures/` | confusion matrices、PR curves、training curves | 报告图表 |
| `outputs/mrl_eye/error_analysis/` | false open / false closed 样例 | 质量分析 |
| `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json` | mouth/yawn 三模型恢复结果 | Stage 7 模型比较 |
| `report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json` | ResNet18 mouth/yawn 最终刷新评估 | 报告指标 source of truth |
| `report_assets/mouth_yawn_evaluation_refresh/figures/` | confusion matrix、PR、ROC、threshold sweep | 报告图表 |
| `report_assets/mouth_yawn_evaluation_refresh/predictions/` | test predictions / probabilities | threshold and ranking analysis |
| Google Drive `Drowsiness_Detection_Colab/outputs/results/*_history.json` | 原始 Stage 7 mouth/yawn training history | 可作为 mouth/yawn training curve source |
| Google Drive `Drowsiness_Detection_Colab/outputs/figures/*_training_curve.png` | 原始 Stage 7 mouth/yawn training curves | 报告训练曲线图来源 |

注意：如果旧文件与恢复后的 final artifacts 不一致，应优先使用 `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` 和 `report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json` 作为最终叙述来源。

---

## 7. 眼部模型评估

### 7.1 候选模型

MRL Eye 评估了：

- ResNet18
- MobileNetV2
- EfficientNet-B0

Source: `outputs/mrl_eye/results/mrl_eye_initial_results.csv`

### 7.2 关键结果

| Model | Best epoch | Test accuracy | Test macro F1 | Closed recall | False open | False closed |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 4 | 98.46% | 98.46% | 98.59% | 89 | 109 |
| MobileNetV2 | 8 | 98.63% | 98.63% | 98.52% | 93 | 84 |
| EfficientNet-B0 | 8 | 98.62% | 98.62% | 98.24% | 111 | 67 |

Source: `outputs/mrl_eye/results/mrl_eye_initial_results.csv`

### 7.3 最终选择

最终 eye runtime model 是 MobileNetV2。

关键 source：

```text
outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json
```

已确认：

- `primary_selected_model = mobilenet_v2`
- `recommended_default_threshold = 0.5`
- `recommended_default_rule = argmax / p_eye_closed >= 0.50`
- runtime checkpoint found: `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`

MobileNetV2 的选择理由：

- test accuracy 和 macro F1 在候选模型中最强或并列最强。
- lightweight architecture 更适合实时眼部 ROI 推理。
- checkpoint 已被 Stage 10 / runtime 代码指定。

Source:

- `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json`
- `src/runtime/stage10_eye_roi_consistency.py`
- `src/runtime/realtime_frame_inference.py`

### 7.4 仍然存在的限制

眼部模型评估仍是 image-level ROI classification。它不能单独证明疲劳，只能说明模型在 held-out MRL Eye split 上识别 open/closed 的能力较强。

需要特别注意：

- 眼镜和反光可能造成错误。
- 运动模糊和低光照可能影响 ROI。
- squinting、低头、侧脸可能不是典型 open/closed。
- 运行时摄像头分布可能与 MRL Eye 数据分布不同。

---

## 8. Mouth/yawn 模型评估

### 8.1 候选模型

Stage 7 mouth/yawn 评估了：

- ResNet18
- MobileNetV2
- EfficientNet-B0

Source: `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`

### 8.2 三模型结果

| Model | Best epoch | Test accuracy | Yawn precision | Yawn recall | Yawn F1 | False yawn | Missed yawn |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 4 | 99.37% | 96.47% | 97.89% | 97.18% | 44 | 26 |
| MobileNetV2 | 4 | 98.75% | 91.74% | 97.48% | 94.52% | 108 | 31 |
| EfficientNet-B0 | 3 | 99.20% | 94.82% | 98.13% | 96.44% | 66 | 23 |

Source: `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`

### 8.3 最终 ResNet18 evaluation refresh

最终报告更建议引用刷新评估 source：

```text
report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json
```

已确认：

| 指标 | 值 |
|---|---:|
| Test samples | 11,154 |
| Label distribution | no_yawn 9,924；yawn 1,230 |
| Accuracy | 99.37% |
| Macro precision | 98.11% |
| Macro recall | 98.72% |
| Macro F1 | 98.41% |
| Weighted F1 | 99.37% |
| Yawn precision | 96.47% |
| Yawn recall | 97.89% |
| Yawn F1 | 97.18% |
| ROC AUC | 99.84% |
| PR AUC / Average Precision | 99.45% |
| Confusion matrix | `[[9880, 44], [26, 1204]]` |

### 8.4 为什么选择 ResNet18

ResNet18 被选为 mouth/yawn runtime model，因为：

- 它在 Stage 7 recovered metrics 中有最高 test accuracy。
- 它有最高 yawn F1。
- 它的 false yawn 数少于 MobileNetV2 和 EfficientNet-B0。
- 它的 checkpoint 是 runtime 指定的 mouth/yawn checkpoint。

需要精确表述：

- EfficientNet-B0 的 yawn recall 更高，missed yawn 更少。
- 但 ResNet18 的整体 test accuracy 和 yawn F1 更高。
- 因此不能写 “EfficientNet-B0 had the best overall result”。更准确是：EfficientNet-B0 had the strongest yawn recall, while ResNet18 was selected for the strongest overall test accuracy and yawn F1.

Source:

- `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`
- `report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json`
- `src/runtime/stage14_mouth_yawn_runtime.py`

---

## 9. 模型选择不只是最高 accuracy

最终选择应该综合：

- test accuracy
- macro F1
- class-specific recall
- class-specific precision
- false positive / false negative 代价
- validation behavior
- checkpoint completeness
- inference speed
- model size
- runtime integration simplicity
- qualitative error patterns
- 与 ROI crop pipeline 的兼容性

VisionGuard 中的实际例子：

- MobileNetV2 被选为 eye specialist，不只是因为指标强，也因为轻量、适合实时 eye ROI 推理，并且 runtime 文件指向它。
- ResNet18 被选为 mouth/yawn specialist，因为它在 mouth/yawn final comparison 中整体 test accuracy 和 yawn F1 更强。
- EfficientNet-B0 是有价值的 comparison model，但不是当前 runtime default。

---

## 10. Error analysis

Aggregate metrics 只告诉你“错了多少”，error analysis 告诉你“为什么错”。

### 10.1 眼部常见错误来源

- squinting 或半闭眼
- 眼镜和反光
- 低光照或强光
- 运动模糊
- 眼部 ROI 偏移
- 侧脸或遮挡

项目相关文件：

- `outputs/mrl_eye/error_analysis/`
- `outputs/mrl_eye/figures/`

### 10.2 嘴部常见错误来源

- 说话造成嘴巴张开
- 微笑或大笑
- 张嘴但不是 yawn
- 头部姿态变化
- mouth landmark 检测失败
- fallback crop 包含过多非嘴部区域

项目相关文件：

- `report_assets/mouth_yawn_evaluation_refresh/error_gallery/`
- `report_assets/mouth_yawn_evaluation_refresh/predictions/`

### 10.3 为什么 keyframes 和 video evidence 有价值

Video Upload 的 evidence figures 和 keyframes 可以帮助观察模型概率随时间变化，以及 temporal fusion 如何把连续证据转成 alert intervals。但这些仍是 runtime demonstration，不是 ground-truth drowsiness accuracy，除非该视频有人工标注的 ground truth。

Source: `src/runtime/system_video_upload_pipeline.py`

---

## 11. Runtime evaluation boundary

必须区分以下几件事：

| 层级 | 含义 | 能否当作系统准确率 |
|---|---|---|
| Offline image-level test metrics | ROI 图像分类测试 | 不能 |
| Video Upload evidence figures | 上传视频的模型概率和 fusion timeline | 不能，除非有 ground truth |
| Realtime Live Monitor behavior | 摄像头实时输出 alerts | 不能 |
| History / Insights analytics | Live Monitor 记录的产品级摘要 | 不能 |
| Full system validation | 需要带 ground-truth drowsiness 标注的独立评估 | 才可能 |

History 和 Insights 页面总结的是 Live Monitor records，不是 accuracy report。

---

## 12. 如何正确报告指标

推荐写法：

- “The selected eye-state specialist model achieved strong test performance for open/closed eye classification.”
- “The mouth/yawn specialist provides `p_yawn` evidence for the later temporal fusion layer.”
- “The runtime system produces warning-candidate intervals based on temporal visual evidence, rather than direct ground-truth drowsiness labels.”
- “Specialist model metrics should be interpreted as ROI-level classification results, not final system-level drowsiness accuracy.”

不推荐写法：

- “The system detects drowsiness with 98% accuracy.”
- “The model proves the driver is drowsy.”
- “Every yawn means drowsiness.”
- “Every closed-eye frame means fatigue.”
- “Warning-candidate intervals are ground truth.”

---

## 13. 初学者检查清单

你应该能够回答：

- Confusion matrix 每个格子代表什么？
- Precision 和 recall 的区别是什么？
- 为什么 macro F1 对类别不平衡更有用？
- 为什么 yawn precision 很重要？
- 为什么 closed-eye recall 很重要？
- 为什么 MobileNetV2 是 eye runtime model？
- 为什么 ResNet18 是 mouth/yawn runtime model？
- 为什么 EfficientNet-B0 不是 runtime default？
- 为什么模型测试分数不等于系统疲劳检测准确率？

---

## 14. 常见错误

| 错误 | 正确做法 |
|---|---|
| 只报告 accuracy | 同时报告 precision、recall、F1 和 confusion matrix |
| 混淆 label mapping | MRL Eye: `0=closed,1=open`; mouth/yawn: `0=no_yawn,1=yawn` |
| 把 validation 结果当成 test 结果 | 明确区分 validation selection 和 held-out test reporting |
| 只看最高 recall 选模型 | 同时看 precision、F1、false positives、runtime suitability |
| 把 EfficientNet-B0 写成最终 runtime model | 它是 comparison model，不是当前 default |
| 使用旧 metric 文件 | 先核对 final artifact 和 source path |
| 把 alert intervals 当 ground truth | alert intervals 是 rule-based runtime outputs |
| 写“系统 99% 准确” | 应写 specialist ROI-level metrics |

---

## 15. 当前需要特别小心的不一致点

1. **Stage 7 epochs / patience**

   完成的 Colab run constants 是：

   - `DEFAULT_EPOCHS = 8`
   - `DEFAULT_PATIENCE = 2`

   Source: `colab_file/stage7_yawdd_training_r.ipynb`

   Local reusable script default 是：

   - `--epochs = 12`
   - `--patience = 3`

   Source: `src/training/train_classifier.py`

2. **Mouth/yawn EfficientNet-B0 结论**

   EfficientNet-B0 的 yawn recall 最高，但 ResNet18 的 test accuracy 和 yawn F1 更高。最终 runtime model 是 ResNet18。

3. **Eye optimizer**

   MRL Eye training script 使用 AdamW，不是 Adam。

   Source: `src/training/train_mrl_eye_baselines.py`

4. **Mouth/yawn training curve**

   本地 `report_assets/mouth_yawn_evaluation_refresh/skipped/training_curve_status.md` 只说明 evaluation refresh 目录自身没有足够 source 重建真实 Stage 7 training curve。Google Drive 原始 Stage 7 输出中已经找到 `Drowsiness_Detection_Colab/outputs/results/resnet18_history.json`、`mobilenet_v2_history.json`、`efficientnet_b0_history.json` 和对应 `outputs/figures/*_training_curve.png`。因此如果报告需要 mouth/yawn training curve，应引用这些云端原始输出；不要从 refresh metrics 反推或伪造。

5. **MRL Eye ROC curve**

   本地和云端都没有找到已生成的 MRL Eye ROC curve image。已有 `outputs/mrl_eye/results/*_test_threshold_sweep.csv` 可以画 9 个阈值点的粗略 threshold ROC，但这不等同于完整 ROC/AUC。完整 ROC/AUC 需要逐样本 `p_eye_closed` scores；当前训练脚本 `src/training/train_mrl_eye_baselines.py` 已经有 prediction logic，但没有把逐样本 MRL Eye predictions 保存成 CSV。若要正式报告 MRL Eye ROC/AUC，应使用现有 checkpoint 重新跑 inference-only prediction pass 并保存 scores 后再计算。
