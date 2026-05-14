# Project Learning Guide: Driver Drowsiness Detection

> 本学习指南基于当前仓库中的 `docs/`、`reports/`、`src/`、`SystemUI/`、`scripts/`、`artifacts/`、`outputs/`、`colab_file/`、`upload_test/` 和 `dataset/` 目录整理。本文主要使用中文说明，并保留关键技术术语的 English labels。  
> 重要边界：本文解释的是当前已经完成的数据预处理（Data Preprocessing）、specialist modules、runtime temporal analysis、rule-based fusion、Stage 17.5 本地 `/video-upload` warning-candidate evidence review UI、Stage 18 frontend-only `/history-48h` demo/local history page，以及 Stage 19 `/` Live Monitor 本地 realtime warning-candidate prototype。它不是最终系统级疲劳检测准确率（Final System-Level Drowsiness Accuracy）报告，也不是 alarm / deployment readiness 报告。

Last updated: 2026-05-13

## 1. Project Overview

### 1.1 已实现事实（Implemented Facts）

本项目目标是构建一个基于深度学习（Deep Learning）的驾驶员疲劳检测与监控系统（Driver Drowsiness Detection and Monitoring System）。当前仓库中的设计不是一个单一的端到端疲劳分类器（Monolithic Classifier），而是一个模块化系统（Modular System）：先训练多个可解释的视觉行为专家模型（Specialist Classifiers），再用 runtime ROI extraction、temporal analysis 和 rule-based fusion 把概率输出组合成 uploaded-video warning-candidate analysis。

当前已经完成并稳定到 Stage 19 Live Monitor prototype 的主要部分是：

| 模块 | 数据来源 | 当前任务 | 输出概念 | 当前状态 |
|---|---|---|---|---|
| 嘴部/打哈欠模块（Mouth/Yawn Specialist） | YawDD Dash 原始视频 + YawDD+ annotation files | `no_yawn` vs `yawn` 二分类（Binary Classification） | `p_yawn` | Stage 7 completed |
| 眼睛开闭模块（Eye Open/Closed Specialist） | MRL Eye dataset | `closed` vs `open` 二分类（Binary Classification） | `p_eye_closed` | Stage 8, Stage 9, Stage 9B completed |
| Runtime temporal analysis | A/B/C/D 和 upload test videos | eye ROI、mouth ROI、temporal warning-candidate analysis | sampled timelines | Stage 10-15 completed as controlled-validation prototype |
| Video Upload Analysis UI | Uploaded short videos | local rule-based warning-candidate review workstation | summary, intervals, tabbed figures, keyframes, technical evidence | Stage 17.5 evidence-review UI polished |
| 48h History UI | Browser local demo history | recent warning-candidate history review | summary cards, charts, timeline, sessions, review queue | Stage 18 frontend-only page completed |
| Live Monitor UI/API | Browser webcam sampled frames | realtime frame evidence and temporal warning-candidate state | `p_eye_closed`, `p_yawn`, ROI/signal quality, realtime candidate state | Stage 19 local prototype completed |

`p_yawn` 表示某一帧或某一嘴部区域裁剪（Mouth ROI Crop）属于打哈欠类别的模型概率。`p_eye_closed` 表示某一眼部图像属于闭眼类别的模型概率。这两个值都是 specialist-module outputs。Stage 17 使用它们生成 rule-based warning-candidate results，不生成最终疲劳真值（final drowsiness truth）或最终系统级准确率。

### 1.2 基于事实的解释（Inferred Interpretation）

项目采用模块化设计是合理的，因为疲劳不是一个单帧视觉标签，而是一个随时间变化的状态。嘴部打哈欠和眼睛闭合是两个互补信号：

- 打哈欠（Yawn）更像低频但强语义的疲劳线索。
- 闭眼（Eye Closed）更像高频、需要时间统计的线索，例如 PERCLOS-like logic。
- 单帧 `closed` 或 `yawn` 只能说明“这一帧像闭眼/打哈欠”，不能直接证明“驾驶员处于疲劳状态”。

因此，当前结果应解释为 specialist-module metrics 和 rule-based drowsiness warning-candidate analysis，而不是 final system-level drowsiness accuracy。

### 1.3 未来工作（Future Work Suggestions）

当前已经有 Stage 15/17 的 rule-based fusion、Stage 17.5 uploaded-video warning-candidate UI、Stage 18 frontend-only 48h history UI，以及 Stage 19 Live Monitor 本地 realtime warning-candidate prototype。更合理的下一步是围绕明确边界继续推进：如果做历史，就接入真实 persisted run records；如果做 Live Monitor，就先设计 alert debounce / alarm policy 和 history ingestion 边界，再实现。当前不能声明 final system-level drowsiness accuracy、deployment readiness、alarm readiness 或 final drowsiness truth。

## 2. Repository Structure and File Map

### 2.1 总体结构

| 路径 | 存储内容 | 支持的 ML pipeline 阶段 | 你应该重点学习什么 |
|---|---|---|---|
| `dataset/` | 原始数据、本地重建数据、图像/视频数据 | Dataset storage / reconstruction input | 数据从哪里来、标签如何对应图像、哪些数据是主线 |
| `artifacts/` | manifests、splits、visual checks、preprocessing outputs、旧结果 | Data preparation / validation / split | 如何把原始数据变成可训练 CSV，如何做 leakage check |
| `reports/` | 数据检查、预处理、划分、训练计划、模型选择报告 | Human-readable evidence | 项目决策依据和质量控制证据 |
| `src/` | Python 源码 | Dataset preparation / preprocessing / training | 实现逻辑、输入输出、label mapping、训练机制 |
| `src/data/` | 数据检查、manifest 构建、subject split、spot check scripts | Data engineering | 数据解析、subject-level split、可视化 sanity check |
| `src/preprocessing/` | YawDD 嘴部 ROI 生成脚本 | Computer vision preprocessing | MediaPipe Face Mesh、fallback crop、mouth ROI |
| `src/training/` | CNN baseline 训练脚本 | Model training / evaluation | PyTorch Dataset/DataLoader、transfer learning、metrics |
| `src/runtime/` | Stage 10-17 runtime scripts | Video inference / temporal analysis / rule-based fusion | eye ROI、mouth ROI、timeline、warning-candidate fusion |
| `src/backend/` | FastAPI backend | Stage 17 upload API / artifact serving | `/api/analyze-video`、session files、安全 URL |
| `SystemUI/` | independent Next.js frontend | Stage 17.5 upload analysis workstation + Stage 18 history page + Stage 19 Live Monitor | `/` Live Monitor、`/video-upload` UI、`/history-48h` UI、Sidebar、summary/interval/keyframe/history review、realtime evidence panel |
| `scripts/` | local helper scripts | Stage 17 launcher | 一键启动 backend + frontend |
| `colab_file/` | Colab notebooks | GPU training / completed run records | Stage 7 和 Stage 9 completed notebook outputs |
| `outputs/` | 同步回本地的训练输出 | Final experiment artifacts | MRL Eye metrics、figures、error analysis、checkpoints |
| `outputs/mrl_eye/` | MRL Eye Stage 9/9B 完整输出 | Eye module model selection | MobileNetV2 选择证据、threshold sweeps、error sheets |
| `outputs/system_video_upload_runs/` | Stage 17 upload-session outputs | Uploaded-video evidence | summary、timeline、figures、keyframes、report |
| `upload_test/` | Stage 17 upload test videos | UI/backend manual validation | A/B/C upload tests，特别是 `C_upload_test.mp4` |
| `checkpoints/` | legacy/local checkpoint folder | Model storage | 当前未发现普通文件；主 checkpoint 在 `outputs/mrl_eye/checkpoints/` |
| `docs/` | 项目结构和当前状态文档 | Project documentation | 如何向队友解释项目结构和当前状态 |

### 2.2 关键文件地图

| 文件 | 作用 | 学习重点 |
|---|---|---|
| `docs/PROJECT_STRUCTURE.md` | 当前仓库结构说明 | 模块化系统、目录职责、artifact flow |
| `docs/PROJECT_CURRENT_STATUS.md` | 当前状态与实验总结 | 哪些已完成，哪些不能过度声明 |
| `reports/yawdd_raw_dash_report.md` | YawDD Dash 原始视频检查 | 原始视频数量、命名异常、subject token 对齐 |
| `reports/yawdd_plus_annotation_format_report.md` | YawDD+ annotation 格式检查 | YOLO bbox 格式、class id、frame index |
| `reports/yawdd_dash_visual_sanity_check.md` | YawDD 可视化 sanity check | 类别语义确认、bbox 不适合作嘴部 ROI |
| `reports/yawdd_dash_reconstruction_report.md` | YawDD Dash frame reconstruction 验证 | 原视频帧索引和 annotation 对齐 |
| `reports/yawdd_dash_mouth_crop_report.md` | 嘴部裁剪报告 | MediaPipe crop、fallback、failed rows、success rate |
| `reports/yawdd_dash_split_report.md` | YawDD subject-level split | leakage prevention、split distribution |
| `reports/mrl_eye_dataset_report.md` | MRL Eye 数据集检查 | label mapping、subject folders、class balance |
| `reports/mrl_eye_split_report.md` | MRL Eye subject split | split checks、subject-level split |
| `reports/mrl_eye_stage9_training_plan.md` | Stage 9 训练计划 | 训练设置、metrics、threshold sweep 设计 |
| `reports/mrl_eye_stage9b_error_analysis.md` | Stage 9B 模型选择 | MobileNetV2 选择、ResNet18 safety reference |
| `reports/stage15_real_mouth_eye_fusion_validation_report.md` | Stage 15 real synchronized rule-based fusion | Stage 12 eye timeline + Stage 14 mouth timeline 如何融合 |
| `reports/stage17_video_upload_detection_mvp_report.md` | Stage 17 upload backend/pipeline report | `/api/analyze-video`、summary、figures、keyframes |
| `reports/stage17_2_manual_review_interpretation_report.md` | Stage 17.2 manual interpretation report | 如何安全解释 C 视频的 warning-candidate intervals |
| `reports/stage17_4_video_upload_mvp_stabilization_report.md` | Stage 17.4 historical stabilization report | launcher、acceptance、demo、限制边界 |
| `docs/STAGE17_5_EYE_EVIDENCE_CALIBRATION.md` | Stage 17.5 eye evidence calibration | weak/moderate/strong eye evidence、strength gate、安全解释 |
| `reports/stage17_5_eye_evidence_calibration_report.md` | Stage 17.5 runtime/report summary | Stage 17.5 backend output interpretation fields |
| `docs/STAGE17_5_VIDEO_UPLOAD_UI_FALLBACK_POLISH.md` | Stage 17.5 keyframe fallback polish | missing optional fields、recent-yawn temporal window、Supporting keyframe |
| `docs/STAGE17_5_VIDEO_UPLOAD_UI_SECOND_PASS_CLEANUP.md` | Stage 17.5 `/video-upload` UI cleanup | compact interval table、metric scope、tabbed figures、fusion vs evidence separation |
| `docs/STAGE18_HISTORY_48H_UI_PAGE_PLAN.md` | Stage 18 48h History page | frontend-only localStorage history、filters、charts、manual review queue |
| `src/runtime/realtime_frame_inference.py` | Stage 19 realtime frame evidence | webcam JPEG frame decode、MediaPipe ROI、eye/mouth model inference、frame-level evidence |
| `src/runtime/realtime_temporal_state.py` | Stage 19 realtime temporal state | mouth activity、recent yawn context/reminder、active eye-warning state、recent sustained eye-warning reminder |
| `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md` | Stage 17 API/result schema | `summary.json`、interval、timeline、keyframe metadata |
| `docs/STAGE17_3_LOCAL_LAUNCH_GUIDE.md` | Stage 17 local launch guide | `make stage17-ui`、backend/frontend URL、排错 |
| `docs/STAGE17_3_VIDEO_UPLOAD_UI_PAGE_REPORT.md` | Stage 17.3 UI report | `/video-upload` 页面结构和 safe wording |
| `docs/STAGE17_4_VIDEO_UPLOAD_UI_ACCEPTANCE_CHECKLIST.md` | Stage 17.4 acceptance checklist | 演示前人工验收项目 |
| `docs/STAGE17_4_DEMO_SCRIPT.md` | Stage 17.4 demo script | 演示流程和安全措辞 |
| `scripts/start_stage17_ui.sh` | one-command local launcher | 启动 backend/frontend，Ctrl+C 一起停止 |
| `Makefile` | project-level command target | `make stage17-ui` |
| `reports/nthuddd2_kaggle_dataset_report.md` | NTHUDDD2 Kaggle 探索 | 为什么不是当前主线 |
| `colab_file/stage7_yawdd_training_r.ipynb` | Completed YawDD Stage 7 run | Stage 7 结果 source of truth |
| `colab_file/stage9_mrl_eye_training_r.ipynb` | Completed MRL Eye Stage 9 run | Stage 9 Colab training record |
| `outputs/mrl_eye/results/mrl_eye_initial_results.csv` | MRL Eye 主结果 CSV | 三个模型 argmax/test metrics |
| `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` | Stage 9B 选择 JSON | selected model、threshold、Stage 10 readiness |
| `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | 选中的 eye specialist checkpoint | 后续 Stage 10 runtime candidate |

## 3. End-to-End System Design

### 3.1 已实现事实（Implemented Facts）

当前系统已经形成了“两个 specialist classifiers + runtime temporal analysis + Stage 17 uploaded-video warning-candidate UI + Stage 18 frontend history review UI + Stage 19 Live Monitor realtime prototype”的本地 MVP：

1. YawDD/YawDD+ Dash mouth/yawn specialist：输入嘴部裁剪图，输出 `p_yawn`。
2. MRL Eye open/closed specialist：输入眼部图像，输出 `p_eye_closed`。
3. Stage 10-15 runtime pipeline：对视频帧抽样，提取 eye/mouth ROI，生成 `p_eye_closed` 和 `p_yawn` timelines，并做 rule-based fusion。
4. Stage 17.3/17.5 Video Upload Analysis UI：通过 FastAPI `/api/analyze-video` 分析上传视频，并在 Next.js `/video-upload` 中展示 compact overview、summary metrics、expandable warning-candidate intervals、tabbed figures、keyframes 和 technical files。
5. Stage 18 48h History UI：在 Next.js `/history-48h` 中用 demo/local `localStorage` 数据展示 recent warning-candidate history、charts、event timeline、sessions 和 manual review queue。
6. Stage 19 Live Monitor：在 Next.js `/` 中使用 browser webcam preview、2 FPS frame sampling、FastAPI realtime frame evidence endpoints 和 session-local temporal warning-candidate state。

当前已经完成的是 rule-based warning-candidate analysis、frontend evidence/history review 和本地 realtime warning-candidate prototype，而不是最终系统级疲劳真值。也就是说，项目现在可以输出或展示 `normal`、`eye_warning_candidate`、`mouth_warning_candidate`、`high_confidence_drowsiness_candidate` 和 `signal_unreliable` 等 warning-candidate states；但它仍然不是 alarm system，不是可部署系统，也不报告 final system-level drowsiness accuracy。

### 3.2 系统设计解释（Inferred Interpretation）

模块化系统设计（Modular System Design）的核心思想是：先让每个模型解决一个更明确、更可监督的视觉子任务，再把这些子任务的输出组合起来。

- 嘴部模块学习“嘴部 ROI 是否表现为打哈欠”。
- 眼部模块学习“眼睛图像是否闭合”。
- Stage 17.1 rule-based fusion 用 recent mouth/yawn evidence 和 sustained eye-warning evidence 生成 high-confidence warning candidates。
- Stage 17.2 interpretation layer 约束解释措辞，避免把 eye-warning evidence 自动写成 verified sustained full eye closure。

后期融合（Late Fusion）指不直接把原始图像拼在一起训练一个大模型，而是融合各 specialist classifier 的 probability outputs，例如：

```text
frame_t -> mouth model -> p_yawn[t]
frame_t -> eye model   -> p_eye_closed[t]
[p_yawn over time, p_eye_closed over time]
  -> Stage 17.1 rule-based fusion
  -> warning-candidate states and intervals
```

时间融合（Temporal Fusion）很关键，因为驾驶疲劳相关线索是时序状态：一次短暂闭眼可能是眨眼，连续较长时间闭眼或频繁闭眼才可能成为 eye-warning evidence；单次张嘴可能是说话或短暂动作，recent mouth/yawn evidence 需要和 sustained eye-warning evidence 组合后才会升级为 high-confidence warning candidate。Stage 17.1 的 sustained-eye gate 专门用于抑制 brief blink-like activity 和 recent-yawn 重叠时的过度升级。

### 3.3 安全关键分类（Safety-Critical Classification）

眼部模块中的 `false_open` 是安全关键错误（Safety-Critical Error）：真实闭眼却预测为开眼，可能让系统错过闭眼片段。`false_closed` 则更像 false alarm tendency：真实开眼却预测为闭眼，可能导致误报警。二者代价不同，因此不能只看 accuracy。

## 4. Dataset Strategy

### 4.1 已实现事实（Implemented Facts）

| 数据集 | 当前用途 | 状态 |
|---|---|---|
| YawDD Dash + YawDD+ annotations | 嘴部/打哈欠 specialist | 当前主线，Stage 7 completed |
| MRL Eye | 眼睛开闭 specialist | 当前主线，Stage 8/9/9B completed |
| Official NTHUDDD2 | 曾考虑 | 因 access constraints 未作为主线使用 |
| Kaggle NTHUDDD2 extracted frames | 曾探索 | 仅探索，不是当前 final active model branch |

YawDD/YawDD+ 的价值是提供驾驶员 Dash 视频和帧级 yawn/no-yawn 标签。MRL Eye 的价值是提供眼睛开闭标签和 subject folders，适合做 subject-level split。

### 4.2 为什么 subject-level split 重要

主体级划分（Subject-Level Split）要求同一个 subject 的所有图像只能出现在 train、validation 或 test 的其中一个 split 中。这样可以降低身份泄漏（Identity Leakage）和帧泄漏（Frame Leakage）。

如果使用 random frame-level split，模型可能在训练集中看到某个驾驶员的相邻帧、相似光照、相同摄像头角度，然后在测试集中再次看到同一个人的近似帧。这会让 test accuracy 虚高，因为模型可能学到了 subject identity 或背景模式，而不是真正泛化到新驾驶员。

### 4.3 数据集偏差（Dataset Bias）

即使 subject-level split 通过，也不代表模型已经覆盖真实世界。YawDD、MRL Eye 和真实车载摄像头可能在 camera angle、lighting、ethnicity、glasses、motion blur、ROI framing 等方面存在 domain gap。因此当前结果代表“在当前数据集和划分协议上的 specialist performance”。

## 5. YawDD/YawDD+ Mouth/Yawn Data Pipeline

### 5.1 Pipeline 总览

| Stage / Step | 已做什么 | 为什么重要 | 证据路径 | 学到的概念 |
|---|---|---|---|---|
| Raw Dash inspection | 检查 `dataset/YawDD_raw/` 下 29 个 Dash `.avi` 视频 | 确认 raw source 和 subject token | `reports/yawdd_raw_dash_report.md` | Dataset Inspection |
| Annotation inspection | 检查 YawDD+ `.txt` annotation 格式 | 理解 frame index、YOLO bbox、class id | `reports/yawdd_plus_annotation_format_report.md` | Annotation Parsing, Class Mapping |
| Visual sanity check | 抽样解码帧并叠加 bbox | 确认 `1=yawn`, `0=no_yawn`，发现 bbox 不适合作 mouth ROI | `reports/yawdd_dash_visual_sanity_check.md` | Visual Sanity Check, ROI Quality |
| Frame reconstruction | 用 YawDD+ frame indices 从原视频重建 Dash frames | 让 annotation 能对应实际图片 | `reports/yawdd_dash_reconstruction_report.md` | Frame Reconstruction |
| Mouth ROI cropping | 用 MediaPipe Face Mesh lip landmarks 生成 mouth crops | 让模型聚焦嘴部，而不是 torso/背景 | `reports/yawdd_dash_mouth_crop_report.md`, `src/preprocessing/generate_yawdd_mouth_crops.py` | Facial Landmarks, Mouth ROI Cropping |
| Fallback crop | Face Mesh 失败时用 lower-face fallback | 提高可用样本数量，同时记录 crop_method | 同上 | Fallback Crop, QC |
| Subject-level split | 按 subject_id 划分 train/val/test | 防止同一 subject 泄漏到多个 split | `reports/yawdd_dash_split_report.md` | Leakage Prevention |

### 5.2 Source data

已核查的 source data：

- Original YawDD Dash videos：`dataset/YawDD_raw/`
- YawDD+ annotation files：`dataset/YawDD+/`
- Reconstructed frames / mouth crops：`dataset/YawDD_plus_reconstructed/`

`reports/yawdd_raw_dash_report.md` 显示 Dash 原始数据包含 29 个 `.avi` 文件：13 female、16 male。部分文件有 `.avi.avi` 或空格命名异常，但 canonical subject token 可以和 YawDD+ subject folders 一一对应。

### 5.3 Annotation interpretation and class mapping

YawDD+ annotation 文件位于 subject folder 的 `labels/` 下。文件名形如：

```text
<8-digit frame index>_<object index>.txt
```

每个文件内容是 YOLO-style bounding box：

```text
<class_id> <x_center> <y_center> <width> <height>
```

已确认 class mapping：

| Class ID | Label |
|---:|---|
| `0` | `no_yawn` |
| `1` | `yawn` |

需要注意：annotation report 初期把 class semantics 标为需要视觉确认；后续 `reports/yawdd_dash_visual_sanity_check.md` 已通过抽样帧确认 class `1` 对应 yawning，class `0` 对应 non-yawning。

### 5.4 为什么原始 YawDD+ bbox 没作为 mouth crops

视觉 sanity check 发现 YawDD+ 原始 bbox 并不稳定定位嘴部或脸部，很多时候覆盖 torso region，例如从下巴到方向盘/身体区域。若直接用这些 bbox 训练，模型可能看不到嘴部 signal，或者学到背景/身体姿态。

因此当前 pipeline 的正确做法是：

1. 使用 YawDD+ annotation 的 frame index 和 class label。
2. 忽略 YawDD+ bbox geometry 作为裁剪依据。
3. 对重建帧重新运行 MediaPipe Face Mesh。
4. 用 lip landmarks 生成嘴部区域裁剪（Mouth ROI Cropping）。
5. 原始 bbox 仅作为 traceability metadata 保存。

### 5.5 MediaPipe Face Mesh mouth ROI

`src/preprocessing/generate_yawdd_mouth_crops.py` 中实现了 mouth ROI generation。脚本使用 MediaPipe FaceLandmarker 检测 facial landmarks，并用 outer lip + inner lip landmark indices 计算 mouth bounding box，再加入 margin 并裁剪到图像边界内。

如果 Face Mesh 没有返回 landmarks，则 fallback 到 OpenCV Haar frontal face detector，并取 face box 下方约 40% 作为 lower-face crop。如果两者都失败，则记录为 `failed`，不保存 crop。

### 5.6 Mouth-crop preprocessing summary

`reports/yawdd_dash_mouth_crop_report.md` 的结果：

| Metric | Value |
|---|---:|
| Total frames processed | 64,378 |
| MediaPipe Face Mesh crops | 64,093 |
| Fallback lower-face crops | 109 |
| Failed crops | 176 |
| Saved trainable crops | 64,202 |
| Success rate | 99.73% |

Saved crop class distribution：

| Class | Count |
|---|---:|
| `no_yawn` | 57,171 |
| `yawn` | 7,031 |

解释：`failed` rows 被排除在 trainable data 之外；`face_mesh` 和 `fallback_lower_face` rows 才进入后续 subject-level split。

### 5.7 Subject-level split and leakage checks

`reports/yawdd_dash_split_report.md` 的 split distribution：

| Split | Subjects | Images | `no_yawn` | `yawn` | Yawn rate |
|---|---:|---:|---:|---:|---:|
| train | 20 | 44,156 | 39,345 | 4,811 | 10.90% |
| val | 4 | 8,892 | 7,902 | 990 | 11.13% |
| test | 5 | 11,154 | 9,924 | 1,230 | 11.03% |

Leakage checks passed：

- No subject appears in more than one split.
- Every split contains both classes.
- All referenced mouth-crop files exist.
- No failed crop rows in trainable data.

## 6. YawDD/YawDD+ Mouth/Yawn Training Pipeline

### 6.1 已实现事实（Implemented Facts）

YawDD Stage 7 在 `colab_file/stage7_yawdd_training_r.ipynb` 中完成。训练输入是 Stage 6 subject-level split 之后的 mouth crops，任务是 binary image classification：`no_yawn` vs `yawn`。

| Setting | Value |
|---|---|
| Framework | PyTorch / torchvision |
| Input | Mouth crops from `artifacts/splits/yawdd_dash_subject_split.csv` |
| Labels | `no_yawn`, `yawn` |
| Image size | `224 x 224` |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Loss | Weighted Cross Entropy |
| Batch size | 32, with practical fallback to 16 |
| Epochs | 12 |
| Early stopping patience | 3 |
| Scheduler | ReduceLROnPlateau |
| Augmentation | mild rotation, brightness/contrast jitter, slight affine scaling |
| Transfer learning | freeze backbone first, then fine-tune full model |
| Architectures | ResNet18, MobileNetV2, EfficientNet-B0 |

### 6.2 训练机制解释（Project-Specific Interpretation）

- 输入图像是嘴部 ROI，而不是整张驾驶员图像。这让 CNN 更专注于 mouth opening、teeth、lip shape 等与 yawn 有关的局部特征。
- 图像 resize 到 `224 x 224` 是因为 torchvision pretrained backbones 通常使用 ImageNet-style input resolution。
- 数据增强（Data Augmentation）只用于训练集，模拟轻微姿态、亮度和尺度变化，帮助模型不要过拟合固定 crop 外观。
- 加权交叉熵（Weighted Cross Entropy）用于处理 class imbalance：`no_yawn` 明显多于 `yawn`。
- 冻结再微调（Freeze-then-Fine-Tune）先训练 classifier head，再释放 backbone，使迁移学习更稳定。
- Early Stopping 用 validation performance 判断是否停止，降低过拟合风险。
- ReduceLROnPlateau 在 validation metric 进入平台期时降低 learning rate。

### 6.3 三个 CNN baseline 的学习意义

| Architecture | 模型家族 | 为什么是合理 baseline | 权衡 | 如何适合 mouth/yawn task |
|---|---|---|---|---|
| ResNet18 | Residual CNN | 结构经典、稳定、容易解释 | 比 MobileNetV2 大，但仍较轻 | 对 mouth texture/shape 有强 baseline 表现 |
| MobileNetV2 | Lightweight CNN | 面向移动端和实时推理 | 更轻更快，但精度可能略低 | 适合作 future real-time mouth specialist candidate |
| EfficientNet-B0 | Efficiency-oriented CNN | 用 compound scaling 追求精度/效率平衡 | 结构更复杂，部署需考虑 runtime | 可检验更现代 backbone 是否提升 yawn recognition |

ResNet18 的核心是残差连接（Residual Connection），帮助较深网络训练更稳定。MobileNetV2 使用轻量化思想，包括 depthwise separable convolution 和 inverted residual ideas。EfficientNet-B0 使用 compound scaling，在 depth、width、resolution 之间做系统平衡。

## 7. YawDD/YawDD+ Results Interpretation

### 7.1 Source-of-truth caveat

`artifacts/results/initial_results.csv` 当前是 stale / not valid source：文件中显示 `N/A` 或 “split manifest has no samples” 等旧错误状态。因此不能把它当作 Stage 7 final results。

已核查的 Stage 7 completed results 来自 `colab_file/stage7_yawdd_training_r.ipynb`。该 notebook 显示 Stage 7 completion status 为 `SUCCESS`，并打印了 final metrics table。

### 7.2 Stage 7 results

| CNN Architecture | Train Accuracy | Validation Accuracy | Test Accuracy | Yawn Precision | Yawn Recall | Yawn F1 |
|---|---:|---:|---:|---:|---:|---:|
| CNN-1: ResNet18 | 98.92% | 98.85% | 99.37% | 96.47% | 97.89% | 97.18% |
| CNN-2: MobileNetV2 | 98.97% | 98.48% | 98.75% | 91.74% | 97.48% | 94.52% |
| CNN-3: EfficientNet-B0 | 98.76% | 99.08% | 99.20% | 94.82% | 98.13% | 96.44% |

### 7.3 如何解释结果

ResNet18 achieved the strongest Stage 7 test accuracy。EfficientNet-B0 achieved the strongest validation accuracy。这个差异说明 model selection 不应只看单一指标：validation accuracy 可以指导训练过程和早停，但最终 test set 是一次性评估的 held-out evidence。

Yawn Precision 表示模型预测为 `yawn` 的样本里有多少是真的 yawn。Yawn Recall 表示所有真实 yawn 样本中有多少被模型抓到。Yawn F1 是 precision 和 recall 的调和平均，适合在类别不平衡时辅助解释。

因为 `no_yawn` 数量远多于 `yawn`，accuracy 很高时仍需要看 yawn recall 和 yawn precision。如果模型总是偏向 `no_yawn`，accuracy 可能仍然不错，但 yawn recall 会变差。

### 7.4 不应过度声明

这些结果只能称为 YawDD/YawDD+ mouth/yawn specialist results。它们不是最终 driver drowsiness detection accuracy，因为：

- 输入是 mouth crops，不是完整实时系统。
- 输出是 `p_yawn`，不是 fatigue score。
- 没有 temporal fusion。
- 没有与 eye module 组合。
- 没有完整 warning-state evaluation。

## 8. MRL Eye Data Pipeline

### 8.1 Dataset root and structure

MRL Eye dataset root：

```text
dataset/mrlEyes_2018_01/
```

已核查结构包含 `annotation.txt`、`stats_2018_01.ods` 和 `s0001/` 到 `s0037/` subject folders。标签映射（Label Mapping）：

| Label | Meaning |
|---:|---|
| `0` | `closed` |
| `1` | `open` |

### 8.2 Stage 8 dataset preparation

`reports/mrl_eye_dataset_report.md` 显示：

| Metric | Value |
|---|---:|
| Total images | 84,898 |
| Trainable images | 84,898 |
| Subjects | 37 |
| Closed images | 41,946 |
| Open images | 42,952 |
| Unreadable images | 0 |
| Unparseable filenames | 0 |

重要输出：

- `artifacts/mappings/mrl_eye_all_images.csv`
- `artifacts/mappings/mrl_eye_trainable.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`
- `artifacts/splits/mrl_eye_subject_split.csv`
- `reports/mrl_eye_dataset_report.md`
- `reports/mrl_eye_split_report.md`
- `artifacts/visual_checks/mrl_eye_closed_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_open_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_by_split_contact_sheet.jpg`

### 8.3 Subject-level split

`reports/mrl_eye_split_report.md` 显示：

| Split | Subjects | Images | Closed | Open |
|---|---:|---:|---:|---:|
| train | 25 | 58,982 | 29,310 | 29,672 |
| val | 6 | 13,029 | 6,333 | 6,696 |
| test | 6 | 12,887 | 6,303 | 6,584 |

Split checks passed：

- Leakage check result: `True`
- Missing split label check result: `True`
- Every image receives exactly one split: `True`
- Every split contains closed and open: `True`
- Missing file check result: `True`

### 8.4 每个阶段教你的概念

| Stage / Step | What was done | Why it matters | Evidence | Concept |
|---|---|---|---|---|
| Dataset inspection | 解析 subject folders、图片、annotation | 确认数据可读、标签可用 | `reports/mrl_eye_dataset_report.md` | Dataset Manifest |
| Label validation | 确认 `0=closed`, `1=open` | 避免训练时标签反转 | 同上 | Binary Label |
| Trainable manifest | 构建所有可训练图像 CSV | 给 DataLoader 提供统一输入 | `artifacts/mappings/mrl_eye_trainable.csv` | Manifest-based Training |
| Subject-level split | 按 subject 分 train/val/test | 减少 identity leakage | `reports/mrl_eye_split_report.md` | Subject-Level Split |
| Visual contact sheets | 保存 closed/open/split 可视化样本 | 人眼检查标签和图像质量 | `artifacts/visual_checks/` | Visual Inspection |

## 9. MRL Eye Training Pipeline

### 9.1 已实现事实（Implemented Facts）

Stage 9 使用 `src/training/train_mrl_eye_baselines.py` 和 `colab_file/stage9_mrl_eye_training_r.ipynb` 训练三个 eye-state specialist baselines。

| Setting | Value |
|---|---|
| Framework | PyTorch / torchvision |
| Input manifest | `artifacts/mappings/mrl_eye_trainable_with_split.csv` |
| Image size | 224 |
| Batch size | 64 |
| Epochs | 10 |
| Freeze epochs | 1 |
| Early stopping patience | 3 |
| Learning rate | `1e-4` |
| Loss | Weighted Cross Entropy from training split |
| Scheduler | ReduceLROnPlateau |
| Checkpoint metric | validation macro F1 |
| Pretrained weights | loaded for all three models |
| Mixed precision | enabled when CUDA was available |
| Architectures | ResNet18, MobileNetV2, EfficientNet-B0 |

### 9.2 Metrics included

Stage 9 reports：

- accuracy
- macro precision / macro recall / macro F1
- weighted F1
- per-class metrics
- confusion matrices
- false-open counts
- false-closed counts
- threshold sweeps for `p_eye_closed`

### 9.3 为什么 validation macro F1 合理

宏平均 F1（Macro F1）先分别计算 closed 和 open 两个类别的 F1，再平均。它不会像 weighted F1 一样被样本更多的类别主导。虽然 MRL Eye 总体 closed/open 数量接近平衡，但不同 subject 内部可能很不平衡，因此 macro F1 比单纯 accuracy 更适合作 checkpoint selection。

### 9.4 false-open 和 false-closed 为什么不同

在本项目中：

- `false_open`: ground truth closed, predicted open。安全关键，因为模型漏掉闭眼帧。
- `false_closed`: ground truth open, predicted closed。更像误报倾向，因为模型把开眼帧说成闭眼。

降低 `p_eye_closed` threshold 会让模型更容易预测 closed，因此通常会提高 closed-eye recall、减少 false_open，但会增加 false_closed。这个 trade-off 必须根据 safety objective、Stage 17.1 sustained-eye gate 和后续 temporal analysis 需求决定。

## 10. MRL Eye Results and Stage 9B Model Selection

### 10.1 Stage 9 argmax results

`outputs/mrl_eye/results/mrl_eye_initial_results.csv` 和 `reports/mrl_eye_stage9b_error_analysis.md` 显示：

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Test Macro F1 | Test Closed Recall | False Open | False Closed |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 99.16% | 98.37% | 98.46% | 98.46% | 98.59% | 89 | 109 |
| MobileNetV2 | 99.33% | 97.91% | 98.63% | 98.63% | 98.52% | 93 | 84 |
| EfficientNet-B0 | 99.44% | 97.91% | 98.62% | 98.62% | 98.24% | 111 | 67 |

### 10.2 Stage 9B threshold and selection results

`outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` 显示：

| Selection item | Value |
|---|---|
| Primary selected model | `mobilenet_v2` |
| Recommended default threshold | `0.50` |
| Recommended default rule | argmax / `p_eye_closed >= 0.50` |
| Safety-prioritized reference | ResNet18 with `p_eye_closed >= 0.30` |
| Stage 10 status | `READY` |
| Required checkpoint | `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` |
| Checkpoint found | `true` |

Threshold comparison from Stage 9B：

| Model | Selected From Val | Test Macro F1 @0.50 | Closed Recall @0.50 | False Open @0.50 | False Closed @0.50 | Test Macro F1 @Selected | Closed Recall @Selected | False Open @Selected | False Closed @Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.30 | 98.46% | 98.59% | 89 | 109 | 97.60% | 99.08% | 58 | 251 |
| MobileNetV2 | 0.30 | 98.63% | 98.52% | 93 | 84 | 98.48% | 98.79% | 76 | 120 |
| EfficientNet-B0 | 0.30 | 98.62% | 98.24% | 111 | 67 | 98.52% | 98.65% | 85 | 106 |

### 10.3 为什么 MobileNetV2 是 primary eye model

MobileNetV2 被选为 primary model，因为它在默认 argmax / `0.50` 决策下具有最好的整体 test accuracy 和 macro F1，同时模型更轻量，更符合实时或近实时推理（Real-Time Suitability）的部署方向。它的 default test metrics 是：

- Test accuracy: 98.63%
- Test macro F1: 98.63%
- Closed-eye recall: 98.52%
- False open: 93
- False closed: 84

### 10.4 为什么保留 ResNet18 threshold 0.30

ResNet18 at threshold `0.30` 是 safety-prioritized reference。它把 false_open 从 89 降到 58，并把 closed-eye recall 从 98.59% 提升到 99.08%。代价是 false_closed 从 109 增加到 251，macro F1 下降到 97.60%。

因此它适合作“更保守、更少漏掉闭眼”的参考方案，但不作为默认方案，因为 false alarm tendency 明显增加。

### 10.5 Threshold Selection 的方法论

阈值候选必须从 validation sweeps 中选择。test set 只能用于 final reporting，不能用于调参。如果看了 test sweep 后再选择 threshold，就是 test-set tuning，会污染最终评估。

## 11. CNN Architecture Knowledge Map

本项目中的 CNN 概念应该通过三个实际 backbone 来理解，而不是脱离项目泛泛学习。

| Concept | 中文解释 | 在本项目中的具体连接 |
|---|---|---|
| Convolutional Layer | 卷积层，用 kernel/filter 在局部图像窗口上提取模式 | 从 mouth crops 中学习嘴部形状，从 eye crops 中学习眼睑/眼裂纹理 |
| Kernel / Filter | 卷积核，学习边缘、纹理、局部形状 | 可学习 lip edge、teeth contrast、eyelid boundary |
| Feature Map | 特征图，卷积输出的空间响应 | 深层 feature map 表示更抽象的 yawn/open/closed clues |
| Channel | 通道，RGB 输入或中间特征维度 | pretrained ImageNet backbone 输入 RGB，即使 MRL Eye 图像视觉上可能接近灰度 |
| Stride | 步幅，控制卷积或 pooling 移动距离 | 影响 feature map spatial resolution |
| Padding | 填充，保持边界信息 | 对 mouth/eye crop 边缘线索有影响 |
| Pooling | 下采样，减少空间尺寸 | 提高平移鲁棒性，但过强下采样可能损失小眼部细节 |
| Batch Normalization | 批归一化，稳定训练 | ResNet/MobileNet/EfficientNet backbone 中常见 |
| Activation Function | 激活函数，引入非线性 | 让网络学习复杂视觉模式而不是线性分类 |
| Fully Connected Layer | 全连接层，最终分类 head | 本项目把 pretrained head 替换为 2-class classifier head |
| Global Average Pooling | 全局平均池化，把空间特征聚合成向量 | 现代 CNN 常在 classifier 前使用 |
| Residual Connection | 残差连接，学习 identity + residual | ResNet18 的核心，让训练更稳定 |
| Depthwise Separable Convolution | 深度可分离卷积，减少参数和计算 | MobileNetV2 的轻量化基础 |
| Inverted Residual | 倒残差结构，MobileNetV2 设计思想 | 支持移动端友好特征提取 |
| Compound Scaling | 同时缩放 depth/width/resolution | EfficientNet-B0 的效率设计思想 |
| Transfer Learning | 迁移学习，用 pretrained backbone | 两个 specialist modules 都使用 torchvision pretrained models |
| Fine-Tuning | 微调，在新任务上继续训练 backbone | 先 freeze，再 full fine-tune，适配 mouth/eye task |

项目级理解：

- ResNet18 代表稳定、强基线、易解释的 residual learning。
- MobileNetV2 代表轻量、部署友好、实时推理候选。
- EfficientNet-B0 代表效率导向的现代 CNN scaling 思路。

## 12. Training Mechanics Knowledge Map

| Concept | 中文解释 | 在本项目中的位置 |
|---|---|---|
| Dataset / Manifest | 用 CSV 描述每张图像路径、标签、split、subject | `artifacts/splits/yawdd_dash_subject_split.csv`, `artifacts/mappings/mrl_eye_trainable_with_split.csv` |
| DataLoader | PyTorch 批量读取数据并应用 transforms | Stage 7/9 training notebooks/scripts |
| Batch | 一次训练迭代中的样本集合 | YawDD batch size 32/16 fallback；MRL Eye batch size 64 |
| Epoch | 模型完整遍历训练集一次 | YawDD up to 12；MRL Eye 10 |
| Forward Pass | 输入图像经过 CNN 得到 logits/probabilities | 输出 `p_yawn` 或 `p_eye_closed` 的基础 |
| Loss Function | 衡量预测和标签差异 | Weighted Cross Entropy |
| Backpropagation | 反向传播计算梯度 | PyTorch 自动求导完成 |
| Gradient | 参数更新方向 | optimizer 根据 gradient 更新模型 |
| Optimizer | 根据 gradient 更新参数 | Adam |
| Learning Rate | 每次更新步长 | `1e-4` |
| Scheduler | 动态调整 learning rate | ReduceLROnPlateau |
| Early Stopping | validation 不再提升时停止 | patience 3 |
| Checkpointing | 保存最佳模型权重 | MRL Eye checkpoints under `outputs/mrl_eye/checkpoints/` |
| Validation Loop | 训练中用于选择模型/早停 | YawDD validation accuracy；MRL Eye validation macro F1 |
| Test Evaluation | 最终 held-out 评估 | 当前报告中的 test accuracy/macro F1 等 |
| Mixed Precision | 使用半精度加速 CUDA training | MRL Eye Stage 9 在 CUDA available 时启用 |

## 13. Metrics and Error Analysis Knowledge Map

| Metric / Concept | 中文解释 | 本项目中如何使用 |
|---|---|---|
| Accuracy | 总体预测正确比例 | 有用，但不能单独作为安全系统判断依据 |
| Precision | 预测为某类时有多少是真的 | Yawn precision 说明 predicted yawn 的可靠性 |
| Recall | 真实某类中有多少被找出 | Yawn recall / closed recall 对漏检很重要 |
| F1-score | precision 和 recall 调和平均 | 类别不平衡时比 accuracy 更有信息量 |
| Macro F1 | 每类 F1 平均，不按样本数加权 | MRL Eye checkpoint metric |
| Weighted F1 | 按 support 加权的 F1 | 反映整体分布下的 F1 |
| Confusion Matrix | 真实类别 vs 预测类别矩阵 | 分析 false_open/false_closed |
| Per-Class Metrics | 每个类别单独 precision/recall/F1 | 避免平均指标掩盖 minority class |
| False Open | true closed, predicted open | 安全关键漏检闭眼帧 |
| False Closed | true open, predicted closed | 误报警倾向 |
| Threshold Sweep | 扫描 `p_eye_closed` 阈值 | 分析 closed recall 和 false alarm trade-off |
| Safety-Oriented Evaluation | 根据错误代价解释指标 | 眼部模块不能只追求最高 accuracy |

### 13.1 为什么 accuracy alone insufficient

驾驶员监控是 safety-related task。假设一个模型 accuracy 很高，但它系统性漏掉闭眼帧，后续 warning-candidate analysis 的证据质量会变差。特别是在 Stage 17.1 sustained-eye gate 中，漏掉连续 eye-warning frames 可能影响 high-confidence warning-candidate escalation。因此必须结合 recall、false_open、threshold sweep、temporal evidence 和 error analysis。

### 13.2 三个层级的 performance

| 层级 | 含义 | 当前是否完成 |
|---|---|---|
| Model-level performance | 单个 CNN 在给定数据集 split 上的分类性能 | 已完成 |
| Specialist-module performance | mouth/yawn 或 eye open/closed module 的任务性能 | 已完成 |
| Rule-based warning-candidate performance | 多模块、时间融合、Stage 17.1/17.5 rule-based fusion/interpretation 在上传视频中的 warning-candidate 输出 | 已完成本地 MVP，但需要更多视频和人工复核 |
| Final system-level performance | 真实系统级疲劳真值、最终准确率、部署级监控性能 | 未完成，future work |

## 14. What Has Been Completed vs What Is Future Work

| Area | Current Status | Evidence / artifact path | What I should learn from it | Future work |
|---|---|---|---|---|
| YawDD frame reconstruction | Completed / READY | `reports/yawdd_dash_reconstruction_report.md`, `artifacts/mappings/yawdd_dash_all_labeled_frames.csv` | frame index 如何从 annotation 对齐原视频 | 无需重建；未来只需复核 source-of-truth |
| YawDD mouth crop generation | Completed / READY | `reports/yawdd_dash_mouth_crop_report.md`, `src/preprocessing/generate_yawdd_mouth_crops.py` | MediaPipe Face Mesh 和 fallback crop | 未来 runtime mouth ROI consistency |
| YawDD subject-level split | Completed / READY | `reports/yawdd_dash_split_report.md`, `artifacts/splits/yawdd_dash_subject_split.csv` | subject-level leakage prevention | 如数据更新才需重建 split |
| YawDD Stage 7 training | Completed | `colab_file/stage7_yawdd_training_r.ipynb` | transfer learning、weighted loss、CNN comparison | 不把 stale CSV 当结果；未来可做 error analysis |
| MRL Eye dataset preparation | Completed / READY | `reports/mrl_eye_dataset_report.md`, `artifacts/mappings/mrl_eye_all_images.csv` | MRL label parsing、manifest construction | 未来 runtime eye crop domain check |
| MRL Eye subject-level split | Completed / READY | `reports/mrl_eye_split_report.md`, `artifacts/splits/mrl_eye_subject_split.csv` | subject-level split with balanced classes | 如数据更新才需重建 split |
| MRL Eye Stage 9 training | Completed | `outputs/mrl_eye/results/mrl_eye_initial_results.csv`, `colab_file/stage9_mrl_eye_training_r.ipynb` | validation macro F1、threshold sweep、confusion matrix | 不需新训练即可进入 Stage 10 |
| MRL Eye Stage 9B model selection | Completed / Stage 10 READY | `reports/mrl_eye_stage9b_error_analysis.md`, `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` | MobileNetV2 default vs ResNet18 safety reference | 已进入 Stage 10+ runtime pipeline |
| NTHUDDD2 exploration | Explored only, not main direction | `reports/nthuddd2_kaggle_dataset_report.md` | Kaggle extracted-frame limitations | 不作为 final active model branch |
| Stage 10 eye ROI runtime consistency | Completed controlled validation | `reports/stage10_runtime_eye_roi_acceptance_report.md`, `outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/` | runtime eye ROI、failure handling、controlled-video acceptance | 需要更多视频域验证 |
| Stage 11 eye temporal analysis | Completed controlled validation | `src/runtime/stage11_eye_temporal_analysis.py` | smoothing、temporal windows、eye timeline | 作为 fusion 输入继续维护 |
| Stage 12 eye alert rule analysis | Completed controlled validation | `src/runtime/stage12_eye_alert_rule_analysis.py` | eye-only alert rule comparison | 不单独作为 final system |
| Stage 13 fusion design | Completed prototype | `reports/stage13_mouth_eye_fusion_design_report.md`, `docs/STAGE13_MOUTH_EYE_FUSION_DESIGN.md` | fusion state schema、rule comparison | 后续被 Stage 15/17 实测管线吸收 |
| Stage 14 mouth/yawn runtime | Completed controlled validation | `reports/stage14_mouth_yawn_runtime_validation_report.md` | recovered mouth checkpoint、runtime mouth ROI、`p_yawn` timeline | 需要更多真实视频验证 |
| Stage 15 real synchronized fusion | Completed rule-based validation | `reports/stage15_real_mouth_eye_fusion_validation_report.md`, `outputs/stage15_real_mouth_eye_fusion/` | Stage 12 eye timeline + Stage 14 mouth timeline 如何融合 | 不声明 final drowsiness truth |
| Stage 17.1 sustained-eye gate | Completed | `reports/stage17_video_upload_detection_mvp_report.md`, `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md` | recent mouth/yawn evidence + sustained eye-warning evidence 才升级 high-confidence | 保持 rule-based，不改成 trained fusion |
| Stage 17.2 interpretation wording | Completed | `reports/stage17_2_manual_review_interpretation_report.md` | eye-warning evidence 的安全解释边界 | 演示和 UI 必须保持 safe wording |
| Stage 17.3 Video Upload Analysis UI | Completed local MVP | `SystemUI/src/app/video-upload/page.tsx`, `docs/STAGE17_3_VIDEO_UPLOAD_UI_PAGE_REPORT.md` | upload workstation、summary、intervals、figures、keyframes、technical files | 后续被 Stage 17.5 UI cleanup 改进 |
| Stage 17.4 launcher and acceptance docs | Completed | `scripts/start_stage17_ui.sh`, `docs/STAGE17_4_VIDEO_UPLOAD_UI_ACCEPTANCE_CHECKLIST.md`, `reports/stage17_4_video_upload_mvp_stabilization_report.md` | `make stage17-ui` 一键启动、demo script、acceptance checklist | 历史稳定包，当前由 Stage 17.5/18 supersede |
| Stage 17.5 eye evidence calibration | Completed | `docs/STAGE17_5_EYE_EVIDENCE_CALIBRATION.md`, `reports/stage17_5_eye_evidence_calibration_report.md`, `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md` | weak/moderate/strong eye evidence、interval eye-strength gate、weak-eye suppression | 不 retrain，不改 `p_eye_closed` / `p_yawn` 公式 |
| Stage 17.5 `/video-upload` UI evidence cleanup | Completed | `docs/STAGE17_5_VIDEO_UPLOAD_UI_SECOND_PASS_CLEANUP.md`, `SystemUI/src/components/video-upload/` | compact interval table、metric scope、fusion state vs descriptive evidence、tabbed figures | 继续保持 safe wording 和 optional-field clarity |
| Stage 18 `/history-48h` frontend history page | Completed frontend-only | `SystemUI/src/app/history-48h/page.tsx`, `SystemUI/src/components/history-48h/`, `docs/STAGE18_HISTORY_48H_UI_PAGE_PLAN.md` | localStorage demo history、filters、charts、timeline、manual review queue | 下一步可接入真实 upload history storage；不代表 webcam 已实现 |
| Stage 19 `/` Live Monitor webcam capture/sampling | Completed local prototype | `SystemUI/src/app/page.tsx`, `SystemUI/src/components/dashboard/LiveVideoCard.tsx` | getUserMedia、mirror preview、2 FPS canvas JPEG sampling、session cleanup | display mirror 不影响 backend raw frame evidence |
| Stage 19 realtime single-frame evidence API | Completed local prototype | `src/runtime/realtime_frame_inference.py`, `src/backend/app.py` | `/api/realtime/health`、session start/frame/stop、`p_eye_closed`、`p_yawn`、ROI/signal quality | 不调用 `/api/analyze-video`，不写 history |
| Stage 19 realtime temporal warning-candidate state | Completed local prototype | `src/runtime/realtime_temporal_state.py`, `SystemUI/src/components/dashboard/LiveVideoCard.tsx` | `mouth_active`、recent yawn context/reminder、`eye_warning_active`、recent sustained eye-warning reminder | reminder 是 display-only，不驱动 high-confidence escalation |
| Final fatigue score | Future work | not available | 如何从 probabilities / warning candidates 到 risk score | 定义 final score 需要额外验证，不能从 Stage 17 直接声明 |
| Alert debounce, alarm output, and webcam history ingestion | Future work | not available | alarm policy、cooldown、manual review、persistence schema | 不要从当前 Stage 19 state 直接写 history 或触发 alarm |
| Deployment | Future work | not available | packaging、hardware tests、security、monitoring | 当前不能声明 deployment readiness |

## 15. Project-Specific Risks and Caveats

| Caveat | Why it matters | How project currently addresses it | What to verify before reporting |
|---|---|---|---|
| Specialist accuracy is not final drowsiness accuracy | `p_yawn`/`p_eye_closed` 不是 fatigue score | 文档明确标注 specialist-module results | 报告标题和结论不要写成 final system accuracy |
| Stale result files | 旧 CSV 可能误导结论 | Stage 7 指定 notebook 为 source of truth | 不使用 `artifacts/results/initial_results.csv` 的 stale rows |
| Random frame-level split misleading | 相邻帧/同 subject 会泄漏 | YawDD 和 MRL Eye 都用 subject-level split | 检查 split report 的 leakage checks |
| Subject-level split reduces but does not eliminate generalization concerns | 数据集仍可能有 domain bias | split 按 subject 做，且有 visual checks | 报告中说明 cross-dataset/real-world still needs verification |
| Mouth/yawn alone does not prove drowsiness | 张嘴可能说话、表情等 | mouth module 仅输出 `p_yawn` | 不把 yawn classifier 当完整 fatigue detector |
| Eye closed/open alone does not prove drowsiness | 单帧闭眼可能是眨眼 | eye module 仅输出 `p_eye_closed` | 需要 temporal fusion 或 PERCLOS-like logic |
| Threshold choices involve safety trade-offs | 降低 threshold 减少 false_open 但增加 false_closed | Stage 9B 提供 default 和 safety reference | 只从 validation 选 threshold，test 只报告 |
| Dataset distributions may differ from real driving | camera/lighting/ROI/domain gap | 当前有 visual checks 和 subject splits | 未来 runtime data 必须做 sanity check |
| MediaPipe mouth crops can fail or fallback | crop_method 不同可能影响训练 | failed rows 排除，fallback rows 记录 | 复核 fallback contact sheets 和 failure distribution |
| High accuracy may hide systematic errors | 某些 subject/glasses/reflection 可能高错 | MRL Eye error contact sheets 已生成 | presentation 前可做更细 qualitative taxonomy |
| Checkpoints should not be committed to normal Git | checkpoint binaries 大，污染 repo | artifact inventory 提醒 use Git LFS if needed | 确认 `.gitignore` / Git LFS policy |
| NTHUDDD2 explored but not main direction | 避免把探索分支写成 final system | docs/status 明确不是主线 | 报告中仅称 exploration |

## 16. How I Should Study This Project

下面是 module-based study flow，不是 day-by-day schedule。

### Module 1: Understand the system goal and modular design

Files to read：`docs/PROJECT_CURRENT_STATUS.md`, `docs/PROJECT_STRUCTURE.md`  
Concepts：Modular System Design, Specialist Classifier, Late Fusion, Temporal Fusion  
Questions：为什么项目不是一个 monolithic classifier？`p_yawn` 和 `p_eye_closed` 为什么不是 fatigue score？  
Evidence：当前状态文档中对 specialist modules、Stage 17 rule-based fusion 和 video-upload UI 的描述。  
Afterward you should explain：当前完成的是 specialist modules 和 uploaded-video warning-candidate MVP，不是 final system-level drowsiness accuracy。

### Module 2: Understand dataset strategy and subject-level split

Files to read：`reports/yawdd_dash_split_report.md`, `reports/mrl_eye_split_report.md`, `reports/nthuddd2_kaggle_dataset_report.md`  
Concepts：Subject-Level Split, Identity Leakage, Frame Leakage, Experimental Validity  
Questions：为什么 random frame split 会虚高？为什么 NTHUDDD2 Kaggle 不作为主线？  
Evidence：split distributions 和 leakage checks。  
Afterward you should explain：subject-level split 如何提高 cross-subject generalization 评估可信度。

### Module 3: Understand YawDD frame reconstruction and annotation interpretation

Files to read：`reports/yawdd_raw_dash_report.md`, `reports/yawdd_plus_annotation_format_report.md`, `reports/yawdd_dash_visual_sanity_check.md`, `reports/yawdd_dash_reconstruction_report.md`  
Concepts：Frame Reconstruction, Annotation Parsing, Class Mapping, Visual Sanity Check  
Questions：YawDD+ 文件名里的 frame index 如何对应 raw `.avi`？class `0/1` 如何确认？  
Evidence：29/29 subject frame index alignment, visual samples。  
Afterward you should explain：为什么可以用 annotation frame index 重建 labeled frames。

### Module 4: Understand MediaPipe mouth ROI preprocessing

Files to read：`src/preprocessing/generate_yawdd_mouth_crops.py`, `reports/yawdd_dash_mouth_crop_report.md`  
Concepts：Facial Landmarks, MediaPipe Face Mesh, Mouth ROI Cropping, Fallback Crop  
Questions：为什么不用 YawDD+ bbox？Face Mesh 失败时如何处理？  
Evidence：success rate 99.73%, crop_method counts, visual QC samples。  
Afterward you should explain：mouth ROI preprocessing 如何把 full frames 转成 trainable mouth crops。

### Module 5: Understand YawDD mouth/yawn CNN training

Files to read：`colab_file/stage7_yawdd_training_r.ipynb`, `src/training/train_classifier.py`, `src/training/run_initial_baselines.py`  
Concepts：CNN, Transfer Learning, Weighted Cross Entropy, Early Stopping  
Questions：为什么比较 ResNet18/MobileNetV2/EfficientNet-B0？为什么 yawn recall 很重要？  
Evidence：notebook final table。  
Afterward you should explain：ResNet18 为什么是当前 strongest Stage 7 test model。

### Module 6: Understand MRL Eye dataset preparation

Files to read：`reports/mrl_eye_dataset_report.md`, `reports/mrl_eye_split_report.md`, `src/data/inspect_mrl_eye.py`, `src/data/build_mrl_eye_manifest.py`, `src/data/split_mrl_eye_subjects.py`  
Concepts：Dataset Manifest, Binary Label, Class Balance, Visual Inspection  
Questions：MRL Eye label `0/1` 含义是什么？split 中每类是否都有样本？  
Evidence：84,898 images, 37 subjects, split checks。  
Afterward you should explain：Stage 8 如何把 MRL Eye 变成可训练 manifest。

### Module 7: Understand MRL Eye CNN training

Files to read：`reports/mrl_eye_stage9_training_plan.md`, `src/training/train_mrl_eye_baselines.py`, `colab_file/stage9_mrl_eye_training_r.ipynb`  
Concepts：Macro F1, Mixed Precision, Checkpoint Selection, Confusion Matrix  
Questions：为什么 checkpoint metric 是 validation macro F1？训练脚本如何计算 false_open？  
Evidence：Stage 9 notebook output 和 `outputs/mrl_eye/results/`。  
Afterward you should explain：三个 CNN 在 eye task 上的性能差异。

### Module 8: Understand threshold selection and safety trade-offs

Files to read：`reports/mrl_eye_stage9b_error_analysis.md`, `outputs/mrl_eye/results/*threshold_sweep.csv`, `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json`  
Concepts：Threshold Selection, Validation Set, Test Set, Safety Trade-off  
Questions：为什么 `0.30` 能减少 false_open？为什么 MobileNetV2 仍默认 `0.50`？  
Evidence：Stage 9B threshold table。  
Afterward you should explain：validation-selected threshold 和 test-tuned threshold 的区别。

### Module 9: Understand metrics and error analysis

Files to read：`outputs/mrl_eye/figures/`, `outputs/mrl_eye/error_analysis/`, `reports/mrl_eye_stage9b_error_analysis.md`  
Concepts：Per-Class Metrics, False Open, False Closed, Safety-Oriented Evaluation  
Questions：accuracy 高时还可能有什么系统性错误？哪些 visual patterns 造成 MRL Eye 错误？  
Evidence：confusion matrices 和 false-open/false-closed contact sheets。  
Afterward you should explain：为什么 safety task 不能只看 accuracy。

### Module 10: Understand Stage 17/17.5 rule-based fusion, UI, and system-level limitations

Files to read：`reports/stage15_real_mouth_eye_fusion_validation_report.md`, `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md`, `docs/STAGE17_5_EYE_EVIDENCE_CALIBRATION.md`, `docs/STAGE17_5_VIDEO_UPLOAD_UI_SECOND_PASS_CLEANUP.md`
Concepts：Rule-Based Fusion, Sustained-Eye Gate, Stage 17.5 Eye-Evidence Calibration, Warning-Candidate Interval Review, Safe Interpretation Wording
Questions：Stage 17.1 为什么要求 recent mouth/yawn evidence plus sustained eye-warning evidence？为什么 brief blink-like activity 要 suppressed from high-confidence escalation？为什么 Stage 17.5 UI 要区分 backend fusion state 和 descriptive eye evidence？
Evidence：Stage 17 schema、C upload validation markers、Stage 17.5 UI validation notes、UI acceptance checklist。
Afterward you should explain：Stage 17/17.5 输出的是 uploaded-video rule-based warning candidates；它不是 webcam，不是 final drowsiness truth，也不是可部署系统。

### Module 11: Understand Stage 18 frontend-only 48h history UI

Files to read：`docs/STAGE18_HISTORY_48H_UI_PAGE_PLAN.md`, `SystemUI/src/app/history-48h/page.tsx`, `SystemUI/src/lib/history48hTypes.ts`, `SystemUI/src/lib/history48hStorage.ts`, `SystemUI/src/lib/history48hUtils.ts`
Concepts：Frontend-Only History, `localStorage`, Demo/Local Data, Candidate Severity Display Score, Manual Review Queue
Questions：`/history-48h` 为什么只能说 warning-candidate history？为什么 localStorage demo history 不能当成 backend truth 或 webcam monitoring？
Evidence：Stage 18 implementation plan、history component files、manual validation checklist。
Afterward you should explain：Stage 18 当前是 history review UI，不是 active webcam capture、backend storage、final drowsiness result 或 deployment-ready monitoring。

## 17. Questions I Should Be Able to Answer

学习完本指南后，你应该能回答下面这些项目特定问题：

- 本项目的最终目标是什么？当前完成到哪个层级？
- 为什么当前设计是 modular system，而不是 monolithic classifier？
- `p_yawn` 和 `p_eye_closed` 分别表示什么？为什么它们不是最终 fatigue score？
- YawDD Dash raw videos 和 YawDD+ annotation files 如何组合使用？
- YawDD+ annotation filename 中的 frame index 为什么可以用来重建帧？
- YawDD+ class `0` 和 `1` 的含义是什么？这个语义是如何确认的？
- 为什么原始 YawDD+ YOLO bounding boxes 没有作为最终 mouth ROI？
- MediaPipe Face Mesh 在 mouth crop generation 中做了什么？
- lower-face fallback crop 什么时候使用？失败 crop 如何处理？
- YawDD mouth crops 的 success rate、class distribution 和 split distribution 是什么？
- 什么是主体级划分（Subject-Level Split）？它如何减少数据泄漏（Data Leakage）？
- 为什么 random frame-level split 在视频/帧数据中可能 misleading？
- YawDD Stage 7 使用了哪些 CNN architectures？为什么要比较它们？
- YawDD Stage 7 中 Weighted Cross Entropy 的作用是什么？
- ResNet18、MobileNetV2、EfficientNet-B0 各代表什么 CNN 设计思想？
- YawDD Stage 7 的 strongest test model 是哪个？EfficientNet-B0 的 best validation accuracy 应如何解释？
- 为什么不能使用 `artifacts/results/initial_results.csv` 作为 YawDD Stage 7 source of truth？
- MRL Eye dataset 的 label mapping 是什么？`0` 和 `1` 分别表示什么？
- MRL Eye Stage 8 做了哪些 manifest 和 split outputs？
- MRL Eye Stage 9 为什么使用 validation macro F1 作为 checkpoint metric？
- Macro F1 和 weighted F1 有什么区别？在本项目中为什么重要？
- 什么是 confusion matrix？如何从中读出 false_open 和 false_closed？
- `false_open` 为什么比普通错误更 safety-critical？
- `false_closed` 为什么更像 false alarm tendency？
- lowering `p_eye_closed` threshold 会对 closed recall、false_open、false_closed 产生什么影响？
- 为什么 threshold candidates 应该从 validation sweep 中选择，而不是 test sweep？
- 为什么 MobileNetV2 被选为 primary MRL Eye model？
- 为什么 ResNet18 with threshold `0.30` 被保留为 safety-prioritized reference？
- 当前 MRL Eye selected runtime checkpoint 是哪个？
- NTHUDDD2 为什么只是 explored branch，而不是当前主线？
- specialist-module performance、model-level performance 和 final system-level performance 有什么区别？
- Stage 17.1 rule-based fusion 如何组合 recent mouth/yawn evidence 和 sustained eye-warning evidence？
- 为什么 brief blink-like activity overlapping recent-yawn 不应自动升级成 high-confidence warning candidate？
- Stage 17.2 为什么要求把 eye-warning evidence 解释为 reduced eye openness、blink-like activity、brief closure、fatigue-like appearance 或 ROI-sensitive cases，而不是直接写成 verified sustained full eye closure？
- Stage 17.3 `/video-upload` 页面包含哪些结果区块？为什么 interval table 和 keyframe metadata 很重要？
- `make stage17-ui` 会启动哪些服务？backend/frontend URL 分别是什么？
- Stage 17.5 `/video-upload` UI 为什么要分开显示 backend fusion state 和 descriptive eye evidence？
- Stage 17.5 为什么把 `recent_yawn_event` 解释成 temporal-window evidence？
- Stage 18 `/history-48h` 使用什么数据源？为什么它必须标成 demo/local history data？
- Stage 19 Live Monitor 为什么要把 current mouth activity、recent yawn context、post-yawn reminder 分开？
- Stage 19 Live Monitor 为什么要把 current eye evidence、active temporal eye-warning candidate、recent sustained eye-warning reminder 分开？
- 为什么 recent sustained eye-warning reminder 只能作为 display-only review note，不能参与 high-confidence escalation？
- 为什么单帧 eye closed/open 或 yawn/no-yawn 不能直接等同于 drowsiness monitoring？

## 18. Final Summary

当前项目已经从两个关键 specialist modules 扩展到 Stage 17.5 本地 video-upload warning-candidate evidence review UI、Stage 18 frontend-only 48h History page，并进一步加入 Stage 19 Live Monitor realtime warning-candidate prototype。它包括数据准备、主体级划分、CNN baseline 训练、runtime ROI extraction、rule-based temporal fusion、FastAPI upload backend、Next.js Video Upload Analysis UI、frontend localStorage history UI、Live Monitor webcam preview/sampling/realtime evidence UI、一键本地启动脚本和验收/演示文档。

第一个完成模块是 YawDD/YawDD+ Dash mouth/yawn specialist。它从原始 Dash `.avi` 视频和 YawDD+ annotation files 出发，重建 labeled frames，确认 class mapping，发现原始 bbox 不适合作 mouth ROI，然后用 MediaPipe Face Mesh 生成嘴部裁剪，并完成 subject-level split 和 Stage 7 CNN training。当前 Stage 7 notebook 结果显示 ResNet18 在 test accuracy 上最强，EfficientNet-B0 在 validation accuracy 上最强。该模块的输出概念是 `p_yawn`。

第二个完成模块是 MRL Eye open/closed specialist。它完成了 84,898 张眼部图像的 dataset preparation、subject-level split、Stage 9 CNN baseline training 和 Stage 9B model selection。MobileNetV2 被选为 primary eye model，默认使用 argmax / `p_eye_closed >= 0.50`。ResNet18 with `p_eye_closed >= 0.30` 被保留为 safety-prioritized reference，因为它减少 false_open、提高 closed-eye recall，但明显增加 false_closed。该模块的输出概念是 `p_eye_closed`。

在 specialist modules 之后，Stage 10-15 建立了 controlled-video runtime analysis 和 real synchronized rule-based fusion。Stage 17.1 引入 sustained-eye gate：high-confidence warning candidate 需要 recent mouth/yawn evidence 加 sustained eye-warning evidence；brief blink-like activity 即使与 recent-yawn 重叠，也会被 conservatively suppressed from high-confidence escalation。Stage 17.2 明确解释边界：eye-warning evidence 不能自动描述成 verified sustained full eye closure。

Stage 17.3/17.4 把上传视频分析整理成可演示的本地 MVP。Stage 17.5 进一步改进 eye-evidence interpretation 和 `/video-upload` evidence review UI：compact result overview、summary metrics、expandable warning-candidate intervals、fusion state vs descriptive eye evidence clarification、tabbed evidence figures、keyframe evidence gallery 和 technical evidence links。FastAPI backend 使用 `POST /api/analyze-video`，`make stage17-ui` 可以一键启动 backend `http://127.0.0.1:8000` 和 frontend `http://127.0.0.1:3000/video-upload`。

Stage 18 新增 `/history-48h` 页面，用 browser `localStorage` key `visionguard.history48h.v1` 保存 demo/local warning-candidate history，并展示 48-hour summary cards、candidate severity trend、event distribution、state breakdown、event timeline、recent sessions 和 manual review queue。这个页面是 frontend-only history review，不代表 webcam 已经实现，也不代表 backend 持久化历史已经接通。

Stage 19 把 `/` 从 Dashboard concept 推进为 Live Monitor 本地 prototype。它支持 browser webcam preview、display-only mirror、2 FPS frame sampling、FastAPI realtime session lifecycle、single-frame `p_eye_closed` / `p_yawn` evidence、ROI/signal quality status，以及 session-local realtime rule-based warning-candidate state。当前 yawn 语义拆成 `mouth_active`、4 秒 recent yawn fusion context、8 秒 post-yawn reminder；eye 语义拆成 current eye evidence、active temporal eye-warning candidate、4 秒 recent sustained eye-warning reminder。两个 reminder 都是 display-only review notes，不应触发 alarm、history write 或 high-confidence escalation。

最重要的技术 lessons 是：

- 数据预处理（Data Preprocessing）和可视化检查（Visual Sanity Check）会直接决定模型是否学到正确信号。
- 主体级划分（Subject-Level Split）比 random frame-level split 更适合评估跨驾驶员泛化。
- 迁移学习（Transfer Learning）、加权交叉熵（Weighted Cross Entropy）、早停（Early Stopping）和学习率调度器（Learning Rate Scheduler）构成了当前 CNN baseline 的核心训练机制。
- 安全相关任务不能只看 accuracy；必须看 recall、F1、confusion matrix、false_open、false_closed 和 threshold sweep。
- Stage 17/17.5 输出是 rule-based drowsiness warning-candidate analysis，必须使用 warning-candidate wording。
- Stage 18 `/history-48h` 当前只展示 demo/local warning-candidate history；它不是 active webcam monitoring，也不是 backend truth storage。
- Stage 19 Live Monitor 当前是 local realtime warning-candidate prototype；它不是 alarm system，不写 `/history-48h`，也不是 deployment-ready monitoring。
- 当前所有准确率都是 specialist-module metrics，不能报告成最终 driver drowsiness detection accuracy。

自然的下一步不是重新训练已有模块，也不是直接宣称部署可用；更合理的是在明确边界内推进：history 方向应把真实 uploaded-video run records 持久化并接入 `/history-48h`，Live Monitor 方向应先设计 alert debounce / alarm policy 和 history ingestion schema，再决定是否接入。未来工作仍应保持 warning-candidate boundary：不输出最终 drowsiness truth、最终系统级准确率或部署可用声明。
