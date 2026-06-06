# VisionGuard 数据预处理技术指南

最后更新：2026-05-26

## 1. 本文档范围

本文档解释 VisionGuard 项目的数据预处理流程，范围从原始数据集开始，到模型训练前的准备状态为止。

本文档覆盖：

- 用于嘴部 / 打哈欠专家模型的 YawDD / YawDD+ Dash 预处理。
- 用于眼部睁闭专家模型的 MRL Eye 预处理。
- 生成的 manifest、trainable manifest、视觉检查结果，以及防止数据泄漏的 subject-level split。
- 每个预处理步骤背后的技术原因。

本文档不覆盖 CNN 架构设计、训练循环、模型选择、运行时推理、时序融合、前端、后端 API、警告逻辑、部署或最终系统评估。

VisionGuard 应被理解为模块化的驾驶员疲劳检测与监测系统，而不是一个单一的 drowsy / not-drowsy 分类器。预处理工作为两个视觉证据信号准备专家数据集：

- 嘴部 / 打哈欠证据，模型推理后对应概念为 `p_yawn`。
- 眼部睁闭证据，模型推理后对应概念为 `p_eye_closed`。

## 2. 高层预处理流程

项目预处理流程如下：

```text
原始数据集
-> 数据集检查
-> 标签含义解释
-> manifest 构建
-> 帧 / 图像提取或过滤
-> 必要时生成 ROI 裁剪图
-> 视觉 sanity check
-> trainable manifest 创建
-> subject-level 训练 / 验证 / 测试划分
-> 模型训练前的输入
```

预处理不只是文件格式转换。它保护项目免受以下常见问题影响：

- 标签解释错误，例如把 yawn / no-yawn 或 open / closed 反过来。
- 训练时临时扫描文件夹导致不可复现。
- 把无效图像或失败 crop 放入训练。
- 使用随机逐帧划分造成 identity leakage。
- 因重复帧或同一 subject 同时出现在多个 split 中而得到虚高指标。
- annotation 文件、提取帧、crop 路径和训练输入之间发生静默漂移。

因此，预处理 artifact 是实验设计的一部分。训练脚本应该读取已记录的 manifest 和 split 文件，而不是临时从原始数据文件夹中发现样本。

## 3. 主数据集 1：YawDD / YawDD+ Dash

YawDD / YawDD+ Dash 用于嘴部 / 打哈欠专家模型。

项目角色：

| 项目 | 含义 |
| --- | --- |
| 专家任务 | `no_yawn` vs `yawn` 分类 |
| 标签映射 | `0 = no_yawn`, `1 = yawn` |
| 后续运行时证据概念 | `p_yawn` |
| 主要原始输入 | 原始 YawDD Dash `.avi` 视频 |
| annotation 输入 | YawDD+ Dash 标签文件 |
| 最终训练输入形式 | 带 train/val/test split 标签的嘴部 ROI crop 图像 |

原始 Dash 视频是完整驾驶员画面，不是嘴部专家模型的最终输入。打哈欠线索主要集中在嘴部区域，因此项目先重建带标签的完整帧，再使用人脸 landmarks 生成嘴部 ROI 裁剪图。

相关路径：

- `dataset/YawDD_raw/`
- `dataset/YawDD+/`
- `dataset/YawDD_plus_reconstructed/`
- `src/data/build_yawdd_dash_mapping.py`
- `src/data/extract_yawdd_dash_labeled_frames.py`
- `src/preprocessing/generate_yawdd_mouth_crops.py`
- `src/preprocessing/precompute_yawdd_mouth_crops.py`
- `src/data/build_yawdd_split.py`
- `reports/yawdd_raw_dash_report.md`
- `reports/yawdd_plus_annotation_format_report.md`
- `reports/yawdd_dash_reconstruction_report.md`
- `reports/yawdd_dash_visual_sanity_check.md`
- `reports/yawdd_dash_mouth_crop_report.md`
- `reports/yawdd_dash_split_report.md`

### 原始 Dash 视频

原始 YawDD Dash 数据以 `.avi` 视频形式提供，每个 subject 对应一个连续视频。当前本地检查到 29 个 Dash 视频：13 个 female subjects 和 16 个 male subjects。

文件名编码了 subject index、gender 和 glasses state。部分原始文件名存在小异常，例如重复 `.avi.avi` 后缀或多余空格。预处理的 mapping 步骤会在匹配视频和 YawDD+ 文件夹之前对这些名称进行规范化。

### YawDD+ annotations

YawDD+ Dash 数据在每个 subject 的 `labels/` 文件夹中包含 annotation 文本文件。文件名包含帧索引：

```text
<8-digit frame index>_<object index>.txt
```

每个文件包含一行 YOLO-style 标注：

```text
<class_id> <x_center> <y_center> <width> <height>
```

重要解释：

- `class_id` 是真正有用的标签。
- `0` 表示 non-yawning。
- `1` 表示 yawning。
- frame index 是原始视频中的 native 0-based frame index。
- YOLO geometry 会被保留作 traceability metadata，但不会作为嘴部 crop 的来源。

视觉 sanity check 发现 YawDD+ 的 box geometry 不是可靠的嘴部区域。在抽样帧中，box 经常覆盖躯干或下半身区域，而不是嘴部。因此项目重建完整帧后，重新计算嘴部 ROI。

### 全部尝试行 vs 可训练行

YawDD 嘴部 crop 预处理区分所有 crop 尝试和训练可用行：

- `artifacts/mappings/yawdd_dash_all_mouth_crops.csv` 记录所有 crop 尝试，包括失败行。
- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` 排除失败 crop，并添加 split 标签。

这个区别很重要：失败 crop 对质量审计有价值，但不能进入 CNN 训练数据。

## 4. YawDD / YawDD+ 分步骤预处理

### 4.1 原始视频和 annotation 检查

第一步是在不训练模型的情况下检查原始 Dash 视频和 annotation 文件夹。

报告：

- `reports/yawdd_raw_dash_report.md`
- `reports/yawdd_plus_annotation_format_report.md`

该步骤验证：

- 原始 source 包含 29 个 Dash `.avi` 视频。
- YawDD+ Dash annotation 文件夹也覆盖 29 个 subjects。
- 对 whitespace 和重复 `.avi` 后缀做规范化后，原始视频 canonical token 与 YawDD+ subject folder token 匹配。
- annotation 文件符合 `<frame>_<object>.txt` 命名模式。
- annotation 行包含二分类 class id 和 YOLO-normalized geometry。
- 一些 subjects 存在 frame-index gaps，因此必须精确使用 frame index，不能使用连续计数器替代。

为什么需要这一步：

- 避免把某个 subject 的 annotation 匹配到错误 raw video。
- 确认 frame extraction 可以由 YawDD+ frame index 驱动。
- 在批量提取前识别 multi-object `_1` annotation 文件。

### 4.2 构建 Dash frame mapping

脚本：

- `src/data/build_yawdd_dash_mapping.py`

默认输出：

- `artifacts/mappings/yawdd_dash_mapping.csv`
- `reports/yawdd_dash_mapping_report.md`

该脚本将每个 YawDD+ Dash subject folder 与对应的原始 YawDD Dash `.avi` 文件配对。

关键代码含义：

- 索引 Dash female / male 文件夹下的原始 `.avi` 文件。
- 通过去除空格和重复 `.avi` 后缀规范化 raw filenames。
- 使用规范化后的 raw video stem 和 YawDD+ folder name 作为 canonical subject token。
- 每个 YawDD+ subject folder 写入一行。
- 记录 mapping confidence 和 notes。

预期字段包括：

- `subject_id`
- `annotation_folder`
- `annotation_txt_path`
- `raw_source_path`
- `mapping_confidence`
- `mapping_notes`

为什么需要这一步：

- 后续提取必须精确知道每个 annotation folder 对应哪个 raw `.avi`。
- mapping table 让这种配对可复现，而不是依赖手工路径判断。
- 文件名异常只需在这里统一处理并记录。

### 4.3 重建带标签的 Dash frames

脚本：

- `src/data/extract_yawdd_dash_labeled_frames.py`

默认输入：

- `artifacts/mappings/yawdd_dash_mapping.csv`

主要输出：

- `dataset/YawDD_plus_reconstructed/Dash/full_frames/<subject_id>/<frame_index>.jpg`
- `dataset/YawDD_plus_reconstructed/Dash/labels_csv/<subject_id>.csv`
- `artifacts/mappings/yawdd_dash_all_labeled_frames.csv`

该步骤重建 YawDD+ annotations 指向的真实图像帧。YawDD+ 提供的是 label files，而不是 image files，因此必须从原始 Dash 视频中解码对应帧。

关键代码含义：

- 扫描每个 subject 的 `labels/` 文件夹。
- 按 frame index 分组 annotation 文件。
- 对每个 frame 优先使用 `_0` annotation 文件。
- 如果存在重复 `_1` 文件，用 `had_duplicate_box` 记录。
- 打开对应 `.avi`，按顺序遍历帧。
- 只保存目标 frame indices。
- 将逐帧 provenance 写入输出 manifest。

重建后的 labeled-frame manifest 包含字段：

- `subject_id`
- `frame_index`
- `image_path`
- `raw_video_path`
- `annotation_txt_path`
- `class_id`
- `binary_label`
- `kept_object_id`
- `had_duplicate_box`
- `yawdd_bbox_raw`
- `extraction_status`
- `notes`

文档记录结果：

- 64,378 个带标签帧。
- 57,347 行 `no_yawn`。
- 7,031 行 `yawn`。
- 10 个重复 `_1` boxes 被标记并忽略。
- 0 个 missing extracted JPEGs。

为什么需要重建：

- annotations 本身不是模型输入。
- 嘴部 cropper 需要真实像素。
- 必须把 YawDD+ frame index 连接到对应 raw video frame。

### 4.4 视觉 sanity check

报告：

- `reports/yawdd_dash_visual_sanity_check.md`

视觉 sanity check 用来确认标签和帧对齐是否符合项目假设。

已确认结论：

- YawDD+ frame indices 与 raw `.avi` frame indices 对齐。
- class `1` 对应可见打哈欠。
- class `0` 对应未打哈欠。
- 抽样检查中的 `_1` 文件是 spurious duplicate detections。
- YawDD+ YOLO boxes 不应作为 mouth crops，因为它们不能可靠定位嘴部。

为什么需要这一步：

- 如果数据集文档不完整，二分类标签很容易被反向解释。
- 使用反向 yawn labels 训练出来的模型仍然会输出数值，但语义是错误的。
- 视觉证据是确认 class `1` 真的表示 yawning、class `0` 真的表示 non-yawning 的最安全方式。

### 4.5 Mouth ROI crop 生成

主要脚本：

- `src/preprocessing/generate_yawdd_mouth_crops.py`

较早 / legacy 预处理入口：

- `src/preprocessing/precompute_yawdd_mouth_crops.py`

当前 Stage 5 cropper 的默认输入：

- `artifacts/mappings/yawdd_dash_all_labeled_frames.csv`

主要输出：

- `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/<subject_id>/<frame_index>.jpg`
- `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`

嘴部 cropper 将重建的 full frames 转换为嘴部 ROI 图像。这一步使嘴部 / 打哈欠专家模型成为聚焦视觉线索的模型，而不是完整驾驶员画面的分类器。

技术流程：

1. 从 labeled-frame manifest 读取一行。
2. 加载重建的 full-frame image。
3. 运行 MediaPipe Face Landmarker / Face Mesh。
4. 使用固定的 outer-lip 和 inner-lip landmark indices。
5. 从 landmark 坐标构建 mouth bounding box。
6. 对 box 进行 expansion / padding。
7. 将 box clip 到图像边界内。
8. 保存 mouth crop。
9. 记录 crop path、crop method、crop bounding box、label、source frame、annotation path 和 notes。

主要 crop method：

- `face_mesh`：MediaPipe landmarks 成功检测，嘴部 ROI 由 lip landmarks 计算。

fallback method：

- `fallback_lower_face`：如果 Face Mesh 失败，OpenCV Haar face detector 尝试检测人脸，并使用人脸下半部分作为 fallback crop。

failure method：

- `failed`：Face Mesh 和 fallback 都无法产生可用 crop，或图像 / crop 无法保存。

文档记录的 Stage 5 结果：

| 指标 | 数值 |
| --- | ---: |
| 处理帧数 | 64,378 |
| MediaPipe Face Mesh crops | 64,093 |
| Fallback lower-face crops | 109 |
| Failed crops | 176 |
| Saved trainable crops | 64,202 |
| Success rate | 99.73% |

为什么 mouth ROI crop 比 full frame 更适合该专家模型：

- yawn label 的视觉证据集中在嘴部。
- full driver frame 包含大量无关背景、方向盘、光照、衣物和 subject identity cues。
- mouth crop 减少无关变化，让分类器更关注证据区域。
- crop manifest 允许审计失败 crop，而不是让它们静默进入训练。

较早的 `precompute_yawdd_mouth_crops.py` 具有历史参考价值。它也使用 MediaPipe-style mouth cropping 和 fallback 逻辑，但当前已文档化的 YawDD Dash Stage 5 输出由 `generate_yawdd_mouth_crops.py` 和 `artifacts/mappings/yawdd_dash_all_mouth_crops.csv` 表示。

### 4.6 Trainable mouth-crop manifest

Artifacts：

- `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`
- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`

区别：

| 文件 | 含义 |
| --- | --- |
| `yawdd_dash_all_mouth_crops.csv` | 所有 crop 尝试，包括 `face_mesh`、`fallback_lower_face` 和 `failed` 行。 |
| `yawdd_dash_all_mouth_crops_trainable.csv` | 过滤失败 crop 和无效 label/path 后的训练可用行，并添加 `split` 列。 |

以下情况的行可能被排除在 trainable data 外：

- `crop_method == failed`
- `binary_label` 不是 `no_yawn` 或 `yawn`
- `mouth_crop_path` 缺失
- 引用的 crop 文件不存在

trainable manifest 仍然保留 source provenance，包括 full frame、raw video、annotation file、class id 和 YawDD+ raw box。这使后续训练和错误分析可追踪。

### 4.7 Subject-level split

脚本：

- `src/data/build_yawdd_split.py`

Artifacts：

- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`
- `artifacts/splits/yawdd_dash_subject_split.csv`
- `reports/yawdd_dash_split_report.md`

YawDD split 按 `subject_id` 划分，而不是按单独 frames 随机划分。脚本会搜索 subject assignment，以平衡：

- train / validation / test subject 数量，
- image proportions，
- yawn rate，
- gender distribution，
- glasses / no-glasses distribution，
- 每个 split 都包含两个类别的要求。

文档记录的 split：

| Split | Subjects | Images | `no_yawn` | `yawn` | Yawn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 20 | 44,156 | 39,345 | 4,811 | 10.90% |
| `val` | 4 | 8,892 | 7,902 | 990 | 11.13% |
| `test` | 5 | 11,154 | 9,924 | 1,230 | 11.03% |

验证检查：

- 29 个 unique trainable subjects。
- 没有 subject 同时出现在多个 split 中。
- trainable data 中没有 failed crop rows。
- 每个 split 都包含两个类别。
- 所有引用的 mouth-crop files 存在。

为什么随机 frame-level split 无效：

- 同一视频中的相邻帧非常相似。
- 随机逐帧划分会让同一个人的脸和相邻帧同时出现在训练和测试中。
- 这会通过测试模型已经见过的身份和场景来抬高评估结果。

该 split 文件是嘴部 / 打哈欠模型训练前的正确边界。

## 5. 主数据集 2：MRL Eye

MRL Eye 用于眼部睁闭专家模型。

项目角色：

| 项目 | 含义 |
| --- | --- |
| 专家任务 | 闭眼 vs 睁眼分类 |
| 标签映射 | `0 = closed`, `1 = open` |
| 后续运行时证据概念 | `p_eye_closed` |
| 主要原始输入 | 按 subject 组织的眼部 crop 图像 |
| 最终训练输入形式 | 带 subject-level split 标签的 trainable image manifest |

相关路径：

- `dataset/mrlEyes_2018_01/`
- `src/data/inspect_mrl_eye.py`
- `src/data/build_mrl_eye_manifest.py`
- `src/data/split_mrl_eye_subjects.py`
- `src/data/spotcheck_mrl_eye.py`
- `reports/mrl_eye_dataset_report.md`
- `reports/mrl_eye_split_report.md`

MRL Eye 已经是眼部图像数据集，因此不需要像 YawDD 那样从 full frame 重建再生成 mouth ROI crop。关键预处理工作是检查、标签解析、manifest 构建、图像可读性过滤、视觉 spot checks 和防泄漏 subject split。

文件名结构编码了以下 metadata：

- subject id，
- image id，
- gender，
- glasses，
- eye state，
- reflections，
- lighting，
- sensor id。

项目会在构建训练 manifest 前验证 annotation mapping。重要 class mapping 是：

| Class id | Label |
| ---: | --- |
| 0 | `closed` |
| 1 | `open` |

Subject-level split 对 MRL Eye 很重要，因为每个 subject 会贡献很多相似眼部图像。随机 image split 同样会带来 identity leakage 风险。

## 6. MRL Eye 分步骤预处理

### 6.1 原始数据集检查

脚本：

- `src/data/inspect_mrl_eye.py`

报告：

- `reports/mrl_eye_dataset_report.md`

该脚本对原始数据集是 read-only 的。它扫描 MRL Eye 文件夹，解析文件名，验证图像可读性，并写入数据集检查报告。

检查内容：

- dataset root 存在。
- `annotation.txt` 确认 `0 = closed` 和 `1 = open`。
- 文件名符合预期 MRL 格式。
- subject id 可以解析。
- 图像可以打开。
- closed 和 open 两个标签都存在。
- 记录 gender、glasses、lighting、reflections 和 sensor id 的 metadata distributions。

文档记录结果：

| 指标 | 数值 |
| --- | ---: |
| Total images | 84,898 |
| Total subjects | 37 |
| Unreadable images | 0 |
| Unparseable filenames | 0 |
| Closed images | 41,946 |
| Open images | 42,952 |

为什么需要这一步：

- 在模型准备前确认 class mapping。
- 防止 malformed filenames 或 unreadable images 进入训练。
- 在 split 前给项目一个稳定的 subjects 和 classes 统计。

### 6.2 构建完整 image manifest

脚本：

- `src/data/build_mrl_eye_manifest.py`

Artifact：

- `artifacts/mappings/mrl_eye_all_images.csv`

manifest 是样本的结构化 CSV 索引。训练时使用 manifest，而不是临时扫描文件夹。

完整 MRL Eye manifest 记录字段包括：

- `image_path`
- `relative_path`
- `filename`
- `subject_id`
- `image_id`
- `gender`
- `glasses`
- `eye_state`
- `label`
- `label_name`
- `reflections`
- `lighting`
- `sensor_id`
- `parse_ok`
- `width`
- `height`
- `extension`
- `is_valid`
- `read_ok`
- `error`

为什么使用 manifest：

- 固定准确的样本列表。
- 记录 parsing 和 validation 状态。
- 让训练可复现。
- 为后续分析保留 labels 和 metadata。

### 6.3 创建 trainable manifest

Artifacts：

- `artifacts/mappings/mrl_eye_trainable.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`

trainable manifest 是通过过滤后可安全用于训练的 full manifest 子集。在当前已记录的 MRL Eye 预处理结果中，全部 84,898 张图像都是 trainable。

readiness criteria 包括：

- 文件名解析成功，
- 图像可读，
- 文件存在，
- label 有效且在 `{0, 1}` 中，
- subject id 存在，
- row 被标记为 valid。

`mrl_eye_trainable.csv` 包含添加 split 前的 trainable image rows。

`mrl_eye_trainable_with_split.csv` 包含相同 trainable rows，并额外添加 `split` 列。它是 MRL Eye 主要训练输入 manifest。

### 6.4 视觉 spot checks

脚本：

- `src/data/spotcheck_mrl_eye.py`

输入：

- `artifacts/mappings/mrl_eye_trainable_with_split.csv`

输出：

- `artifacts/visual_checks/mrl_eye_closed_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_open_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_by_split_contact_sheet.jpg`

该脚本从样本行中生成 contact sheets。contact sheet 是带标签的小图网格。

contact sheet 的作用：

- 快速发现 `closed` 和 `open` 标签是否反转。
- 检查图像是否是合理的 eye crops。
- 验证 train、validation 和 test split 中是否有合理样本。
- 在训练前发现 unreadable、corrupted 或 unexpected images。

视觉 spot checks 不能替代定量检查，但能发现 CSV summary 看不到的问题。

### 6.5 Subject-level split

脚本：

- `src/data/split_mrl_eye_subjects.py`

Artifacts：

- `artifacts/splits/mrl_eye_subject_split.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`
- `reports/mrl_eye_split_report.md`

split 脚本读取 `artifacts/mappings/mrl_eye_trainable.csv`，并将每个 subject 分配到以下一个 split：

- `train`
- `val`
- `test`

split search 会在保持 subject group 不被拆开的前提下，尝试匹配约 70/15/15 的图像比例，并保证每个 split 都包含两个类别。

文档记录的 split：

| Split | Subjects | Images | Closed | Open |
| --- | ---: | ---: | ---: | ---: |
| `train` | 25 | 58,982 | 29,310 | 29,672 |
| `val` | 6 | 13,029 | 6,333 | 6,696 |
| `test` | 6 | 12,887 | 6,303 | 6,584 |

验证检查：

- leakage check 通过，
- 没有 missing split labels，
- 每张图像正好获得一个 split，
- 每个 split 都包含 closed 和 open samples，
- 引用的 image files 存在。

为什么 subject separation 很重要：

- MRL Eye 每个 subject 有很多样本。
- 眼部外观可能编码 identity、lighting、camera 和 sensor conditions。
- held-out split 包含训练时未见过的 subjects，评估才更诚实。

该 split manifest 是眼部专家模型训练前的边界。

## 7. 关于 NTHUDDD2 的简要说明

NTHUDDD2 / Kaggle extracted-frame 数据曾被探索，但不是当前主系统方向。

相关路径：

- `reports/nthuddd2_kaggle_dataset_report.md`
- `docs/archive/reports/nthu_dataset_report.md`
- `src/data/build_nthuddd2_kaggle_manifest.py`
- `src/data/split_nthuddd2_kaggle_subject.py`
- `artifacts/mappings/nthuddd2_kaggle_all_images.csv`
- `artifacts/mappings/nthuddd2_kaggle_all_images_trainable.csv`
- `artifacts/mappings/nthuddd2_kaggle_all_images_trainable_with_split.csv`
- `artifacts/splits/nthuddd2_kaggle_subject_split.csv`

探索过的 Kaggle 分支是二分类 extracted-frame 数据集，标签为：

- `notdrowsy = 0`
- `drowsy = 1`

它不是官方 raw-video NTHU-DDD protocol，且本地只解析出 4 个 subjects。它不应被呈现为最终 VisionGuard evidence pipeline。当前主系统使用 MRL Eye 作为眼部专家数据集，使用 YawDD / YawDD+ Dash 作为嘴部 / 打哈欠专家数据集，并在模型推理后进行基于规则的运行时融合。

这个区分很重要，因为如果围绕 NTHUDDD2 描述 VisionGuard，会让项目看起来像一个单一 binary drowsiness image classifier，而这不是当前架构。

## 8. 生成 artifact 及其含义

下表列出当前仓库中的主要预处理 artifacts。

| Artifact path | Dataset / module | 内容 | 为什么存在 | 模型训练前用途 |
| --- | --- | --- | --- | --- |
| `artifacts/mappings/yawdd_dash_all_labeled_frames.csv` | YawDD mouth/yawn | 64,378 行重建 full-frame 记录，包含 subject、frame index、image path、raw video path、annotation path、class id、binary label、duplicate-box flag 和 extraction status | 将 YawDD+ labels 连接到真实解码的 Dash frames | mouth ROI crop 生成的 source manifest |
| `artifacts/mappings/yawdd_dash_all_mouth_crops.csv` | YawDD mouth/yawn | 64,378 行 crop attempt，包括 `face_mesh`、`fallback_lower_face` 和 `failed` crop methods | 审计每次 crop 尝试并保留失败信息 | trainable crop filtering 和 split construction 的输入 |
| `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` | YawDD mouth/yawn | 64,202 行 trainable mouth-crop 记录，包含有效 crop paths 和 `split` 列 | 移除 failed crop rows，同时保留 provenance | mouth/yawn CNN 训练的 ready row list |
| `artifacts/splits/yawdd_dash_subject_split.csv` | YawDD mouth/yawn | row-level split manifest，包含 subject、split、gender、glasses、full-frame path、crop path、label、class id、crop method 和 provenance fields | 显式记录 subject-level train/val/test assignment | 确认 leakage-safe split membership 并提供 split-aware sample rows |
| `artifacts/mappings/mrl_eye_all_images.csv` | MRL Eye | 84,898 行完整 image manifest，包含解析出的 filename metadata、label、image size、validity 和 read status | 固定完整解析图像列表 | 过滤 trainable MRL Eye samples 的来源 |
| `artifacts/mappings/mrl_eye_trainable.csv` | MRL Eye | 84,898 行验证和过滤后的 trainable records | 提供 split 前的干净 eye image set | MRL subject split 脚本输入 |
| `artifacts/mappings/mrl_eye_trainable_with_split.csv` | MRL Eye | 84,898 行 trainable records 加 `split` 标签 | 将干净样本与 train/val/test assignment 合并 | eye open/closed CNN 训练的 ready row list |
| `artifacts/splits/mrl_eye_subject_split.csv` | MRL Eye | 37 行 subject-level split，包含每个 subject 的 image count 和 class ratio | 记录每个 subject 属于 train、validation 还是 test | 验证 subject separation 和 split balance |

检查时，本表中的必要 artifact 在当前仓库中均存在。

## 9. 预处理背后的技术概念

### Manifest

manifest 是样本的 CSV 索引，记录路径、标签、metadata、validation flags 和 provenance。训练使用 manifest 可以保证样本列表可复现。

### Trainable manifest

trainable manifest 是 full manifest 中通过过滤的子集。它排除 unreadable、malformed、missing labels、missing files 或 failed crops 的行。

### Label mapping

label mapping 定义每个 numeric class id 的含义。在本项目中：

- YawDD：`0 = no_yawn`, `1 = yawn`。
- MRL Eye：`0 = closed`, `1 = open`。
- NTHUDDD2 exploratory branch：`0 = notdrowsy`, `1 = drowsy`。

任何模型训练前都必须确认 label mapping。

### ROI crop

ROI 是 region of interest，即感兴趣区域。对于 YawDD mouth/yawn 训练，ROI 是嘴部 crop。它让模型输入聚焦于打哈欠相关视觉证据。

### MediaPipe landmarks

MediaPipe landmarks 是图像上检测到的人脸关键点。mouth cropper 使用唇部 landmarks 构建嘴部 bounding box。如果 landmarks 失败，项目会尝试 lower-face fallback crop。

### Visual sanity check

visual sanity check 是对解码图像和标签做小规模人工或半人工检查，用于确认 labels 和 frame indices 符合代码假设。

### Contact sheet

contact sheet 是带标签的样本图像网格。它帮助人工快速检查标签正确性、样本质量和 split 合理性。

### Subject-level split

subject-level split 将每个人只分配到一个 split。这样可以防止模型在训练和测试时看到同一个人的图像。

### Data leakage

data leakage 指 validation 或 test set 的信息进入训练。在视频和人脸 / 眼部数据集中，随机 frame-level split 是常见泄漏来源，因为相邻帧和同一 subject 可能出现在多个 split 中。

### Class balance

class balance 描述每个类别的样本数量。YawDD mouth/yawn 是不平衡数据，因为 yawn frames 是少数。split 过程会尝试在 train、validation 和 test 中保持相似 yawn rate。

### Failed crop / invalid row

failed crop 指没有产生可用 ROI image 的行。invalid row 可能存在 bad label、missing subject id、missing file、unreadable image 或 parsing failure。这些行应该被审计，而不是进入训练。

### Reproducible preprocessing artifact

reproducible preprocessing artifact 是保存下来的 CSV、report 或 contact sheet，用于记录某个预处理步骤的输出。它让团队成员不必重新运行昂贵或破坏性的操作，就能检查 pipeline。

## 10. 模型训练前已经准备好的内容

在预处理 / 训练边界处，项目有两个主要 ready-for-training 输入。

### 嘴部 / 打哈欠专家模型输入

训练应使用 YawDD mouth/yawn trainable mouth-crop 数据：

- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`
- 或 `artifacts/splits/yawdd_dash_subject_split.csv` 中等价的 split rows

每个训练行应提供：

- mouth crop path，
- label（`no_yawn` 或 `yawn`），
- class id（`0` 或 `1`），
- subject id，
- split（`train`、`val` 或 `test`），
- crop method 和 provenance，供后续分析使用。

当该文件存在并通过 split / leakage 检查后，训练边界才开始。

### 眼部睁闭专家模型输入

训练应使用 MRL Eye trainable image 数据：

- `artifacts/mappings/mrl_eye_trainable_with_split.csv`

每个训练行应提供：

- eye image path，
- label name（`closed` 或 `open`），
- class id（`0` 或 `1`），
- subject id，
- split（`train`、`val` 或 `test`），
- parsed metadata，例如 lighting、reflections、glasses 和 sensor id。

当该文件存在，并且 `reports/mrl_eye_split_report.md` 确认 leakage-safe splitting 后，训练边界才开始。

本文档在这里停止。模型架构、优化设置、指标和 checkpoint 选择属于训练文档范围。

## 11. 常见错误以及本项目如何避免

| 常见错误 | 本项目如何避免 |
| --- | --- |
| 把专家模型指标当作最终系统级驾驶员疲劳准确率 | 文档将专家数据集与运行时 warning-candidate analysis 分开描述。 |
| 把 VisionGuard 称为单一 drowsy/not-drowsy classifier | 项目使用独立 mouth/yawn 和 eye-state 专家模型，再加 rule-based temporal fusion。 |
| 把 NTHUDDD2 当作主项目方向 | NTHUDDD2 被记录为 explored branch，当前主方向是 YawDD + MRL Eye。 |
| 反转 YawDD 标签 | visual sanity checks 确认 `0 = no_yawn`，`1 = yawn`。 |
| 反转 MRL Eye 标签 | `annotation.txt` 和 inspection scripts 确认 `0 = closed`，`1 = open`。 |
| 使用 YawDD+ YOLO boxes 作为嘴部 crops | visual checks 显示这些 boxes 不是可靠 mouth ROI，因此使用 MediaPipe mouth landmarks。 |
| 训练 failed crops | failed crop rows 保存在 all-attempt manifest 中，但会从 trainable manifest 排除。 |
| 对同一 subject 使用随机 frame-level split | subject-level split 脚本将每个 subject 只分配到一个 split。 |
| 依赖 stale result files | 解读输出前应检查当前 reports 和 manifest headers。 |
| 将 raw datasets、generated bulk crops、checkpoints 或 large outputs 提交到普通 Git | raw 和 generated large assets 应保留在被忽略的 dataset / artifact / output / checkpoint 位置，除非明确 curated。 |

## 12. 可复现性和验证清单

开始或 review 模型训练前，使用以下 checklist。

### 已检查 source files

- `docs/PROJECT_STRUCTURE.md`
- `docs/PROJECT_CURRENT_STATUS.md`
- `docs/tech_learning/PROJECT_LEARNING_GUIDE.md`
- `src/data/build_yawdd_dash_mapping.py`
- `src/data/extract_yawdd_dash_labeled_frames.py`
- `src/preprocessing/generate_yawdd_mouth_crops.py`
- `src/preprocessing/precompute_yawdd_mouth_crops.py`
- `src/data/build_yawdd_split.py`
- `src/data/inspect_mrl_eye.py`
- `src/data/build_mrl_eye_manifest.py`
- `src/data/split_mrl_eye_subjects.py`
- `src/data/spotcheck_mrl_eye.py`

### 已检查 reports

- `reports/yawdd_raw_dash_report.md`
- `reports/yawdd_plus_annotation_format_report.md`
- `reports/yawdd_dash_reconstruction_report.md`
- `reports/yawdd_dash_visual_sanity_check.md`
- `reports/yawdd_dash_mouth_crop_report.md`
- `reports/yawdd_dash_split_report.md`
- `reports/mrl_eye_dataset_report.md`
- `reports/mrl_eye_split_report.md`
- `reports/nthuddd2_kaggle_dataset_report.md`
- `docs/archive/reports/nthu_dataset_report.md`

### 已检查 manifest headers

- `artifacts/mappings/yawdd_dash_all_labeled_frames.csv`
- `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`
- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`
- `artifacts/splits/yawdd_dash_subject_split.csv`
- `artifacts/mappings/mrl_eye_all_images.csv`
- `artifacts/mappings/mrl_eye_trainable.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`
- `artifacts/splits/mrl_eye_subject_split.csv`

### 验证问题

- YawDD labels 是否已确认为 `0 = no_yawn`, `1 = yawn`？
- MRL Eye labels 是否已确认为 `0 = closed`, `1 = open`？
- YawDD frame reconstruction 是否使用 annotation frame index，而不是 running counter？
- YawDD+ `_1` duplicate object files 是否在 labeling 中被忽略，但为了 traceability 被记录？
- YawDD mouth crops 是否来自 MediaPipe mouth landmarks 或 fallback lower-face logic，而不是原始 YawDD+ YOLO boxes？
- failed mouth crops 是否已从 trainable rows 中排除？
- YawDD 和 MRL split files 是否保证每个 subject 只在一个 split 中？
- 每个 split 是否都包含两个类别？
- reports 是否确认引用的 image / crop paths 存在？
- review 预处理时是否没有意外运行模型训练？
- review 期间是否没有修改 source code、dataset、checkpoint、runtime、frontend、backend、report 或已有文档？

如果任何答案不确定，应先检查相关 report 和 artifact header，再开始训练。
