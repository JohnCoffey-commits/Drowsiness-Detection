# Evidence-Fused Driver Drowsiness Warning-Candidate System

Final Technical Report | Drowsiness Detection Project

Generated location: `docs/final/final_report_en.pdf`

Claim boundary: this report describes specialist-module metrics and rule-based warning-candidate behavior. It does not claim final system-level driver drowsiness accuracy, clinical validation, real-road validation, or deployment readiness.

## Abstract

This report documents a modular driver drowsiness warning-candidate prototype. The project deliberately avoids a single end-to-end drowsy/not-drowsy classifier. Instead, it separates visible facial evidence into two specialist channels: a mouth/yawn model trained on YawDD/YawDD+ Dash evidence and an eye open/closed model trained on MRL Eye. The two model outputs, `p_yawn` and `p_eye_closed`, are then processed by temporal rules, quality gates, and a tiered fusion layer.

The main training evidence comes from `colab_file/stage7_yawdd_training_r.ipynb` and `colab_file/stage9_mrl_eye_training_r.ipynb`. In Stage 7, the selected mouth specialist is ResNet18, with 99.37% test accuracy and 97.18% yawn F1. In Stage 9/9B, the selected eye specialist is MobileNetV2, with 98.63% test accuracy, 98.63% macro F1, and 98.52% closed-eye recall. Later stages focus on runtime evidence alignment rather than new model training.

The system now includes a FastAPI backend, a Next.js user interface, uploaded-video analysis, a real-time webcam Live Monitor prototype, event history, insights, and a local SQLite summary archive. The correct claim boundary remains important: the current system emits rule-based warning-candidate states for review and demonstration. It does not claim final system-level driver drowsiness accuracy, clinical validation, road deployment readiness, or a learned fusion classifier.

![Figure 1. Project-level pipeline: data preparation, specialist models, temporal rules, fusion, and application-facing outputs.](figures/fig01_system_pipeline.png)

## 1. Introduction and Background

Driver drowsiness detection is difficult because drowsiness is a temporal human state rather than a label that can be reliably inferred from a single frame. Accident statistics and countermeasure discussions by NHTSA treat drowsy driving as a safety problem, but ground-truth identification in real vehicles remains challenging. Ocular measures such as PERCLOS have also been studied as indicators of alertness reduction, especially because sustained eyelid closure can correspond to lapses in visual attention.

This project follows the same general direction but uses a more careful implementation boundary. It does not measure physical eyelid aperture directly. The runtime eye signal is a CNN probability, `p_eye_closed`, computed from extracted eye regions. For that reason, the report uses PERCLOS-like or PERCLOS-inspired terminology rather than claiming a true PERCLOS measurement. This distinction matters because model probability, face tracking quality, lighting, glasses, head pose, and ROI quality can all affect the resulting signal.

A dual-evidence design was chosen because mouth and eye behavior have complementary weaknesses. A yawn-like mouth opening may be caused by speaking, expression, or a transient pose. A high closed-eye probability may be caused by blinking, squinting, reflection, partial occlusion, or tracking failure. Training the two specialist tasks separately keeps labels interpretable, and delaying the final decision to a temporal fusion layer makes the warning logic easier to audit.

## 2. Overview of the Architecture/System

The repository structure follows the modular architecture described in `docs/PROJECT_STRUCTURE.md`. Local or reconstructed data are stored under `dataset/`, mappings and intermediate artifacts under `artifacts/`, training and runtime outputs under `outputs/`, stage reports under `reports/`, core processing and inference code under `src/`, the FastAPI service under `src/backend/`, and the Next.js interface under `SystemUI/`.

The system can be understood as four layers. The data layer reconstructs and validates specialist datasets. The training layer compares ResNet18, MobileNetV2, and EfficientNet-B0 baselines for mouth/yawn and eye open/closed classification. The runtime layer extracts face landmarks and ROIs, computes frame-level specialist probabilities, applies temporal rules, and aligns mouth-eye evidence. The application layer exposes uploaded-video analysis, Live Monitor sessions, event history, insights, and local archive summaries.

The backend entry point is `src/backend/app.py`. Uploaded-video analysis is handled by `src/runtime/system_video_upload_pipeline.py`, which connects Stage 10 eye ROI extraction, Stage 11 temporal preparation, a Stage 12-style eye adapter, Stage 14 mouth inference, F5 fusion, and keyframe extraction. Real-time evidence comes from `src/runtime/realtime_frame_inference.py`, while session-local temporal state is maintained in `src/runtime/realtime_temporal_state.py`.

![Figure 2. Data processing flow: YawDD/YawDD+ Dash mouth evidence and MRL Eye evidence are prepared independently with leakage-aware subject-level splits.](figures/fig02_data_processing_flow.png)

## 3. Data Processing and Model Training

The YawDD/YawDD+ Dash mouth branch was reconstructed into 64,378 labeled frames across 29 subjects. The class distribution is imbalanced but usable: 57,347 frames are labeled `no_yawn` and 7,031 frames are labeled `yawn`. The mouth crop stage processed all reconstructed frames, producing 64,202 trainable crops with a 99.73% success rate. The split is subject-level rather than frame-random: 44,156 images for training, 8,892 for validation, and 11,154 for testing, with yawn rates close to 11% across the splits.

The MRL Eye branch contains 84,898 eye images across 37 subjects, with 41,946 closed-eye images and 42,952 open-eye images. Local parsing found no unreadable images. The subject-level split contains 58,982 training images, 13,029 validation images, and 12,887 test images. This split is more conservative than a random image-level split because the same person does not appear across training and evaluation partitions.

Both training notebooks use PyTorch and torchvision. The mouth/yawn models use 224 by 224 inputs, Adam with learning rate `1e-4`, weighted cross entropy, ReduceLROnPlateau, early stopping, and mild augmentation. The eye models use 224 by 224 inputs, pretrained backbones, weighted cross entropy, validation macro F1 checkpointing, and a threshold analysis after the default argmax evaluation.

## 4. Fusion and Runtime Decision Logic

Stage 12 selected the eye temporal rule `quality_gated_perclos_mean_ge_0.60_consec`. The rule computes a rolling PERCLOS-like binary mean and requires the value to be at least 0.60 for at least two sampled frames. If the no-face ratio in a five-frame window exceeds 0.20, the system marks the eye signal as unreliable. This prevents face tracking failure from being interpreted as eye-closure evidence.

Stage 14 applies the trained mouth specialist to runtime mouth crops and computes `p_yawn = softmax(logits)[1]`. A sampled row with `p_yawn >= 0.50` is treated as a yawn event. The runtime state also keeps a recent-yawn context window, because the eye warning and mouth event may not occur in the exact same sampled frame. Recent-yawn context is evidence for fusion, not proof that the current frame contains an ongoing yawn.

The F5 fusion layer is rule-based and quality-aware. If the eye signal is unreliable and no recent yawn exists, the output is `signal_unreliable`. If the eye signal is unreliable but recent yawn evidence exists, the output is `mouth_warning_candidate`. If an eye warning and recent yawn evidence overlap, the output is `high_confidence_drowsiness_candidate`. If only the eye warning is present, the output is `eye_warning_candidate`; if only yawn context is present, the output is `mouth_warning_candidate`; otherwise the output is `normal`.

Stage 17.1 and Stage 17.5 tighten this behavior by adding sustained-eye evidence and evidence-strength gates. These additions reduce the chance that a brief blink-like event or weak eye signal will be escalated merely because it overlaps with recent-yawn context. The real-time Live Monitor uses the same evidence semantics but maintains session-local state at approximately 2 FPS.

![Figure 3. F5 fusion logic: signal quality is checked before mouth-eye evidence is elevated into warning-candidate states.](figures/fig03_fusion_logic.png)

## 5. Results and Evaluation

Stage 7 results show that ResNet18 is the best mouth/yawn specialist under the project’s test split. Its test accuracy is 99.37%, and its yawn F1 is 97.18%. EfficientNet-B0 has the highest validation accuracy, but its test yawn F1 is lower than ResNet18. Because the mouth data are imbalanced, yawn precision, yawn recall, and yawn F1 are more informative than accuracy alone.

Stage 9/9B results support MobileNetV2 as the primary eye specialist. Under the default decision rule, MobileNetV2 reaches 98.63% test accuracy and 98.63% macro F1, with a reasonable balance between false-open and false-closed errors. A lower threshold can improve closed-eye recall, but it also increases false-closed predictions; the project therefore keeps the threshold-adjusted results as safety-oriented references rather than the default deployment choice.

![Figure 4. Specialist model performance. These are module-level results, not final system-level drowsiness accuracy.](figures/fig04_model_performance.png)

| Model | Train Acc | Val Acc | Test Acc | Yawn Precision | Yawn Recall | Yawn F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 98.92% | 98.85% | 99.37% | 96.47% | 97.89% | 97.18% |
| MobileNetV2 | 98.97% | 98.48% | 98.75% | 91.74% | 97.48% | 94.52% |
| EfficientNet-B0 | 98.76% | 99.08% | 99.20% | 94.82% | 98.13% | 96.44% |

| Model | Test Acc | Macro F1 | Closed Recall | False Open | False Closed | Val Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 98.46% | 98.46% | 98.59% | 89 | 109 | 0.30 |
| MobileNetV2 | 98.63% | 98.63% | 98.52% | 93 | 84 | 0.30 |
| EfficientNet-B0 | 98.62% | 98.62% | 98.24% | 111 | 67 | 0.30 |

## 5.1 Controlled Fusion Validation

Stage 15 synchronizes real Stage 12 eye timelines with real Stage 14 mouth inference timelines and applies the F5 fusion rule to four controlled videos. The normal-open baseline produces only normal states. The realistic drowsy simulation contains mouth warnings, eye warnings, and a small number of high-confidence candidates where recent-yawn and eye-warning evidence overlap. The mild head-motion video includes signal-unreliable states, which is the intended behavior when the eye signal quality is weak. The controlled long open/closed video produces eye-warning candidates without mouth evidence.

For the realistic drowsy simulation, the manually observed yawn interval is around 14.3 to 16.8 seconds. Stage 14 marks 12 out of 12 rows in that interval as yawn events, with a mean `p_yawn` of approximately 0.981. Stage 17 uploaded-video validation further demonstrates that the same evidence chain can be packaged into an API response with sampled frames, warning candidates, yawn events, figures, and keyframes for review.

![Figure 5. Stage 15 fusion state counts for A/B/C/D controlled validation videos.](figures/fig05_stage15_fusion_counts.png)

| Video | Normal | Eye | Mouth | High Confidence | Signal Unreliable |
| --- | --- | --- | --- | --- | --- |
| A_normal_open_baseline | 70 | 0 | 0 | 0 | 0 |
| B_realistic_drowsy_simulation | 49 | 18 | 30 | 6 | 0 |
| C_mild_head_motion | 76 | 7 | 0 | 0 | 12 |
| D_controlled_long_open_closed | 54 | 65 | 0 | 0 | 0 |

## 6. Discussion and Conclusions

The project’s main engineering strength is its explicit separation between specialist model evidence and system-level interpretation. The mouth model reports yawn evidence. The eye model reports closed-eye evidence. The temporal and fusion layers convert those signals into warning-candidate states with quality checks. This separation makes the pipeline easier to audit and prevents single-frame probabilities from being overstated as final drowsiness labels.

The main limitation is generalization. The mouth model is trained on reconstructed Dash mouth crops, the eye model is trained on MRL Eye images, and runtime video inference relies on MediaPipe-derived ROIs. These domains are related but not identical. The A/B/C/D validation confirms that the pipeline behaves plausibly in a small controlled setting, but it does not establish road-level performance, driver-level fatigue ground truth, or robustness across camera position, lighting, glasses, occlusion, skin tone, and vehicle conditions.

The current system is therefore best described as a local, review-oriented driver drowsiness warning-candidate prototype. A publication-grade or deployment-grade extension would need more synchronized mouth-eye video data, temporal ground-truth annotation, subject/camera/lighting-stratified evaluation, and possibly a learned temporal fusion model after sufficient annotated data are available.

## References

[1] NHTSA. Drowsy Driving: Countermeasures That Work. https://www.nhtsa.gov/book/countermeasures-that-work/drowsy-driving

[2] Dinges, D. F., Mallis, M. M., Maislin, G., & Powell, J. W. Evaluation of techniques for ocular measurement as an index of fatigue and as the basis for alertness management. NHTSA, 1998. https://rosap.ntl.bts.gov/view/dot/2518

[3] FMCSA/NHTSA. PERCLOS: A Valid Psychophysiological Measure of Alertness. https://ntlsearch.bts.gov/ntl/md.do?id=51369

[4] Abtahi, S., Omidyeganeh, M., Shirmohammadi, S., & Hariri, B. YawDD: A Yawning Detection Dataset. ACM MMSys Workshop, 2014.

[5] MRL. MRL Eye Dataset. https://mrl.cs.vsb.cz/eyedataset.html

[6] Google MediaPipe. MediaPipe Face Mesh. https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh

[7] He, K., Zhang, X., Ren, S., & Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016.

[8] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR, 2018.

[9] Tan, M., & Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.

[10] Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS, 2019.

## Appendices

Primary internal evidence files include `docs/PROJECT_STRUCTURE.md`, `docs/PROJECT_CURRENT_STATUS.md`, `docs/archive/stage16/reports/stage16_final_integration_summary_report.md`, `reports/stage15_real_mouth_eye_fusion_validation_report.md`, `reports/stage14_mouth_yawn_runtime_validation_report.md`, `reports/stage12_eye_alert_rule_analysis_report.md`, `reports/mrl_eye_stage9b_error_analysis.md`, `reports/yawdd_dash_split_report.md`, `reports/mrl_eye_split_report.md`, `colab_file/stage7_yawdd_training_r.ipynb`, and `colab_file/stage9_mrl_eye_training_r.ipynb`.

Capability-use audit: research-writing-assistant routing, paper-orchestration, writing-core, figures/diagram handling, and verification guidance were used. The English report reuses existing verified project evidence and existing English-labeled figures. Verification includes PDF generation, file inspection, PDF metadata/page check, text extraction, page rendering, and visual inspection of rendered output. Remaining risk: the report inherits all limitations of the underlying stage reports and notebooks; it does not add new experiments.
