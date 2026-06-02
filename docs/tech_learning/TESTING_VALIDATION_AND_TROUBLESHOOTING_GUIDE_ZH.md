# Testing, Validation, and Troubleshooting Guide

中文标题：测试、验证与故障排查指南

## 1. 本文目的

本文解释如何验证 VisionGuard 的各个系统部分是否工作，以及遇到问题时如何定位。它关注 operational validation 和 debugging，不是模型准确率评估报告。

核心原则：能打开页面、能上传视频、能生成 alert，不等于证明系统有最终 drowsiness accuracy。

## 2. 本项目中“测试”的含义

VisionGuard 的测试可以分成多个层次：

| 测试层 | 验证什么 | 不证明什么 |
|---|---|---|
| 数据/预处理检查 | manifests、splits、crops、样本数量是否存在合理 | 不证明 runtime 正确 |
| 模型 artifact 检查 | checkpoint 和 metrics files 是否存在 | 不证明 full-system accuracy |
| Backend API 检查 | FastAPI endpoints 是否 reachable | 不证明模型判断正确 |
| Realtime runtime 检查 | Live Monitor frame flow 是否工作 | 不证明 warning 是 ground truth |
| Video Upload 检查 | upload pipeline 是否产出 summary/timeline/figures/keyframes | 不证明视频真实疲劳标签 |
| Frontend UI 检查 | 页面、按钮、导出和状态是否可用 | 不证明算法准确率 |
| Archive/History/Insights 检查 | records 是否保存和汇总 | 不证明 evaluation metrics |
| Report evidence 检查 | figure/table 来源是否可追踪 | 不证明未做过的实验 |

Source: `docs/PROJECT_CURRENT_STATUS.md`, `docs/AI_PROJECT_CONTEXT.md`

## 3. 测试前的只读安全检查

排查问题前建议先运行：

```bash
git status --short
```

安全原则：

- 不要随意删除 localStorage 或 SQLite archive；
- 不要为了清空 UI 计数删除 `data/visionguard_archive.sqlite`；
- 不要覆盖 `outputs/`、`artifacts/` 或 checkpoints；
- 不要在故障排查时顺手改 threshold；
- 不要把重训模型当作普通 debugging 手段；
- 先确认 URL、CORS、backend process、checkpoint path，再考虑代码问题。

## 4. Data 和 Artifact 检查

常见只读检查：

| 检查目标 | 路径 |
|---|---|
| MRL Eye trainable manifest | `artifacts/mappings/mrl_eye_trainable_with_split.csv` |
| YawDD/YAWDD+ mouth crops manifest | `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` |
| recovered Stage 7 mouth/yawn artifacts | `artifacts/recovered_stage7_mouth_yawn/` |
| mouth/yawn final refresh metrics | `report_assets/mouth_yawn_evaluation_refresh/` |
| MRL Eye results | `outputs/mrl_eye/results/` |
| MRL Eye figures | `outputs/mrl_eye/figures/` |
| eye checkpoint | `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` |
| mouth/yawn checkpoint | `checkpoints/resnet18_best.pt` |

Source: `docs/PROJECT_STRUCTURE.md`, `docs/tech_learning/MODEL_EVALUATION_AND_SELECTION_GUIDE_ZH.md`

如果文件缺失，不要立即重新训练。先确认是否在正确 project root、是否有外部 artifacts 未同步、是否路径写错。

## 5. Backend Health Checks

确认的 backend endpoints：

| Endpoint | 用途 |
|---|---|
| `/` | backend root/status response |
| `/api/realtime/health` | realtime inference service health |
| `/api/archive/health` | local archive health |
| `/api/archive/records` | archive records list |
| `/api/analyze-video` | video upload analysis |

Source: `src/backend/app.py`

安全检查示例：

```bash
curl -fsS http://127.0.0.1:8000/api/realtime/health
curl -fsS http://127.0.0.1:8000/api/archive/health
```

这些命令验证服务是否 reachable，不验证模型 accuracy。

## 6. Realtime Live Monitor 验证

Live Monitor 验证重点：

1. 浏览器 camera permission 是否允许；
2. `/api/realtime/session/start` 是否成功；
3. `/api/realtime/frame` 是否持续返回；
4. response 是否包含 frame evidence 和 temporal state；
5. UI risk display 是否更新；
6. warning overlay 是否按状态出现；
7. sound/critical acknowledgement 是否按 UI 逻辑工作；
8. stop camera 后 `/api/realtime/session/stop` 是否成功；
9. stable events 是否进入 notification/history ingestion；
10. History 页面是否显示 Live Monitor records。

Source: `SystemUI/src/components/dashboard/LiveVideoCard.tsx`, `SystemUI/src/components/dashboard/LiveMonitorPage.tsx`, `src/backend/app.py`

边界：看到 warning overlay 只说明 runtime rule 触发了 warning-candidate，不说明该帧或该段视频是人工真值疲劳。

## 7. Video Upload 验证

Video Upload 验证重点：

1. 文件选择是否成功；
2. `/api/analyze-video` 是否返回 response；
3. `outputs/system_video_upload_runs/{session_id}/` 是否创建；
4. `summary.json` 是否可读；
5. `timeline.csv` / `fusion_timeline.csv` 是否存在；
6. Alert Intervals 是否显示；
7. Evidence Timeline figures 是否加载；
8. keyframes metadata 和 thumbnail 是否存在；
9. `Download report` 是否生成 HTML；
10. Technical Details 是否链接到 backend artifacts。

Source: `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx`, `SystemUI/src/components/video-upload/EvidenceFigures.tsx`, `src/runtime/system_video_upload_pipeline.py`, `src/runtime/keyframe_extractor.py`

边界：上传分析成功是 pipeline validation，不是 labelled drowsiness accuracy validation。

## 8. History 和 Insights 验证

History/Insights 验证重点：

- Live Monitor 事件是否写入 `visionguard.history48h.v1`；
- backend archive 是否包含 source=`live_monitor` records；
- History default time window 是否为 48h；
- Recent Drives 是否显示 session/drive；
- local/archive merge 是否 dedupe；
- Insights 是否从 Live Monitor scope 生成 summary；
- Video Upload results 不应被默认当作 History/Insights Live Monitor 统计。

Source: `SystemUI/src/components/history-48h/History48hPage.tsx`, `SystemUI/src/components/insights/InsightsPage.tsx`, `SystemUI/src/lib/history48hStorage.ts`, `SystemUI/src/lib/backendArchiveApi.ts`

## 9. Frontend Build 和 Lint 验证

确认命令：

```bash
cd SystemUI
npm run lint
npm run build
```

`npm run lint` 检查 ESLint 规则；`npm run build` 检查 Next.js production build。它们能发现 TypeScript/React/build 问题，但不能证明 backend 可用，也不能证明模型准确率。

Source: `SystemUI/package.json`

## 10. Deployment 验证

远程测试验证重点：

- local backend `/api/realtime/health` 可访问；
- local backend `/api/archive/health` 可访问；
- Cloudflare tunnel URL health endpoints 可访问；
- Vercel `NEXT_PUBLIC_API_BASE_URL` 指向当前 tunnel；
- backend CORS 包含 exact Vercel origin；
- Vercel env 改动后已 redeploy；
- `scripts/deployment_preflight.sh` 通过。

Source: `docs/DEPLOYMENT_RUNBOOK.md`, `scripts/deployment_preflight.sh`

## 11. Troubleshooting Matrix

| Symptom | Likely Cause | Where to Check | Safe Fix | What Not To Do |
|---|---|---|---|---|
| CORS error | allowed origins 不包含 frontend origin | browser console, `VISIONGUARD_ALLOWED_ORIGINS` | 更新 env 并重启 backend | 不要改模型代码 |
| Backend unreachable | backend process 停止或 URL 错 | terminal, `/api/realtime/health` | 启动 backend 或修正 URL | 不要清空 history |
| Upload fails | 文件过大、backend stopped、pipeline error | backend logs, run folder | 检查 logs 和 artifact | 不要重训模型 |
| No face detected | camera angle/light/face visibility 问题 | Live Monitor UI, backend response | 调整摄像头和光照 | 不要把 no-face 当 safe |
| No keyframes | 没有 warning intervals 或 artifact missing | run folder, keyframes endpoint | 检查 summary/intervals | 不要伪造 keyframes |
| Evidence figures missing | figure artifact path 不存在或 backend URL 错 | run `figures/`, network tab | 检查 artifact URL | 不要用前端自制图替代 |
| History empty | 没有 Live Monitor stable event 或 archive unavailable | localStorage, archive health | 先生成 Live Monitor event | 不要删除 DB |
| Insights empty | 没有 Live Monitor records | History page, archive records | 先确认 data scope | 不要混入 upload records |
| Archive write rejected | payload unsafe 或 token mismatch | backend response, archive code | 检查 payload/token | 不要存 raw frame/base64 |
| Build fails | TS/ESLint/Next 问题 | build output | 修复前端代码 | 不要忽略错误部署 |
| Checkpoint missing | 文件未同步 | checkpoint paths | 恢复 checkpoint | 不要随手 retrain |

## 12. Validation Boundary

需要明确区分：

- health check pass ≠ 模型准确率；
- upload analysis pass ≠ ground-truth drowsiness detection；
- frontend warning displayed ≠ 人工真值 alert；
- History/Insights charts ≠ model evaluation metrics；
- specialist model metrics ≠ full-system accuracy；
- deployment success ≠ safety certification。

## 13. 初学者检查清单

- 我是否先看了 `git status --short`？
- 我是否确认 backend URL 是当前 URL？
- 我是否确认 CORS origin 精确匹配？
- 我是否确认 checkpoint 存在？
- 我是否能区分 localStorage 和 SQLite archive？
- 我是否能解释 upload figures 是 runtime evidence？
- 我是否避免把 demo success 写成 accuracy？

## 14. 常见错误

- 删除 archive/localStorage 来修复 UI count；
- 为了 demo 好看调整 threshold；
- backend URL 错时去重训模型；
- 把工作 demo 当成 accuracy evaluation；
- 报告问题时不写 source path；
- 把 upload artifacts 和 History analytics 混为一谈；
- tunnel 没通过就更新 Vercel。
