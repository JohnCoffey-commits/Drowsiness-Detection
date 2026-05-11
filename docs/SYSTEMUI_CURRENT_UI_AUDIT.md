# SystemUI 当前界面审计

生成日期：2026-05-10

## 1. 总览

`SystemUI/` 是一个独立的 Next.js 前端项目，用于展示驾驶员疲劳/困倦监测系统的仪表盘界面。

当前技术栈：

| 项目 | 当前情况 |
|---|---|
| 框架 | Next.js `16.2.5`，App Router |
| React | React `19.2.4` |
| 语言 | TypeScript |
| 样式 | Tailwind CSS 4、`tw-animate-css`、shadcn/base-ui 风格组件 |
| 图表 | `recharts` |
| 图标 | `lucide-react` |
| UI 名称 | `VisionGuard` |
| 当前数据来源 | 大部分页面使用 `SystemUI/src/lib/mockData.ts` 中的 mock 数据 |
| 后端接入 | 只有 `/video-upload` 页面显式调用外部 FastAPI 后端 `POST /api/analyze-video` |

重要说明：

- 当前 SystemUI 不是 Python 模型逻辑本身。
- 除 `/video-upload` 外，大部分界面是展示型 mock dashboard。
- 当前没有在 SystemUI 内部实现 Next.js API route。
- `SystemUI/.next/` 是构建产物，`SystemUI/node_modules/` 是依赖目录，不属于主要源码。

## 2. 顶层结构

主要源码路径：

| 路径 | 作用 |
|---|---|
| `SystemUI/src/app/` | Next.js App Router 页面目录 |
| `SystemUI/src/components/dashboard/` | 仪表盘专用组件 |
| `SystemUI/src/components/ui/` | 通用 UI 基础组件 |
| `SystemUI/src/lib/mockData.ts` | 所有 mock 页面数据、类型、格式化工具 |
| `SystemUI/public/` | 静态图片资源，如 `eye.png`、`yawn.png` |
| `SystemUI/package.json` | 前端依赖和脚本 |
| `SystemUI/src/app/globals.css` | 全局样式、Tailwind、暗色模式覆盖 |

## 3. 全局布局与导航

### `SystemUI/src/app/layout.tsx`

全站根布局：

- 使用 Google `Inter` 字体。
- 引入全局 CSS。
- 页面左侧固定显示 `Sidebar`。
- 主内容区域左边距固定为 `260px`，对应 sidebar 宽度。
- 注入一段 `themeScript`，从 `localStorage` 读取 `visionguard-theme`，初始化 dark mode。

页面结构大致为：

```text
html
  body
    flex min-h-screen
      Sidebar
      main content area
```

### `SystemUI/src/components/dashboard/Sidebar.tsx`

左侧导航栏，品牌名为 `VisionGuard`，副标题为 `Driver Drowsiness System`。

当前导航项：

| 路由 | 菜单名 | 作用 |
|---|---|---|
| `/` | Dashboard | 总览仪表盘 |
| `/live-monitor` | Live Monitor | 模拟实时监控 |
| `/history-48h` | 48h History | 最近 48 小时历史统计 |
| `/session-review` | Session Review | 单次驾驶 session 回看 |
| `/insights` | Insights | 行为/疲劳模式洞察 |
| `/model-details` | Model Details | 模型结构和模块指标展示 |
| `/video-upload` | Video Upload | Stage 17 视频上传分析 MVP 页面 |
| `/alerts` | Alerts | 告警记录管理 |
| `/settings` | Settings | 输入源、模型、阈值和导出设置 |

底部固定显示：

- `System Status`
- `All systems operational`

### `SystemUI/src/components/dashboard/TopBar.tsx`

所有主要页面顶部栏：

- 显示页面标题。
- 显示 `LIVE` 或 `OFFLINE` 状态，目前来自 mock 数据 `dashboardData.status.isLive`。
- 显示驾驶时长计时器，基于 mock 的 `sessionStartedSecondsAgo` 每秒递增。
- 提供 `ThemeToggle` 日/夜模式切换。
- 显示驾驶员 mock 头像缩写和名字：`JC` / `John Carter`。

## 4. 页面清单

## 4.1 Dashboard 首页 `/`

源码：

- `SystemUI/src/app/page.tsx`

页面定位：

- 系统总览首页。
- 展示模拟实时视频、闭眼/yawn 计数、风险仪表、趋势图和最近事件。

主要内容：

| 区块 | 组件/数据 | 当前展示 |
|---|---|---|
| 顶部栏 | `TopBar` | `Live Monitoring Dashboard` |
| 视频卡片 | `LiveVideoCard` | 使用 Unsplash 驾驶员图片作为背景，叠加眼睛 ROI 框、嘴部 ROI 框、Face Tracking、REC、FPS |
| 状态指标 | `StatusMetricCard` | `EYES CLOSED` 事件数、`YAWN` 事件数 |
| 风险卡片 | `DrowsinessRiskCard` | 仪表盘式 `Drowsiness Risk`，当前 mock 为 `Medium / 62` |
| 趋势图 | `DrowsinessLevelChart` | 最近 1 小时 drowsiness score 折线图，数据在客户端随机生成 |
| 最近事件 | `RecentEventsList` | Normal / Eyes Closed / Yawning 事件流 |

数据来源：

- `dashboardData`
- `DrowsinessLevelChart` 内部随机生成 60 个时间点数据。

当前状态：

- 主要是 mock/demo dashboard。
- 未直接调用 Python 模型或真实后端。

## 4.2 Live Monitor `/live-monitor`

源码：

- `SystemUI/src/app/live-monitor/page.tsx`

页面定位：

- 模拟实时运行监控界面。
- 显示视频、ROI 状态、模型信号和事件流。

主要内容：

| 区块 | 当前展示 |
|---|---|
| 视频区 | 复用 `LiveVideoCard`，显示模拟驾驶员画面和 ROI overlay |
| Tracking Status | Face tracking、Mouth ROI、Left eye ROI、Right eye ROI，当前均为 Tracking/Locked |
| Model Signals | Yawn probability、Eye-closure probability 两个进度条 |
| 模块状态 | mouthState、eyeState、fatigueState |
| 处理性能 | FPS、frame latency |
| Temporal window | 当前 mock 为 30 秒 |
| Input source | Demo video |
| ROI Previews | Mouth crop、Left eye crop、Right eye crop 的占位预览 |
| Runtime Event Stream | 最新运行事件，例如 eye closure duration increased、face tracking recovered |

数据来源：

- `liveMonitorData`

当前状态：

- 页面表现为实时监控，但内容来自 mock 数据。
- 没有真正接入摄像头或实时 Python pipeline。

## 4.3 48h History `/history-48h`

源码：

- `SystemUI/src/app/history-48h/page.tsx`

页面定位：

- 最近 48 小时驾驶疲劳历史统计。

主要内容：

| 区块 | 当前展示 |
|---|---|
| Summary metrics | Monitored time、Warning events、Yawn events、Eye closures |
| 48-Hour Fatigue Risk Trend | fatigueScore 折线图 |
| High-Risk Time Periods | 高风险时间段列表，包含原因和 peak score |
| Warning and Danger Events | warning/danger 事件柱状图 |
| Yawn and Eye-Closure Counts | yawns/eyeClosures 柱状图 |
| Driving Sessions | session 表格：session id、start、duration、risk、max score、events |

数据来源：

- `history48hData`

当前状态：

- 历史分析页面为 mock 数据展示。
- 没有从 Stage 17 输出目录读取真实 batch 结果。

## 4.4 Session Review `/session-review`

源码：

- `SystemUI/src/app/session-review/page.tsx`

页面定位：

- 单次驾驶 session 的回看和事件复盘。

主要内容：

| 区块 | 当前展示 |
|---|---|
| Session selector | 下拉选择 `S-1048` 或 `S-1046` |
| Session metrics | Duration、Average score、Warnings、Fatigue cues |
| Fatigue Score Timeline | 当前 session 的 fatigueScore 折线图 |
| Session Video | 视频回放占位区，显示 `Video snapshot stream`，尚未接入真实视频 |
| Important Events | session timeline 事件列表 |
| Key Moments | 关键时刻卡片，占位图标，描述 peak fatigue / eye closure peak 等 |

数据来源：

- `sessionReviewData`

当前状态：

- 交互点：可以切换 mock session。
- 视频和关键截图仍是占位，不读取 Stage 17 keyframes。

## 4.5 Insights `/insights`

源码：

- `SystemUI/src/app/insights/page.tsx`

页面定位：

- 驾驶行为/疲劳模式洞察。

主要内容：

| 区块 | 当前展示 |
|---|---|
| Summary metrics | Late-night risk、Fatigue build-up、Dominant cue |
| High-Risk Time-of-Day Distribution | 不同时段 riskEvents 柱状图 |
| Repeated Fatigue Periods | Late-night sessions、End-of-session drift、Post-duration yawning |
| Yawn Frequency Trend | driving duration 分段 yawn 数 |
| Eye-Closure Trend | driving duration 分段 closure 数 |
| Observations | 用户可读的观察建议 |

数据来源：

- `insightsData`

当前状态：

- 全部是 mock 洞察。
- 尚未从真实 Stage 12/14/15/17 输出汇总生成。

## 4.6 Model Details `/model-details`

源码：

- `SystemUI/src/app/model-details/page.tsx`

页面定位：

- 展示项目的模块结构、模型选择和 specialist-module 指标。

主要内容：

| 区块 | 当前展示 |
|---|---|
| Selected eye model | `MobileNetV2`，MRL Eye specialist |
| Default threshold | `0.50`，`argmax / p_eye_closed >= 0.50` |
| Safety reference | `ResNet18 with threshold 0.30` |
| System Architecture Summary | YawDD/YawDD+ mouth/yawn specialist、MRL Eye specialist、temporal fusion |
| Accuracy note | 明确说明这些是 specialist-module results，不是 final system-level drowsiness accuracy |
| Mouth/Yawn Module table | ResNet18、MobileNetV2、EfficientNet-B0 训练/验证/测试准确率 |
| MRL Eye Module table | ResNet18、MobileNetV2、EfficientNet-B0 训练/验证/测试准确率 |

数据来源：

- `modelDetailsData`

当前状态：

- 这页已经有比较合适的安全边界提醒：指标不是最终系统级准确率。
- 页面展示的是静态 mock/硬编码项目结果，不动态读取 reports。

## 4.7 Video Upload `/video-upload`

源码：

- `SystemUI/src/app/video-upload/page.tsx`

页面定位：

- Stage 17 视频上传分析 MVP 页面。
- 当前 SystemUI 中唯一明确连接后端 API 的页面。

主要交互：

| 控件 | 作用 |
|---|---|
| Backend URL 输入框 | 默认 `http://127.0.0.1:8000` |
| Video file 文件选择 | 接受 `.mp4,.mov,.avi,.m4v,video/*` |
| Analyze video 按钮 | 将文件通过 `FormData` 发送到后端 |

调用后端：

```text
POST {backendUrl}/api/analyze-video
```

期待响应结构：

- `session_id`
- `status`
- `warning_counts`
- `fusion_figure_url`
- `report_url`
- `keyframes`
- `summary`

结果展示：

| 区块 | 当前展示 |
|---|---|
| 状态徽章 | ready / completed 等 |
| 安全提示 | 明确说明 MVP 不输出最终 drowsy/not-drowsy truth，不声称最终系统级准确率 |
| Summary metrics | Normal、Eye-warning、Mouth-warning、High-confidence、Signal unreliable |
| Fusion Timeline | 显示后端返回的 fusion timeline 图片 |
| Keyframe Gallery | 显示后端返回的 warning-candidate keyframes |

当前状态：

- 这是 Stage 17 上传 MVP 的前端入口。
- 它不直接运行模型，只调用外部 FastAPI 后端。
- 没有轮询任务状态；当前实现是同步提交、等待响应。
- 没有上传进度条，只有 loading 状态。
- 没有显示 interval 表格或完整 timeline 表格。

## 4.8 Alerts `/alerts`

源码：

- `SystemUI/src/app/alerts/page.tsx`

页面定位：

- 告警记录列表和过滤。

主要内容：

| 区块 | 当前展示 |
|---|---|
| Unresolved | 未解决告警数 |
| Danger alerts | danger 告警数 |
| Warning alerts | warning 告警数 |
| Alert History | 告警表格 |
| Severity filter | all / normal / warning / danger |
| Time filter | Last 48 hours / Today / All mock data |

表格列：

- Alert id
- Timestamp
- Severity
- Reason
- Fatigue score
- Status

数据来源：

- `alertsData`

当前状态：

- 支持前端过滤。
- 数据是 mock，不接入 Stage 17 输出或后端告警数据库。

## 4.9 Settings `/settings`

源码：

- `SystemUI/src/app/settings/page.tsx`

页面定位：

- 展示输入源、设备状态、模型配置、阈值和导出选项。

主要内容：

| 区块 | 当前展示 |
|---|---|
| Input Source | Demo video / Webcam 单选 |
| System and Device Status | camera、processing、storage、lastSync |
| Model Configuration | mouth/yawn model、eye model 下拉框 |
| Alert Sensitivity | sensitivity range slider |
| Thresholds | yawn threshold、eye threshold、danger score |
| Temporal Window | window length 秒数输入 |
| Export Options | Session CSV、Alert log、Summary report 按钮 |

数据来源：

- `settingsData`

当前状态：

- 多数控件只是本地 uncontrolled input/select，不会保存配置。
- 页面文字明确说明 export actions 等待 future backend integration。
- 虽然 UI 有 Webcam 选项，但当前项目边界下没有实现 webcam/real-time 检测。

## 5. 共享组件

### Dashboard 组件

| 组件 | 作用 |
|---|---|
| `Sidebar` | 左侧固定导航和系统状态 |
| `TopBar` | 页面标题、LIVE 状态、计时器、主题切换、驾驶员信息 |
| `ThemeToggle` | light/dark 切换，使用 `localStorage` key `visionguard-theme` |
| `PageChrome` | 提供 `PageMain`、`Panel`、`MetricTile`、`ToneBadge`、`riskTone` |
| `LiveVideoCard` | 模拟驾驶员画面，叠加 ROI 框、REC、FPS |
| `StatusMetricCard` | Dashboard 上的 EYES CLOSED / YAWN 指标卡 |
| `DrowsinessRiskCard` | 风险仪表盘卡片 |
| `DrowsinessLevelChart` | 最近 1 小时 drowsiness score 折线图，客户端随机生成 |
| `RecentEventsList` | 最近事件列表，时间根据当前时间动态计算 |
| `ChartMount` | 避免 Recharts hydration 问题，挂载后再渲染图表 |

### UI 基础组件

| 组件 | 作用 |
|---|---|
| `badge.tsx` | 基于 Base UI 和 cva 的 Badge |
| `button.tsx` | 基于 Base UI Button 和 cva 的 Button |
| `card.tsx` | Card、CardHeader、CardTitle、CardContent 等 |
| `progress.tsx` | Base UI Progress 包装 |

## 6. 当前数据模型

`SystemUI/src/lib/mockData.ts` 定义了主要类型和 mock 数据：

| 类型/数据 | 作用 |
|---|---|
| `DashboardData` / `dashboardData` | 首页状态、驾驶员、事件、风险 |
| `LiveMonitorData` / `liveMonitorData` | 实时监控页面状态 |
| `History48hData` / `history48hData` | 48 小时历史 |
| `DrivingSession` / `sessionReviewData` | session 回看 |
| `InsightData` / `insightsData` | 洞察页面 |
| `modelDetailsData` | 模型细节和 specialist 指标 |
| `AlertRecord` / `alertsData` | 告警列表 |
| `SettingsData` / `settingsData` | 设置页 |
| `formatHMS`、`formatClock`、`formatHM` | 时间格式化 |

需要注意：

- mock 数据中有 `fatigueScore`、`danger`、`Drowsiness Risk` 等展示词。
- `/video-upload` 页面相对更符合 Stage 17 的安全措辞，使用 warning-candidate 相关表达。
- 如果后续做正式 demo，建议统一把“final drowsy/not-drowsy”风险相关措辞收敛为 warning-candidate 语言。

## 7. 静态资源

`SystemUI/public/` 当前包含：

| 文件 | 用途 |
|---|---|
| `eye.png` | `StatusMetricCard` 眼睛事件图标 |
| `yawn.png` | `StatusMetricCard` yawn 事件图标 |
| `file.svg`、`globe.svg`、`next.svg`、`vercel.svg`、`window.svg` | create-next-app 默认资源，目前主 UI 中未明显使用 |

`LiveVideoCard` 还使用一个外部 Unsplash 图片 URL 作为模拟驾驶员背景。

## 8. 当前后端/真实 pipeline 接入情况

| 页面 | 是否接入后端 | 说明 |
|---|---|---|
| `/video-upload` | 是 | 调用 `POST http://127.0.0.1:8000/api/analyze-video` |
| 其他页面 | 否 | 使用 `mockData.ts` |

仓库根目录中已有 Python 后端：

- `src/backend/app.py`

但它不在 `SystemUI/` 内。SystemUI 只是通过可编辑的 Backend URL 访问它。

当前 `/video-upload` 页面可以展示：

- warning count metrics
- fusion timeline figure
- keyframe gallery

尚未展示：

- 详细 interval 表格
- `summary.json` 的完整字段
- `timeline.csv` / `fusion_timeline.csv` 表格视图
- report markdown 内容
- 后端异步任务进度

## 9. 运行方式

在 `SystemUI/` 目录下：

```bash
npm run dev
```

默认访问：

```text
http://localhost:3000
```

如果要使用 `/video-upload`，还需要在仓库根目录启动 Python 后端，例如：

```bash
source .venv-stage10/bin/activate
python src/backend/app.py --host 127.0.0.1 --port 8000
```

然后在 UI 的 Backend URL 输入框保持：

```text
http://127.0.0.1:8000
```

## 10. 当前界面的总体判断

当前 SystemUI 已经具备完整 dashboard 雏形：

- 有左侧导航。
- 有首页、实时监控、历史、session 回看、洞察、模型详情、上传分析、告警、设置等页面。
- 页面视觉结构统一，组件复用清晰。
- `/video-upload` 已经是 Stage 17 上传 MVP 的前端入口。

但从项目真实功能角度看：

- 绝大多数页面仍是 mock 展示。
- 真实 Python pipeline 只通过 `/video-upload` 页面接入。
- 还没有 live webcam / real-time detection。
- 还没有把 Stage 17 batch 结果或历史输出接入 History、Alerts、Session Review、Insights 等页面。
- 当前 UI 不能被描述为 deployment-ready。

## 11. 建议的后续 UI 整合方向

优先级较高的下一步：

1. 在 `/video-upload` 结果区增加 warning interval 表格，显示 start/end、state、reason、keyframe 链接。
2. 在 `/video-upload` 展示 Stage 17.1/17.2 的 sustained-eye gate 与 manual-review safe wording。
3. 把 `summary.json` 的核心字段完整展示出来，例如 suppressed brief-eye-warning frames、high-confidence intervals、signal-unreliable intervals。
4. 将 `/session-review` 的 key moments 改为读取 upload run 的 keyframes。
5. 将 `/alerts` 改为读取真实 upload run 的 warning candidates，而不是 mock alerts。
6. 保持安全措辞：使用 warning candidate / high-confidence warning candidate / signal unreliable，不声称最终困倦判断。

不建议现在做的事：

- 不要把 UI 文案改成“final drowsiness detected”。
- 不要声称最终系统级准确率。
- 不要把 webcam 选项描述为已实现。
- 不要把 mock History/Insights 当作真实结果。

