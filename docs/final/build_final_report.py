#!/usr/bin/env python3
"""Build the final Chinese PDF report and figures for the drowsiness project."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "final"
FIG_DIR = OUT_DIR / "figures"
PDF_PATH = OUT_DIR / "final_report.pdf"
MD_PATH = OUT_DIR / "final_report.md"

FONT_PATHS = [
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
]

PALETTE = {
    "navy": "#1D3557",
    "blue": "#457B9D",
    "cyan": "#A8DADC",
    "paper": "#F7FAFC",
    "red": "#E63946",
    "orange": "#F18F01",
    "green": "#59A14F",
    "purple": "#A23B72",
    "gray": "#5E6A75",
    "light_gray": "#EEF2F6",
    "white": "#FFFFFF",
}

STAGE7 = [
    ("ResNet18", 98.92, 98.85, 99.37, 96.47, 97.89, 97.18),
    ("MobileNetV2", 98.97, 98.48, 98.75, 91.74, 97.48, 94.52),
    ("EfficientNet-B0", 98.76, 99.08, 99.20, 94.82, 98.13, 96.44),
]

STAGE9 = [
    ("ResNet18", 98.46, 98.46, 98.59, 89, 109, 0.30),
    ("MobileNetV2", 98.63, 98.63, 98.52, 93, 84, 0.30),
    ("EfficientNet-B0", 98.62, 98.62, 98.24, 111, 67, 0.30),
]

FUSION_COUNTS = [
    ("A_normal_open_baseline", 70, 0, 0, 0, 0),
    ("B_realistic_drowsy_simulation", 49, 18, 30, 6, 0),
    ("C_mild_head_motion", 76, 7, 0, 0, 12),
    ("D_controlled_long_open_closed", 54, 65, 0, 0, 0),
]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    chosen = next((p for p in FONT_PATHS if p.exists()), None)
    if chosen is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(chosen), size=size, index=0)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text_by_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    fnt: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for token in text.split(" "):
        candidate = token if not current else f"{current} {token}"
        if text_size(draw, candidate, fnt)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = token
    if current:
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = "#111827",
    align: str = "center",
    line_gap: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    max_width = x2 - x1 - 28
    lines: list[str] = []
    for raw in text.split("\n"):
        lines.extend(wrap_text_by_width(draw, raw, fnt, max_width))
    line_heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(line_heights) + line_gap * max(0, len(lines) - 1)
    y = y1 + max(0, (y2 - y1 - total_h) // 2)
    for line, h in zip(lines, line_heights):
        w, _ = text_size(draw, line, fnt)
        if align == "left":
            x = x1 + 18
        elif align == "right":
            x = x2 - 18 - w
        else:
            x = x1 + (x2 - x1 - w) // 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += h + line_gap


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    width: int = 3,
    radius: int = 22,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str = "#334155",
    width: int = 4,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    length = 18
    spread = math.pi / 7
    p1 = (
        end[0] - length * math.cos(angle - spread),
        end[1] - length * math.sin(angle - spread),
    )
    p2 = (
        end[0] - length * math.cos(angle + spread),
        end[1] - length * math.sin(angle + spread),
    )
    draw.polygon([end, p1, p2], fill=fill)


def canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (1800, 1050), hex_to_rgb(PALETTE["paper"]))
    draw = ImageDraw.Draw(img)
    draw.text((70, 44), title, font=font(42, True), fill=hex_to_rgb(PALETTE["navy"]))
    if subtitle:
        draw.text((72, 102), subtitle, font=font(25), fill=hex_to_rgb(PALETTE["gray"]))
    draw.line([(70, 142), (1730, 142)], fill=hex_to_rgb(PALETTE["cyan"]), width=4)
    return img, draw


def save(img: Image.Image, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    img.save(path, "PNG", dpi=(300, 300))
    return path


def fig_system_pipeline() -> Path:
    img, draw = canvas(
        "Figure 1. Project-level System Pipeline",
        "Specialist visual evidence, temporal rules, and review-oriented applications",
    )
    title_font = font(24, True)
    body_font = font(19)
    boxes = [
        ((80, 230, 340, 390), "Datasets\nYawDD/YawDD+ Dash\nMRL Eye", PALETTE["blue"]),
        ((420, 230, 680, 390), "Preprocessing\nframe rebuild\nmouth / eye ROI\nsubject split", PALETTE["cyan"]),
        ((760, 175, 1050, 330), "Mouth Specialist\nResNet18\np_yawn", PALETTE["orange"]),
        ((760, 405, 1050, 560), "Eye Specialist\nMobileNetV2\np_eye_closed", PALETTE["purple"]),
        ((1130, 280, 1400, 455), "Temporal Rules\nPERCLOS-like eye rule\nrecent yawn window\nquality gate", PALETTE["green"]),
        ((1480, 220, 1720, 520), "Applications\nVideo Upload\nLive Monitor\n48h History\nInsights\nSQLite archive", PALETTE["red"]),
    ]
    for box, text, color in boxes:
        rounded_box(draw, box, "#FFFFFF", color, width=5)
        lines = text.split("\n")
        draw_wrapped_text(draw, (box[0] + 12, box[1] + 16, box[2] - 12, box[1] + 62), lines[0], title_font, color, line_gap=3)
        if len(lines) > 1:
            draw_wrapped_text(
                draw,
                (box[0] + 12, box[1] + 68, box[2] - 12, box[3] - 14),
                "\n".join(lines[1:]),
                body_font,
                "#1F2937",
                line_gap=5,
            )
    arrow(draw, (340, 310), (420, 310))
    arrow(draw, (680, 310), (760, 255))
    arrow(draw, (680, 310), (760, 485))
    arrow(draw, (1050, 255), (1130, 340))
    arrow(draw, (1050, 485), (1130, 390))
    arrow(draw, (1400, 370), (1480, 370))
    rounded_box(draw, (360, 720, 1440, 900), "#FFFFFF", PALETTE["navy"], width=4)
    draw_wrapped_text(
        draw,
        (390, 735, 1410, 890),
        "Core boundary: the system reports rule-based warning-candidate states, not final system-level driver drowsiness truth.",
        font(28, True),
        PALETTE["navy"],
    )
    return save(img, "fig01_system_pipeline.png")


def fig_data_processing() -> Path:
    img, draw = canvas(
        "Figure 2. Data Processing Flow",
        "Two leakage-aware specialist datasets are prepared independently before fusion",
    )
    body = font(18)
    lane_font = font(29, True)
    step_font = font(21, True)
    lanes = [
        ("YawDD/YawDD+ Dash mouth branch", 205, PALETTE["orange"]),
        ("MRL Eye open/closed branch", 600, PALETTE["purple"]),
    ]
    for label, y, color in lanes:
        draw.text((90, y - 70), label, font=lane_font, fill=hex_to_rgb(color))
        steps = [
            ("Raw / annotations", "YawDD videos + YawDD+ frame labels" if y < 400 else "84,898 PNG eye images"),
            ("Manifest + QC", "64,378 labeled frames" if y < 400 else "37 subjects; 0 unreadable images"),
            ("ROI / trainable set", "64,202 mouth crops; 99.73% success" if y < 400 else "closed/open labels; subject manifest"),
            ("Subject split", "train 44,156 / val 8,892 / test 11,154" if y < 400 else "train 58,982 / val 13,029 / test 12,887"),
            ("Specialist output", "p_yawn" if y < 400 else "p_eye_closed"),
        ]
        x = 85
        box_w = 265
        step_gap = 55
        for idx, (t, b) in enumerate(steps):
            box = (x, y, x + box_w, y + 170)
            rounded_box(draw, box, "#FFFFFF", color, width=4)
            draw_wrapped_text(draw, (x + 8, y + 16, x + box_w - 8, y + 68), t, step_font, color)
            draw_wrapped_text(draw, (x + 12, y + 72, x + box_w - 12, y + 158), b, body, "#1F2937", line_gap=5)
            if idx < len(steps) - 1:
                arrow(draw, (x + box_w, y + 85), (x + box_w + 42, y + 85), color)
            x += box_w + step_gap
    return save(img, "fig02_data_processing_flow.png")


def fig_fusion_logic() -> Path:
    img, draw = canvas(
        "Figure 3. F5 Fusion Logic",
        "Tiered quality-aware rule used for synchronized mouth-eye warning candidates",
    )
    decision = font(22, True)
    body = font(21)
    nodes = {
        "start": ((735, 170, 1065, 260), "Aligned timeline row\np_eye_closed + p_yawn"),
        "quality": ((720, 320, 1080, 435), "Eye signal unreliable?"),
        "recent": ((230, 505, 590, 620), "Recent yawn event?"),
        "eye": ((720, 505, 1080, 620), "Eye warning candidate?"),
        "high": ((1200, 505, 1620, 620), "Eye warning + recent yawn?"),
        "signal": ((160, 760, 625, 875), "signal_unreliable\nquality marker, not drowsiness evidence"),
        "mouth": ((700, 760, 1085, 875), "mouth_warning_candidate\nrecent/current yawn evidence"),
        "eyeout": ((1165, 760, 1545, 875), "eye_warning_candidate\ntemporal eye evidence only"),
        "highout": ((1210, 300, 1660, 415), "high_confidence_drowsiness_candidate\nreview cue, not final truth"),
    }
    for key, (box, text) in nodes.items():
        color = PALETTE["navy"]
        if key in {"signal"}:
            color = PALETTE["gray"]
        if key in {"mouth"}:
            color = PALETTE["orange"]
        if key in {"eyeout", "eye"}:
            color = PALETTE["purple"]
        if key in {"high", "highout"}:
            color = PALETTE["red"]
        rounded_box(draw, box, "#FFFFFF", color, width=4)
        draw_wrapped_text(draw, box, text, decision if "?" in text else body, color if "?" in text else "#1F2937")
    arrow(draw, (900, 260), (900, 320))
    arrow(draw, (720, 378), (590, 560), PALETTE["gray"])
    draw.text((620, 438), "yes", font=font(20, True), fill=hex_to_rgb(PALETTE["gray"]))
    arrow(draw, (900, 435), (900, 505), PALETTE["purple"])
    draw.text((925, 465), "no", font=font(20, True), fill=hex_to_rgb(PALETTE["purple"]))
    arrow(draw, (410, 620), (395, 760), PALETTE["gray"])
    draw.text((315, 665), "no", font=font(20, True), fill=hex_to_rgb(PALETTE["gray"]))
    arrow(draw, (590, 562), (700, 815), PALETTE["orange"])
    draw.text((604, 650), "yes", font=font(20, True), fill=hex_to_rgb(PALETTE["orange"]))
    arrow(draw, (1080, 562), (1200, 560), PALETTE["red"])
    arrow(draw, (1410, 505), (1430, 415), PALETTE["red"])
    draw.text((1115, 530), "yes + recent yawn", font=font(20, True), fill=hex_to_rgb(PALETTE["red"]))
    arrow(draw, (900, 620), (900, 760), PALETTE["orange"])
    draw.text((925, 665), "no eye, recent yawn", font=font(20, True), fill=hex_to_rgb(PALETTE["orange"]))
    arrow(draw, (1080, 620), (1270, 760), PALETTE["purple"])
    draw.text((1118, 690), "eye only", font=font(20, True), fill=hex_to_rgb(PALETTE["purple"]))
    return save(img, "fig03_fusion_logic.png")


def fig_model_performance() -> Path:
    img, draw = canvas(
        "Figure 4. Specialist Model Performance",
        "Stage 7 mouth/yawn and Stage 9 MRL Eye baseline comparison",
    )
    axis = "#334155"
    small = font(18)
    title = font(28, True)

    panels = [
        (90, 220, 790, 840, "Mouth/Yawn Test Accuracy", [(m, acc) for m, _, _, acc, *_ in STAGE7], PALETTE["orange"]),
        (1010, 220, 1710, 840, "Eye Test Macro F1", [(m, f1) for m, _, f1, *_ in STAGE9], PALETTE["purple"]),
    ]
    for x1, y1, x2, y2, heading, data, color in panels:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=20, fill="#FFFFFF", outline=hex_to_rgb(PALETTE["light_gray"]), width=3)
        draw.text((x1 + 28, y1 + 28), heading, font=title, fill=hex_to_rgb(PALETTE["navy"]))
        chart = (x1 + 75, y1 + 110, x2 - 50, y2 - 90)
        draw.line([(chart[0], chart[3]), (chart[2], chart[3])], fill=axis, width=3)
        draw.line([(chart[0], chart[1]), (chart[0], chart[3])], fill=axis, width=3)
        min_v = 94 if "Mouth" in heading else 97
        max_v = 100
        bar_w = 115
        gap = 80
        for i, (name, value) in enumerate(data):
            x = chart[0] + 55 + i * (bar_w + gap)
            h = int((value - min_v) / (max_v - min_v) * (chart[3] - chart[1]))
            y = chart[3] - h
            draw.rounded_rectangle((x, y, x + bar_w, chart[3]), radius=12, fill=hex_to_rgb(color))
            draw.text((x - 8, y - 36), f"{value:.2f}%", font=small, fill=hex_to_rgb(PALETTE["navy"]))
            for line_i, line in enumerate(textwrap.wrap(name, width=12)):
                tw, _ = text_size(draw, line, small)
                draw.text((x + (bar_w - tw) / 2, chart[3] + 16 + line_i * 24), line, font=small, fill=axis)
        draw.text((chart[0] - 12, chart[1] - 4), f"{max_v}%", font=small, fill=axis)
        draw.text((chart[0] - 12, chart[3] - 12), f"{min_v}%", font=small, fill=axis)
    return save(img, "fig04_model_performance.png")


def fig_stage15_counts() -> Path:
    img, draw = canvas(
        "Figure 5. Stage 15 Fusion State Counts",
        "Controlled-realistic A/B/C/D validation with F5 tiered quality-aware fusion",
    )
    colors_map = [
        ("Normal", PALETTE["green"]),
        ("Eye warning", PALETTE["purple"]),
        ("Mouth warning", PALETTE["orange"]),
        ("High confidence", PALETTE["red"]),
        ("Signal unreliable", PALETTE["gray"]),
    ]
    chart = (220, 220, 1650, 820)
    draw.line([(chart[0], chart[3]), (chart[2], chart[3])], fill="#334155", width=3)
    draw.line([(chart[0], chart[1]), (chart[0], chart[3])], fill="#334155", width=3)
    max_total = max(sum(row[1:]) for row in FUSION_COUNTS)
    y_scale = (chart[3] - chart[1]) / max_total
    bar_w = 185
    gap = 125
    for i, row in enumerate(FUSION_COUNTS):
        slug, *vals = row
        x = chart[0] + 80 + i * (bar_w + gap)
        y = chart[3]
        for val, (_, col) in zip(vals, colors_map):
            h = val * y_scale
            draw.rectangle((x, y - h, x + bar_w, y), fill=hex_to_rgb(col))
            if val >= 6:
                draw_wrapped_text(draw, (x, int(y - h), x + bar_w, int(y)), str(val), font(20, True), "#FFFFFF")
            y -= h
        for line_i, line in enumerate(textwrap.wrap(slug.replace("_", " "), width=18)):
            tw, _ = text_size(draw, line, font(17))
            draw.text((x + (bar_w - tw) / 2, chart[3] + 18 + line_i * 22), line, font=font(17), fill="#334155")
    lx, ly = 230, 870
    for label, col in colors_map:
        draw.rectangle((lx, ly, lx + 28, ly + 28), fill=hex_to_rgb(col))
        draw.text((lx + 40, ly), label, font=font(20), fill="#334155")
        lx += 285
    return save(img, "fig05_stage15_fusion_counts.png")


def generate_figures() -> list[Path]:
    return [
        fig_system_pipeline(),
        fig_data_processing(),
        fig_fusion_logic(),
        fig_model_performance(),
        fig_stage15_counts(),
    ]


def markdown_report() -> str:
    return """# 基于眼-嘴双通道证据融合的驾驶疲劳候选预警系统报告

作者：Drowsiness Detection Project Group  
生成位置：`docs/final/final_report.pdf`  
生成日期：2026-05-21  

## Abstract / 摘要

本文报告一个模块化驾驶疲劳检测原型系统。项目没有采用端到端“疲劳/非疲劳”单一分类器，而是将可见驾驶员行为拆分为两个可解释的视觉证据通道：基于 YawDD/YawDD+ Dash 数据的嘴部打哈欠识别模块输出 `p_yawn`，基于 MRL Eye 数据的眼部闭合识别模块输出 `p_eye_closed`。两个专门模型的结果再经过运行时 ROI 提取、时序平滑、PERCLOS-like 规则和质量门控，形成 rule-based warning-candidate timeline。当前系统已经扩展到 FastAPI 后端、Next.js 前端、视频上传分析、实时 webcam Live Monitor、本地 48h history/insights 页面与 SQLite 摘要归档，但所有输出仍限定为“疲劳候选预警”而非最终驾驶疲劳真值。

项目的核心训练结果来自 `colab_file/stage7_yawdd_training_r.ipynb` 和 `colab_file/stage9_mrl_eye_training_r.ipynb`。Stage 7 中，YawDD 嘴部模型以 ResNet18 取得最高测试准确率 99.37%，yawn F1 为 97.18%。Stage 9/9B 中，MRL Eye 眼部模型选择 MobileNetV2 作为主模型，测试准确率 98.63%，macro F1 为 98.63%，closed-eye recall 为 98.52%。后续 Stage 12-17 没有重新训练模型，而是围绕信号质量、时序持续性和人类审阅边界设计融合逻辑。该设计在 A/B/C/D 小规模受控视频上达成预期行为：正常视频不触发候选告警，真实打哈欠片段产生嘴部候选，长闭眼片段产生眼部候选，头部遮挡被标注为 signal-unreliable。

![Project-level system pipeline](figures/fig01_system_pipeline.png)

## 1. Introduction and Background / 引言与背景

驾驶疲劳检测的难点不只在分类模型本身，还在于“疲劳”这一状态很难由单帧视觉证据直接定义。困倦驾驶事故识别通常依赖事后调查和行为证据，漏报风险较高。PERCLOS 相关研究将眼睑闭合时间比例作为警觉性下降的重要指标，NHTSA 相关报告也将 Perclose/PERCLOS 与视觉注意力 lapses 建立了实验联系。本项目吸收这一思路，但没有直接测量真实眼睑开合百分比，而是将 CNN 的 `p_eye_closed` 作为闭眼概率代理，因此项目文档中使用 “PERCLOS-like” 或 “PERCLOS-inspired” 的表述更准确。

项目采用双通道行为证据的原因也较充分。打哈欠和眼部闭合都可作为疲劳相关行为线索，但单独使用任一信号都存在歧义：张嘴可能来自说话、表情或短暂动作，闭眼概率升高可能来自眨眼、眯眼、反光、眼镜、头部姿态和 ROI 偏差。将嘴部 yawn evidence 与眼部 temporal eye-warning evidence 分离建模，再在时序层进行规则融合，可以让每个模块的训练标签更清楚，也能把“模型分类结果”和“系统级解释”隔离开来。

本报告的写作边界与项目代码保持一致。训练指标只用于评价 specialist model，运行时状态只表示 warning-candidate，不能写成最终系统级驾驶疲劳准确率、临床结论、真实道路验证或部署就绪性。

## 2. Overview of the Architecture/System / 系统架构概述

项目整体结构以 `docs/PROJECT_STRUCTURE.md` 为主线。`dataset/` 存放本地原始或重建数据，`artifacts/` 存放映射表、划分文件和中间结果，`outputs/` 存放训练与运行时证据，`reports/` 存放各阶段人类可读报告，`src/` 负责数据处理、训练和运行时推理，`src/backend/` 提供 FastAPI 服务，`SystemUI/` 提供 Next.js 前端。

系统可概括为四层。数据层中，YawDD/YawDD+ Dash 被重建为 64,378 个带标签帧，并通过 MediaPipe Face Mesh 嘴唇 landmarks 生成 64,202 个可训练嘴部 crop；MRL Eye 则提供 84,898 张眼部图片，标签为 `0 = closed` 和 `1 = open`。训练层中，两个专门任务都使用 ResNet18、MobileNetV2、EfficientNet-B0 进行迁移学习 baseline 比较。运行时层中，MediaPipe FaceLandmarker 从完整人脸视频提取眼部和嘴部 ROI，眼部模型输出 `p_eye_closed`，嘴部模型输出 `p_yawn`，Stage 12 使用质量门控的 rolling PERCLOS-like 规则，Stage 13-15 使用 `F5_tiered_quality_aware_fusion` 生成融合状态。应用层中，Stage 17 负责上传视频分析，Stage 19 负责实时 webcam 原型，Stage 20-22 增加本地账号、主题、通知、history/insights 和 SQLite 摘要归档。

当前后端入口为 `src/backend/app.py`，关键接口包括 `POST /api/analyze-video`、`GET /api/realtime/health`、`POST /api/realtime/session/start`、`POST /api/realtime/frame`、`POST /api/realtime/session/stop` 以及本地 archive API。实时单帧证据由 `src/runtime/realtime_frame_inference.py` 生成，session-local temporal state 由 `src/runtime/realtime_temporal_state.py` 维护。视频上传完整流水线由 `src/runtime/system_video_upload_pipeline.py` 串联 Stage 10、Stage 11、Stage 12-style adapter、Stage 14、F5 fusion 和 keyframe extraction。

![Data processing flow](figures/fig02_data_processing_flow.png)

## 3. Data Processing and Model Training / 数据处理与模型训练

YawDD/YawDD+ Dash 嘴部数据处理质量较高。重建阶段得到 29 个 subject、64,378 个标注帧，其中 `no_yawn` 为 57,347 帧，`yawn` 为 7,031 帧。嘴部 crop 阶段处理 64,378 帧，MediaPipe Face Mesh 成功 crop 64,093 帧，fallback lower-face crop 109 帧，失败 176 帧，成功率为 99.73%。subject-level split 避免同一 subject 跨 train/val/test 泄漏：train 44,156 张，val 8,892 张，test 11,154 张，三个 split 的 yawn rate 均约 11%。

MRL Eye 数据集经本地检查包含 84,898 张图片、37 个 subject，closed 41,946 张，open 42,952 张。subject-level split 为 train 58,982、val 13,029、test 12,887，三个 split 均包含 closed/open，泄漏检查通过。MRL Eye 的类别比例整体接近均衡，但 subject 内部分布差异较大，因此 subject-level split 比随机 image-level split 更适合作为当前项目的保守评估策略。

两个训练 notebook 均使用 PyTorch / torchvision。Stage 7 嘴部训练采用 224×224 输入、Adam、学习率 `1e-4`、weighted cross entropy、ReduceLROnPlateau、early stopping patience 3，并使用训练集上的轻量旋转、亮度/对比度扰动和仿射缩放。Stage 9 眼部训练采用 224 输入、batch size 64、最多 10 个 epoch、freeze epoch 1、weighted cross entropy、validation macro F1 作为 checkpoint metric，并要求 pretrained weights 成功加载。

## 4. Fusion and Runtime Decision Logic / 融合层逻辑判断

Stage 12 的眼部时序规则选择为 `quality_gated_perclos_mean_ge_0.60_consec`。该规则要求 rolling PERCLOS-like mean-binary ratio 大于等于 0.60，并持续至少 2 个 sampled frames；若 5 帧窗口内 no-face ratio 大于 0.20，则标记为 `signal_unreliable`。这使系统避免把追踪失败当作疲劳证据，也能抑制正常视频中的短暂单帧波动。

Stage 14 的嘴部运行时逻辑从 full-face video 中提取 mouth/lip ROI，并使用恢复的 Stage 7 ResNet18 checkpoint 计算 `p_yawn = softmax(logits)[1]`。`p_yawn >= 0.50` 的 sampled row 记为 yawn event，后续时间窗口内会保留 recent-yawn context。recent-yawn context 是融合上下文，不等于当前帧必然正在打哈欠。

F5 fusion 的核心思想是分层处理质量、嘴部证据和眼部证据。若 eye signal 不可靠且没有 recent yawn，则输出 `signal_unreliable`；若 eye signal 不可靠但存在 recent yawn，则输出 `mouth_warning_candidate`；若 eye warning 与 recent yawn 共同出现，则输出 `high_confidence_drowsiness_candidate`；若只有 eye warning，则输出 `eye_warning_candidate`；若只有 recent yawn，则输出 `mouth_warning_candidate`；否则输出 `normal`。Stage 17.1/17.5 又增加持续眼部证据和强度门控，避免 brief blink-like 或 weak eye evidence 与 recent yawn 偶然重叠时被过度升级。

![F5 fusion logic](figures/fig03_fusion_logic.png)

实时 Live Monitor 使用相同的模型语义，但处理方式更偏向 session-local state。单帧后端只返回 `p_eye_closed`、`p_yawn`、ROI 状态和 signal quality；时序状态由 `RealtimeTemporalState` 在当前会话内维护。Live Monitor 默认 2 FPS 采样，使用 yawn on/off hysteresis、eye warning enter/exit rolling mean、sustained eye-warning 判断、recent reminder 和 cooldown 逻辑驱动前端 overlay、sound cue、risk gauge 和 dashboard event。该路径不存储 raw frame、raw image、raw video 或 blob，只保存轻量 summary/event records。

## 5. Results and Evaluation / 结果与评估

Stage 7 训练结果直接来自 `colab_file/stage7_yawdd_training_r.ipynb` 与恢复后的 `artifacts/recovered_stage7_mouth_yawn/initial_results.csv`。ResNet18 因测试准确率和 yawn F1 表现最佳，被选为嘴部打哈欠专门模型。EfficientNet-B0 的 validation accuracy 更高，但测试集整体表现略低于 ResNet18。考虑到类别不平衡，报告不应只写 accuracy，yawn precision/recall/F1 和 confusion matrix 更能说明模型是否漏掉少数类 yawn。

| Mouth/Yawn Model | Train Acc | Val Acc | Test Acc | Yawn Precision | Yawn Recall | Yawn F1 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 98.92% | 98.85% | 99.37% | 96.47% | 97.89% | 97.18% |
| MobileNetV2 | 98.97% | 98.48% | 98.75% | 91.74% | 97.48% | 94.52% |
| EfficientNet-B0 | 98.76% | 99.08% | 99.20% | 94.82% | 98.13% | 96.44% |

Stage 9/9B 训练和模型选择结果显示，MobileNetV2 是当前主眼部模型，因为默认阈值下 test accuracy、macro F1、误报/漏报平衡和实时部署适配性最好。ResNet18 at `p_eye_closed >= 0.30` 被保留为 safety-prioritized reference：closed recall 提升到 99.08%，false-open 降到 58，但 false-closed 增加到 251，因此不适合作为默认设置。这里的 false-open 是 true closed 被预测为 open，安全意义上更敏感；false-closed 是 true open 被预测为 closed，主要体现误报倾向。

| Eye Model | Test Acc | Test Macro F1 | Closed Recall | False Open | False Closed | Val-selected Threshold |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 98.46% | 98.46% | 98.59% | 89 | 109 | 0.30 |
| MobileNetV2 | 98.63% | 98.63% | 98.52% | 93 | 84 | 0.30 |
| EfficientNet-B0 | 98.62% | 98.62% | 98.24% | 111 | 67 | 0.30 |

![Specialist model performance](figures/fig04_model_performance.png)

Stage 15 使用真实 Stage 12 eye timeline 和真实 Stage 14 model-generated `p_yawn` timeline 完成同步融合。A/B/C/D 受控视频的 F5 融合结果如下。B 视频中，用户手动观察到 14.3s-16.8s 左右存在打哈欠；Stage 14 在该窗口内 12/12 行触发 yawn-event，mean `p_yawn` 约为 0.981。Stage 15 的 high-confidence candidate 出现在 recent-yawn evidence 与 eye-warning evidence 发生重叠的时段。C 视频中，头部运动、头发/手遮挡等问题被部分归入 signal quality，而不是直接升级为疲劳结论。

| Video | Normal | Eye Warning | Mouth Warning | High Confidence Candidate | Signal Unreliable |
|---|---:|---:|---:|---:|---:|
| A_normal_open_baseline | 70 | 0 | 0 | 0 | 0 |
| B_realistic_drowsy_simulation | 49 | 18 | 30 | 6 | 0 |
| C_mild_head_motion | 76 | 7 | 0 | 0 | 12 |
| D_controlled_long_open_closed | 54 | 65 | 0 | 0 | 0 |

![Stage 15 fusion counts](figures/fig05_stage15_fusion_counts.png)

Stage 17 将上述 pipeline 封装为 uploaded-video MVP。后端验证 `B_realistic_drowsy_simulation.mp4` 时得到 103 个 sampled frames、18 个 eye-warning candidate frames、30 个 yawn warning candidate frames、6 个 critical/high-confidence eye warning candidate frames、14 个 yawn events 和 3 个 keyframes。`upload_test/C_upload_test.mp4` 的本地 UI 验证 markers 包括 9 个 high-confidence warning candidate frames、8 个 suppressed brief-eye escalation frames、4 个 keyframes 和 3 张 figures。Stage 19 Live Monitor 则将同一模型证据链迁移到 webcam session，支持 2 FPS 自动采样、实时 frame endpoint、session-local temporal state、overlay、sound cue、risk gauge、history ingestion 和 SQLite summary archive。

## 6. Discussion and Conclusions / 讨论与结论

该项目的主要工程优势在于边界控制清楚。训练指标只声称为 specialist-module performance，运行时输出只声称为 warning-candidate，前端和后端也保留了永久解释文本，避免把单帧概率或规则状态误写成最终驾驶疲劳检测。数据划分采用 subject-level split，也比随机 frame split 更能降低身份和相邻帧泄漏风险。

系统的主要风险来自泛化能力和真实世界验证不足。YawDD 嘴部模型训练于重建 Dash mouth crops，MRL Eye 眼部模型训练于眼部 crop 图片，运行时视频再由 MediaPipe 生成 ROI；训练域和运行时域并不完全一致。Stage 10-15 的 A/B/C/D 视频说明 pipeline 能在小规模受控场景下工作，但 subject 数、光照、摄像头、遮挡、眼镜反光、头部姿态、真实驾驶环境和疲劳 ground truth 都不足。当前系统也没有训练 learned fusion classifier，F5 是规则融合，适合 demo 和审阅辅助，不适合写成生产级疲劳检测器。

结论上，本项目已经完成一个结构完整、证据链较清楚的本地驾驶疲劳候选预警原型：嘴部模型能识别打哈欠证据，眼部模型能识别闭眼证据，时序规则能处理持续性，质量门控能隔离 no-face/ROI failure，前后端能支持上传视频和实时 webcam 原型。若要把项目提升到可发表或可部署级别，下一步应采集更多同步眼-嘴视频，建立 temporal ground-truth fatigue/warning annotation，按 subject/camera/lighting 条件做分层评估，并在有足够标注后再考虑 learned temporal fusion。

## References / 参考文献

[1] NHTSA. Drowsy Driving: Countermeasures That Work. https://www.nhtsa.gov/book/countermeasures-that-work/drowsy-driving  
[2] Dinges, D. F., Mallis, M. M., Maislin, G., & Powell, J. W. Evaluation of techniques for ocular measurement as an index of fatigue and as the basis for alertness management. NHTSA, 1998. https://rosap.ntl.bts.gov/view/dot/2518  
[3] FMCSA/NHTSA. PERCLOS: A Valid Psychophysiological Measure of Alertness. https://ntlsearch.bts.gov/ntl/md.do?id=51369  
[4] Abtahi, S., Omidyeganeh, M., Shirmohammadi, S., & Hariri, B. YawDD: A Yawning Detection Dataset. ACM MMSys Workshop, 2014. https://www.site.uottawa.ca/~shervin/pubs/CogniVue-Dataset-ACM-MMSys2014.pdf  
[5] MRL. MRL Eye Dataset. https://mrl.cs.vsb.cz/eyedataset.html  
[6] Google MediaPipe. MediaPipe Face Mesh. https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh  
[7] He, K., Zhang, X., Ren, S., & Sun, J. Deep Residual Learning for Image Recognition. CVPR 2016.  
[8] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.  
[9] Tan, M., & Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.  
[10] Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.

## Appendices / 附录

主要内部证据文件包括 `docs/PROJECT_STRUCTURE.md`、`docs/PROJECT_CURRENT_STATUS.md`、`reports/stage16_final_integration_summary_report.md`、`reports/stage15_real_mouth_eye_fusion_validation_report.md`、`reports/stage14_mouth_yawn_runtime_validation_report.md`、`reports/stage12_eye_alert_rule_analysis_report.md`、`reports/mrl_eye_stage9b_error_analysis.md`、`reports/yawdd_dash_split_report.md`、`reports/mrl_eye_split_report.md`、`colab_file/stage7_yawdd_training_r.ipynb` 和 `colab_file/stage9_mrl_eye_training_r.ipynb`。

能力使用审计：本次生成使用 research-writing-assistant、figures-diagram、figures-python、latex-output、documents 技能约束。实际产物包括 PDF、Markdown 源稿、5 张 PNG 可视化和构建脚本。验证包括生成脚本执行、PDF 文件完整性检查、PDF 文本抽取、页面渲染抽样和输出文件清单检查。剩余风险是该报告没有引入新的实验结果，所有项目结论均依赖现有 stage reports、notebook 输出和本地源码证据。
"""


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="main", frames=[frame], onPage=draw_page_frame),
            ]
        )

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            text = flowable.getPlainText()
            style_name = flowable.style.name
            if style_name in {"Heading1", "Heading2"}:
                level = 0 if style_name == "Heading1" else 1
                key = f"h{level}-{abs(hash(text))}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (level, text, self.page, key))


def draw_page_frame(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor(PALETTE["light_gray"]))
    canvas.setLineWidth(0.7)
    canvas.line(doc.leftMargin, 282 * mm, doc.leftMargin + doc.width, 282 * mm)
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor(PALETTE["gray"]))
    canvas.drawString(doc.leftMargin, 286 * mm, "Drowsiness Detection Final Report")
    canvas.drawRightString(doc.leftMargin + doc.width, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def register_fonts() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    base = ParagraphStyle(
        "BaseChinese",
        parent=sample["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=16.5,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
        wordWrap="CJK",
    )
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base,
            fontSize=24,
            leading=31,
            alignment=TA_CENTER,
            textColor=colors.HexColor(PALETTE["navy"]),
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base,
            fontSize=11,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor(PALETTE["gray"]),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base,
            fontSize=16,
            leading=22,
            textColor=colors.HexColor(PALETTE["navy"]),
            spaceBefore=15,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base,
            fontSize=13,
            leading=19,
            textColor=colors.HexColor(PALETTE["blue"]),
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": base,
        "small": ParagraphStyle(
            "Small",
            parent=base,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor(PALETTE["gray"]),
            alignment=TA_LEFT,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor(PALETTE["gray"]),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "toc_h": ParagraphStyle(
            "TOCHeading",
            parent=base,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor(PALETTE["navy"]),
            spaceAfter=20,
        ),
    }


def p(text: str, st: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), st)


def metric_table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PALETTE["navy"])),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_figure(story: list, path: Path, caption: str, st: dict[str, ParagraphStyle], width_mm: float = 160) -> None:
    with Image.open(path) as pil_img:
        raw_w, raw_h = pil_img.size
    draw_w = width_mm * mm
    draw_h = raw_h * draw_w / raw_w
    img = RLImage(str(path), width=draw_w, height=draw_h)
    img.hAlign = "CENTER"
    story.extend([Spacer(1, 5), img, p(caption, st["caption"]), Spacer(1, 6)])


def build_pdf(figures: list[Path]) -> None:
    register_fonts()
    st = styles()
    doc = ReportDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
    )

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1",
            fontName="STSong-Light",
            fontSize=10.5,
            leading=15,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
        ),
        ParagraphStyle(
            "TOCLevel2",
            fontName="STSong-Light",
            fontSize=9.5,
            leading=13,
            leftIndent=14,
            firstLineIndent=0,
            spaceBefore=2,
        ),
    ]

    story: list = []
    story.extend(
        [
            Spacer(1, 45 * mm),
            p("基于眼-嘴双通道证据融合的<br/>驾驶疲劳候选预警系统报告", st["title"]),
            p("Final Technical Report | Drowsiness Detection Project", st["subtitle"]),
            Spacer(1, 18 * mm),
            metric_table(
                [
                    ["Item", "Value"],
                    ["Project type", "Modular driver drowsiness warning-candidate prototype"],
                    ["Main evidence", "YawDD/YawDD+ Dash mouth/yawn + MRL Eye open/closed"],
                    ["Runtime output", "Rule-based warning-candidate states"],
                    ["Generated at", "docs/final/final_report.pdf"],
                ],
                col_widths=[38 * mm, 120 * mm],
            ),
            Spacer(1, 16 * mm),
            p(
                "Claim boundary: this document reports specialist-module metrics and controlled-validation warning-candidate behavior. It does not claim final system-level drowsiness accuracy, deployment readiness, clinical validation, or real-world road validation.",
                st["small"],
            ),
            PageBreak(),
            p("Table of Contents", st["toc_h"]),
            toc,
            PageBreak(),
        ]
    )

    sections = [
        (
            "Abstract / 摘要",
            [
                "本文报告一个模块化驾驶疲劳检测原型系统。项目没有采用端到端“疲劳/非疲劳”单一分类器，而是将可见驾驶员行为拆分为两个可解释的视觉证据通道：基于 YawDD/YawDD+ Dash 数据的嘴部打哈欠识别模块输出 p_yawn，基于 MRL Eye 数据的眼部闭合识别模块输出 p_eye_closed。两个专门模型的结果再经过运行时 ROI 提取、时序平滑、PERCLOS-like 规则和质量门控，形成 rule-based warning-candidate timeline。",
                "项目的核心训练结果来自 colab_file/stage7_yawdd_training_r.ipynb 和 colab_file/stage9_mrl_eye_training_r.ipynb。Stage 7 中，YawDD 嘴部模型以 ResNet18 取得最高测试准确率 99.37%，yawn F1 为 97.18%。Stage 9/9B 中，MRL Eye 眼部模型选择 MobileNetV2 作为主模型，测试准确率 98.63%，macro F1 为 98.63%，closed-eye recall 为 98.52%。后续 Stage 12-17 没有重新训练模型，而是围绕信号质量、时序持续性和人类审阅边界设计融合逻辑。",
            ],
        ),
        (
            "1. Introduction and Background / 引言与背景",
            [
                "驾驶疲劳检测的难点不只在分类模型本身，还在于“疲劳”这一状态很难由单帧视觉证据直接定义。困倦驾驶事故识别通常依赖事后调查和行为证据，漏报风险较高。PERCLOS 相关研究将眼睑闭合时间比例作为警觉性下降的重要指标，NHTSA 相关报告也将 Perclose/PERCLOS 与视觉注意力 lapses 建立了实验联系。",
                "本项目吸收这一思路，但没有直接测量真实眼睑开合百分比，而是将 CNN 的 p_eye_closed 作为闭眼概率代理，因此项目文档中使用 “PERCLOS-like” 或 “PERCLOS-inspired” 的表述更准确。打哈欠和眼部闭合都可作为疲劳相关行为线索，但单独使用任一信号都存在歧义。将嘴部 yawn evidence 与眼部 temporal eye-warning evidence 分离建模，再在时序层进行规则融合，可以让每个模块的训练标签更清楚，也能把模型分类结果和系统级解释隔离开来。",
            ],
        ),
        (
            "2. Overview of the Architecture/System / 系统架构概述",
            [
                "项目整体结构以 docs/PROJECT_STRUCTURE.md 为主线。dataset/ 存放本地原始或重建数据，artifacts/ 存放映射表、划分文件和中间结果，outputs/ 存放训练与运行时证据，reports/ 存放各阶段人类可读报告，src/ 负责数据处理、训练和运行时推理，src/backend/ 提供 FastAPI 服务，SystemUI/ 提供 Next.js 前端。",
                "系统可概括为四层：数据层、训练层、运行时层和应用层。数据层中，YawDD/YawDD+ Dash 被重建为 64,378 个带标签帧，并通过 MediaPipe Face Mesh 嘴唇 landmarks 生成 64,202 个可训练嘴部 crop；MRL Eye 提供 84,898 张眼部图片。训练层使用 ResNet18、MobileNetV2、EfficientNet-B0 进行迁移学习 baseline 比较。运行时层从完整人脸视频提取眼部和嘴部 ROI，并通过 F5 tiered quality-aware fusion 生成融合状态。应用层支持视频上传分析、Live Monitor、48h History、Insights 和 SQLite 摘要归档。",
            ],
        ),
        (
            "3. Data Processing and Model Training / 数据处理与模型训练",
            [
                "YawDD/YawDD+ Dash 嘴部数据处理质量较高。重建阶段得到 29 个 subject、64,378 个标注帧，其中 no_yawn 为 57,347 帧，yawn 为 7,031 帧。嘴部 crop 阶段处理 64,378 帧，MediaPipe Face Mesh 成功 crop 64,093 帧，fallback lower-face crop 109 帧，失败 176 帧，成功率为 99.73%。subject-level split 避免同一 subject 跨 train/val/test 泄漏。",
                "MRL Eye 数据集经本地检查包含 84,898 张图片、37 个 subject，closed 41,946 张，open 42,952 张。subject-level split 为 train 58,982、val 13,029、test 12,887，三个 split 均包含 closed/open，泄漏检查通过。MRL Eye 的类别比例整体接近均衡，但 subject 内部分布差异较大，因此 subject-level split 比随机 image-level split 更适合作为当前项目的保守评估策略。",
            ],
        ),
        (
            "4. Fusion and Runtime Decision Logic / 融合层逻辑判断",
            [
                "Stage 12 的眼部时序规则选择为 quality_gated_perclos_mean_ge_0.60_consec。该规则要求 rolling PERCLOS-like mean-binary ratio 大于等于 0.60，并持续至少 2 个 sampled frames；若 5 帧窗口内 no-face ratio 大于 0.20，则标记为 signal_unreliable。Stage 14 的嘴部运行时逻辑使用恢复的 Stage 7 ResNet18 checkpoint 计算 p_yawn = softmax(logits)[1]。p_yawn >= 0.50 的 sampled row 记为 yawn event，后续时间窗口内会保留 recent-yawn context。",
                "F5 fusion 的核心思想是分层处理质量、嘴部证据和眼部证据。若 eye signal 不可靠且没有 recent yawn，则输出 signal_unreliable；若 eye signal 不可靠但存在 recent yawn，则输出 mouth_warning_candidate；若 eye warning 与 recent yawn 共同出现，则输出 high_confidence_drowsiness_candidate；若只有 eye warning，则输出 eye_warning_candidate；若只有 recent yawn，则输出 mouth_warning_candidate；否则输出 normal。Stage 17.1/17.5 又增加持续眼部证据和强度门控，避免 brief blink-like 或 weak eye evidence 与 recent yawn 偶然重叠时被过度升级。",
            ],
        ),
        (
            "5. Results and Evaluation / 结果与评估",
            [
                "Stage 7 训练结果直接来自 colab_file/stage7_yawdd_training_r.ipynb 与恢复后的 artifacts/recovered_stage7_mouth_yawn/initial_results.csv。ResNet18 因测试准确率和 yawn F1 表现最佳，被选为嘴部打哈欠专门模型。EfficientNet-B0 的 validation accuracy 更高，但测试集整体表现略低于 ResNet18。考虑到类别不平衡，报告不应只写 accuracy，yawn precision/recall/F1 和 confusion matrix 更能说明模型是否漏掉少数类 yawn。",
                "Stage 9/9B 训练和模型选择结果显示，MobileNetV2 是当前主眼部模型，因为默认阈值下 test accuracy、macro F1、误报/漏报平衡和实时部署适配性最好。ResNet18 at p_eye_closed >= 0.30 被保留为 safety-prioritized reference：closed recall 提升到 99.08%，false-open 降到 58，但 false-closed 增加到 251，因此不适合作为默认设置。",
                "Stage 15 使用真实 Stage 12 eye timeline 和真实 Stage 14 model-generated p_yawn timeline 完成同步融合。B 视频中，用户手动观察到 14.3s-16.8s 左右存在打哈欠；Stage 14 在该窗口内 12/12 行触发 yawn-event，mean p_yawn 约为 0.981。Stage 15 的 high-confidence candidate 出现在 recent-yawn evidence 与 eye-warning evidence 发生重叠的时段。C 视频中，头部运动、头发/手遮挡等问题被部分归入 signal quality，而不是直接升级为疲劳结论。",
            ],
        ),
        (
            "6. Discussion and Conclusions / 讨论与结论",
            [
                "该项目的主要工程优势在于边界控制清楚。训练指标只声称为 specialist-module performance，运行时输出只声称为 warning-candidate，前端和后端也保留了永久解释文本，避免把单帧概率或规则状态误写成最终驾驶疲劳检测。数据划分采用 subject-level split，也比随机 frame split 更能降低身份和相邻帧泄漏风险。",
                "系统的主要风险来自泛化能力和真实世界验证不足。YawDD 嘴部模型训练于重建 Dash mouth crops，MRL Eye 眼部模型训练于眼部 crop 图片，运行时视频再由 MediaPipe 生成 ROI；训练域和运行时域并不完全一致。Stage 10-15 的 A/B/C/D 视频说明 pipeline 能在小规模受控场景下工作，但 subject 数、光照、摄像头、遮挡、眼镜反光、头部姿态、真实驾驶环境和疲劳 ground truth 都不足。",
                "结论上，本项目已经完成一个结构完整、证据链较清楚的本地驾驶疲劳候选预警原型。若要把项目提升到可发表或可部署级别，下一步应采集更多同步眼-嘴视频，建立 temporal ground-truth fatigue/warning annotation，按 subject/camera/lighting 条件做分层评估，并在有足够标注后再考虑 learned temporal fusion。",
            ],
        ),
    ]

    fig_map = {
        "Abstract / 摘要": (figures[0], "Figure 1. 项目整体 pipeline：从数据集、预处理、专门模型到时序规则和应用层。"),
        "2. Overview of the Architecture/System / 系统架构概述": (figures[1], "Figure 2. 数据处理流程：YawDD/YawDD+ 嘴部分支与 MRL Eye 眼部分支独立准备并采用 subject-level split。"),
        "4. Fusion and Runtime Decision Logic / 融合层逻辑判断": (figures[2], "Figure 3. F5 融合层逻辑判断：质量优先、嘴部证据和眼部证据分层升级。"),
        "5. Results and Evaluation / 结果与评估": (figures[3], "Figure 4. 专门模型测试性能对比。"),
    }

    for heading, paragraphs in sections:
        story.append(p(heading, st["h1"]))
        for para in paragraphs:
            story.append(p(para, st["body"]))
        if heading in fig_map:
            add_figure(story, *fig_map[heading], st)
        if heading == "5. Results and Evaluation / 结果与评估":
            story.append(p("Table 1. Stage 7 mouth/yawn specialist results", st["h2"]))
            story.append(
                metric_table(
                    [["Model", "Train Acc", "Val Acc", "Test Acc", "Yawn Precision", "Yawn Recall", "Yawn F1"]]
                    + [[m, f"{ta:.2f}%", f"{va:.2f}%", f"{te:.2f}%", f"{pr:.2f}%", f"{rc:.2f}%", f"{f1:.2f}%"] for m, ta, va, te, pr, rc, f1 in STAGE7],
                    col_widths=[30 * mm, 20 * mm, 20 * mm, 20 * mm, 25 * mm, 24 * mm, 20 * mm],
                )
            )
            story.append(Spacer(1, 8))
            story.append(p("Table 2. Stage 9/9B MRL Eye specialist results", st["h2"]))
            story.append(
                metric_table(
                    [["Model", "Test Acc", "Macro F1", "Closed Recall", "False Open", "False Closed", "Val Threshold"]]
                    + [[m, f"{acc:.2f}%", f"{f1:.2f}%", f"{rec:.2f}%", str(fo), str(fc), f"{thr:.2f}"] for m, acc, f1, rec, fo, fc, thr in STAGE9],
                    col_widths=[32 * mm, 22 * mm, 22 * mm, 25 * mm, 24 * mm, 25 * mm, 24 * mm],
                )
            )
            story.append(Spacer(1, 8))
            add_figure(story, figures[4], "Figure 5. Stage 15 A/B/C/D 受控验证视频的融合状态计数。", st)
            story.append(p("Table 3. Stage 15 F5 fusion counts", st["h2"]))
            story.append(
                metric_table(
                    [["Video", "Normal", "Eye", "Mouth", "High Confidence", "Signal Unreliable"]]
                    + [[v, str(n), str(e), str(m), str(h), str(s)] for v, n, e, m, h, s in FUSION_COUNTS],
                    col_widths=[58 * mm, 20 * mm, 20 * mm, 22 * mm, 34 * mm, 32 * mm],
                )
            )

    story.append(PageBreak())
    story.append(p("References / 参考文献", st["h1"]))
    refs = [
        "NHTSA. Drowsy Driving: Countermeasures That Work. https://www.nhtsa.gov/book/countermeasures-that-work/drowsy-driving",
        "Dinges, D. F., Mallis, M. M., Maislin, G., & Powell, J. W. Evaluation of techniques for ocular measurement as an index of fatigue and as the basis for alertness management. NHTSA, 1998. https://rosap.ntl.bts.gov/view/dot/2518",
        "FMCSA/NHTSA. PERCLOS: A Valid Psychophysiological Measure of Alertness. https://ntlsearch.bts.gov/ntl/md.do?id=51369",
        "Abtahi, S., Omidyeganeh, M., Shirmohammadi, S., & Hariri, B. YawDD: A Yawning Detection Dataset. ACM MMSys Workshop, 2014.",
        "MRL. MRL Eye Dataset. https://mrl.cs.vsb.cz/eyedataset.html",
        "Google MediaPipe. MediaPipe Face Mesh. https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh",
        "He, K., Zhang, X., Ren, S., & Sun, J. Deep Residual Learning for Image Recognition. CVPR 2016.",
        "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.",
        "Tan, M., & Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.",
        "Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.",
    ]
    story.append(
        ListFlowable(
            [ListItem(p(ref, st["body"]), leftIndent=10) for ref in refs],
            bulletType="1",
            start="1",
            leftIndent=16,
        )
    )
    story.append(p("Appendices / 附录", st["h1"]))
    for para in [
        "主要内部证据文件包括 docs/PROJECT_STRUCTURE.md、docs/PROJECT_CURRENT_STATUS.md、reports/stage16_final_integration_summary_report.md、reports/stage15_real_mouth_eye_fusion_validation_report.md、reports/stage14_mouth_yawn_runtime_validation_report.md、reports/stage12_eye_alert_rule_analysis_report.md、reports/mrl_eye_stage9b_error_analysis.md、reports/yawdd_dash_split_report.md、reports/mrl_eye_split_report.md、colab_file/stage7_yawdd_training_r.ipynb 和 colab_file/stage9_mrl_eye_training_r.ipynb。",
        "能力使用审计：本次生成使用 research-writing-assistant、figures-diagram、figures-python、latex-output、documents 技能约束。实际产物包括 PDF、Markdown 源稿、5 张 PNG 可视化和构建脚本。验证包括生成脚本执行、PDF 文件完整性检查、PDF 文本抽取、页面渲染抽样和输出文件清单检查。剩余风险是该报告没有引入新的实验结果，所有项目结论均依赖现有 stage reports、notebook 输出和本地源码证据。",
    ]:
        story.append(p(para, st["body"]))

    doc.multiBuild(story)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = generate_figures()
    MD_PATH.write_text(markdown_report(), encoding="utf-8")
    build_pdf(figures)
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {MD_PATH}")
    for figure in figures:
        print(f"Wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
