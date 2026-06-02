# Stage 13 Manual B Yawn Annotation Log

## 1. Purpose

This is a manual annotation sanity check for Stage 13 rule-based mouth-eye fusion.

It uses a user-confirmed yawn interval for `B_realistic_drowsy_simulation.mp4` to check whether the Stage 13 fusion rule behaves as expected when a mouth/yawn timeline is available.

This is not runtime mouth/yawn model inference and not final system-level drowsiness accuracy.

## 2. Manual Annotation

| Field | Value |
| --- | --- |
| Video | `B_realistic_drowsy_simulation.mp4` |
| Yawning interval | `14.3s` to `16.8s` |
| Source | User manual video review |
| Mouth source label | `manual_video_review_annotation` |

## 3. Generated Mouth Timeline

Manual mouth timeline:

```text
artifacts/audits/stage13_manual_B_yawn_annotation_2026-05-09/manual_mouth_timeline_B_yawn_14p3_16p8.csv
```

The timeline was generated on the same sampled timestamps used by the Stage 12 eye alert timelines for A/B/C/D.

For `A_normal_open_baseline`, `C_mild_head_motion`, and `D_controlled_long_open_closed`:

- `p_yawn = 0.05`
- `yawn_event = false`
- `recent_yawn_event = false`
- `mouth_signal_status = ok`
- `mouth_source = manual_video_review_annotation`

For `B_realistic_drowsy_simulation`:

- Rows sampled between the confirmed yawn interval use `p_yawn = 0.95` and `yawn_event = true`.
- Other rows use `p_yawn = 0.05` and `yawn_event = false`.
- The sampled yawn rows start at `14.381352s` and end at `16.674031s`.
- `recent_yawn_event` is true from `14.381352s` through `21.259390s`, using the 8.0 second recent-yawn window.

## 4. Stage 13 Rerun

Command:

```bash
python src/runtime/stage13_mouth_eye_fusion_design.py \
  --stage12-output-dir outputs/stage12_eye_alert_rule_analysis \
  --mouth-timeline artifacts/audits/stage13_manual_B_yawn_annotation_2026-05-09/manual_mouth_timeline_B_yawn_14p3_16p8.csv \
  --output-dir outputs/stage13_mouth_eye_fusion_manual_B_yawn_annotation \
  --recent-yawn-window-sec 8.0 \
  --yawn-threshold 0.50 \
  --videos A_normal_open_baseline,B_realistic_drowsy_simulation,C_mild_head_motion,D_controlled_long_open_closed
```

Output directory:

```text
outputs/stage13_mouth_eye_fusion_manual_B_yawn_annotation/
```

Result: passed.

## 5. B Result Interpretation

For `B_realistic_drowsy_simulation`, the manual mouth annotation produced:

| Metric | Value |
| --- | ---: |
| `yawn_event` rows | 12 |
| First sampled yawn timestamp | 14.381352 |
| Last sampled yawn timestamp | 16.674031 |
| `recent_yawn_event` rows | 34 |
| First recent-yawn timestamp | 14.381352 |
| Last recent-yawn timestamp | 21.259390 |
| High-confidence candidate rows | 6 |
| First high-confidence timestamp | 16.882456 |
| Last high-confidence timestamp | 17.924583 |

During the manually confirmed yawn interval, the fusion state becomes `mouth_warning_candidate` because the mouth annotation is active while the Stage 12 eye warning is not yet active for most sampled rows.

Immediately after the yawn interval, the recent-yawn window remains active and overlaps with Stage 12 eye-warning rows. The recommended Stage 13 rule therefore emits `high_confidence_drowsiness_candidate` from `16.882456s` to `17.924583s`.

This better matches the user-confirmed behavior than the synthetic demo because the mouth/yawn signal now comes from a manually reviewed yawn interval in the real B video. It still does not validate automatic mouth/yawn detection.

## 6. Limitations

- Manual annotation is not model inference.
- There is still no automatic runtime mouth/yawn pipeline.
- This is not final system-level drowsiness accuracy.
- This does not validate deployment readiness.
- True synchronized validation requires Stage 14 runtime mouth/yawn inference to produce automatic `p_yawn` timelines.

## 7. Next Recommended Step

Implement runtime mouth/yawn inference to produce automatic `p_yawn` timelines for the same videos, then rerun Stage 13 with model-generated mouth/yawn timelines.
