# Preprocessing Report

Input dataset: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/dataset/YawDD+/dataset/Dash`
Output directory: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/preprocessed/yawdd_dash_mouth`

## Summary

- Total labeled images discovered: 0
- Successful MediaPipe mouth crops: 0
- Fallback crops: 0
- Failed samples removed: 0
- Unlabeled images skipped: 0

## Notes

MediaPipe Face Mesh is used to crop the lip landmark region. When landmarks are unavailable, the script falls back to a detected lower-face crop, then to a centered lower-face crop.

No labeled image files were available under the requested YawDD+ Dash path, so no mouth crops were generated.
