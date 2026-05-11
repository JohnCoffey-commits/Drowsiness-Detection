# Initial Experiment Summary

## A. Task Type

**Image Classification**

## B. Experimental settings

The initial Stage 7 experiment uses the reconstructed YawDD+ Dash mouth-crop dataset uploaded to Google Drive and trains directly from a copied local Colab workspace under `/content/` for faster I/O. Labels are read from the existing Stage 6 subject-level split manifest and are defined as `no_yawn` and `yawn`. The mouth crops come from the completed Stage 5 preprocessing pipeline, where the mouth ROI was extracted from MediaPipe Face Mesh landmarks with fallback lower-face crops already handled before Stage 7. Training uses the existing leakage-safe subject-level split, the three CNN baselines CNN-1 ResNet18, CNN-2 MobileNetV2, and CNN-3 EfficientNet-B0, Adam optimizer, learning rate 1e-4, batch size 32 with fallback to 16 if needed, weighted cross-entropy loss, up to 12 epochs, early stopping patience 3, ReduceLROnPlateau scheduling, and realistic training-only augmentation with small rotation, mild brightness/contrast jitter, and slight affine scaling.

## C. Initial Results table

| CNN Architecture | Train Accuracy | Validation Accuracy | Test Accuracy |
|---|---:|---:|---:|
| CNN-1 (ResNet18) | 98.92% | 98.85% | 99.37% |
| CNN-2 (MobileNetV2) | 98.97% | 98.48% | 98.75% |
| CNN-3 (EfficientNet-B0) | 98.76% | 99.08% | 99.20% |

## D. Short interpretation

CNN-1 (ResNet18) achieved the strongest test accuracy in this initial Stage 7 run.
The best model shows a train-validation gap of 0.06 percentage points, which is the main early signal to monitor for overfitting.
Class imbalance still matters because `no_yawn` is more frequent than `yawn`, so the weighted cross-entropy setting and the test confusion matrix should be reviewed together rather than relying on accuracy alone.
Because the split is done at subject level rather than image level, the evaluation is more challenging and more realistic than a random frame split.
The next step should be focused error analysis on the saved confusion matrices and misclassified subjects, followed by controlled hyperparameter tuning or additional data balancing experiments.
