# Initial Experiment Summary

## A. Task Type

**Image Classification**

## B. Experimental settings

The initial experiment uses YawDD+ Dash as the intended primary dataset for binary mouth-focused yawning classification. Samples are mapped to `yawn` for yawning frames and `no-yawn` for normal or talking frames when those labels are recoverable from the inspected folder or file names. Each image is preprocessed with MediaPipe Face Mesh to crop the mouth region from lip landmarks, with a lower-face fallback crop when landmarks are unavailable, then resized to 224 x 224 and normalized with ImageNet statistics. Splitting is performed at subject level using approximately 70% training, 15% validation, and 15% test subjects to avoid frame leakage. The three baselines are CNN-1 ResNet18, CNN-2 MobileNetV2, and CNN-3 EfficientNet-B0, trained in PyTorch with Adam, learning rate 1e-4, weighted cross entropy, batch size 32 with fallback to 16, 12 epochs, early stopping patience 3, ReduceLROnPlateau scheduling, mild rotation/brightness/contrast/scale augmentation, and a two-stage transfer-learning schedule that freezes the backbone before full fine-tuning.

## C. Initial Results table

| CNN Architecture        | Train Accuracy | Validation Accuracy | Test Accuracy |
| ----------------------- | -------------: | ------------------: | ------------: |
| CNN-1 (ResNet18)        |            N/A |                 N/A |           N/A |
| CNN-2 (MobileNetV2)     |            N/A |                 N/A |           N/A |
| CNN-3 (EfficientNet-B0) |            N/A |                 N/A |           N/A |

## D. Short interpretation

No CNN result is available yet because the inspected local YawDD+ Dash folder does not currently contain image files.
The dataset appears incomplete for the requested supervised image-classification experiment because only annotation text files were found.
NTHUDDD2 has recoverable labels and subject IDs, but it was intentionally not substituted for the initial YawDD+ Dash training source.
Mouth cropping cannot be evaluated until paired Dash image frames are present.
The next step is to add the missing YawDD+ Dash images, rerun preprocessing, build the subject split, and then run the three CNN baselines.
