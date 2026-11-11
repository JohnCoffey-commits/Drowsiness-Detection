# Model assets

The three assets were downloaded and verified locally. They remain excluded from Git and Docker build context. Only authorized operators should obtain the private checkpoints.

| File and required location | Bytes | Source |
| --- | ---: | --- |
| `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | 9156491 | Owner-provided Drive eye checkpoint |
| `checkpoints/resnet18_best.pt` | 44790859 | Owner-provided Drive mouth checkpoint |
| `artifacts/models/face_landmarker.task` | 3758596 | Google MediaPipe face landmarker float16 version 1 |

The exact SHA-256 identities are executable in `scripts/verify_assets.py`. Eye: `a3335fc60cf91013a999b2e55246ee1a700019de075e729a58f7627a6863d0ca`; mouth: `fda7b4c1709c022a4e8942e74f538c2f122baa2ec39db84423adaff6f5fddf47`; face: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

The official face asset is available at https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task . Private Drive identifiers and the operator's local paths are retained in the local preparation manifest, not needed for the public code review. The mouth file belongs in `checkpoints/`, even though its source Drive folder used `outputs/checkpoints/`.

Both checkpoint archives passed integrity checks and safe `weights_only=True` loading. Eye metadata describes MobileNet V2, 224-pixel inputs, closed/open labels 0/1. Mouth metadata describes ResNet18, 224-pixel inputs, no_yawn/yawn labels 0/1. Existing loaders and finite crop predictions were verified. The Goal 3 eye-loader change explicitly uses safe weights-only loading.

On 11 November the authorized 27.64-second single-person video completed the native CPU pipeline with 166 sampled frames and four keyframes, in 39.45 seconds. This initial execution preceded the final learning commit and is not the final clean-clone result. Inputs, facial crops and keyframes remain local. A separate demonstration MP4 was downloaded but is a UI recording and was not treated as an inference sample.

File presence, hash identity, successful deserialization, readiness and complete video inference are different checks. A successful run is not a labelled accuracy study or a driving-safety validation.
