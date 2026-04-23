from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MOUTH_LANDMARKS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tokenize(value: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", value.lower()) if part]


def infer_label(image_path: Path) -> str | None:
    tokens = set(tokenize(" ".join(image_path.parts)))
    if "yawning" in tokens or "yawn" in tokens:
        return "yawn"
    if {"normal", "talking", "talk", "noyawn", "notyawn", "no"} & tokens:
        return "no-yawn"
    return None


@dataclass(frozen=True)
class Sample:
    subject_id: str
    original_path: Path
    label: str


def discover_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        label = infer_label(image_path)
        if label is None:
            continue
        try:
            subject_id = image_path.relative_to(root).parts[0]
        except IndexError:
            subject_id = image_path.parent.name
        samples.append(Sample(subject_id=subject_id, original_path=image_path, label=label))
    return samples


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def mouth_box_from_landmarks(face_landmarks, width: int, height: int) -> tuple[int, int, int, int]:
    xs = []
    ys = []
    for idx in MOUTH_LANDMARKS:
        landmark = face_landmarks.landmark[idx]
        xs.append(int(round(landmark.x * width)))
        ys.append(int(round(landmark.y * height)))
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    margin = max(10, int(round(0.10 * max(x2 - x1, y2 - y1))))
    return clamp_box(x1 - margin, y1 - margin, x2 + margin, y2 + margin, width, height)


def lower_face_fallback_box(width: int, height: int, face_box: tuple[int, int, int, int] | None = None) -> tuple[int, int, int, int]:
    if face_box:
        x1, y1, x2, y2 = face_box
        face_height = y2 - y1
        return clamp_box(x1, y1 + int(face_height * 0.55), x2, y2, width, height)
    crop_w = int(width * 0.60)
    crop_h = int(height * 0.35)
    cx = width // 2
    y1 = int(height * 0.55)
    return clamp_box(cx - crop_w // 2, y1, cx + crop_w // 2, y1 + crop_h, width, height)


def detect_face_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    import cv2

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(x + w), int(y + h)


def save_crop(image_rgb: np.ndarray, box: tuple[int, int, int, int], output_path: Path, size: int) -> None:
    import cv2
    from PIL import Image

    x1, y1, x2, y2 = box
    crop = image_rgb[y1:y2, x1:x2]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(resized).save(output_path, quality=95)


def precompute(root: Path, output_dir: Path, manifest_path: Path, failures_path: Path, size: int, max_images: int | None) -> dict[str, int]:
    samples = discover_samples(root)
    if max_images is not None:
        samples = samples[:max_images]

    stats = {
        "total_images": len(samples),
        "successful_landmark_crops": 0,
        "fallback_crops": 0,
        "failed_samples_removed": 0,
        "unlabeled_images_skipped": sum(
            1
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and infer_label(p) is None
        ),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="") as manifest_file, failures_path.open("w", newline="") as failure_file:
        manifest = csv.DictWriter(
            manifest_file, fieldnames=["subject_id", "original_path", "processed_path", "label", "crop_method"]
        )
        failures = csv.DictWriter(failure_file, fieldnames=["original_path", "reason"])
        manifest.writeheader()
        failures.writeheader()

        if not samples:
            return stats

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise SystemExit("MediaPipe is required for preprocessing. Install dependencies from requirements.txt.") from exc

        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("OpenCV is required for preprocessing. Install dependencies from requirements.txt.") from exc

        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            for sample in samples:
                image_bgr = cv2.imread(str(sample.original_path))
                if image_bgr is None:
                    stats["failed_samples_removed"] += 1
                    failures.writerow({"original_path": sample.original_path, "reason": "cv2_read_failed"})
                    continue

                height, width = image_bgr.shape[:2]
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(image_rgb)
                crop_method = "mediapipe_mouth"
                box: tuple[int, int, int, int] | None = None

                if result.multi_face_landmarks:
                    box = mouth_box_from_landmarks(result.multi_face_landmarks[0], width, height)
                    stats["successful_landmark_crops"] += 1
                else:
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    face_box = detect_face_box(gray)
                    box = lower_face_fallback_box(width, height, face_box)
                    crop_method = "fallback_lower_face" if face_box else "fallback_center_lower_face"
                    stats["fallback_crops"] += 1
                    failures.writerow({"original_path": sample.original_path, "reason": crop_method})

                relative_stem = sample.original_path.relative_to(root).with_suffix("")
                processed_path = output_dir / sample.label / f"{relative_stem}.jpg"
                try:
                    save_crop(image_rgb, box, processed_path, size)
                except Exception as exc:  # noqa: BLE001 - keep preprocessing robust and logged.
                    stats["failed_samples_removed"] += 1
                    failures.writerow({"original_path": sample.original_path, "reason": f"crop_save_failed:{exc}"})
                    continue

                manifest.writerow(
                    {
                        "subject_id": sample.subject_id,
                        "original_path": sample.original_path,
                        "processed_path": processed_path,
                        "label": sample.label,
                        "crop_method": crop_method,
                    }
                )

    return stats


def write_report(path: Path, root: Path, output_dir: Path, stats: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preprocessing Report",
        "",
        f"Input dataset: `{root}`",
        f"Output directory: `{output_dir}`",
        "",
        "## Summary",
        "",
        f"- Total labeled images discovered: {stats['total_images']}",
        f"- Successful MediaPipe mouth crops: {stats['successful_landmark_crops']}",
        f"- Fallback crops: {stats['fallback_crops']}",
        f"- Failed samples removed: {stats['failed_samples_removed']}",
        f"- Unlabeled images skipped: {stats['unlabeled_images_skipped']}",
        "",
        "## Notes",
        "",
        "MediaPipe Face Mesh is used to crop the lip landmark region. When landmarks are unavailable, the script falls back to a detected lower-face crop, then to a centered lower-face crop.",
    ]
    if stats["total_images"] == 0:
        lines.extend(
            [
                "",
                "No labeled image files were available under the requested YawDD+ Dash path, so no mouth crops were generated.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute YawDD+ Dash mouth crops with MediaPipe Face Mesh.")
    parser.add_argument("--input-root", type=Path, default=repo_root() / "dataset" / "YawDD+" / "dataset" / "Dash")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "artifacts" / "preprocessed" / "yawdd_dash_mouth")
    parser.add_argument("--manifest", type=Path, default=repo_root() / "artifacts" / "preprocessed" / "yawdd_dash_mouth" / "manifest.csv")
    parser.add_argument("--failures", type=Path, default=repo_root() / "artifacts" / "preprocessed" / "yawdd_dash_mouth" / "preprocessing_failures.csv")
    parser.add_argument("--report", type=Path, default=repo_root() / "reports" / "preprocessing_report.md")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    stats = precompute(args.input_root, args.output_dir, args.manifest, args.failures, args.image_size, args.max_images)
    (args.output_dir / "preprocessing_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    write_report(args.report, args.input_root, args.output_dir, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
