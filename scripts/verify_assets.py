"""Verify privately prepared model files without loading or deserializing them."""

import argparse
import hashlib
import json
from pathlib import Path

ASSETS = {
    "outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt": (
        9156491,
        "a3335fc60cf91013a999b2e55246ee1a700019de075e729a58f7627a6863d0ca",
    ),
    "checkpoints/resnet18_best.pt": (
        44790859,
        "fda7b4c1709c022a4e8942e74f538c2f122baa2ec39db84423adaff6f5fddf47",
    ),
    "artifacts/models/face_landmarker.task": (
        3758596,
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}


def verify(root: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for relative, (size, expected) in ASSETS.items():
        path = root / relative
        actual = None
        if path.is_file():
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        valid = actual == expected and path.stat().st_size == size
        results.append({"path": relative, "valid": valid, "sha256": actual})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    results = verify(parser.parse_args().root)
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if all(row["valid"] for row in results) else 1)
