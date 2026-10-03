"""
Convert a raw dataset into the 5-class layout used by this project.

Supports the Kaggle "Garbage Classification" dataset (12 classes) by mapping:
    biological -> organic
    brown-glass / green-glass / white-glass -> glass
    cardboard / paper -> paper
    metal -> metal
    plastic -> plastic
Other classes (battery, clothes, shoes, trash) are skipped.

Usage:
    python prepare_dataset.py --src path/to/garbage_classification --dst dataset
You can also just drop your own images straight into dataset/<class>/ and skip this script.
"""
import argparse
import shutil
from pathlib import Path

MAPPING = {
    "biological": "organic", "organic": "organic",
    "brown-glass": "glass", "green-glass": "glass", "white-glass": "glass", "glass": "glass",
    "cardboard": "paper", "paper": "paper",
    "metal": "metal",
    "plastic": "plastic",
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing one sub-folder per original class")
    ap.add_argument("--dst", default="dataset")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    counts = {}
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        target = MAPPING.get(folder.name.lower())
        if not target:
            print(f"skip  {folder.name}")
            continue
        out = dst / target
        out.mkdir(parents=True, exist_ok=True)
        for img in folder.iterdir():
            if img.suffix.lower() in EXTS:
                shutil.copy2(img, out / f"{folder.name}_{img.name}")
                counts[target] = counts.get(target, 0) + 1
    print("\nImages per class:")
    for k, v in sorted(counts.items()):
        print(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
