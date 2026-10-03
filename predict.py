"""Classify one image from the command line.

Usage:  python predict.py path/to/photo.jpg
"""
import sys
from classifier import classify

if len(sys.argv) != 2:
    sys.exit("Usage: python predict.py <image_path>")

r = classify(sys.argv[1])
print(f"\nWaste type : {r['label'].upper()}  ({r['confidence']:.1%})")
print(f"Bin        : {r.get('bin', '-')}")
print(f"Tip        : {r.get('tip', '-')}\n")
for p in r["predictions"]:
    print(f"  {p['label']:8s} {p['confidence']:.1%}")
