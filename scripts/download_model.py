#!/usr/bin/env python3
"""
Downloads a YOLOv8 pretrained model (pt) from ultralytics and saves it to models/.
Requires internet access.
"""
import os
from pathlib import Path
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="yolov8n.pt", help="Model name / filename to save")
args = parser.parse_args()

os.makedirs("models", exist_ok=True)
target = Path("models") / args.name

print("Downloading COCO pretrained model (yolov8n)...")
# Use ultralytics to get weights
model = YOLO("yolov8n.pt")  # this will download if not present in cache
model.save(str(target))
print(f"Saved to {target}")
