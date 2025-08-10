#!/usr/bin/env python3
"""
Export an ultralytics YOLOv8 model to ONNX.
Requires ultralytics installed (`pip install ultralytics`).
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="models/yolov8n.pt")
parser.add_argument("--onnx", default="models/yolov8n.onnx")
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

os.makedirs("models", exist_ok=True)
print(f"Loading weights: {args.weights}")
y = YOLO(args.weights)
print(f"Exporting to ONNX: {args.onnx} (imgsz={args.imgsz})")
y.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False, device="cpu")
# ultralytics export will produce a file with same base name under runs/ or the model dir
# Move or copy to desired path if not placed
# If ultralytics saved to 'yolov8n.onnx' in cwd, move it:
if not Path(args.onnx).exists():
    # try common output
    possible = Path("yolov8n.onnx")
    if possible.exists():
        possible.replace(args.onnx)
print("Export done.")
