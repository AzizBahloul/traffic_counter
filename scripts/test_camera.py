#!/usr/bin/env python3
import cv2
from ultralytics import YOLO

# Quick test to verify camera works and ultralytics models load (not needed for ONNX runtime)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

model = YOLO("yolov8n.pt")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=640, conf=0.25)
    annotated = results[0].plot()
    cv2.imshow("camera", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
