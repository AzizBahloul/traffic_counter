# Traffic Counter

A real-time traffic counter using YOLOv8 ONNX model for object detection and OpenCV for video processing.

Features:
- Perform object detection (person, vehicle classes) using ONNX Runtime (CPU or GPU/CUDA).
- Draw labeled bounding boxes around detected objects.
- Simple CLI for video/camera input and output file writing.

## Prerequisites

- C++17 compiler (e.g. `g++`).
- [CMake](https://cmake.org/) (>=3.16).
- [OpenCV](https://opencv.org/) (with videoio support).
- [ONNX Runtime](https://onnxruntime.ai/) libraries and headers installed under `/usr/local` or set `ONNXRUNTIME_ROOT`.
- (Optional) CUDA Toolkit to enable GPU inference.

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url> traffic_counter
   cd traffic_counter
   ```

2. (Optional) Export YOLOv8 ONNX model:
   ```bash
   python scripts/export_yolo.py --weights yolov8n.pt --output models/yolov8n.onnx
   ```

3. Build the C++ application:
   ```bash
   mkdir -p build && cd build
   cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
   make -j$(nproc)
   ```

## Usage

```bash
./build/traffic_counter [--input <video|camera>] [--model <path>] [--confidence <0.25>] [--output <out.mp4>]
``` 

- `--input`: Path to video file or `camera` (default: `camera`).
- `--model`: ONNX model path (default: `models/yolov8n.onnx`).
- `--confidence`: Minimum confidence threshold for detections.
- `--output`: Optional path to save processed video.

## GPU Inference

If ONNX Runtime CUDA provider is found during build, GPU inference will be enabled automatically.

## Project Structure

```
├── CMakeLists.txt
├── src/                # C++ source code
├── scripts/            # Python model export scripts
├── models/             # ONNX model files
├── build/              # CMake build output
├── .vscode/            # VS Code editor settings
├── README.md
└── .gitignore
```

## License

MIT License