#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "ObjectDetector.h"
#include "TrafficCounter.h"
#include "utils.h"

void printUsage() {
    std::cout << "Usage: traffic_counter [--input <video|camera>] [--model <path>] [--confidence <0.25>] [--output <out.mp4>]\n";
}

int main(int argc, char** argv) {
    std::string inputArg = "camera";
    std::string modelPath = "models/yolov8n.onnx";
    std::string outputPath = "";
    float conf = 0.25f;
    int inputSize = 640;

    // Very simple CLI parsing
    for (int i=1;i<argc;i++) {
        std::string a = argv[i];
        if (a == "--input" && i+1<argc) { inputArg = argv[++i]; }
        else if (a == "--model" && i+1<argc) { modelPath = argv[++i]; }
        else if (a == "--output" && i+1<argc) { outputPath = argv[++i]; }
        else if (a == "--confidence" && i+1<argc) { conf = std::stof(argv[++i]); }
        else if (a == "--size" && i+1<argc) { inputSize = std::stoi(argv[++i]); }
        else if (a == "--help") { printUsage(); return 0; }
    }

    cv::VideoCapture cap;
    if (inputArg == "camera" || inputArg == "0") {
        cap.open(0);
    } else {
        cap.open(inputArg);
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input: " << inputArg << std::endl;
        return -1;
    }

    ObjectDetector detector(modelPath, inputSize, conf, 0.45f, {2,3,5,7}); // track vehicle classes by default
    TrafficCounter counter(0.6f, "results/counts.csv");
    counter.setDetector(&detector);

    int frameW = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameH = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25;

    cv::VideoWriter writer;
    if (!outputPath.empty()) {
        writer.open(outputPath, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(frameW, frameH));
        if (!writer.isOpened()) {
            std::cerr << "Failed to open output writer: " << outputPath << std::endl;
        }
    }

    cv::Mat frame;
    double lastTime = now_ms();
    while (true) {
        if (!cap.read(frame)) break;
        double t0 = now_ms();
        cv::Mat out = counter.processFrame(frame);
        double t1 = now_ms();
        double dt = (t1 - t0) / 1000.0;
        double fpsReal = 1.0 / std::max(1e-6, dt);

        // draw FPS & counts
        drawInfo(out, counter.totalCount(), fpsReal);

        cv::imshow("Traffic Counter", out);
        if (writer.isOpened()) writer.write(out);

        int key = cv::waitKey(1);
        if (key == 27) break;
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    cv::destroyAllWindows();
    return 0;
}
