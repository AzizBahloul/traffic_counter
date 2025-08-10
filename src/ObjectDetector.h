#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include "utils.h"

#ifdef USE_CUDA
#define ORT_USE_CUDA
#endif

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelPath, int inputSize = 640, float confThreshold = 0.25f, float nmsIou = 0.45f, const std::vector<int>& classes = {});
    ~ObjectDetector();

    std::vector<Detection> detect(const cv::Mat& image);

private:
    void preprocess(const cv::Mat& image, std::vector<float>& outTensor, int& outW, int& outH, float& scale, int& padW, int& padH);
    std::vector<Detection> postprocess(const cv::Mat& image, std::vector<float>& output, int outW, int outH, float scale, int padW, int padH);

    Ort::Env env_;
    Ort::Session* session_;
    Ort::SessionOptions sessionOptions_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;

    int inputSize_;
    float confThreshold_;
    float nmsIou_;
    std::vector<int> classesFilter_;
};
