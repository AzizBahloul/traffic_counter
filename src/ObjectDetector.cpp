#include "ObjectDetector.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>

ObjectDetector::ObjectDetector(const std::string& modelPath, int inputSize, float confThreshold, float nmsIou, const std::vector<int>& classes)
    : env_(ORT_LOGGING_LEVEL_WARNING, "TrafficCounter"),
      session_(nullptr),
      inputSize_(inputSize),
      confThreshold_(confThreshold),
      nmsIou_(nmsIou),
      classesFilter_(classes)
{
    sessionOptions_.SetIntraOpNumThreads(1);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    sessionOptions_.AppendExecutionProvider_CUDA(0);
#endif

    // Create session
    session_ = new Ort::Session(env_, modelPath.c_str(), sessionOptions_);

    // Get input & output names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session_->GetInputCount();
    size_t numOutputNodes = session_->GetOutputCount();

    inputNames_.clear();
    for (size_t i = 0; i < numInputNodes; ++i) {
        Ort::AllocatorWithDefaultOptions alloc;
        auto name_alloc = session_->GetInputNameAllocated(i, alloc);
        inputNames_.push_back(name_alloc.get() ? std::string(name_alloc.get()) : std::string("") );
    }

    outputNames_.clear();
    for (size_t i = 0; i < numOutputNodes; ++i) {
        Ort::AllocatorWithDefaultOptions alloc;
        auto name_alloc = session_->GetOutputNameAllocated(i, alloc);
        outputNames_.push_back(name_alloc.get() ? std::string(name_alloc.get()) : std::string("") );
    }

}

ObjectDetector::~ObjectDetector() {
    if (session_) delete session_;
}

void ObjectDetector::preprocess(const cv::Mat& image, std::vector<float>& outTensor, int& outW, int& outH, float& scale, int& padW, int& padH) {
    // letterbox to inputSize_ keeping aspect ratio
    int w = image.cols, h = image.rows;
    float r = std::min((float)inputSize_ / w, (float)inputSize_ / h);
    int nw = (int)round(w * r);
    int nh = (int)round(h * r);
    padW = (inputSize_ - nw) / 2;
    padH = (inputSize_ - nh) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh));
    cv::Mat canvas = cv::Mat::zeros(inputSize_, inputSize_, CV_8UC3);
    resized.copyTo(canvas(cv::Rect(padW, padH, nw, nh)));

    // Convert BGR to RGB, normalize to 0..1
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    // CHW float vector
    outTensor.resize(3 * inputSize_ * inputSize_);
    int idx = 0;
    // split channels
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < inputSize_; ++y) {
            for (int x = 0; x < inputSize_; ++x) {
                outTensor[idx++] = rgb.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    outW = inputSize_;
    outH = inputSize_;
    scale = r;
}

static float iou(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int w = x2 - x1;
    int h = y2 - y1;
    if (w <= 0 || h <= 0) return 0.f;
    float inter = (float)w * h;
    float unionArea = (float)(a.area() + b.area() - inter);
    return inter / unionArea;
}

std::vector<Detection> ObjectDetector::postprocess(const cv::Mat& image, std::vector<float>& output, int outW, int outH, float scale, int padW, int padH) {
    // Typical YOLOv8 ONNX export outputs shape [1, N, 85] => [x, y, w, h, confidence, class_probs...]
    // We'll parse by reading consecutive blocks of (5 + num_classes)
    std::vector<Detection> all;

    if (output.size() < 6) return all;

    // Determine stride: assume second dimension size is known by measuring (we don't have shape info here).
    // But typical layout: output.size() = N*(5+num_classes)
    // Let's assume num_classes is 80 (COCO); but to be tolerant, infer num_classes by reading first row length using heuristic:
    size_t total = output.size();
    // Try to guess num_classes by looking for values >1 or <0 (not reliable). Instead, use known COCO classes 80 -> 85
    // We'll assume 80 classes if divisible.
    int candidates[] = {80, 20, 1}; // fallback
    int num_classes = 80;
    bool ok = false;
    for (int nc : candidates) {
        size_t stride = 5 + nc;
        if (total % stride == 0) { num_classes = nc; ok = true; break; }
    }
    if (!ok) {
        // assume stride = 85
        num_classes = 80;
    }
    int stride = 5 + num_classes;
    size_t N = output.size() / stride;

    for (size_t i = 0; i < N; ++i) {
        float bx = output[i * stride + 0];
        float by = output[i * stride + 1];
        float bw = output[i * stride + 2];
        float bh = output[i * stride + 3];
        float obj_conf = output[i * stride + 4];

        // class scores
        float best_conf = 0.f;
        int best_cls = -1;
        for (int c = 0; c < num_classes; ++c) {
            float cls_conf = output[i * stride + 5 + c];
            if (cls_conf > best_conf) { best_conf = cls_conf; best_cls = c; }
        }

        float score = obj_conf * best_conf;
        if (score < confThreshold_) continue;

        // Convert xywh (center) in the inputSize_ scale to original image coords.
        // Coordinates currently relative to model input grid (0..inputSize)
        float x1 = (bx - bw / 2.0f - padW) / scale;
        float y1 = (by - bh / 2.0f - padH) / scale;
        float x2 = (bx + bw / 2.0f - padW) / scale;
        float y2 = (by + bh / 2.0f - padH) / scale;

        cv::Rect rect;
        rect.x = std::max(0, (int)std::floor(x1));
        rect.y = std::max(0, (int)std::floor(y1));
        rect.width = std::min(image.cols - rect.x, (int)std::ceil(x2) - rect.x);
        rect.height = std::min(image.rows - rect.y, (int)std::ceil(y2) - rect.y);

        if (rect.width <= 0 || rect.height <= 0) continue;
        // filter classes if provided
        if (!classesFilter_.empty()) {
            bool okc = false;
            for (int c : classesFilter_) if (c == best_cls) { okc = true; break; }
            if (!okc) continue;
        }

        Detection d;
        d.box = rect;
        d.score = score;
        d.class_id = best_cls;
        all.push_back(d);
    }

    // NMS
    std::sort(all.begin(), all.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> keep;
    std::vector<bool> removed(all.size(), false);

    for (size_t i = 0; i < all.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(all[i]);
        for (size_t j = i + 1; j < all.size(); ++j) {
            if (removed[j]) continue;
            if (all[i].class_id != all[j].class_id) continue;
            if (iou(all[i].box, all[j].box) > nmsIou_) removed[j] = true;
        }
    }

    return keep;
}

std::vector<Detection> ObjectDetector::detect(const cv::Mat& image) {
    std::vector<float> inputTensorValues;
    int outW, outH, padW, padH;
    float scale;
    preprocess(image, inputTensorValues, outW, outH, scale, padW, padH);

    // Prepare input tensor
    std::vector<int64_t> inputShape = {1, 3, outH, outW};
    size_t inputTensorSize = inputTensorValues.size();

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size());

    // Run
    // Convert input/output names to const char* arrays for ONNX Runtime
    std::vector<const char*> inputNamePtrs, outputNamePtrs;
    for (const auto& s : inputNames_) inputNamePtrs.push_back(s.c_str());
    for (const auto& s : outputNames_) outputNamePtrs.push_back(s.c_str());

    auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(), &inputTensor, 1, outputNamePtrs.data(), outputNamePtrs.size());

    // Expect one output
    std::vector<float> output;
    // For each output tensor, copy data
    for (auto &ot : outputTensors) {
        float* outData = ot.GetTensorMutableData<float>();
        size_t totalLen = 1;
        auto shape = ot.GetTensorTypeAndShapeInfo().GetShape();
        for (auto d : shape) totalLen *= (size_t)d;
        output.insert(output.end(), outData, outData + totalLen);
    }

    return postprocess(image, output, outW, outH, scale, padW, padH);
}
