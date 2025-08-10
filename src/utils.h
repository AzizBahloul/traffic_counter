#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

inline double now_ms() {
    using namespace std::chrono;
    return (double)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void drawDetections(cv::Mat &frame, const std::vector<Detection>& dets, const std::vector<std::string>& classNames);
void drawInfo(cv::Mat &frame, int total_count, double fps);

std::vector<int> argSortDesc(const std::vector<float>& v);

// Simple CSV append
inline void append_csv(const std::string &path, const std::string &line) {
    std::ofstream f;
    f.open(path, std::ios::app);
    if (f.is_open()) {
        f << line << "\n";
        f.close();
    }
}
