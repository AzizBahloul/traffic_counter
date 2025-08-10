#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "ObjectDetector.h"
#include "utils.h"

// Simple centroid tracker for vehicle counting
struct Track {
    int id;
    cv::Point centroid;
    cv::Rect bbox;
    int disappeared;
    bool counted;
    std::vector<cv::Point> trace;
};

class TrafficCounter {
public:
    TrafficCounter(float lineYRelative = 0.6f, const std::string& csvPath = "results/counts.csv");
    void setDetector(ObjectDetector* detector);
    cv::Mat processFrame(const cv::Mat& frame);

    int totalCount() const { return total_count_; }

private:
    void update(const std::vector<Detection>& detections);
    int registerTrack(const cv::Rect& bbox);
    void deregister(int trackID);

    std::unordered_map<int, Track> tracks_;
    int nextTrackID_;
    int maxDisappeared_;
    int maxDistance_;
    int minFramesToCount_;
    int total_count_;
    float lineYRel_;
    int frameWidth_;
    int frameHeight_;
    ObjectDetector* detector_;
    std::string csvPath_;
};
