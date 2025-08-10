#include "TrafficCounter.h"
#include <algorithm>
#include <cmath>
#include <sstream>

TrafficCounter::TrafficCounter(float lineYRelative, const std::string& csvPath)
: nextTrackID_(0), maxDisappeared_(30), maxDistance_(80), minFramesToCount_(3), total_count_(0), lineYRel_(lineYRelative), frameWidth_(0), frameHeight_(0), detector_(nullptr), csvPath_(csvPath)
{
    // create CSV header
    append_csv(csvPath_, "timestamp,track_id,class_id,counted,center_x,center_y");
}

void TrafficCounter::setDetector(ObjectDetector* detector) {
    detector_ = detector;
}

int TrafficCounter::registerTrack(const cv::Rect& bbox) {
    Track t;
    t.id = nextTrackID_++;
    t.bbox = bbox;
    t.centroid = cv::Point(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
    t.disappeared = 0;
    t.counted = false;
    t.trace.push_back(t.centroid);
    tracks_[t.id] = t;
    return t.id;
}

void TrafficCounter::deregister(int trackID) {
    tracks_.erase(trackID);
}

void TrafficCounter::update(const std::vector<Detection>& detections) {
    if (frameWidth_ == 0 || frameHeight_ == 0) return;

    // If there are no tracks, register all detections
    if (tracks_.empty()) {
        for (const auto& d : detections) {
            registerTrack(d.box);
        }
        return;
    }

    // Build arrays of centroids
    std::vector<cv::Point> inputCentroids;
    std::vector<cv::Rect> rects;
    for (const auto& d : detections) {
        inputCentroids.emplace_back(d.box.x + d.box.width/2, d.box.y + d.box.height/2);
        rects.push_back(d.box);
    }

    // Existing track centroids
    std::vector<int> trackIDs;
    std::vector<cv::Point> trackCentroids;
    for (auto &kv : tracks_) {
        trackIDs.push_back(kv.first);
        trackCentroids.push_back(kv.second.centroid);
    }

    // Distance matrix
    cv::Mat dist(trackCentroids.size(), inputCentroids.size(), CV_32F, cv::Scalar(0));
    for (size_t i = 0; i < trackCentroids.size(); ++i) {
        for (size_t j = 0; j < inputCentroids.size(); ++j) {
            float d = cv::norm(trackCentroids[i] - inputCentroids[j]);
            dist.at<float>(i,j) = d;
        }
    }

    // Greedy assignment
    std::vector<int> assignedTracks(trackCentroids.size(), -1);
    std::vector<int> assignedInputs(inputCentroids.size(), -1);

    for (size_t iter = 0; iter < std::min(trackCentroids.size(), inputCentroids.size()); ++iter) {
        double minVal;
        cv::Point minLoc;
        cv::minMaxLoc(dist, &minVal, nullptr, &minLoc, nullptr);
        int tr = minLoc.y;
        int in = minLoc.x;
        if (minVal > maxDistance_) break;
        assignedTracks[tr] = in;
        assignedInputs[in] = tr;
        // invalidate row and column
        for (int c = 0; c < dist.cols; ++c) dist.at<float>(tr,c) = 1e6;
        for (int r = 0; r < dist.rows; ++r) dist.at<float>(r,in) = 1e6;
    }

    // Update assigned tracks
    for (size_t i = 0; i < assignedTracks.size(); ++i) {
        int in = assignedTracks[i];
        int trackID = trackIDs[i];
        if (in != -1) {
            tracks_[trackID].bbox = rects[in];
            tracks_[trackID].centroid = inputCentroids[in];
            tracks_[trackID].disappeared = 0;
            tracks_[trackID].trace.push_back(inputCentroids[in]);
        } else {
            // mark disappeared
            tracks_[trackID].disappeared += 1;
            if (tracks_[trackID].disappeared > maxDisappeared_) deregister(trackID);
        }
    }

    // Register new inputs that were not assigned
    for (size_t j = 0; j < assignedInputs.size(); ++j) {
        if (assignedInputs[j] == -1) {
            registerTrack(rects[j]);
        }
    }

    // Counting: check line crossing
    int lineY = (int)(lineYRel_ * frameHeight_);
    for (auto &kv : tracks_) {
        Track &t = kv.second;
        if (t.counted) continue;
        if (t.trace.size() < (size_t)minFramesToCount_) continue;
        // simple last two points
        cv::Point p1 = t.trace[t.trace.size() - 2];
        cv::Point p2 = t.trace.back();
        // crossing from above to below
        if ((p1.y < lineY && p2.y >= lineY) || (p1.y > lineY && p2.y <= lineY)) {
            t.counted = true;
            total_count_++;
            // append CSV
            std::ostringstream ss;
            ss << (long long)now_ms() << "," << t.id << "," << 0 /*class placeholder*/ << "," << 1 << "," << t.centroid.x << "," << t.centroid.y;
            append_csv(csvPath_, ss.str());
        }
    }
}

cv::Mat TrafficCounter::processFrame(const cv::Mat& frame) {
    if (!detector_) return frame.clone();
    // Run detector and draw only bounding boxes with labels
    std::vector<Detection> dets = detector_->detect(frame);
    cv::Mat out = frame.clone();
    drawDetections(out, dets, {"person","bicycle","car","motorbike","aeroplane","bus","train","truck"});
    return out;
}
