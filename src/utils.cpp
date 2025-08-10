#include "utils.h"
#include <iomanip>

void drawDetections(cv::Mat &frame, const std::vector<Detection>& dets, const std::vector<std::string>& classNames) {
    for (const auto& d : dets) {
        cv::rectangle(frame, d.box, cv::Scalar(0,255,0), 2);
        std::ostringstream label;
        std::string name = (d.class_id >=0 && d.class_id < (int)classNames.size()) ? classNames[d.class_id] : std::to_string(d.class_id);
        label << name << " " << std::fixed << std::setprecision(2) << d.score;
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int top = std::max(d.box.y, labelSize.height);
        cv::rectangle(frame, cv::Point(d.box.x, top - labelSize.height - 4), cv::Point(d.box.x + labelSize.width, top + baseline - 4), cv::Scalar(0,255,0), cv::FILLED);
        cv::putText(frame, label.str(), cv::Point(d.box.x, top - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
}

void drawInfo(cv::Mat &frame, int total_count, double fps) {
    std::ostringstream ss;
    ss << "Count: " << total_count;
    cv::putText(frame, ss.str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,200,255), 2);
    if (fps > 0.0) {
        std::ostringstream s2;
        s2 << std::fixed << std::setprecision(1) << fps << " FPS";
        cv::putText(frame, s2.str(), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,200,255), 2);
    }
}

std::vector<int> argSortDesc(const std::vector<float>& v) {
    std::vector<int> idx(v.size());
    for (size_t i=0;i<v.size();++i) idx[i] = (int)i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    return idx;
}
