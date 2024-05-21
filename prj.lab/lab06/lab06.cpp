#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

struct Circle {
    int x, y, r;
};

cv::Mat detectCirclesHough(cv::Mat inputImage, std::vector<Circle>& detectedCircles) {
    cv::Mat gray;
    cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows / 16, 80, 10, 5, 10);

    for (size_t i = 0; i < circles.size(); i++) {
        Circle circle;
        circle.x = cvRound(circles[i][0]);
        circle.y = cvRound(circles[i][1]);
        circle.r = cvRound(circles[i][2]);
        detectedCircles.push_back(circle);
    }

    return gray;
}

void drawCircles(cv::Mat& image, const std::vector<Circle>& circles, cv::Scalar color) {
    for (size_t i = 0; i < circles.size(); i++) {
        cv::circle(image, cv::Point(circles[i].x, circles[i].y), circles[i].r, color, 2);
    }
}

float calculateIoU(Circle gt, Circle det) {
    float r1 = gt.r, r2 = det.r;
    float x1 = gt.x, y1 = gt.y, x2 = det.x, y2 = det.y;

    float d = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    if (d > r1 + r2) return 0.0f;
    if (d <= abs(r1 - r2)) return (r1 < r2) ? CV_PI * r1 * r1 : CV_PI * r2 * r2;

    float part1 = r1 * r1 * acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1));
    float part2 = r2 * r2 * acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2));
    float part3 = 0.5 * sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2));

    float interArea = part1 + part2 - part3;
    float unionArea = CV_PI * (r1 * r1 + r2 * r2) - interArea;

    return interArea / unionArea;
}

void computeFROC(const std::vector<Circle>& groundTruth, const std::vector<Circle>& detected, double iouThreshold) {
    int TP = 0, FP = 0, FN = 0;
    std::vector<bool> detectedFlags(detected.size(), false);

    for (const auto& gt : groundTruth) {
        bool foundMatch = false;
        for (size_t i = 0; i < detected.size(); i++) {
            if (detectedFlags[i]) continue;
            float iou = calculateIoU(gt, detected[i]);
            if (iou >= iouThreshold) {
                TP++;
                detectedFlags[i] = true;
                foundMatch = true;
                break;
            }
        }
        if (!foundMatch) FN++;
    }

    for (size_t i = 0; i < detected.size(); i++) {
        if (!detectedFlags[i]) FP++;
    }

    cout << "True Positives: " << TP << endl;
    cout << "False Positives: " << FP << endl;
    cout << "False Negatives: " << FN << endl;

    double sensitivity = (double)TP / (TP + FN);
    double avgFPPerImage = (double)FP;

    cout << "Sensitivity: " << sensitivity << endl;
    cout << "Avg FP per Image: " << avgFPPerImage << endl;
}

int main(int argc, const char* argv[]) {
    Mat testImage;
    testImage = imread("grid.png", IMREAD_COLOR);
    if (testImage.empty()) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }

    std::vector<Circle> groundTruth;
    std::vector<Circle> detected;

    cv::Mat detectedGray = detectCirclesHough(testImage, detected);

    groundTruth = detected;

    drawCircles(testImage, detected, cv::Scalar(0, 0, 255));

    double iouThreshold = 0.5;
    computeFROC(groundTruth, detected, iouThreshold);

    cv::imshow("Test Image", detectedGray);
    cv::imshow("Detected Objects", testImage);
    cv::waitKey(100000);

    return 0;
}
