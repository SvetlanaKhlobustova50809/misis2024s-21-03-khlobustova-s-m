#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

struct Circle {
    int x, y, r;
};

cv::Mat testImage;
std::vector<Circle> groundTruth;

cv::Mat generateTestImage(int numObjects, cv::Range sizeRange, cv::Range contrastRange, int blurSize) {
    cv::Mat image(550, 550, CV_8UC3, cv::Scalar(0, 0, 0));

    double contrastStep = 2.2;
    double sizeStep = 0.5;

    int circleDistance = 40;

    for (int i = 0; i < 10; i++) {
        double contrast = contrastRange.end + i * contrastStep;

        for (int j = 0; j < 10; j++) {
            double size = sizeRange.start + j * sizeStep + 3;

            if ((j + 1) * 50 + size <= image.cols && (i + 1) * 50 + size <= image.rows) {
                cv::Mat circleMask = cv::Mat::zeros(image.size(), CV_8U);
                cv::circle(circleMask, cv::Point((j + 1) * 50, (i + 1) * 50), size, cv::Scalar(255), -1);

                cv::Mat circle;
                image.copyTo(circle);
                circle.setTo(cv::Scalar(255, 255, 255), circleMask);
                circle.convertTo(circle, -1, contrast / 30.0);
                circle.copyTo(image, circleMask);
                groundTruth.push_back({ (j + 1) * 50, (i + 1) * 50, static_cast<int>(size) });
            }
        }
    }

    cv::threshold(image, image, 20, 100, cv::THRESH_BINARY);
    cv::GaussianBlur(image, image, cv::Size(blurSize, blurSize), 0);


    Mat noisyImage = image.clone();
    RNG rng;
    rng.fill(noisyImage, RNG::NORMAL, 15 * 3, 15);
  
    return image + noisyImage;
}

cv::Mat thresholdBinary(cv::Mat inputImage, int thresholdValue) {
    cv::Mat binaryImage;
    cv::cvtColor(inputImage, binaryImage, cv::COLOR_BGR2GRAY);
    cv::threshold(binaryImage, binaryImage, thresholdValue, 255, cv::THRESH_BINARY);
    return binaryImage;
}

cv::Mat adaptiveBinary(cv::Mat inputImage, int blockSize, double C) {
    cv::Mat binaryImage;
    cv::cvtColor(inputImage, binaryImage, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(binaryImage, binaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
    return binaryImage;
}

void tuneBinaryParameters(cv::Mat inputImage) {
    int thresholdValue = 127;
    int blockSize = 11;
    double C = 2.0;
    int adaptive1 = 21, adaptive2 = -10;

    cv::Mat binaryImage, adaptive;

    cvNamedWindow("Binary Parameters", cv::WINDOW_GUI_NORMAL);
    cvCreateTrackbar("Threshold Value", "Binary Parameters", &thresholdValue, 255);
    cvCreateTrackbar("adaptive1 Value", "Binary Parameters", &adaptive1, 51, [](int value) {
        if (value % 2 == 0) {
            cvSetTrackbarPos("adaptive1 Value", "Binary Parameters", value + 1);
        }
        });
    cvSetTrackbarMin("adaptive1 Value", "Binary Parameters", 3);
    cvCreateTrackbar("adaptive2 Value", "Binary Parameters", &adaptive2, 10);

    while (true) {
        binaryImage = thresholdBinary(inputImage, thresholdValue);
        adaptive = adaptiveBinary(inputImage, adaptive1, adaptive2);
        cv::imshow("Binary Image", binaryImage);
        cv::imshow("Adaptive Thresholded Image", adaptive);

        char key = cv::waitKey(10);
        if (key == 27) // Нажатие ESC для выхода
            break;
    }
}

cv::Mat otsuBinary(cv::Mat inputImage) {
    cv::Mat binaryImage;
    cv::cvtColor(inputImage, binaryImage, cv::COLOR_BGR2GRAY);
    cv::threshold(binaryImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binaryImage;
}

float calculateIoU(cv::Rect box1, cv::Rect box2) {
    cv::Rect intersection = box1 & box2;
    cv::Rect unionRect = box1 | box2;

    float intersectionArea = intersection.area();
    float unionArea = unionRect.area();

    float iou = intersectionArea / unionArea;

    return iou;
}

void evaluateDetection(const vector<Circle>& groundTruth, const vector<Circle>& detected, float threshold, int& TP, int& FP, int& FN) {
    TP = 0;
    FN = 0;
    vector<bool> detectedMatched(detected.size(), false);

    for (const auto& gt : groundTruth) {
        bool matchFound = false;
        for (size_t i = 0; i < detected.size(); ++i) {
            if (!detectedMatched[i]) {
                Rect gtRect(gt.x - gt.r, gt.y - gt.r, 2 * gt.r, 2 * gt.r);
                Rect dtRect(detected[i].x - detected[i].r, detected[i].y - detected[i].r, 2 * detected[i].r, 2 * detected[i].r);

                float iou = calculateIoU(gtRect, dtRect);

                if (iou > threshold) {
                    matchFound = true;
                    detectedMatched[i] = true;
                    break;
                }
            }
        }
        if (matchFound) {
            TP++;
        }
        else {
            FN++;
        }
    }

    FP = std::count(detectedMatched.begin(), detectedMatched.end(), false);
}


int main(int argc, const char* argv[]) {
    testImage = generateTestImage(100, cv::Range(3, 7), cv::Range(25, 5), 35);
    cv::imshow("Test Image", testImage);

    cv::Mat thresholded = thresholdBinary(testImage, 100);
    cv::Mat adaptive = adaptiveBinary(testImage, 21, 10);
    cv::Mat otsu = otsuBinary(testImage);
    Mat ad;
    cv::imshow("Otsu Thresholded Image", otsu);
    tuneBinaryParameters(testImage);
    imshow("test", testImage);

    Mat gray;
    cvtColor(testImage, gray, COLOR_BGR2GRAY);

    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    vector<Vec3f> circles;
    vector<Circle> detected;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 200, 30, 0, 0);

    Mat binary;
    threshold(blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    Mat morph;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(binary, morph, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat output = testImage.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(output, contours, (int)i, Scalar(0, 0, 255), 2);

        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);

        detected.push_back({ static_cast<int>(center.x), static_cast<int>(center.y), static_cast<int>(radius) });
    }

    imshow("Detected Contours", output);

    float qualityThreshold = 0.5;
    int TP = 0, FP = 0, FN = 0;

    evaluateDetection(groundTruth, detected, qualityThreshold, TP, FP, FN);
    cv::imshow("Test Image", testImage);

    std::cout << "True Positives (TP): " << TP << std::endl;
    std::cout << "False Positives (FP): " << FP << std::endl;
    std::cout << "False Negatives (FN): " << FN << std::endl;

    float averageIoU = static_cast<float>(TP) / (TP + FN + FP);
    std::cout << "Average IoU: " << averageIoU << std::endl;

    cv::waitKey(100000);

}
