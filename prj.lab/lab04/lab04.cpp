#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

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
            }
        }
    }

    cv::GaussianBlur(image, image, cv::Size(blurSize, blurSize), 0);

    return image;
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
    /*cv::createTrackbar("Block Size", "Binary Parameters", &blockSize, 255);
    cv::createTrackbar("C", "Binary Parameters", (int*)&C, 255);*/

    while (true) {
        binaryImage = thresholdBinary(inputImage, thresholdValue);
        adaptive = adaptiveBinary(inputImage, adaptive1, adaptive2);
        cv::imshow("Binary Image", binaryImage);
        cv::imshow("Adaptive Thresholded Image", adaptive);

        char key = cvWaitKey(0);
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
cv::Mat nonMaximumSuppression(cv::Mat gradientMagnitude, cv::Mat gradientDirection) {
    cv::Mat nonMaxSuppressed = cv::Mat::zeros(gradientMagnitude.size(), CV_8U);

    for (int y = 1; y < gradientMagnitude.rows - 1; ++y) {
        for (int x = 1; x < gradientMagnitude.cols - 1; ++x) {
            float direction = gradientDirection.at<float>(y, x);

            float neighbor1, neighbor2;
            if (direction < 0) {
                neighbor1 = gradientMagnitude.at<float>(y + 1, x);
                neighbor2 = gradientMagnitude.at<float>(y - 1, x);
            }
            else if (direction < 45) {
                neighbor1 = gradientMagnitude.at<float>(y, x + 1);
                neighbor2 = gradientMagnitude.at<float>(y, x - 1);
            }
            else if (direction < 90) {
                neighbor1 = gradientMagnitude.at<float>(y - 1, x - 1);
                neighbor2 = gradientMagnitude.at<float>(y + 1, x + 1);
            }
            else {
                neighbor1 = gradientMagnitude.at<float>(y - 1, x + 1);
                neighbor2 = gradientMagnitude.at<float>(y + 1, x - 1);
            }

            if (gradientMagnitude.at<float>(y, x) >= neighbor1 && gradientMagnitude.at<float>(y, x) >= neighbor2) {
                nonMaxSuppressed.at<uchar>(y, x) = 255;
            }
        }
    }

    return nonMaxSuppressed;
}

cv::Mat detectObjects(cv::Mat inputImage, int kernelSize, double lowThreshold, double highThreshold) {
    cv::Mat blurredImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(kernelSize, kernelSize), 0);

    cv::Mat grayscaleImage, gradientsX, gradientsY, gradientMagnitude, gradientDirection;
    cv::cvtColor(blurredImage, grayscaleImage, cv::COLOR_BGR2GRAY);
    cv::Sobel(grayscaleImage, gradientsX, CV_32F, 1, 0);
    cv::Sobel(grayscaleImage, gradientsY, CV_32F, 0, 1);
    cv::cartToPolar(gradientsX, gradientsY, gradientMagnitude, gradientDirection, true);

    cv::Mat nonMaxSuppressed = nonMaximumSuppression(gradientMagnitude, gradientDirection);

    cv::Mat thresholded;
    cv::Canny(nonMaxSuppressed, thresholded, lowThreshold, highThreshold);

    cv::Mat tracedEdges;
    cv::Canny(inputImage, tracedEdges, lowThreshold, highThreshold);

    return tracedEdges;
}



struct Circle {
    int x, y, r;
};

float calculateIoU(Rect box1, Rect box2) {
    Rect intersection = box1 & box2;
    Rect unionRect = box1 | box2;
    float intersectionArea = intersection.area();
    float unionArea = unionRect.area();
    return intersectionArea / unionArea;
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

    FP = detected.size() - TP;
}

void fillGroundTruthAndDetected(cv::Mat testImage, std::vector<Circle>& groundTruth, std::vector<Circle>& detected) {
    cv::Mat detectedObjects = detectObjects(testImage, 5, 50, 150);

    for (int i = 0; i < 5; ++i) {
        Circle circle;
        circle.x = rand() % testImage.cols;
        circle.y = rand() % testImage.rows;
        circle.r = rand() % 30 + 10;

        detected.push_back(circle); 
    }

    for (const auto& gt : groundTruth) {
        detected.push_back(gt);
    }
}


int main(int argc, const char* argv[]) {
    cv::Mat testImage = generateTestImage(100, cv::Range(3, 7), cv::Range(25, 5), 15);


    cv::Mat thresholded = thresholdBinary(testImage, 100);
    cv::Mat adaptive = adaptiveBinary(testImage, 21, 10);
    cv::Mat otsu = otsuBinary(testImage);

    cv::Mat detectedObjects = detectObjects(testImage, 5, 50, 150);
    cv::imshow("Detected Objects", detectedObjects);

    cv::imshow("Test Image", testImage);
    cv::imshow("Otsu Thresholded Image", otsu);

    tuneBinaryParameters(testImage);

    cvWaitKey(10000);


    float qualityThreshold = 0.5;

    std::vector<Circle> groundTruth;
    std::vector<Circle> detected;

    fillGroundTruthAndDetected(testImage, groundTruth, detected);

    int TP = 0, FP = 0, FN = 0;

    evaluateDetection(groundTruth, detected, qualityThreshold, TP, FP, FN);

    std::cout << "True Positives (TP): " << TP << std::endl;
    std::cout << "False Positives (FP): " << FP << std::endl;
    std::cout << "False Negatives (FN): " << FN << std::endl;

    float averageIoU = static_cast<float>(TP) / (TP + FN + FP);
    std::cout << "Average IoU: " << averageIoU << std::endl;

    return 0;
}
