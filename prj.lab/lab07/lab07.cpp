#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

struct Circle {
    int x, y, r;
};

std::vector<Circle> groundTruth;

void generateTestImage(int numObjects, cv::Range sizeRange, cv::Range contrastRange, int blurSize) {
    cv::Mat image(550, 550, CV_8UC3, cv::Scalar(0, 0, 0));

    double contrastStep = 2.2;
    double sizeStep = 0.5;

    for (int i = 0; i < 10; i++) {
        double contrast = contrastRange.end + i * contrastStep;

        for (int j = 0; j < 10; j++) {
            double size = sizeRange.start + j * sizeStep + 3;

            if ((j + 1) * 50 + size <= image.cols && (i + 1) * 50 + size <= image.rows) {
                groundTruth.push_back({ (j + 1) * 50, (i + 1) * 50, static_cast<int>(size) });
            }
        }
    }
}

Mat segmentImageKMeans(const Mat& inputImage, int k) {
    Mat samples(inputImage.rows * inputImage.cols, 3, CV_32F);
    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            for (int z = 0; z < 3; z++) {
                samples.at<float>(y + x * inputImage.rows, z) = inputImage.at<Vec3b>(y, x)[z];
            }
        }
    }

    Mat labels;
    int attempts = 5;
    Mat centers;
    kmeans(samples, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), attempts, KMEANS_PP_CENTERS, centers);

    Mat segmentedImage(inputImage.size(), inputImage.type());
    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * inputImage.rows, 0);
            segmentedImage.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
            segmentedImage.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
            segmentedImage.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }

    return segmentedImage;
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

int main() {
    generateTestImage(100, cv::Range(3, 7), cv::Range(25, 5), 15);
    Mat testImage = imread("C:/Users/Svt/Desktop/ProcImage/pictures/lab04_test1.png", IMREAD_COLOR);
    if (testImage.empty()) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }

    int k = 3; // Количество кластеров
    Mat segmented = segmentImageKMeans(testImage, k);

    imshow("Test Image", testImage);
    imshow("Segmented Image", segmented);
    imwrite("C:/Users/Svt/Desktop/ProcImage/pictures/lab07_segm3.png", segmented);

    Mat gray;
    cvtColor(segmented, gray, COLOR_BGR2GRAY);

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

    Mat output = segmented.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(output, contours, (int)i, Scalar(0, 0, 255), 2);

        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);

        detected.push_back({ static_cast<int>(center.x), static_cast<int>(center.y), static_cast<int>(radius) });
    }

    imshow("Detected Contours", output);
    imwrite("C:/Users/Svt/Desktop/ProcImage/pictures/lab07_segm_res3.png", output);

    float qualityThreshold = 0.5;
    int TP = 0, FP = 0, FN = 0;

    evaluateDetection(groundTruth, detected, qualityThreshold, TP, FP, FN);
    //cv::imshow("Test Image", testImage);

    std::cout << "True Positives (TP): " << TP << std::endl;
    std::cout << "False Positives (FP): " << FP << std::endl;
    std::cout << "False Negatives (FN): " << FN << std::endl;

    float averageIoU = static_cast<float>(TP) / (TP + FN + FP);
    std::cout << "Average IoU: " << averageIoU << std::endl;


    waitKey(0);

    return 0;
}
