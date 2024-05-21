#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

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

double evaluateSegmentation(const Mat& groundTruth, const Mat& segmented) {
    CV_Assert(groundTruth.size() == segmented.size() && groundTruth.type() == segmented.type());

    int correctPixels = 0;
    int totalPixels = groundTruth.rows * groundTruth.cols;

    for (int y = 0; y < groundTruth.rows; y++) {
        for (int x = 0; x < groundTruth.cols; x++) {
            if (groundTruth.at<Vec3b>(y, x) == segmented.at<Vec3b>(y, x)) {
                correctPixels++;
            }
        }
    }

    return static_cast<double>(correctPixels) / totalPixels;
}

int main() {
    Mat testImage = imread("grid.png", IMREAD_COLOR);
    if (testImage.empty()) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }

    int k = 2;
    Mat segmented = segmentImageKMeans(testImage, k);

    double accuracy = evaluateSegmentation(testImage, segmented);
    cout << "Segmentation Accuracy: " << accuracy << endl;

    imshow("Test Image", testImage);
    imshow("Segmented Image", segmented);
    waitKey(0);

    return 0;
}
