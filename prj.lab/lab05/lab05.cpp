#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Функция для генерации изображения круга на фоне квадрата
Mat generateTestImage(int side, int circleRadius, Scalar bgColor, Scalar circleColor) {
    Mat testImage(side, side, CV_8UC1, bgColor);
    circle(testImage, Point(side / 2, side / 2), circleRadius, circleColor, -1);
    return testImage;
}

int main() {
    int side = 99;
    int circleRadius = 25;
    vector<Scalar> colors = { {0, 127},
                              {127, 0},
                              {255, 0},
                              {255, 127},
                              {0, 255},
                              {127, 255} };
    int rows = 2, cols = 3;

    Mat combinedImages(rows * side, cols * side, CV_8UC1, Scalar(0));

    vector<Mat> testImages;
    int index = 0;
    int ind = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Scalar bgColor = colors[ind][0];
            Scalar circleColor = colors[ind][1];
            Mat testImage = generateTestImage(side, circleRadius, bgColor, circleColor);
            testImage.copyTo(combinedImages(Rect(j * side, i * side, side, side)));
            testImages.push_back(testImage);
            ind++;
        }
    }

    imshow("Test Images", combinedImages);

    Mat kernel1 = (Mat_<float>(2, 2) << 1, 0, 0, -1);
    Mat kernel2 = (Mat_<float>(2, 2) << 0, 1, -1, 0);

    vector<Mat> I1_images, I2_images;
    for (const auto& img : testImages) {
        Mat I1, I2;
        filter2D(img, I1, CV_32F, kernel1);
        filter2D(img, I2, CV_32F, kernel2);
        I1_images.push_back(I1);
        I2_images.push_back(I2);
    }

    Mat I1_combined(rows * side, cols * side, CV_32F, Scalar(0));
    Mat I2_combined(rows * side, cols * side, CV_32F, Scalar(0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            I1_images[idx].copyTo(I1_combined(Rect(j * side, i * side, side, side)));
            I2_images[idx].copyTo(I2_combined(Rect(j * side, i * side, side, side)));
        }
    }

    imshow("I1 Images", I1_combined);
    imshow("I2 Images", I2_combined);

    vector<Mat> I3_images;
    for (size_t i = 0; i < I1_images.size(); ++i) {
        Mat I3;
        magnitude(I1_images[i], I2_images[i], I3);
        I3_images.push_back(I3);
    }

    Mat I3_combined(rows * side, cols * side, CV_32F, Scalar(0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            I3_images[idx].copyTo(I3_combined(Rect(j * side, i * side, side, side)));
        }
    }

    imshow("I3 Images", I3_combined);

    Mat visualized(rows * side, cols * side, CV_8UC3, Scalar(0, 0, 0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            vector<Mat> channels = { I1_images[idx], I2_images[idx], I3_images[idx] };
            Mat merged;
            merge(channels, merged);
            merged.convertTo(merged, CV_8UC3);
            merged.copyTo(visualized(Rect(j * side, i * side, side, side)));
        }
    }

    imshow("Visualized Image", visualized);
    waitKey(100000);

    return 0;
}
