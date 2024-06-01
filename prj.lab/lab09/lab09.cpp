#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Функция для конвертации sRGB в linRGB
Mat convertToLinRGB(const Mat& inputImage) {
    Mat linRGBImage;
    inputImage.convertTo(linRGBImage, CV_32F, 1.0 / 255);

    for (int y = 0; y < linRGBImage.rows; y++) {
        for (int x = 0; x < linRGBImage.cols; x++) {
            Vec3f& color = linRGBImage.at<Vec3f>(y, x);
            for (int c = 0; c < 3; c++) {
                if (color[c] <= 0.04045) {
                    color[c] /= 12.92;
                }
                else {
                    color[c] = pow((color[c] + 0.055) / 1.055, 2.4);
                }
            }
        }
    }

    linRGBImage.convertTo(linRGBImage, CV_8UC3, 255);
    return linRGBImage;
}

// Функция для вычисления и отображения гистограммы
void showHistogram(const Mat& image, const string& windowName) {
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX);

    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow(windowName, histImage);
}

int main(int argc, char** argv) {
    Mat testImage = imread("grid.png", IMREAD_COLOR);
    if (testImage.empty()) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }

    // Конвертация изображения в linRGB
    Mat linRGBImage = convertToLinRGB(testImage);

    // Отображение оригинального изображения и его linRGB версии
    imshow("Original Image", testImage);
    imshow("linRGB Image", linRGBImage);

    // Вычисление и отображение гистограммы
    showHistogram(testImage, "Histogram - Original Image");
    showHistogram(linRGBImage, "Histogram - linRGB Image");

    waitKey(0);
    return 0;
}
