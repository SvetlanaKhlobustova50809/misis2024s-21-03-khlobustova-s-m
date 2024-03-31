#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat autoContrastSingleChannel(Mat img) {
    if (img.depth() != CV_8U) {
        cout << "Image depth is not 8-bit. Converting to 8-bit depth." << endl;
        img.convertTo(img, CV_8U);
    }

    Mat hist;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    float alpha = 0.1, beta = 0.1;
    float sum = 0;
    int totalPixels = img.rows * img.cols;
    int blackThreshold = 0, whiteThreshold = 255;

    for (int i = 0; i < histSize; i++) {
        sum += hist.at<float>(i);
        if (sum >= alpha * totalPixels) {
            blackThreshold = i;
            break;
        }
    }

    sum = 0;
    for (int i = histSize - 1; i >= 0; i--) {
        sum += hist.at<float>(i);
        if (sum >= beta * totalPixels) {
            whiteThreshold = i;
            break;
        }
    }

    Mat result;
    img.convertTo(result, CV_8U, 255.0 / (whiteThreshold - blackThreshold), -255.0 * blackThreshold / (whiteThreshold - blackThreshold));
    return result;
}

// Функция для автоконтрастирования цветного изображения
Mat autoContrastColor(Mat img) {
    vector<Mat> channels;
    split(img, channels);

    // Применение автоконтрастирования для каждого канала
    for (int i = 0; i < channels.size(); i++) {
        channels[i] = autoContrastSingleChannel(channels[i]);
    }

    Mat result;
    merge(channels, result);
    return result;
}

void showHistogram(Mat& img) {
    // Разделение изображения на каналы
    vector<Mat> bgr_planes;
    split(img, bgr_planes);

    // Создание гистограмм
    int histSize = 256;
    float range[] = { 0, 256 }; // диапазон значений
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // Рисование гистограмм
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // Нормализация результатов до диапазона [0, histImage.rows]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Рисование для каждого канала
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, LINE_AA);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, LINE_AA);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, LINE_AA);
    }

    // Отображение гистограммы
    imshow("Histogram", histImage);
    waitKey(10000);
}

int main(int argc, char* argv[]) {
    Mat img = imread("C:/Users/Svt/Desktop/khlobustova/prj.lab/pictures/test_image2.jpg");


    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat result;
    if (img.channels() == 1) {
        result = autoContrastSingleChannel(img);
    }
    else {
        result = autoContrastColor(img);
    }
    showHistogram(img);
    showHistogram(result);

    imshow("Original Image", img);
    imshow("Auto-Contrast Image", result);
    waitKey(10000);
}
