#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat generateTestImage(int side, int innerSquareSide, int circleRadius, vector<int> brightnessLevels) {
    Mat testImage(side, side, CV_8UC1, Scalar(0));
    Rect innerSquare((side - innerSquareSide) / 2, (side - innerSquareSide) / 2, innerSquareSide, innerSquareSide);
    testImage(innerSquare).setTo(brightnessLevels[1]);
    circle(testImage, Point(side / 2, side / 2), circleRadius, brightnessLevels[2], -1);
    return testImage;
}

Mat drawHistogram(const Mat& image) {
    Mat histImage(256, 256, CV_8UC1, Scalar(230));
    Mat hist;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Рисование столбиков гистограммы
    for (int i = 0; i < histSize; i++) {
        rectangle(histImage, Point(i, histImage.rows), Point(i + 1, histImage.rows - cvRound(hist.at<float>(i))), Scalar(0), 1);
    }

    return histImage;
}

Mat addNoise(const Mat& image, double stddev) {
    Mat noisyImage = image.clone();
    Mat noise(image.size(), CV_8U);
    randn(noise, Scalar::all(0), Scalar::all(stddev));
    noisyImage += noise;
    return noisyImage;
}

int main() {
    // Параметры тестовых изображений
    int side = 256;
    int innerSquareSide = 209;
    int circleRadius = 83;
    vector<int> brightnessLevels1 = { 0, 127, 255 };
    vector<int> brightnessLevels2 = { 20, 127, 235 };
    vector<int> brightnessLevels3 = { 55, 127, 200 };
    vector<int> brightnessLevels4 = { 90, 127, 165 };

    // Генерация тестовых изображений
    Mat testImage1 = generateTestImage(side, innerSquareSide, circleRadius, brightnessLevels1);
    Mat testImage2 = generateTestImage(side, innerSquareSide, circleRadius, brightnessLevels2);
    Mat testImage3 = generateTestImage(side, innerSquareSide, circleRadius, brightnessLevels3);
    Mat testImage4 = generateTestImage(side, innerSquareSide, circleRadius, brightnessLevels4);

    // Значения среднеквадратичного отклонения для шума
    vector<double> stddevValues = { 3.0, 7.0, 15.0 };

    // Генерация зашумленных изображений и их гистограмм
    vector<Mat> noisyTestImages;
    vector<Mat> histImages;
    for (double stddev : stddevValues) {
        Mat noisyTestImage1 = addNoise(testImage1, stddev);
        Mat noisyTestImage2 = addNoise(testImage2, stddev);
        Mat noisyTestImage3 = addNoise(testImage3, stddev);
        Mat noisyTestImage4 = addNoise(testImage4, stddev);

        noisyTestImages.push_back(noisyTestImage1);
        noisyTestImages.push_back(noisyTestImage2);
        noisyTestImages.push_back(noisyTestImage3);
        noisyTestImages.push_back(noisyTestImage4);

        Mat histImage1 = drawHistogram(noisyTestImage1);
        Mat histImage2 = drawHistogram(noisyTestImage2);
        Mat histImage3 = drawHistogram(noisyTestImage3);
        Mat histImage4 = drawHistogram(noisyTestImage4);

        histImages.push_back(histImage1);
        histImages.push_back(histImage2);
        histImages.push_back(histImage3);
        histImages.push_back(histImage4);
    }

    // Соединение изображений в одно окно
    Mat combinedTop;
    hconcat(noisyTestImages[0], noisyTestImages[1], combinedTop);
    hconcat(combinedTop, noisyTestImages[2], combinedTop);
    hconcat(combinedTop, noisyTestImages[3], combinedTop);

    Mat combinedBottom;
    hconcat(histImages[0], histImages[1], combinedBottom);
    hconcat(combinedBottom, histImages[2], combinedBottom);
    hconcat(combinedBottom, histImages[3], combinedBottom);

    Mat combined;
    vconcat(combinedTop, combinedBottom, combined);

    // Отображение сгенерированных изображений
    imshow("Combined Images", combined);
    waitKey(10000);

    return 0;
}
