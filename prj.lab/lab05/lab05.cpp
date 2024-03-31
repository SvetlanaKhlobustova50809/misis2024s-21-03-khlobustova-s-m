#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

int main() {
    Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255)); // Белый фон
    rectangle(img, Point(50, 50), Point(150, 150), Scalar(0, 0, 255), -1); // Красный прямоугольник
    circle(img, Point(250, 250), 50, Scalar(0, 255, 0), -1); // Зеленый круг
    line(img, Point(100, 300), Point(300, 100), Scalar(255, 0, 0), 2); // Синяя линия


    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }
    double alpha = 0.5, beta = 0.5;
    Mat new_image = Mat::zeros(img.size(), img.type());

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            for (int c = 0; c < img.channels(); c++) {
                new_image.at<Vec3b>(y, x)[c] =
                    saturate_cast<uchar>(alpha * img.at<Vec3b>(y, x)[c] + beta);
            }
        }
    }


    //Mat result;
    //// Проверка на одноканальное или цветное изображение
    //if (img.channels() == 1) {
    //    result = autoContrastSingleChannel(img);
    //}
    //else {
    //    result = autoContrastColor(img);
    //}

    // Отображение результатов
    imshow("Original Image", img);
    imshow("Auto-Contrast Image", new_image);
    waitKey(10000);

    

    return 0;
}
