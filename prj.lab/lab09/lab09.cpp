#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat applyGrayWorld(const Mat& src) {
    Mat dst = src.clone();

    Scalar meanValue = mean(src);

    double avgGray = (meanValue[0] + meanValue[1] + meanValue[2]) / 3.0;
    double kB = avgGray / meanValue[0];
    double kG = avgGray / meanValue[1];
    double kR = avgGray / meanValue[2];

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b intensity = src.at<Vec3b>(y, x);
            dst.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(intensity[0] * kB);
            dst.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(intensity[1] * kG);
            dst.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(intensity[2] * kR);
        }
    }

    return dst;
}

int main() {
    Mat srcImage = imread("C:/Users/Svt/Desktop/ProcImage/pictures/lab03_test2.jpg", IMREAD_COLOR);
    if (srcImage.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat balancedImage = applyGrayWorld(srcImage);

    imshow("Original Image", srcImage);
    imshow("Gray World Balanced Image", balancedImage);
    imwrite("C:/Users/Svt/Desktop/ProcImage/pictures/lab09_gray_world_balanced.jpg", balancedImage);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
