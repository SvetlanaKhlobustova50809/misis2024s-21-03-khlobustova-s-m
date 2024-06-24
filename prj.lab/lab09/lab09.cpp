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

double calculateMSE(const Mat& src1, const Mat& src2) {
    Mat s1;
    absdiff(src1, src2, s1);       // |src1 - src2|
    s1.convertTo(s1, CV_32F);  // convert to float
    s1 = s1.mul(s1);           // |src1 - src2|^2

    Scalar s = sum(s1);        // sum elements per channel

    double mse = (s[0] + s[1] + s[2]) / (double)(src1.channels() * src1.total());
    return mse;
}

double calculateColorfulnessIndex(const Mat& src) {
    Mat lab_image;
    cvtColor(src, lab_image, COLOR_BGR2Lab);

    Scalar meanLab, stddevLab;
    meanStdDev(lab_image, meanLab, stddevLab);

    double colorfulness = sqrt(stddevLab[1] * stddevLab[1] + stddevLab[2] * stddevLab[2]) + 0.3 * sqrt(meanLab[1] * meanLab[1] + meanLab[2] * meanLab[2]);

    return colorfulness;
}

void measureQuality(const Mat& srcImage, const Mat& balancedImage) {
    double originalColorfulness = calculateColorfulnessIndex(srcImage);
    double balancedColorfulness = calculateColorfulnessIndex(balancedImage);
    double mse = calculateMSE(srcImage, balancedImage);

    cout << "Original Image Colorfulness Index: " << originalColorfulness << endl;
    cout << "Balanced Image Colorfulness Index: " << balancedColorfulness << endl;
    cout << "Mean Squared Error (MSE): " << mse << endl;
}

int main() {
    Mat srcImage = imread("C:/Users/Svt/Desktop/ProcImage/pictures/lab03_test1.jpg", IMREAD_COLOR);
    if (srcImage.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat balancedImage = applyGrayWorld(srcImage);

    imshow("Original Image", srcImage);
    imshow("Gray World Balanced Image", balancedImage);
    imwrite("C:/Users/Svt/Desktop/ProcImage/pictures/lab09_gray_world_balanced1.jpg", balancedImage);

    measureQuality(srcImage, balancedImage);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
