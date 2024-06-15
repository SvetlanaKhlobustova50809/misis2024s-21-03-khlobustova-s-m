#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <queue>

using namespace cv;
using namespace std;

Mat1f createPatternImage(int sideLength, int circleRadius, int bgColor, int fgColor) {
    Mat1f patternImage(sideLength, sideLength, bgColor);
    cv::Point centerPoint(sideLength / 2, sideLength / 2);
    cv::circle(patternImage, centerPoint, circleRadius, fgColor, cv::FILLED);
    return patternImage;
}

Mat1f assembleTestImage(int black, int gray, int white) {
    int sideLength = 99;
    int circleRadius = 25;
    Mat1f finalImage;
    vector<vector<Mat1f>> imageMatrix(2, vector<Mat1f>(3));

    imageMatrix[0][0] = createPatternImage(sideLength, circleRadius, black, white);
    imageMatrix[0][1] = createPatternImage(sideLength, circleRadius, gray, black);
    imageMatrix[0][2] = createPatternImage(sideLength, circleRadius, white, black);
    imageMatrix[1][0] = createPatternImage(sideLength, circleRadius, white, gray);
    imageMatrix[1][1] = createPatternImage(sideLength, circleRadius, black, white);
    imageMatrix[1][2] = createPatternImage(sideLength, circleRadius, gray, white);

    vconcat(imageMatrix[0][0], imageMatrix[1][0], finalImage);
    for (int i = 1; i < 3; i++) {
        Mat1f tempConcat;
        vconcat(imageMatrix[0][i], imageMatrix[1][i], tempConcat);
        hconcat(finalImage, tempConcat, finalImage);
    }
    return finalImage;
}

int main(int argc, char* argv[]) {
    Mat1f testPattern = assembleTestImage(0, 127, 255);
    Mat core1 = (Mat_<float>(2, 2) << 1, 0, 0, -1);
    Mat core2 = (Mat_<float>(2, 2) << 0, 1, -1, 0);
    Mat1f filteredImage1, filteredImage2;
    Mat1f combinedImage = testPattern.clone();

    filter2D(testPattern, filteredImage1, -1, core1);
    filter2D(testPattern, filteredImage2, -1, core2);

    double minVal, maxVal;
    minMaxLoc(filteredImage1, &minVal, &maxVal);
    filteredImage1 = filteredImage1 * (255 / (maxVal - minVal)) + maxVal / 2;
    minMaxLoc(filteredImage2, &minVal, &maxVal);
    filteredImage2 = filteredImage2 * (255 / (maxVal - minVal)) + maxVal / 2;

    for (int row = 0; row < testPattern.rows; row++) {
        for (int col = 0; col < testPattern.cols; col++) {
            combinedImage(row, col) = sqrt(pow(filteredImage1(row, col), 2) + pow(filteredImage2(row, col), 2));
        }
    }
    vector<Mat1b> channels(3);
    channels[0] = Mat1b(filteredImage1.clone());
    channels[1] = Mat1b(filteredImage2.clone());
    channels[2] = Mat1b(combinedImage.clone());

    Mat colorResult;
    merge(channels, colorResult);

    imshow("Test", testPattern);
    imshow("Combined Result", colorResult);
    waitKey(0);

    return 0;
}
