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

    for (int i = 0; i < histSize; i++) {
        rectangle(histImage, Point(i, histImage.rows), Point(i + 1, histImage.rows - cvRound(hist.at<float>(i))), Scalar(0), 1);
    }

    return histImage;
}

Mat addNoise(const Mat& image, double stddev) {
    Mat noisyImage = image.clone();
    Mat im = image.clone();
    RNG rng;
    rng.fill(noisyImage, RNG::NORMAL, stddev, stddev);
    return im + noisyImage;
}

int main() {
    int side = 256;
    int innerSquareSide = 209;
    int circleRadius = 83;
    vector<int> brightnessLevels1 = { 0, 127, 255 };
    vector<int> brightnessLevels2 = { 20, 127, 235 };
    vector<int> brightnessLevels3 = { 55, 127, 200 };
    vector<int> brightnessLevels4 = { 90, 127, 165 };

    Mat testImage1 = generateTestImage(side, innerSquareSide, circleRadius, brightnessLevels1);
    Mat testImage2 = generateTestImage(side, innerSquareSide, circleRadius, brightnessLevels2);
    Mat testImage3 = generateTestImage(side, innerSquareSide, circleRadius, brightnessLevels3);
    Mat testImage4 = generateTestImage(side, innerSquareSide, circleRadius, brightnessLevels4);

    Mat combinedImages;

    Mat combinedTop;
    hconcat(testImage1, testImage2, combinedTop);
    hconcat(combinedTop, testImage3, combinedTop);
    hconcat(combinedTop, testImage4, combinedTop);

    vector<double> stddevValues = { 3.0, 7.0, 15.0 };

    vector<Mat> noisyTestImages(4);
    vector<Mat> histImages(4);
    Mat combinedBottom;
    Mat combinedHistograms;
    for (double stddev : stddevValues) {
        Mat noisyTestImage1 = addNoise(testImage1, stddev);
        Mat noisyTestImage2 = addNoise(testImage2, stddev);
        Mat noisyTestImage3 = addNoise(testImage3, stddev);
        Mat noisyTestImage4 = addNoise(testImage4, stddev);

        noisyTestImages[0] = (noisyTestImage1);
        noisyTestImages[1] = (noisyTestImage2);
        noisyTestImages[2] = (noisyTestImage3);
        noisyTestImages[3] = (noisyTestImage4);

        Mat histImage1 = drawHistogram(noisyTestImage1);
        Mat histImage2 = drawHistogram(noisyTestImage2);
        Mat histImage3 = drawHistogram(noisyTestImage3);
        Mat histImage4 = drawHistogram(noisyTestImage4);

        histImages[0] = (histImage1);
        histImages[1] = (histImage2);
        histImages[2] = (histImage3);
        histImages[3] = (histImage4);
        
        hconcat(noisyTestImages[0], noisyTestImages[1], combinedBottom);
        hconcat(combinedBottom, noisyTestImages[2], combinedBottom);
        hconcat(combinedBottom, noisyTestImages[3], combinedBottom);
        //imshow("", combinedBottom);
        //waitKey(10000);

       
        hconcat(histImages[0], histImages[1], combinedHistograms);
        hconcat(combinedHistograms, histImages[2], combinedHistograms);
        hconcat(combinedHistograms, histImages[3], combinedHistograms);

        vconcat(combinedTop, combinedBottom, combinedImages);
        vconcat(combinedImages, combinedHistograms, combinedImages);

        combinedTop = combinedImages;
    }

    string outputFileName = "C:/Users/Svt/Desktop/khlobustova/prj.lab/lab02/combined_image.png";
    imwrite(outputFileName, combinedImages);
    cout << "Combined image saved as " << outputFileName << endl;

    return 0;
}
