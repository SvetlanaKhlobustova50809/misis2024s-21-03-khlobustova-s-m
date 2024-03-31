#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat generateTestImage(int side, int innerSquareSide, int circleRadius, vector<Scalar> colors) {
    Mat testImage(side, side, CV_8UC3, Scalar(0, 0, 0));
    Rect innerSquare((side - innerSquareSide) / 2, (side - innerSquareSide) / 2, innerSquareSide, innerSquareSide);
    testImage(innerSquare).setTo(colors[1]);
    circle(testImage, Point(side / 2, side / 2), circleRadius, colors[2], -1);
    return testImage;
}

int main() {
    int side = 99;
    int innerSquareSide = 99;
    int circleRadius = 25;

    vector<vector<Scalar>> colors = {
        {0, 127, 255},
        {0, 255, 127},
        {127, 0, 255},
        {127, 255, 0},
        {255, 0, 127},
        {255, 127, 0}
    };

    Mat combinedImages;

    for (int i = 0; i < colors.size(); ++i) {
        Mat testImage = generateTestImage(side, innerSquareSide, circleRadius, colors[i]);

        Mat kernel1 = (Mat_<float>(2, 2) << 1, 0, 0, -1);
        Mat result1;
        filter2D(testImage, result1, -1, kernel1);

        Mat kernel2 = (Mat_<float>(2, 2) << 0, 1, -1, 0);
        Mat result2;
        filter2D(testImage, result2, -1, kernel2);

        Mat combinedResult;
        hconcat(testImage, result1, combinedResult);
        hconcat(combinedResult, result2, combinedResult);

        if (i == 0)
            combinedImages = combinedResult.clone();
        else {
            Mat combined;
            hconcat(combinedImages, combinedResult, combinedImages);
        }
    }

    string outputFileName = "combined_image_color.png";
    imwrite(outputFileName, combinedImages);
    cout << "Combined image saved as " << outputFileName << endl;

    return 0;
}
