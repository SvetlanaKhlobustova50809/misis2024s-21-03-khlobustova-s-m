#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

Mat convertToLinearRGB(const Mat& img) {
    Mat linearImg = img.clone();
    linearImg.convertTo(linearImg, CV_64FC3);
    normalize(linearImg, linearImg, 0, 1, NORM_MINMAX);

    for (int i = 0; i < linearImg.rows; ++i) {
        for (int j = 0; j < linearImg.cols; ++j) {
            Vec3d& pixel = linearImg.at<Vec3d>(i, j); 

            for (int c = 0; c < 3; ++c) {
                double& color = pixel.val[c];

                if (color <= 0.04045) {
                    color /= 12.92;
                }
                else {
                    color = pow((color + 0.055) / 1.055, 2.4);
                }
            }
        }
    }

    return linearImg;
}

Point2f projectToPoint(const Vec3d& point, const Vec3d& basis1, const Vec3d& basis2) {
    float x = static_cast<float>(point.dot(basis1));
    float y = static_cast<float>(point.dot(basis2));
    return Point2f(x, y);
}

Mat createProjectionHistogram(const vector<Point2f>& projectedPoints, const vector<Vec3b>& colors, int histWidth, int histHeight) {
    Mat histogram(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));

    float minX = FLT_MAX, maxX = -FLT_MAX;
    float minY = FLT_MAX, maxY = -FLT_MAX;

    for (const auto& point : projectedPoints) {
        minX = min(minX, point.x);
        maxX = max(maxX, point.x);
        minY = min(minY, point.y);
        maxY = max(maxY, point.y);
    }

    for (size_t i = 0; i < projectedPoints.size(); i++) {
        int x = cvRound(histWidth * (projectedPoints[i].x - minX) / (maxX - minX));
        int y = cvRound(histHeight * (projectedPoints[i].y - minY) / (maxY - minY));
        if (x >= 0 && x < histWidth && y >= 0 && y < histHeight) {
            histogram.at<Vec3b>(histHeight - 1 - y, x) = colors[i];
        }
    }

    return histogram;
}

int main() {
    const Mat inputImage = imread("C:/Users/Svt/Desktop/ProcImage/pictures/lab03_test1.jpg", IMREAD_COLOR);

    if (inputImage.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat linearImage = convertToLinearRGB(inputImage);

    Vec3d basis1(1, -1, 0);
    Vec3d basis2(1, 1, -2);

    vector<Point2f> projectedPoints;
    vector<Vec3b> linearColors, originalColors;

    for (int y = 0; y < linearImage.rows; ++y) {
        for (int x = 0; x < linearImage.cols; ++x) {
            Vec3d intensity = linearImage.at<Vec3d>(y, x);

            Vec3d vector(intensity[0] - 0.5, intensity[1] - 0.5, intensity[2] - 0.5);

            Point2f projectedPoint = projectToPoint(vector, basis1, basis2);

            projectedPoints.push_back(projectedPoint);
            linearColors.push_back(intensity);
            originalColors.push_back(inputImage.at<Vec3b>(y, x));
        }
    }

    int histWidth = 256;
    int histHeight = 256;

    Mat projectionHist = createProjectionHistogram(projectedPoints, originalColors, histWidth, histHeight);
    Mat projectionHistLin = createProjectionHistogram(projectedPoints, linearColors, histWidth, histHeight);

    namedWindow("Projection Histogram LinRGB", WINDOW_NORMAL);
    imshow("Projection Histogram LinRGB", projectionHistLin);
    imwrite("C:/Users/Svt/Desktop/ProcImage/pictures/lab08_hist1.png", projectionHistLin);

    namedWindow("Projection Histogram", WINDOW_NORMAL);
    imshow("Projection Histogram", projectionHist);
    imwrite("C:/Users/Svt/Desktop/ProcImage/pictures/lab08_proj_hist1.png", projectionHist);

    ofstream outFile("C:/Users/Svt/Desktop/ProcImage/pictures/lab08_projected_points_colors1.txt");
    if (!outFile) {
        cout << "Could not open file for writing!" << endl;
        return -1;
    }
    for (size_t i = 0; i < projectedPoints.size(); ++i) {
        outFile << projectedPoints[i].x << " " << projectedPoints[i].y << " "
            << static_cast<int>(linearColors[i][0]) << " "
            << static_cast<int>(linearColors[i][1]) << " "
            << static_cast<int>(linearColors[i][2]) << endl;
    }
    outFile.close();

    waitKey(0);
    destroyAllWindows();

    return 0;
}
