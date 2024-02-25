#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void gammaCorrection(Mat& img, double gamma) {
    CV_Assert(img.depth() != sizeof(uchar));

    uchar lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = saturate_cast<uchar>(pow((double)(i / 255.0), gamma) * 255.0);
    }

    const int channels = img.channels();
    MatIterator_<uchar> it, end;
    for (it = img.begin<uchar>(), end = img.end<uchar>(); it != end; it++) {
        *it = lut[(*it)];
    }
}

int main(int argc, char* argv[]) {
    int s = 3;
    int h = 30;
    double gamma = 2.4;
    string outputFilename;

    if (argc > 1) {
        s = atoi(argv[1]);
    }
    if (argc > 2) {
        h = atoi(argv[2]);
    }
    if (argc > 3) {
        gamma = atof(argv[3]);
    }
    if (argc > 4) {
        outputFilename = argv[4];
    }

    Mat gradientImg(h, 256 * s, CV_8UC1, Scalar(0));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < 256 * s; j++) {
            gradientImg.at<uchar>(i, j) = j / s;
        }
    }

    Mat gammaCorrectedImg = gradientImg.clone();
    gammaCorrection(gammaCorrectedImg, gamma);

    if (outputFilename.empty()) {
        Mat combinedImg;
        vconcat(gradientImg, gammaCorrectedImg, combinedImg);
        imshow("Gradient and gamma image", combinedImg);
        waitKey(0);
    }
    else {
        imwrite(outputFilename, gammaCorrectedImg);
        cout << "Image saved as " << outputFilename << endl;
    }

    return 0;
}
