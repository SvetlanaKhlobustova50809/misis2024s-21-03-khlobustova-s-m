#include <opencv2/opencv.hpp>

cv::Mat generateTestImage(int numObjects, cv::Range sizeRange, cv::Range contrastRange, int blurSize) {
    cv::Mat image(550, 550, CV_8UC3, cv::Scalar(0, 0, 0));

    double contrastStep = 2.2;
    double sizeStep = 0.5;

    int circleDistance = 40;

    for (int i = 0; i < 10; i++) {
        double contrast = contrastRange.end + i * contrastStep;

        for (int j = 0; j < 10; j++) {
            double size = sizeRange.start + j * sizeStep + 3;

            if ((j + 1) * 50 + size <= image.cols && (i + 1) * 50 + size <= image.rows) {
                cv::Mat circleMask = cv::Mat::zeros(image.size(), CV_8U);
                cv::circle(circleMask, cv::Point((j + 1) * 50, (i + 1) * 50), size, cv::Scalar(255), -1);

                cv::Mat circle;
                image.copyTo(circle);
                circle.setTo(cv::Scalar(255, 255, 255), circleMask);
                circle.convertTo(circle, -1, contrast / 30.0);
                circle.copyTo(image, circleMask);
            }
        }
    }

    cv::GaussianBlur(image, image, cv::Size(blurSize, blurSize), 0);

    return image;
}

cv::Mat thresholdBinary(cv::Mat inputImage, int thresholdValue) {
    cv::Mat binaryImage;
    cv::cvtColor(inputImage, binaryImage, cv::COLOR_BGR2GRAY);
    cv::threshold(binaryImage, binaryImage, thresholdValue, 255, cv::THRESH_BINARY);
    return binaryImage;
}

cv::Mat adaptiveBinary(cv::Mat inputImage, int blockSize, double C) {
    cv::Mat binaryImage;
    cv::cvtColor(inputImage, binaryImage, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(binaryImage, binaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
    return binaryImage;
}

void tuneBinaryParameters(cv::Mat inputImage) {
    int thresholdValue = 127;
    int blockSize = 11;
    double C = 2.0;
    int adaptive1 = 21, adaptive2 = -10;

    cv::Mat binaryImage, adaptive;

    cv::namedWindow("Binary Parameters", cv::WINDOW_GUI_NORMAL);
    cv::createTrackbar("Threshold Value", "Binary Parameters", &thresholdValue, 255);
    cv::createTrackbar("adaptive1 Value", "Binary Parameters", &adaptive1, 51, [](int value, void* userdata) {
        if (value % 2 == 0) {
            cv::setTrackbarPos("adaptive1 Value", "Binary Parameters", value + 1);
        }
        }, nullptr);
    cv::setTrackbarMin("adaptive1 Value", "Binary Parameters", 3);

    cv::createTrackbar("adaptive2 Value", "Binary Parameters", &adaptive2, 10);
    /*cv::createTrackbar("Block Size", "Binary Parameters", &blockSize, 255);
    cv::createTrackbar("C", "Binary Parameters", (int*)&C, 255);*/

    while (true) {
        binaryImage = thresholdBinary(inputImage, thresholdValue);
        adaptive = adaptiveBinary(inputImage, adaptive1, adaptive2);
        cv::imshow("Binary Image", binaryImage);
        cv::imshow("Adaptive Thresholded Image", adaptive);

        char key = cv::waitKey(10);
        if (key == 27) // Нажатие ESC для выхода
            break;
    }
}

cv::Mat otsuBinary(cv::Mat inputImage) {
    cv::Mat binaryImage;
    cv::cvtColor(inputImage, binaryImage, cv::COLOR_BGR2GRAY);
    cv::threshold(binaryImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binaryImage;
}

int main(int argc, const char* argv[]) {
    cv::Mat testImage = generateTestImage(100, cv::Range(3, 7), cv::Range(25, 5), 15);

    //cv::imshow("Test Image", testImage);
    //cv::Mat thresholded = thresholdBinary(testImage, 100);
    //cv::Mat adaptive = adaptiveBinary(testImage, 21, 10);
    cv::Mat otsu = otsuBinary(testImage);

     //cv::imshow("Thresholded Image", thresholded);
     //cv::imshow("Adaptive Thresholded Image", adaptive);
     cv::imshow("Otsu Thresholded Image", otsu);

    tuneBinaryParameters(testImage);
    cv::waitKey(100000);
}

