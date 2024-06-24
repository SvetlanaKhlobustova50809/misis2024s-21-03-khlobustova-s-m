#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
bool f = true;

Mat autoContrastSingleChannel(Mat img) {
    if (img.depth() != CV_8U) {
        cout << "Image depth is not 8-bit. Converting to 8-bit depth." << endl;
        img.convertTo(img, CV_8U);
    }

    Mat hist;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    float alpha = 0.15, beta = 0.0001;
    float sum = 0;
    int totalPixels = img.rows * img.cols;
    int blackThreshold = 0, whiteThreshold = 255;

    for (int i = 0; i < histSize; i++) {
        sum += hist.at<float>(i);
        if (sum >= alpha * totalPixels) {
            blackThreshold = i;
            break;
        }
    }

    sum = 0;
    for (int i = histSize - 1; i >= 0; i--) {
        sum += hist.at<float>(i);
        if (sum >= beta * totalPixels) {
            whiteThreshold = i;
            break;
        }
    }

    Mat result;
    img.convertTo(result, CV_8U, 255.0 / (whiteThreshold - blackThreshold), -255.0 * blackThreshold / (whiteThreshold - blackThreshold));
    return result;
}
//
//Mat autoContrastJoint(Mat img) {
//    if (img.depth() != CV_8U) {
//        cout << "Image depth is not 8-bit. Converting to 8-bit depth." << endl;
//        img.convertTo(img, CV_8U);
//    }
//
//    // Создаем объединенную гистограмму
//    vector<Mat> channels;
//    split(img, channels);
//
//    Mat combinedHist = Mat::zeros(256, 1, CV_32F);
//    for (int i = 0; i < channels.size(); i++) {
//        Mat hist;
//        int histSize = 256;
//        float range[] = { 0, 256 };
//        const float* histRange = { range };
//        bool uniform = true, accumulate = false;
//
//        calcHist(&channels[i], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
//        combinedHist += hist;
//    }
//
//    // Определяем старые пороги (clow и chigh)
//    float alpha = 0.15, beta = 0.0001;
//    float sum = 0;
//    int totalPixels = img.rows * img.cols * img.channels();
//    int clow = 0, chigh = 255;
//
//    for (int i = 0; i < 256; i++) {
//        sum += combinedHist.at<float>(i);
//        if (sum >= alpha * totalPixels) {
//            clow = i;
//            break;
//        }
//    }
//
//    sum = 0;
//    for (int i = 255; i >= 0; i--) {
//        sum += combinedHist.at<float>(i);
//        if (sum >= beta * totalPixels) {
//            chigh = i;
//            break;
//        }
//    }
//
//    // Устанавливаем новые пороги (cmin и cmax)
//    int cmin = 0, cmax = 255;
//
//    // Применяем автоконтрастирование
//    img.convertTo(img, CV_32F);
//    Mat result = img.clone();
//    result.forEach<Vec3f>([&](Vec3f& pixel, const int* position) -> void {
//        for (int i = 0; i < 3; i++) {
//            pixel[i] = cmin + (pixel[i] - clow) * (cmax - cmin) / (chigh - clow);
//            pixel[i] = std::min(std::max(pixel[i], static_cast<float>(cmin)), static_cast<float>(cmax)); // Ограничение значений диапазоном [cmin, cmax]
//        }
//        });
//    result.convertTo(result, CV_8U);
//
//    return result;
//}

//Mat autoContrastColor(Mat img) {
//    vector<Mat> channels;
//    split(img, channels);
//
//    for (int i = 0; i < channels.size(); i++) {
//        channels[i] = autoContrastSingleChannel(channels[i]);
//    }
//
//    Mat result;
//    merge(channels, result);
//    return result;
//}

Mat autoContrastColor(Mat img, bool useJointContrast = false) {
    if (useJointContrast) {
        if (img.depth() != CV_8U) {
            cout << "Image depth is not 8-bit. Converting to 8-bit depth." << endl;
            img.convertTo(img, CV_8U);
        }

        // Создаем объединенную гистограмму
        vector<Mat> channels;
        split(img, channels);

        Mat combinedHist = Mat::zeros(256, 1, CV_32F);
        for (int i = 0; i < channels.size(); i++) {
            Mat hist;
            int histSize = 256;
            float range[] = { 0, 256 };
            const float* histRange = { range };
            bool uniform = true, accumulate = false;

            calcHist(&channels[i], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
            combinedHist += hist;
        }

        // Определяем старые пороги (clow и chigh)
        float alpha = 0.15, beta = 0.0001;
        float sum = 0;
        int totalPixels = img.rows * img.cols * img.channels();
        int clow = 0, chigh = 255;

        for (int i = 0; i < 256; i++) {
            sum += combinedHist.at<float>(i);
            if (sum >= alpha * totalPixels) {
                clow = i;
                break;
            }
        }

        sum = 0;
        for (int i = 255; i >= 0; i--) {
            sum += combinedHist.at<float>(i);
            if (sum >= beta * totalPixels) {
                chigh = i;
                break;
            }
        }

        // Устанавливаем новые пороги (cmin и cmax)
        int cmin = 0, cmax = 255;

        // Применяем автоконтрастирование
        img.convertTo(img, CV_32F);
        Mat result = img.clone();
        result.forEach<Vec3f>([&](Vec3f& pixel, const int* position) -> void {
            for (int i = 0; i < 3; i++) {
                pixel[i] = cmin + (pixel[i] - clow) * (cmax - cmin) / (chigh - clow);
                pixel[i] = std::min(std::max(pixel[i], static_cast<float>(cmin)), static_cast<float>(cmax)); // Ограничение значений диапазоном [cmin, cmax]
            }
            });
        result.convertTo(result, CV_8U);

        return result;
    }
    else {
        vector<Mat> channels;
        split(img, channels);

        for (int i = 0; i < channels.size(); i++) {
            channels[i] = autoContrastSingleChannel(channels[i]);
        }

        Mat result;
        merge(channels, result);
        return result;
    }
}

void showHistogram(Mat& img) {
    vector<Mat> bgr_planes;
    split(img, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, LINE_AA);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, LINE_AA);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, LINE_AA);
    }

    imshow("Histogram", histImage);

    if (f) {
        string outputFileName = "C:/Users/Svt/Desktop/ProcImage/pictures/lab03_gist2.png";
        imwrite(outputFileName, histImage);
    }
}

int main(int argc, char* argv[]) {
    Mat img = imread("C:/Users/Svt/Desktop/ProcImage/pictures/lab03_test2.jpg");

    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat result;
    if (img.channels() == 1) {
        result = autoContrastSingleChannel(img);
    }
    else {
        bool useJointContrast = true; // Или false, если хотите использовать поканальное контрастирование
        result = autoContrastColor(img, useJointContrast);
    }
    showHistogram(img);
    showHistogram(result);

    imshow("Original Image", img);
    imshow("Auto-Contrast Image", result);
    string outputFileName = "C:/Users/Svt/Desktop/ProcImage/pictures/lab03_res2.png";
    imwrite(outputFileName, result);
    waitKey(100000);

}
