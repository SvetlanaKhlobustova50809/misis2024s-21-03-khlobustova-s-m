#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;

vector<string> data_links;
vector<vector<Point>> srcContours;

typedef struct
{
	string data;
	vector<Point> location;
} decodedObject;


Mat transformPerspective(Mat& im, vector<Point>& points)
{
	const int outputWidth = 200;
	const int outputHeight = 200;

	vector<Point2f> dstPoints = { Point2f(0, 0), Point2f(outputWidth - 1, 0), Point2f(outputWidth - 1, outputHeight - 1), Point2f(0, outputHeight - 1) };

	vector<Point2f> srcPoints;
	for (const Point& pt : points)
	{
		srcPoints.push_back(Point2f(pt.x, pt.y));
	}

	Mat transformMatrix = getPerspectiveTransform(srcPoints, dstPoints);

	Mat transformed;
	warpPerspective(im, transformed, transformMatrix, Size(outputWidth, outputHeight));

	return transformed;
}

void display(Mat& im, vector<decodedObject>& decodedObjects)
{
	for (int i = 0; i < decodedObjects.size(); i++)
	{
		vector<Point> points = decodedObjects[i].location;
		srcContours.push_back(decodedObjects[i].location);
		vector<Point> hull;

		if (points.size() > 4)
			convexHull(points, hull);
		else
			hull = points;

		int n = hull.size();

		for (int j = 0; j < n; j++)
		{
			line(im, hull[j], hull[(j + 1) % n], Scalar(255, 0, 0), 5);
		}
	}

}

void Find_QR_Rect(Mat src, vector<Mat>& ROI_Rect, vector<decodedObject>& decodedObjects)
{
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	Mat blur;
	GaussianBlur(gray, blur, Size(3, 3), 0);

	Mat bin;
	threshold(blur, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imwrite("C:/Users/Svt/Desktop/khlobustova/prj.cw/input/bin1.png", bin);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 1));
	Mat open;
	morphologyEx(bin, open, MORPH_OPEN, kernel);

	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(21, 1));
	Mat close;
	morphologyEx(open, close, MORPH_CLOSE, kernel1);
	imwrite("C:/Users/Svt/Desktop/khlobustova/prj.cw/input/kernel1.png", close);

	vector<vector<Point>>MaxContours;
	findContours(close, MaxContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	bool f = true;

	for (int i = 0; i < MaxContours.size(); i++)
	{
		Mat mask = Mat::zeros(src.size(), CV_8UC3);
		mask = Scalar::all(255);

		vector<Point> poly;
		approxPolyDP(MaxContours[i], poly, 0.02 * arcLength(MaxContours[i], true), true);

		double area = contourArea(MaxContours[i]);

		if (area > 6000 && area < 100000)
		{
			RotatedRect MaxRect = minAreaRect(MaxContours[i]);
			double ratio = MaxRect.size.width / MaxRect.size.height;

			if (ratio > 0.8 && ratio < 1.2)
			{
				Rect MaxBox = MaxRect.boundingRect();
				Mat ROI = src(Rect(MaxBox.tl(), MaxBox.br()));
				ROI.copyTo(mask(MaxBox));
				ROI_Rect.push_back(mask);
				f = false;
			}
		}
	}
	if (f) {
		Mat edges;
		Canny(gray, edges, 100, 200);

		vector<vector<Point>> contours;
		findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		for (const auto& contour : contours)
		{
			vector<Point> poly;
			approxPolyDP(contour, poly, 0.02 * arcLength(contour, true), true);

			if (poly.size() == 4)
			{
				Mat transformed = transformPerspective(src, poly);

				QRCodeDetector qrDecoder = QRCodeDetector::QRCodeDetector();
				vector<Point> bbox;
				string data = qrDecoder.detectAndDecode(transformed, bbox);

				if (!data.empty())
				{
					decodedObject obj;
					obj.data = data;

					data_links.push_back(obj.data);

					for (const Point& pt : poly)
					{
						obj.location.push_back(pt);
					}

					decodedObjects.push_back(obj);
				}
			}
		}
		display(src, decodedObjects);
	}
}

int Dectect_QR_Rect(Mat src, Mat& canvas, vector<Mat>& ROI_Rect)
{
	vector<vector<Point>>QR_Rect;

	for (int i = 0; i < ROI_Rect.size(); i++)
	{
		Mat gray;
		cvtColor(ROI_Rect[i], gray, COLOR_BGR2GRAY);

		Mat bin;
		threshold(gray, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

		vector<vector<Point>>contours;
		vector<Vec4i>hierarchy;
		findContours(bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

		int ParentIndex = -1;
		int cn = 0;

		vector<Point>rect_points;
		for (int i = 0; i < contours.size(); i++) {
			if (hierarchy[i][2] != -1 && cn == 0) {
				ParentIndex = i;
				cn++;
			}
			else if (hierarchy[i][2] != -1 && cn == 1) {
				cn++;
			}
			else if (hierarchy[i][2] == -1) {
				ParentIndex = -1;
				cn = 0;
			}

			if (hierarchy[i][2] != -1 && cn == 2) {
				drawContours(canvas, contours, ParentIndex, Scalar::all(255), -1);

				RotatedRect rect;

				rect = minAreaRect(contours[ParentIndex]);

				rect_points.push_back(rect.center);

			}
		}

		for (int i = 0; i < rect_points.size(); i++) {
			line(canvas, rect_points[i], rect_points[(i + 1) % rect_points.size()], Scalar::all(255), 5);
		}

		QR_Rect.push_back(rect_points);
	}

	return QR_Rect.size();
}

vector<string> readFile(const string& filename) {
	ifstream file(filename);
	vector<string> lines;
	string line;
	while (getline(file, line)) {
		lines.push_back(line);
	}
	file.close();
	return lines;
}

vector<vector<Point>> parseFile(const string& filename, int val) {
	vector<vector<Point>> layoutContours;
	ifstream file(filename);
	string line;
	vector<Point> points;
	int cou = 0;

	while (getline(file, line)) {
		// Удаление пробелов и символов ; [ ]
		line.erase(remove_if(line.begin(), line.end(), [](char c) { return isspace(c) || c == '[' || c == ']' || c == ';'; }), line.end());
		
		// Разделение строки на координаты
		if (!line.empty()) {
			stringstream pointStream(line);
			string coord;
			vector<int> coords;

			while (getline(pointStream, coord, ',')) {
				coords.push_back(stoi(coord));
				if (coords.size() == 2) {
					points.push_back(Point(coords[0], coords[1]));
					cou++;
					if ((val == 4) && (cou == val)) {
						layoutContours.push_back(points);
						points.clear();
						cou = 0;
					}
				}
			}
		}
	}
	if (val == 1) {
		layoutContours.push_back(points);
		points.clear();
	}

	//for (auto i : layoutContours)
	//	cout << i << endl;

	return layoutContours;
}

void evaluateQuality(const Mat& src, string latoutPath) {

	/*Mat hsv;
	cvtColor(layout, hsv, COLOR_BGR2HSV);

	Scalar lowerRed1(0, 100, 100);
	Scalar upperRed1(10, 255, 255);
	Scalar lowerRed2(160, 100, 100);
	Scalar upperRed2(180, 255, 255);

	Mat redMask1, redMask2, redMask;
	inRange(hsv, lowerRed1, upperRed1, redMask1);
	inRange(hsv, lowerRed2, upperRed2, redMask2);


	redMask = redMask1 | redMask2;

	vector<vector<Point>> layoutContours2;
	findContours(redMask, layoutContours2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);*/
	//for (auto i : layoutContours2)
	//	cout << i << endl;


	long long val = 4;
	if (srcContours.size() == 1)
		val = 1;


	vector<vector<Point>> layoutContours = parseFile(latoutPath, val);

	double TP = 0, FP = 0, FN = 0;

	vector<bool> srcMatched(srcContours.size(), false);
	vector<bool> layoutMatched(layoutContours.size(), false);

	for (size_t i = 0; i < layoutContours.size(); ++i) {
		bool matchFound = false;
		Rect layoutRect = boundingRect(layoutContours[i]);

		for (size_t j = 0; j < srcContours.size(); ++j) {
			Rect srcRect = boundingRect(srcContours[j]);
			if ((layoutRect & srcRect).area() > 0.5 * layoutRect.area()) {
				matchFound = true;
				srcMatched[j] = true;
				layoutMatched[i] = true;
				break;
			}
		}

		if (matchFound) {
			TP++;
		}
		else {
			FN++;
		}
	}

	for (bool matched : srcMatched) {
		if (!matched) {
			FP++;
		}
	}

	//Rect rect1 = boundingRect(layoutContours[0]);
	//Rect rect2 = boundingRect(srcContours[0]);
	//Rect intersection = rect1 & rect2;
	//double intersectionArea = intersection.area();
	//double unionArea = rect1.area() + rect2.area() - intersectionArea;
	//cout << (unionArea > 0) ? (intersectionArea / unionArea) : 0;

	cout << "True Positives (TP): " << TP << endl;
	cout << "False Positives (FP): " << FP << endl;
	cout << "False Negatives (FN): " << FN << endl;

	std::cout << "Average IoU: " << (TP) / (TP + FN + FP) << std::endl;
}

void calculateQualityMetrics(const vector<string>& src, const vector<string>& layout) {
	double TP = 0, FP = 0, FN = 0;

	vector<string> sortedSrc = src;
	vector<string> sortedLayout = layout;
	sort(sortedSrc.begin(), sortedSrc.end());
	sort(sortedLayout.begin(), sortedLayout.end());

	auto srcIt = sortedSrc.begin();
	auto layoutIt = sortedLayout.begin();

	while (srcIt != sortedSrc.end() && layoutIt != sortedLayout.end()) {
		if (*srcIt == *layoutIt) {
			TP++;
			srcIt++;
			layoutIt++;
		}
		else if (*srcIt < *layoutIt) {
			FP++;
			srcIt++;
		}
		else {
			FN++;
			layoutIt++;
		}
	}

	FP += distance(srcIt, sortedSrc.end());
	FN += distance(layoutIt, sortedLayout.end());

	cout << "True Positives (TP): " << TP << endl;
	cout << "False Positives (FP): " << FP << endl;
	cout << "False Negatives (FN): " << FN << endl;
	cout << "Quality assessment: " << (TP) / (sortedLayout.size()) << endl;
}


int main(int argc, char* argv[]) {
	//string base = "C:/Users/Svt/Desktop/khlobustova/prj.cw";
	string base = ".";
	string test, src, layout, src_data, layout_data;
	test = "/input/test1.png";
	src = "/output/src1.png";
	layout = "/input/input_layout1.txt";
	src_data = "/output/src_data1.txt";
	layout_data = "/input/layout_data1.txt";

	if (argc > 1) {
		test = argv[1];
	}
	if (argc > 2) {
		src = argv[2];
	}
	if (argc > 3) {
		layout = argv[3];
	}
	if (argc > 4) {
		src_data = argv[4];
	}
	if (argc > 5) {
		layout_data = argv[5];
	}

	Mat source = imread(base + test);

	if (source.empty())
	{
		std::cout << "No image data!" << endl;
		std::system("pause");
		return 0;
	}

	string outputAns = base + src;

	vector<Mat>ROI_Rect;
	vector<decodedObject> decodedObjects;
	Find_QR_Rect(source, ROI_Rect, decodedObjects);
	QRCodeDetector qrDet = QRCodeDetector::QRCodeDetector();
	Mat bbox, rectifiedImage;
	string data;
	bool showing = false;

	Mat canvas = Mat::zeros(source.size(), source.type());

	if (ROI_Rect.size() != 0) {
		int flag = Dectect_QR_Rect(source, canvas, ROI_Rect);

		if (flag <= 0)
		{
			std::cout << "Can not detect QR code!" << endl;
			std::system("pause");
			return 0;
		}

		for (auto i : ROI_Rect) {
			Mat bbox, rectifiedImage;
			data = qrDet.detectAndDecode(i, bbox, rectifiedImage);
			if (data.length() != 0) {
				data_links.push_back(data);
				showing = true;
			}
		}

		Mat gray;
		cvtColor(canvas, gray, COLOR_BGR2GRAY);

		vector<vector<Point>>contours;
		findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		srcContours = contours;

		Point2f points[4];

		for (int i = 0; i < contours.size(); i++)
		{
			RotatedRect rect = minAreaRect(contours[i]);

			rect.points(points);

			for (int j = 0; j < 4; j++)
			{
				line(source, points[j], points[(j + 1) % 4], Scalar(255, 0, 0), 5);
			}
		}
	}

	std::ofstream out;
	out.open(base + src_data);

	if (out.is_open())
	{
		for (string links : data_links)
			out << links << std::endl;
	}
	out.close();

	imshow("source", source);
	cv::imwrite(outputAns, source);

	evaluateQuality(source, base + layout);

	string srcFile = base + src_data;
	string layoutFile = base + layout_data;

	vector<string> srcData = readFile(srcFile);
	vector<string> layoutData = readFile(layoutFile);

	calculateQualityMetrics(srcData, layoutData);

	//for (auto i : srcContours)
	//	cout << i << endl;

	cv::waitKey(0);
	cv::destroyAllWindows();
	std::system("pause");

	return 0;
}

