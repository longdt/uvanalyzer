/*
 * UVAnalyzer.cpp
 *
 *  Created on: Mar 17, 2015
 *      Author: thienlong
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "UVAnalyzer.h"

#include "uvproc.h"
#include <iostream>
using namespace cv;

namespace icr {

UVAnalyzer::UVAnalyzer() {

}

UVAnalyzer::~UVAnalyzer() {
	// TODO Auto-generated destructor stub
}

bool needLocalThreshold(cv::Mat& binary) {
	std::vector<int> projects;
	projectVertical(binary, projects);
	int startX = 0;
	for (; startX < binary.cols && projects[startX] < 10; ++startX) {}
	int endX = binary.cols - 1;
	for (; endX >= 0 && projects[endX] < 10; --endX) {}
	return (startX < 20);
}

void localThreshold(cv::Mat& src, cv::Mat& output) {
	Rect roi(0, 272, src.cols, 152);
	Mat roiMid = src(roi);
	Mat mOutput = output(roi);
	double thres = threshold(roiMid, mOutput, 0, 255, THRESH_OTSU | THRESH_BINARY);
	roi.y += roi.height;
	roi.height = src.rows - roi.y;
	Mat roiBot = src(roi);
	Mat outBot = output(roi);
	threshold(roiBot, outBot, thres, 255, THRESH_BINARY);
	roi.y = 0;
	roi.height = 272;
	Mat roiTop = src(roi);
	Mat outTop = output(roi);
	threshold(roiTop, outTop, 0, 255, THRESH_OTSU | THRESH_BINARY);
}

bool detectAreas(cv::Mat& src, vector<Rect>& rects) {
	std::vector<int> projects;
	projectVertical(src, projects);
	int thresX = src.rows / 6;
	int startX = 10;
	for (; startX < src.cols && projects[startX] < thresX; ++startX) {}
	int endX = src.cols - 10;
	for (; endX >= 0 && projects[endX] < thresX; --endX) {}
	//horizontal
	projectHorizontal(src, projects);
	int thresY = src.cols / 4;
	int startY = 10;
	for (; startY < src.rows && projects[startY] < thresY; ++startY) {}
	int endY = src.rows - 10;
	for (; endY >= 0&& projects[endY] < thresY; --endY) {}
	line( src, Point(startX, 0), Point(startX, src.rows), Scalar(255,255, 255), 1, CV_AA);
	line( src, Point(endX, 0), Point(endX, src.rows), Scalar(255,255, 255), 1, CV_AA);
	line( src, Point(0, startY), Point(src.cols, startY), Scalar(255,255, 255), 1, CV_AA);
	line( src, Point(0, endY), Point(src.cols, endY), Scalar(255,255, 255), 1, CV_AA);
	if (endY - startY < src.rows * 0.28 || endX - startX < src.cols * 0.8) {
		return false;
	}
	return true;
}

float UVAnalyzer::checkValid(cv::Mat& uvImg) {
	Mat dst;
	adjustAutoLevels(uvImg, dst);
	imshow("levels", dst);
	drawHist(dst);
	Mat bi;
	threshold(dst, bi, 0, 255, THRESH_OTSU | THRESH_BINARY);
	if (needLocalThreshold(bi)) {
		std::cout << "perform local threshold" << std::endl;
		localThreshold(uvImg, bi);
	}
	vector<Rect> rects;
	if (!detectAreas(bi, rects)) {
		return 0;
	}
	imshow("bi", bi);
	return 1;
}



} /* namespace icr */
