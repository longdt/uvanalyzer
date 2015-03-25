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
#define EPICK_HEIGHT 65
#define EPICK_WIDTH 200
#define EPICK_ON_BOTTOM 500

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

#define MID_Y 272
#define MID_HEIGHT 152
double thresholdMid(cv::Mat& src, cv::Mat& output) {
	Rect roi(0, MID_Y, src.cols, MID_HEIGHT);
	Mat roiMid = src(roi);
	Mat outputMid = output(roi);
	double thres = threshold(roiMid, outputMid, 0, 255, THRESH_OTSU | THRESH_BINARY);
	std::vector<int> projects;
	projectVertical(outputMid, projects);
	int thresX = roiMid.rows / 7;
	int startX = 0;
	int endX = roiMid.cols - 1;
	for (; startX < roiMid.cols && projects[startX] < thresX; ++startX) {}
	for (; endX >= 0 && projects[endX] < thresX; --endX) {}
	if (startX > 20 || roiMid.cols - endX > 25) {
		return thres;
	}
	std::cout << "Problem" << std::endl;
	average<int> avg;
	for (int r = 0; r < roiMid.rows; ++r) {
		for (int c = 0; c <= 20; ++c) {
			if (outputMid.at<uchar>(r,c) > 0) {
				avg.update(roiMid.at<uchar>(r,c));
			}
		}
	}
	for (int r = 0; r < roiMid.rows; ++r) {
		for (int c = roiMid.cols - 25; c < roiMid.cols; ++c) {
			if (outputMid.at<uchar>(r, c) > 0) {
				avg.update(roiMid.at<uchar>(r, c));
			}
		}
	}
	thres = avg.sdeviation() + avg.mean() + avg.deviation();
	threshold(roiMid, outputMid, thres, 255, THRESH_BINARY);
	return thres;
}

double thresholdTop(cv::Mat& src, cv::Mat& output) {
	Rect roi(0, 0, src.cols, MID_Y);
	Mat roiTop = src(roi);
	Mat outputTop = output(roi);
	double thres = threshold(roiTop, outputTop, 0, 255, THRESH_OTSU | THRESH_BINARY);
	std::vector<int> projects;
	projectVertical(outputTop, projects);
	int thresX = roiTop.rows / 7;
	int startX = 0;
	int endX = roiTop.cols - startX;
	for (; startX < roiTop.cols && projects[startX] < thresX; ++startX) {}
	for (; endX >= 0 && projects[endX] < thresX; --endX) {}
	if (startX > 20 || roiTop.cols - endX > 25) {
		return thres;
	}
	std::cout << "Problem" << std::endl;
	average<int> avg;
	for (int r = 0; r < roiTop.rows; ++r) {
		for (int c = 0; c <= 20; ++c) {
			if (outputTop.at<uchar>(r,c) > 0) {
				avg.update(roiTop.at<uchar>(r,c));
			}
		}
	}
	for (int r = 0; r < roiTop.rows; ++r) {
		for (int c = roiTop.cols - 25; c < roiTop.cols; ++c) {
			if (outputTop.at<uchar>(r, c) > 0) {
				avg.update(roiTop.at<uchar>(r, c));
			}
		}
	}
	thres = avg.sdeviation() + avg.mean() + avg.deviation();
	threshold(roiTop, outputTop, thres, 255, THRESH_BINARY);
	return thres;
}

void localThreshold(cv::Mat& src, cv::Mat& output) {
	double thres = thresholdMid(src, output);
	Rect roi(0, MID_Y + MID_HEIGHT, src.cols, src.rows - MID_Y - MID_HEIGHT);
	Mat roiBot = src(roi);
	Mat outBot = output(roi);
	threshold(roiBot, outBot, thres, 255, THRESH_BINARY);
	thresholdTop(src, output);
}



void detectAreas(cv::Mat& src, vector<Rect>& rects) {
	std::vector<int> projects;
	//vertical
	projectVertical(src, projects);
	int thresX = src.rows / 7;
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

	//debug draw
	line( src, Point(startX, 0), Point(startX, src.rows), Scalar(255,255, 255), 1, CV_AA);
	line( src, Point(endX, 0), Point(endX, src.rows), Scalar(255,255, 255), 1, CV_AA);
	line( src, Point(0, startY), Point(src.cols, startY), Scalar(255,255, 255), 1, CV_AA);
	line( src, Point(0, endY), Point(src.cols, endY), Scalar(255,255, 255), 1, CV_AA);

	Rect roi(startX, startY, endX - startX, endY - startY);
	Mat temp = src(roi);
	if (endY - startY < src.rows * 0.28 || endX - startX < src.cols * 0.8) {
		throw std::bad_exception();
	} else if (endY - startY < 4 * EPICK_HEIGHT) {
		//form 2 epick
		std::cout << "form 2" << std::endl;
	}
	if (endY > EPICK_ON_BOTTOM) {
		//option epick on bottom
		std::cout <<"epick on bottom" << std::endl;
	}
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
	imshow("bi", bi);
	detectAreas(bi, rects);
	return 1;
}



} /* namespace icr */
