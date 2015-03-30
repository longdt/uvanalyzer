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

Rect findBoundingRect(cv::Mat& src, float thresholdX, float thresholdY) {
	std::vector<int> projects;
	//vertical
	projectVertical(src, projects);
	int startX = 0;
	for (; startX < src.cols && projects[startX] < thresholdX; ++startX) {}
	if (startX == src.cols) {
		return Rect();
	}
	int endX = src.cols - 1;
	for (; endX >= 0 && projects[endX] < thresholdX; --endX) {}
	if (endX == -1) {
		return Rect();
	}
	//horizontal
	projectHorizontal(src, projects);
	int startY = 0;
	for (; startY < src.rows && projects[startY] < thresholdY; ++startY) {}
	if (startY == src.rows) {
		return Rect();
	}
	int endY = src.rows - 1;
	for (; endY >= 0 && projects[endY] < thresholdY; --endY) {}
	if (endY == -1) {
		return Rect();
	}
	return Rect(startX, startY, endX - startX + 1, endY - startY + 1);
}

#define FORM1 1
#define FORM2 2
#define FORM3 3
int detectForm(cv::Mat& src, cv::Rect& area, vector<Rect>& rects) {
	float form2Score = area.width > 6 * EPICK_WIDTH ? 0.5 : 0;
	Rect botRoi(area.x + area.width - 2 * EPICK_WIDTH, area.y + area.height - EPICK_HEIGHT / 2, 2 * EPICK_WIDTH, EPICK_HEIGHT / 2);
	Mat botRight = src(botRoi);
	float botScore = countNonZero(botRight) / (float) (botRight.rows * botRight.cols);
	if (botScore < 0.3 && area.height < 3.5 * EPICK_HEIGHT) {
		std::cout << "form 2" << std::endl;
		return FORM2;
	}
	Rect topRoi(area.x, area.y, 4 * EPICK_WIDTH, EPICK_HEIGHT / 2);
	Mat topLeft = src(topRoi);
	float topScore = countNonZero(topLeft) / (float) (topLeft.rows * topLeft.cols);

	Rect midRoi(area.x + area.width - 2 * EPICK_WIDTH, area.y + 1.25 * EPICK_HEIGHT, 2 * EPICK_WIDTH, EPICK_HEIGHT / 2);
	Mat midRight = src(midRoi);
	float midScore = countNonZero(midRight) / (float) (midRight.rows * midRight.cols);
	form2Score = (form2Score + (3 - topScore - botScore - midScore) / 3) / 2;
	if (form2Score > 0.7) {
		std::cout << "form 3" << std::endl;
		return FORM3;
	}
	return FORM1;
}

void findForm2Areas(cv::Mat& src, cv::Rect& area, vector<Rect>& rects) {
	Rect roi = area;
	roi.width = roi.width -  2.2 * EPICK_WIDTH;
	Mat temp = src(roi);
	Rect r = findBoundingRect(temp, src.rows / 7.0f, src.cols / 4.0f);
	if (r.width == 0 || r.height == 0) {
		throw std::bad_exception();
	}
	r.x += roi.x;
	r.y += roi.y;
	rects.push_back(r);

	//top right
	roi.height = 1.5 * EPICK_HEIGHT;
	roi.y = roi.y - 1.25 * EPICK_HEIGHT;
	roi.x += roi.width;
	roi.width = 2.2 * EPICK_WIDTH;
//	rectangle(src, roi, Scalar(255,255, 255)); //debug
	temp = src(roi);
	r = findBoundingRect(temp, 0.5 * EPICK_HEIGHT, EPICK_WIDTH);
	if (r.width == 0 || r.height == 0) {
		throw std::bad_exception();
	}
	r.x += roi.x;
	r.y += roi.y;
	rects.push_back(r);

	roi.y = r.y + 2.3 * EPICK_HEIGHT;
//	rectangle(src, roi, Scalar(255,255, 255)); //debug
	temp = src(roi);
	r = findBoundingRect(temp, 0.5 * EPICK_HEIGHT, EPICK_WIDTH);
	if (r.width == 0 || r.height == 0) {
		throw std::bad_exception();
	}
	r.x += roi.x;
	r.y += roi.y;
	rects.push_back(r);
}

void findForm3Areas(cv::Mat& src, cv::Rect& area, vector<Rect>& rects) {
	Rect roi = area;
	roi.width = roi.width -  2.2 * EPICK_WIDTH;
	Mat temp = src(roi);
	Rect r = findBoundingRect(temp, src.rows / 7.0f, src.cols / 4.0f);
	if (r.width == 0 || r.height == 0) {
		throw std::bad_exception();
	}
	r.x += roi.x;
	r.y += roi.y;
	rects.push_back(r);

	//top right
	roi.height = 1.5 * EPICK_HEIGHT;
	roi.x += roi.width;
	roi.width = 2.2 * EPICK_WIDTH;
	temp = src(roi);
	r = findBoundingRect(temp, 0.5 * EPICK_HEIGHT, EPICK_WIDTH);
	if (r.width == 0 || r.height == 0) {
		throw std::bad_exception();
	}
	r.x += roi.x;
	r.y += roi.y;
	rects.push_back(r);

	roi.y += 2.3 * EPICK_HEIGHT;
//	rectangle(src, roi, Scalar(255,255, 255)); //debug
	temp = src(roi);
	r = findBoundingRect(temp, 0.5 * EPICK_HEIGHT, EPICK_WIDTH);
	if (r.width == 0 || r.height == 0) {
		throw std::bad_exception();
	}
	r.x += roi.x;
	r.y += roi.y;
	rects.push_back(r);
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

	if (endY - startY < src.rows * 0.28 || endX - startX < src.cols * 0.8) {
		throw std::bad_exception();
	} else if (endY > EPICK_ON_BOTTOM) {
		//option epick on bottom
		std::cout <<"epick on bottom" << std::endl;
		int startYBE = endY - 2 * EPICK_HEIGHT;
		for (; startYBE < src.rows && projects[startYBE] < thresY; ++startYBE) {}
		rects.push_back(Rect(startX, startYBE, endX- startX, endY - startYBE));
		for (endY = endY - 2 * EPICK_HEIGHT; endY >= 0&& projects[endY] < thresY; --endY) {}
		rects.push_back(Rect(startX, startY, endX- startX, endY - startY));
		return;
	}
	//detect form 2
	Rect area(startX, startY, endX- startX, endY - startY);
	int type = detectForm(src, area, rects);
	switch (type) {
		case FORM1:
			rects.push_back(area);
			break;
		case FORM2:
			findForm2Areas(src, area, rects);
			break;
		case FORM3:
			findForm3Areas(src, area, rects);
			break;
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
	//debug print rects
	for (Rect rect : rects) {
		rectangle(uvImg, rect, Scalar(255,255, 255));
	}
	imshow("src", uvImg);
	imshow("bi", bi);
	return 1;
}



} /* namespace icr */
