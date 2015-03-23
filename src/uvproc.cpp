/*
 * uvproc.cpp
 *
 *  Created on: Mar 17, 2015
 *      Author: thienlong
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "uvproc.h"
#include <iostream>

using namespace cv;

#include "opencv2/core/core.hpp"
void adjustAutoLevels(cv::Mat& src, cv::Mat& dst) {
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	Mat hist;
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform,
			accumulate);
	vector<float> accHist(histSize, 0);
	accHist[0] = hist.at<float>(0);
	for (int i = 1; i < histSize; i++) {
		accHist[i] = accHist[i - 1] + hist.at<float>(i);
	}
	//find low, high pixel value
	int totalPixels = src.rows * src.cols;
	int lowP = 0;
	float level = 0.01 * totalPixels;
	for (; lowP < histSize && accHist[lowP] < level; ++lowP) {
	}
	int highP = 255;
	for (; highP > 0 && (totalPixels - accHist[highP]) < level; --highP) {
	}
	float alpha = 255.0f / (highP - lowP);
	float beta = -alpha * lowP;
	dst = Mat::zeros(src.size(), src.type());
	int value = 0;
	for (int r = 0; r < src.rows; ++r) {
		for (int c = 0; c < src.cols; ++c) {
			uchar p = src.at<uchar>(r, c);
			value = (int) (alpha * src.at<uchar>(r, c) + beta);
			if (value > 255) {
				value = 255;
			} else if (value < 0) {
				value = 0;
			}
			dst.at<uchar>(r, c) = value;
		}
	}
}

void projectVertical(cv::Mat &input, std::vector<int> &output) {
	output.resize(input.cols);
	for (int c = 0; c < input.cols; ++c) {
		int sum = 0;
		for (int r = 0; r < input.rows; ++r) {
			if (input.at<uchar>(r, c) > 0) {
				sum += 1;
			}
		}
		output[c] = sum;
	}
}

void projectHorizontal(cv::Mat &input, std::vector<int> &output) {
	output.resize(input.rows);
	for (int r = 0; r < input.rows; ++r) {
		int sum = 0;
		for (int c = 0; c < input.cols; ++c) {
			if (input.at<uchar>(r, c) > 0) {
				sum += 1;
			}
		}
		output[r] = sum;
	}
}

void drawHist(cv::Mat& src) {
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	Mat hist;
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform,
			accumulate);
	// Draw the histograms for B, G and R
	int hist_w = 512;
	int hist_h = 100;
	int bin_w = cvRound((double) hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	/// Draw for each channel
	for (int i = 1; i < histSize; i++) {
		line(histImage,
				Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
	}
	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);
}
