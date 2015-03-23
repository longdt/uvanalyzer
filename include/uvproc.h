/*
 * uvproc.h
 *
 *  Created on: Mar 17, 2015
 *      Author: thienlong
 */

#ifndef INCLUDE_UVPROC_H_
#define INCLUDE_UVPROC_H_
#include "opencv2/core/core.hpp"
#include <vector>

void adjustAutoLevels(cv::Mat& src, cv::Mat& dst);
void drawHist(cv::Mat& src);
void projectVertical(cv::Mat &input, std::vector<int> &output);
void projectHorizontal(cv::Mat &input, std::vector<int> &output);

#endif /* INCLUDE_UVPROC_H_ */
