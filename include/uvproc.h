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

template<class T> class average {
private:
	std::vector<T> values;
	T sum;
public:
	average() : sum(0) {}
	void update(T value) {
		values.push_back(value);
		sum += value;
	}

	double mean() {
		return sum /(double) values.size();
	}

	T max() {
		auto iter = std::max_element(values.begin(), values.end());
		return *iter;
	}

	double deviation() {
		double m = mean();
		double devSum(0);
		for (T v : values) {
			devSum += abs(m - v);
		}
		return devSum / values.size();
	}

	double sdeviation() {
		double m = mean();
		double devSum(0);
		for (T v : values) {
			devSum += (m - v) * (m - v);
		}
		return sqrt(devSum / (double) values.size());
	}

	int size() {
		return values.size();
	}
};

void adjustAutoLevels(cv::Mat& src, cv::Mat& dst);
void drawHist(cv::Mat& src);
void projectVertical(cv::Mat &input, std::vector<int> &output);
void projectHorizontal(cv::Mat &input, std::vector<int> &output);

#endif /* INCLUDE_UVPROC_H_ */
