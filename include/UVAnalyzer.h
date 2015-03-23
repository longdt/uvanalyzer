/*
 * UVAnalyzer.h
 *
 *  Created on: Mar 17, 2015
 *      Author: thienlong
 */

#ifndef UVANALYZER_H_
#define UVANALYZER_H_
#include <opencv2/core/core.hpp>
namespace icr {

class UVAnalyzer {
public:
	UVAnalyzer();
	float checkValid(cv::Mat& uvImg);

	virtual ~UVAnalyzer();
};

} /* namespace icr */

#endif /* UVANALYZER_H_ */
