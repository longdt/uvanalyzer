/*
 * com_eprotea_icrengine_UVChecker.cpp
 *
 *  Created on: Apr 9, 2015
 *      Author: thienlong
 */

#include "com_eprotea_icrengine_UVChecker.h"

#include <opencv2/highgui/highgui.hpp>
#include "UVAnalyzer.h"

#include "opencv2/core/core.hpp"
#include <exception>
using cv::imdecode;
using cv::Mat;
using icr::UVAnalyzer;
using std::bad_exception;

JNIEXPORT jboolean JNICALL Java_com_eprotea_icrengine_UVChecker_checkValid(
		JNIEnv *env, jclass classObj, jbyteArray uvImg) {
	jbyte* imgData = env->GetByteArrayElements(uvImg, NULL);
	int length = env->GetArrayLength(uvImg);
	Mat img = imdecode(Mat(1, length, CV_8UC1, imgData), 0);
	env->ReleaseByteArrayElements(uvImg, imgData, 0);
	if (img.empty()) {
		jclass Exception = env->FindClass("com/eprotea/icrengine/ICRException");
		env->ThrowNew(Exception,"invalid image data format");
		return JNI_FALSE;
	}
	UVAnalyzer analyzer;
	try {
		return analyzer.checkValid(img) > 0.5 ? JNI_TRUE : JNI_FALSE;
	} catch (const bad_exception &e) {
		return JNI_FALSE;
	}
}
