/*
 * test.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: thienlong
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <vector>

#include "uvproc.h"

#include "UVAnalyzer.h"
using boost::filesystem::directory_iterator;
using boost::filesystem::path;
using icr::UVAnalyzer;
using namespace cv;
using namespace std;

int main(int argc, char **argv) {
	path p("/media/thienlong/linux/UV Cheque/UV/");
	if (!exists(p) || !is_directory(p)) {
		return 0;
	}
	vector<path> files;
	std::copy(directory_iterator(p), directory_iterator(), std::back_inserter(files));
	std::sort(files.begin(), files.end());
	UVAnalyzer analyzer;
	for (auto iter = files.begin(), iterend = files.end(); iter != iterend; ++iter) {
		string file = iter->string();
//		if (file.compare("/media/thienlong/linux/UV Cheque/UV/abc_00005_04.TIF") != 0) continue;
		cout << file << endl;
		Mat src = cv::imread(file, 0);
		if (src.empty()) {
			continue;
		}
		imshow("src", src);
		try {
			cout << (analyzer.checkValid(src) ? "valid" : "invalid") << endl;
		} catch (const exception &e) {
			cout << "invalid" << endl;
		}
		waitKey(0);
	}
}
