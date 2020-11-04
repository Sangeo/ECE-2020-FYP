/**
	Multi-processing attempt
	multi_processing.cpp
	Purpose: Getting color signals from the face using multi-processing and haar-cascade classifiers

	@author: Jerry Li
	@version: v1 27/4/2020
*/
// haar cascade reference: https://docs.opencv.org/4.3.0/db/d28/tutorial_cascade_classifier.html

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/matx.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <queue>
#include <future>

using namespace cv;
using namespace std;

class realTimeVideo {
public:
	Mat frame;
	vector<Mat> frameBuffer;

	int buffLen, sampleLen;
	int pos, newpos;
	int idx;
};


void getFrames(VideoCapture* pcapture,realTimeVideo vid) {
	Mat f;
	cout << "entered getFrames" << endl;
	while (1) {
		pcapture->read(f);
		
		if (!f.empty()) {
			vid.frame = f.clone();
			vid.frameBuffer.push_back(f);
		}
		cout << "current queue size = " << vid.frameBuffer.size() << endl;
	}
}

void showFrames(realTimeVideo vid) {
	cout << "entered showFrames" << endl;
	waitKey(50);
	while (1) {
		if (!vid.frame.empty()) {
			imshow("frame", vid.frame);
		}
		if (waitKey(5) == 27) break;
	}
}

void processFrames(realTimeVideo vid) {
	cout << "entered procFrames" << endl;
	Mat proc;
	while (1) {

		//do some processing here
		//GaussianBlur(proc, proc, proc.size(),0);
		//imshow("proc frame", proc);
		//waitKey(5);
		if (vid.frameBuffer.size() >= 10) {
			vid.frameBuffer.pop_back();
		}
		else {
			break;
		}
	}
}

int main() {
	// start threads here 
	VideoCapture cap(0);
	cap.set(CAP_PROP_FPS, 30);
	cap.set(CAP_PROP_FRAME_WIDTH, 1920 / 3);
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080 / 3);
	if (!cap.isOpened()) return 0;
	realTimeVideo v1;

	cout << "entered main" << endl;
	thread t1(getFrames, &cap, v1);
	thread t2(showFrames, v1);
	thread t3(processFrames, v1);

	t1.join();
	t2.join();
	t3.join();

	return 0;
}