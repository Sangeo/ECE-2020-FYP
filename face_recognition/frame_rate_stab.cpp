#include "opencv_helper.h"
#include <Windows.h>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

int main(int argc, const char** argv) {

	//introduce the module
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nPress Esc to quit program\nThis program demonstrates signal decomposition by color in openCV with facial recognition in a video stream");
	parser.printMessage();

	//-- 1. Load the cascades
	String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
	if (!face_cascade.load(face_cascade_name)) {
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	//Get video input/output setup
	cv::VideoCapture capture(1); // 1 for front facing webcam
	//capture.set(CAP_PROP_FPS, 30);
	//capture.set(CAP_PROP_FRAME_WIDTH, 1280);
	//capture.set(CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(CAP_PROP_AUTOFOCUS, 1);
	capture.set(CAP_PROP_AUTO_WB, 0);

	cout << "CAPTURE FOMRAT IS: " << capture.get(CAP_PROP_FORMAT) << endl;
	if (!capture.isOpened()) {
		return -1;
	}
	capture.set(CAP_PROP_FOURCC, ('D', 'I', 'V', 'X'));

	Mat frame;
	Mat newFrame;
	deque<Mat> frameQ;
	const int numFrames = 300; //10 seconds worth of frames
	const int FPS = capture.get(CAP_PROP_FPS);
	const int msPerFrame = 33;
	cout << "Frames per second according to capture.get(): " << FPS << endl;

	
	for (size_t i = 0; i < numFrames; i++) {

		Clock::time_point start = Clock::now();
		if (capture.read(frame)) {
			if (!frame.empty()) {
				frameQ.push_back(frame.clone());
			}
		}
		else {
			break;
		}
		Clock::time_point end = Clock::now();
		auto ms = chrono::duration_cast<milliseconds>(end - start);
		int captureTime = ms.count();
		
		cout << "time taken to read frame " << i << "is: " << captureTime << endl;
	}

	for (size_t i = 0; i < numFrames; i++) {
		Mat t = frameQ.front();
		imshow("frame", t);
		if (waitKey(33) > 0) {
			break;
		}
		frameQ.pop_front();
	}

	frameQ.clear();
	capture.release();
	return 0;
}
