#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <time.h>
#include <chrono>

using namespace cv;
using namespace std;
typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

int main(int argc, const char** argv)
{
	//start default camera
	VideoCapture cap(0);
	cap.set(CAP_PROP_FPS, 30);
	double format = cap.get(CAP_PROP_FORMAT);
	double mode = cap.get(CAP_PROP_MODE);
	cout << "Video format is: " << format << "and capture mode is: " << mode << "\n" << endl;
	cap.set(CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
	//check if camera has started
	if (!cap.isOpened())
	{
		cout << "no camera" << endl;
		return -1;
	}

	//code to check for frame rate 
	double fps = cap.get(CAP_PROP_FPS);
	cout << "Frames per second using CV_CAP_PROP_FPS: " << fps << endl;

	//define number of frames to capture
	int numFrame = 0;
	Mat frame;
	
	Clock::time_point start = Clock::now();
	for (;;)
	{
		cap.read(frame);
		
		if (frame.empty())
		{
			cout << "frame is empty lmao" << endl;
		}
		else
		{
			imshow("frame", frame);
			numFrame++;
		}
		if (waitKey(5) == 27) break;
	}
	Clock::time_point end = Clock::now();
	// Time elapsed
	milliseconds ms = chrono::duration_cast<milliseconds>(end -start);
	cout << "Time taken : " << ms.count() << "ms\n" << endl;
	// Calculate frames per second
	fps = numFrame / (ms.count()/1000);
	cout << "Estimated frames per second : " << fps << endl;


	// Release video
	cap.release();
	return 0;

}
