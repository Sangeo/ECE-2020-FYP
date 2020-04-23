#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
	//start default camera
	VideoCapture cap(0);
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
	time_t start, end; //define time values 
	
	time(&start);
	for (;;)
	{
		Mat frame;
		cap.read(frame);
		double timestamp = cap.get(CAP_PROP_POS_MSEC);
		if (frame.empty())
		{
			cout << "frame is empty lmao" << endl;
		}
		else
		{
			imshow("frame", frame);
			cout << "timestamp: " << timestamp << endl;
			numFrame++;
		}
		if (waitKey(5) == 27) break;
	}
	time(&end);
	// Time elapsed
	double seconds = difftime(end, start);
	cout << "Time taken : " << seconds << " seconds" << endl;
	// Calculate frames per second
	fps = numFrame / seconds;
	cout << "Estimated frames per second : " << fps << endl;


	// Release video
	cap.release();
	return 0;

}
