// C++ program to detects face in a video using color
// Goal: To extract color signals from the face recognition file
// Altered:
// - Does not mask the eyes ( we don't need this)
// - Uses a rectangle as a display rather than an elipse
// - Get a signal decomposition for the image in terms of blue/green/red
// - Obtained signal and outputs into excel where required.

// To-do list:
// - Amplify correct color spectrums
// - Apply filter to remove noise from color image
// - Plot color signals and extract peak-to-peak time to find heart rate.
// reference: https://docs.opencv.org/4.3.0/db/d28/tutorial_cascade_classifier.html


#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/matx.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
//#include "\Users\jerry\vcpkg\installed\x86-windows\include\matplotlibcpp.h"

using namespace std;
using namespace cv;

void writeCSV(string filename, Mat m); //legacy do not use
void write_CSV(string filename, vector<double> arr, double fps);
void detectAndDisplay(Mat frame, int cSel);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
vector<double> OUTPUT_CSV_VAR;

int main(int argc, const char** argv)
{
	//Introducing the module
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nPress Esc to quit program\nThis program demonstrates signal decomposition by color in openCV with facial recognition in a video stream");
	parser.printMessage();

	//prompt to select color
	cout << "Please select the color for signal decomposition\n";
	cout << "1 for blue, 2 for green, 3 for red : \n";
	int color_sel;
	cin >> color_sel;


	//-- 1. Load the cascades
	String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
	String eyes_cascade_name = samples::findFile(parser.get<String>("eyes_cascade"));
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "--(!)Error loading eyes cascade\n";
		return -1;
	};

	//-- 2. Read the video stream
	VideoCapture capture(0);
	capture.set(CAP_PROP_FPS, 30);
	//capture.set(CAP_PROP_FRAME_WIDTH, 1920/2);
	//capture.set(CAP_PROP_FRAME_HEIGHT, 1080/2);
	// check that the camera is open
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}
	// this frame will store all information about the video captured by the camera
	Mat frame;

	// to find the time taken to do all calculations/capture frames
	typedef chrono::high_resolution_clock Clock;
	typedef chrono::milliseconds milliseconds;
	Clock::time_point start = Clock::now();
	for (;;)
	{
		if (capture.read(frame))
		{
			detectAndDisplay(frame, color_sel);
		}
		if (waitKey(1) == 27)
		{
			break; // if escape is pressed at any time
		}
	}
	Clock::time_point end = Clock::now();
	milliseconds ms = chrono::duration_cast<milliseconds>(end - start);
	cout << "ran for duration: " << ms.count() << "ms\n" << endl;
	long long duration = ms.count();
	long long n = OUTPUT_CSV_VAR.size();
	cout << n << "\n" << endl;
	long long fps = n / (duration/1000); // gives frames per second
	cout << "Frames per second: " << fps << "\n" << endl;
	write_CSV("output_file2.csv", OUTPUT_CSV_VAR, fps);

	capture.release();
	return 0;
}

void detectAndDisplay(Mat frame, int cSel)
{
	// Declaration of variables
	//-- converts video feed into grayscale, and then detecks edges.
	Mat frame_gray;
	Mat frame_clone = frame.clone();
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);

	//-- This is used for color separation for later
	Mat zeroMatrix = Mat::zeros(Size(frame_clone.cols, frame_clone.rows), CV_8UC1);
	Mat faceROI;

	// This will split the current frame (stored in frame_clone) into three different color 
	// channels each time we get a new frame
	Mat splt[3];
	cv::split(frame_clone, splt);
	Mat colorImg; 
	// colorImg will be the output which stores the color separated channels
	// split the color channels into blue, green and red according to BGR format of 
	vector<Mat> B;
	B.push_back(splt[0]);
	B.push_back(zeroMatrix);
	B.push_back(zeroMatrix);
	vector<Mat> G;
	G.push_back(zeroMatrix);
	G.push_back(splt[1]);
	G.push_back(zeroMatrix);
	vector<Mat> R;
	R.push_back(zeroMatrix);
	R.push_back(zeroMatrix);
	R.push_back(splt[2]);
	//switch select for color selection
	switch (cSel)
	{
	case 1: {
		//showing blue spectrum
		cv::merge(B, colorImg);
		//imshow("Blue", colorImg);
		break;
	}
	case 2: {
		//showing red spectrum
		cv::merge(G, colorImg);
		//imshow("Green", colorImg);
		break;
	}
	case 3: {
		//showing red spectrum
		cv::merge(R, colorImg);
		//imshow("Red", colorImg);
		break;
	}
	default: break;
	}
	// faceROI Detection
	for (size_t i = 0; i < faces.size(); i++)
	{
		// displays a rectangle around the face 
		Point topLC(faces[i].x, faces[i].y + faces[i].height);
		Point botRC(faces[i].x + faces[i].width, faces[i].y);
		rectangle(frame, topLC, botRC, Scalar(0, 0, 255), 1, LINE_4, 0);
		// locate region of interest (ROI) for the forehead
		Point innerTopLC(faces[i].x + (faces[i].width * 3 / 10), faces[i].y + faces[i].height / 20);
		Point innerBotRC(faces[i].x + (faces[i].width * 7 / 10), faces[i].y + (faces[i].height * 1 / 5));
		Rect myROI(innerTopLC, innerBotRC);

		faceROI = colorImg(myROI).clone();
		if (!faceROI.empty()) {
			vector<Mat> temp;
			split(faceROI, temp);//resplits the channels (extracting the color green for default/testing cases)
			Scalar averageColor = mean(temp[cSel - 1]);
			double s = sum(averageColor)[0];
			OUTPUT_CSV_VAR.push_back(s);
		}
	}
	imshow("Capture - Face detection", frame);
}

void writeCSV(string filename, Mat m) //legacy do not use
{
	ofstream myfile;
	myfile.open(filename.c_str());
	myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
}

void write_CSV(string filename, vector<double> arr, double fps)
{
	ofstream myfile;
	myfile.open(filename.c_str());
	int vsize = arr.size();
	for (int n = 0; n < vsize; n++)
	{
		myfile << fps << ";" << n << ";" << arr[n] << endl;
	}
}