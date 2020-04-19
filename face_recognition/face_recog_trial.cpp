// C++ program to detects face in a video using color
// Goal: To extract color signals from the face recognition file
// Altered:
// - Does not mask the eyes ( we don't need this)
// - Uses a rectangle as a display rather than an elipse
// - Get a signal decomposition for the image in terms of blue/green/red

// To-do list:
// - Make sure that the signal obtained from the face is not discontinuous if the face is not visible (using last frame maybe...)
// - Extract color signals
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
//#include "\Users\jerry\vcpkg\installed\x86-windows\include\matplotlibcpp.h"

using namespace std;
using namespace cv;

void writeCSV(string filename, Mat m);
void detectAndDisplay(Mat frame, int cSel);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int argc, const char** argv)
{
	
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nPress Esc to quit program\nThis program demonstrates signal decomposition by color in openCV with facial recognition in a video stream");
	parser.printMessage();

	cout << "Please select the color for signal decomposition\n";
	cout << "1 for blue, 2 for green, 3 for red :____________\n";
	int color_sel;
	cin >> color_sel;

	String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
	String eyes_cascade_name = samples::findFile(parser.get<String>("eyes_cascade"));
	//-- 1. Load the cascades
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
	int camera_device = parser.get<int>("camera");
	VideoCapture capture;
	//-- 2. Read the video stream
	capture.open(camera_device);
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}
	Mat frame;

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame, color_sel);
		if (waitKey(5) == 27)
		{
			break; // escape
		}
	}
	return 0;
}

void detectAndDisplay(Mat frame, int cSel)
{
	ofstream outData;
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
	Mat faceColor;
	Mat faceROI;

	for (size_t i = 0; i < faces.size(); i++)
	{
		// Formatting for the facial recognition component
		Point topLC(faces[i].x, faces[i].y + faces[i].height);
		Point botRC(faces[i].x + faces[i].width, faces[i].y);
		rectangle(frame, topLC, botRC, Scalar(0, 0, 255), 1, LINE_4, 0);
		Point innerTopLC(faces[i].x + (faces[i].width * 3 / 10), faces[i].y + faces[i].height / 20);
		Point innerBotRC(faces[i].x + (faces[i].width * 7 / 10), faces[i].y + (faces[i].height * 1 / 5));
		rectangle(frame, innerTopLC, innerBotRC, Scalar(255, 0, 0), 1, LINE_4, 0);
		Rect myROI(innerTopLC, innerBotRC);
		Rect fixedROI(100, 100, 500, 200);
		// This will split the current frame (stored in frame_clone) into three different color channels
		Mat splt[3];
		cv::split(frame, splt);
		Mat colorImg; //This will be the output which stores the color separated channels
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
		switch (cSel)
		{
		case 1: {
			//showing blue spectrum
			cv::merge(B, colorImg);
			imshow("Blue", colorImg);
			break;
		}
		case 2: {
			//showing red spectrum
			cv::merge(G, colorImg);
			imshow("Green", colorImg);
			break;
		}
		case 3: {
			//showing red spectrum
			cv::merge(R, colorImg);
			imshow("Red", colorImg);
			break;
		}
		default: break;
		}
		Mat cloneImg = colorImg.clone();
		faceROI = cloneImg(myROI).clone();
		
	}
	//live plot of the frames
	//flip(frame, frame, 1);
	imshow("Capture - Face detection", frame);
	if (!faceROI.empty()) {
		imshow("face", faceROI);
		vector<Mat> temp;
		split(faceROI, temp);//resplits the channels (extracting the color green for default/testing cases)
		writeCSV("output_file.csv", temp[1]);
	}

}

void writeCSV(string filename, Mat m)
{
	ofstream myfile;
	myfile.open(filename.c_str());
	myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
}