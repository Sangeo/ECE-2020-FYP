// C++ program to detects face in a video using color
// Goal: To extract color signals from the face recognition file
// Altered:
// - Does not mask the eyes ( we don't need this)
// - Uses a rectangle as a display rather than an elipse
// - Get a signal decomposition for the image in terms of blue/green/red
// - Obtained signal and outputs into excel where required.
// - Able to obtain a higher sampling rate after discovering resize functino of opencv

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
#include <vector>
#include <future>

using namespace std;
using namespace cv;

Mat splitColor(Mat frameC, Mat zeros, int cSel);
void write_CSV(string filename, vector<double> arr, double fps);
void detectAndDisplay(Mat frame, int cSel, Scalar lower, Scalar upper);
Mat skinDetection(Mat frame, Rect originalFaceRect, Scalar lBound, Scalar uBound);
typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

CascadeClassifier face_cascade;
vector<double> OUTPUT_CSV_VAR;

int main(int argc, const char** argv) {
	//Introducing the module
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
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
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	//-- 2. Read the video stream
	VideoCapture capture(0); //inputs into the Mat frame as CV_8UC3 format (unsigned integer)
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(CAP_PROP_FRAME_HEIGHT, 1080);

	// check that the camera is open
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}
	// this frame will store all information about the video captured by the camera
	Mat frame;

	cv::Scalar lower = cv::Scalar(0, 48, 80);
	cv::Scalar upper = cv::Scalar(20, 255, 255);

	for (;;)
	{
		if (capture.read(frame))
		{
			detectAndDisplay(frame, color_sel, lower, upper);
		}

		if (waitKey(1) == 27)
		{
			break; // if escape is pressed at any time
		}
	}

	// to find the time taken to do all calculations/capture frames
	int fps = 30; //this is constant for now, will find a way to adjust fps dynamically
	OUTPUT_CSV_VAR.erase(OUTPUT_CSV_VAR.begin(), OUTPUT_CSV_VAR.begin() + 150); //get rid of the first 150 frames from the recording
	write_CSV("output_file2.csv", OUTPUT_CSV_VAR, fps);

	capture.release();
	return 0;
}

void detectAndDisplay(Mat frame, int cSel, Scalar lBound, Scalar uBound) {
	Mat frameClone = frame.clone();
	Mat procFrame; //frame used for face recognition
	// resize the frame upon entry
	const double scaleFactor = 1.0 / 9;
	cv::resize(frameClone, procFrame, cv::Size(), scaleFactor, scaleFactor);

	// clone the frame for processing
	Mat frame_gray;
	cvtColor(procFrame, frame_gray, COLOR_BGR2GRAY); // convert the current frame into grayscale
	equalizeHist(frame_gray, frame_gray); // equalise the grayscale img
	//-- Detect faces
	std::vector<int> numDetections;
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray.clone(), faces, numDetections, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(20, 20));
	// finds the best face possible on the current frame
	int bestIndex = std::distance(
		numDetections.begin(),
		std::max_element(numDetections.begin(), numDetections.end()));
	
	Mat faceROI;

	// ensure that the frame is only processed when a face is obtained)
	if (!faces.empty()) {

		Rect bestFaceRect = faces[bestIndex]; //this contains the rectangle for the best face detected
		Rect originalFaceRect = Rect(bestFaceRect.tl() * (1 / scaleFactor), bestFaceRect.br() * (1 / scaleFactor)); //rescaling back to normal size
		// this obtains the forehead region of the face (will adjust dynamically)
		Point innerTopLC(originalFaceRect.x + (originalFaceRect.width * 3 / 10), originalFaceRect.y + originalFaceRect.height / 20);
		Point innerBotRC(originalFaceRect.x + (originalFaceRect.width * 7 / 10), originalFaceRect.y + (originalFaceRect.height * 1 / 5));
		Rect myROI(innerTopLC, innerBotRC);
		// draw on the frame clone
		rectangle(frameClone, originalFaceRect, Scalar(0, 0, 255), 1, LINE_4, 0);
		//-- This is used for color separation for later
		Mat zeroMatrix = Mat::zeros(Size(frameClone.cols, frameClone.rows), CV_8UC1);

		Mat colorImg = splitColor(frameClone, zeroMatrix, cSel);
		faceROI = colorImg(myROI).clone();
		vector<Mat> temp;
		split(faceROI, temp);//resplits the channels (extracting the color green for default/testing cases)
		Scalar averageColor = mean(temp[cSel - 1]); //takes the average of the color along a selected spectrum B/R/G
		double s = sum(averageColor)[0];
		OUTPUT_CSV_VAR.push_back(s);
	}
	// show what is obtained
	imshow("Capture - Face detection", frameClone);
	if (!faceROI.empty()) imshow("face ROI", faceROI);

}

Mat splitColor(Mat frameC, Mat zeros, int cSel) {
	// This will split the current frame (stored in frame_clone) into three different color 
	// channels each time we get a new frame
	if (!frameC.empty())
	{
		Mat splt[3];
		cv::split(frameC, splt);
		Mat colorImg;
		// colorImg will be the output which stores the color separated channels
		// split the color channels into blue, green and red according to BGR format of 
		vector<Mat> B;
		B.push_back(splt[0]);
		B.push_back(zeros);
		B.push_back(zeros);
		vector<Mat> G;
		G.push_back(zeros);
		G.push_back(splt[1]);
		G.push_back(zeros);
		vector<Mat> R;
		R.push_back(zeros);
		R.push_back(zeros);
		R.push_back(splt[2]);
		//switch select for color selection
		switch (cSel)
		{
		case 1: {
			//showing blue spectrum
			cv::merge(B, colorImg);
			break;
		}
		case 2: {
			//showing red spectrum
			cv::merge(G, colorImg);
			break;
		}
		case 3: {
			//showing red spectrum
			cv::merge(R, colorImg);
			break;
		}
		default: break;
		}
		return colorImg;
	}
}

/** Function to get skin color values on the face region of interest
	This function was written to obtain a larger sample size when doing calculations
	for signal averaging

	Mat skin = skinDetection(frame, originalFaceRect, lBound, uBound);
	imshow("skin", skin);

*/
Mat skinDetection(Mat frame, Rect originalFaceRect, Scalar lBound, Scalar uBound) {

	cv::Mat normalFace;
	cv::Mat faceRegion;
	cv::Mat HSVFrame;
	cv::cvtColor(frame, HSVFrame, cv::COLOR_BGR2HSV);
	cv::Mat skinMask;
	cv::Mat skin;

	//zone out faceRegion
	faceRegion = HSVFrame(originalFaceRect).clone();
	cv::inRange(faceRegion, lBound, uBound, skinMask);
	normalFace = frame(originalFaceRect).clone();

	//apply morphology to image to remove noise and ensure cleaner frame
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
	cv::erode(skinMask, skinMask, kernel, cv::Point(-1, -1), 2);
	cv::dilate(skinMask, skinMask, kernel, cv::Point(-1, -1), 2);
	cv::GaussianBlur(skinMask, skinMask, cv::Size(), 3, 3);
	//bitwise and to get the actual skin color back;
	cv::bitwise_and(normalFace, normalFace, skin, skinMask);

	return skin;
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