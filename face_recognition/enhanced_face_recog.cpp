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
void detectAndDisplay(Mat frame, int cSel);

typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

CascadeClassifier face_cascade;
vector<double> OUTPUT_CSV_VAR;

int main(int argc, const char** argv)
{
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
	// to find the time taken to do all calculations/capture frames

	/*Clock::time_point start = Clock::now();*/
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
	//Clock::time_point end = Clock::now();
	//milliseconds ms = chrono::duration_cast<milliseconds>(end - start);
	//cout << "ran for duration: " << ms.count() << "ms\n" << endl;
	//long long duration = ms.count();
	//long long n = OUTPUT_CSV_VAR.size();
	//cout << "Number of frames captured: " << n << "\n" << endl;
	//long long fps = n / (duration / 1000); // gives frames per second
	//cout << "Frames per second: " << fps << "\n" << endl;
	int fps = 30;
	OUTPUT_CSV_VAR.erase(OUTPUT_CSV_VAR.begin(), OUTPUT_CSV_VAR.begin() + 150);
	write_CSV("output_file2.csv", OUTPUT_CSV_VAR, fps);

	capture.release();
	return 0;
}

void detectAndDisplay(Mat frame, int cSel)
{
	Mat frameClone = frame.clone();
	Mat procFrame; //frame used for face recognition
	// resize the frame upon entry
	const double scaleFactor = 1.0 / 9.0;
	cv::resize(frameClone, procFrame, cv::Size(), scaleFactor, scaleFactor);
	// clone the frame for processing
	Mat frame_gray;
	cvtColor(procFrame, frame_gray, COLOR_BGR2GRAY); // convert the current frame into grayscale
	equalizeHist(frame_gray, frame_gray); // equalise the grayscale img
	//-- Detect faces
	std::vector<int> numDetections;
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray.clone(), faces, numDetections, 1.1);
	// finds the best face possible on the current frame
	int bestIndex = std::distance(
		numDetections.begin(),
		std::max_element(numDetections.begin(), numDetections.end()));
	Rect myROI;
	Rect originalFace;
	// ensure that the frame is only processed when a face is obtained)
	if (!faces.empty()) {

		cv::Rect bestFace = faces[bestIndex]; //this contains the rectangle for the best face detected
		cv::Rect originalFace = Rect(bestFace.tl() * (1 / scaleFactor), bestFace.br() * (1 / scaleFactor)); //rescaling back to normal size
		// this obtains the forehead region of the face (will adjust dynamically)
		Point innerTopLC(originalFace.x + (originalFace.width * 3 / 10), originalFace.y + originalFace.height / 20);
		Point innerBotRC(originalFace.x + (originalFace.width * 7 / 10), originalFace.y + (originalFace.height * 1 / 5));
		Rect myROI(innerTopLC, innerBotRC);
		// draw on the frame clone
		rectangle(frameClone, originalFace, Scalar(0, 0, 255), 1, LINE_4, 0);
		//-- This is used for color separation for later
		Mat zeroMatrix = Mat::zeros(Size(frameClone.cols, frameClone.rows), CV_8UC1);

		Mat colorImg = splitColor(frameClone, zeroMatrix, cSel);
		Mat faceROI = colorImg(myROI).clone();

		vector<Mat> temp;
		split(faceROI, temp);//resplits the channels (extracting the color green for default/testing cases)
		Scalar averageColor = mean(temp[cSel - 1]);
		double s = sum(averageColor)[0];
		OUTPUT_CSV_VAR.push_back(s);
	}
	// show what is obtained
	imshow("Capture - Face detection", frameClone);
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
		return colorImg;
	}
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