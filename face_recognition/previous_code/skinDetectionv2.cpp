#include "opencv2/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/matx.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <future>
#include <deque>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "fftw3.h"
#include "Python.h"
#include "matplotlibcpp.h"
#include <algorithm>
#include <tuple>

cv::CascadeClassifier face_cascade;
using namespace cv;
std::deque<double> topLeftX;
std::deque<double> topLeftY;
std::deque<double> botRightX;
std::deque<double> botRightY;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

cv::Mat detectAndDisplay(
	cv::Mat frame);

cv::Rect findPoints(
	std::vector<cv::Rect> faces,
	int bestIndex,
	double scaleFactor);

cv::Mat skinDetection(
	cv::Mat frameC,
	cv::Rect originalFaceRect,
	cv::Rect stuff);

int main(int argc, const char** argv) {

	//introduce the facial recog
	cv::CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("ECSE FYP 2020");
	parser.printMessage();

	//-- 1. Load the cascades
	cv::String face_cascade_name = cv::samples::findFile(parser.get<cv::String>("face_cascade"));
	if (!face_cascade.load(face_cascade_name)) {          
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	cv::VideoCapture capture(0);

	capture.set(cv::CAP_PROP_FPS, 30);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(cv::CAP_PROP_AUTOFOCUS, 1);
	capture.set(cv::CAP_PROP_AUTO_WB, 0);
	std::cout << "CAPTURE FOMRAT IS: " << capture.get(cv::CAP_PROP_FORMAT) << std::endl;

	const double scaleFactor = 1.0 / 3.0;

	Mat frame;
	while (true) {
		Clock::time_point start = Clock::now();
		if (capture.read(frame)) {
			Mat skinFrame = detectAndDisplay(frame);
			if (!skinFrame.empty())
				imshow("Skin Frame", skinFrame);
			Mat resizedFrame;
			resize(frame.clone(), resizedFrame, cv::Size(), scaleFactor, scaleFactor);
			if (!resizedFrame.empty())
				//imshow("Frame", resizedFrame);
			if (cv::waitKey(1) > 0)break;
		}
		Clock::time_point end = Clock::now();
		auto ms = std::chrono::duration_cast<milliseconds>(end - start);
		double captureTime = ms.count() / 1000.0;
		std::cout << "Skin/ROI Detection Took: " << captureTime << std::endl;

	}

	return -99;
}

/** Function detectAndDisplay
	Detects a face in a video feed from camera and
	return the frame of information which contains the color information in it.
	Input: current frame (raw)
	Output: current skin (raw);
*/
Mat detectAndDisplay(Mat frame) {

	Mat frameClone = frame.clone();
	Mat procFrame;
	Mat frameGray;
	std::vector<cv::Rect> faces;
	std::vector<int> numDetections;

	//downsizing image before processing
	const double scaleFactor = 1.0 / 7.0;
	resize(frameClone, procFrame, cv::Size(), scaleFactor, scaleFactor);
	//convert the image into a grayscale image which will be equalised for face detection
	cvtColor(procFrame, frameGray, COLOR_BGR2GRAY); // convert the current frame into grayscale
	equalizeHist(frameGray, frameGray); // equalise the grayscale img
	//use the Haar cascade classifier to process the image with training files.
	face_cascade.detectMultiScale(frameGray.clone(), faces, numDetections, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	// finds the best face possible on the current frame
	int bestIndex = std::distance(
		numDetections.begin(),
		std::max_element(numDetections.begin(), numDetections.end()));

	cv::Rect faceROI;
	cv::Mat trueROI;
	if (!faces.empty()) {
		//find the faceROI using a moving average to ease the noise out	
		faceROI = findPoints(faces, bestIndex, scaleFactor);

		if (!faceROI.empty()) {
			// draws on the current face region of interest
			// draws on the current face region of interest
			Point2i tempTL, tempBR;
			tempTL.x = faceROI.tl().x - 40;
			if (tempTL.x <= 0) {
				tempTL.x = faceROI.tl().x;
			}
			tempTL.y = faceROI.tl().y - 40;
			if (tempTL.y <= 0) {
				tempTL.y = faceROI.tl().y;
			}
			tempBR.x = faceROI.br().x + 40;
			if (tempBR.x >= frameClone.cols) {
				tempBR.x = faceROI.br().x;
				std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
			}
			tempBR.y = faceROI.br().y + 40;
			if (tempBR.y >= frameClone.rows) {
				tempBR.y = faceROI.br().y;
				std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
			}

			Rect tempRect(tempTL, tempBR);
			//rectangle(frameClone, tempRect, Scalar(0, 0, 255), 1, LINE_4, 0);


			trueROI = skinDetection(frameClone, tempRect, faceROI);
		}
	}
	/*imshow("Detected Face", frameClone);
	cv::waitKey(1);*/
	if (!trueROI.empty()) {
		return trueROI;
	}
	else {
		Mat zeros;
		return zeros;
	}
}

/** Function: findPoints
	locates the best face ROI coordinates using a simple
	moving average to eliminate noise generated by the casscade classifier

	uses a window size of 8 frames to ensure coordinates do not jump around

	Input:
	- vector of face locations,
	- the best face obtained,
	- the scale factor used to downsize the original image
	Output:
	- current face ROI coordinates, scaled to original image size
*/
Rect findPoints(std::vector<Rect> faces, int bestIndex, double scaleFactor) {

	double tlX = floor(faces[bestIndex].tl().x * (1 / scaleFactor));
	double tlY = floor(faces[bestIndex].tl().y * (1 / scaleFactor));
	double brX = floor(faces[bestIndex].br().x * (1 / scaleFactor));
	double brY = floor(faces[bestIndex].br().y * (1 / scaleFactor));

	//temporary variables
	double avgtlX;
	double avgtlY;
	double avgbrX;
	double avgbrY;
	double sumTX = 0;
	double sumTY = 0;
	double sumBX = 0;
	double sumBY = 0;
	//if the queue size is above a certain number we start to take the average of the frames
	int frameWindow = 7;
	if (topLeftX.size() >= frameWindow) {

		//Take the sum of all elements in the current frame window
		for (int i = 0; i < frameWindow; i++) {
			sumTX += topLeftX[i];
			sumTY += topLeftY[i];
			sumBX += botRightX[i];
			sumBY += botRightY[i];
		}

		//calculate the running average (flooring the number)
		avgtlX = std::floor(sumTX / frameWindow);
		avgtlY = std::floor(sumTY / frameWindow);
		avgbrX = std::floor(sumBX / frameWindow);
		avgbrY = std::floor(sumBY / frameWindow);

		//pop the front of the queue
		topLeftX.pop_front();
		topLeftY.pop_front();
		botRightX.pop_front();
		botRightY.pop_front();
		//add the current number into the queue
		topLeftX.push_back(tlX);
		topLeftY.push_back(tlY);
		botRightX.push_back(brX);
		botRightY.push_back(brY);

		//cout << avgtlX << "," << avgtlY << "," << avgbrX << "," << avgbrY << endl;
		//set the average as the points used for the face ROI rectangles
		Point avgTopLeft = Point(avgtlX, avgtlY);
		Point avgBotRight = Point(avgbrX, avgbrY);
		Rect faceROI = Rect(avgTopLeft, avgBotRight);

		return faceROI;
	}
	else {
		//add to the queue the current face positions
		topLeftX.push_back(tlX);
		topLeftY.push_back(tlY);
		botRightX.push_back(brX);
		botRightY.push_back(brY);
		Point avgTopLeft = Point(tlX, tlY);
		Point avgBotRight = Point(brX, brY);
		Rect faceROI = Rect(avgTopLeft, avgBotRight);

		return faceROI;
	}

	return Rect();
}


/** Function: skinDetection
	Input: Current frame, face ROI
	Output: Skin information

	Obtains skin in the detected face ROI, two thresholds are applied to the frame
	These following values work best under a bright white lamp
	1. YCrCb values
	lBound = (0, 133, 70);
	uBound = (255, 173, 127);

	2. HSV values
	lBound = (0, 30, 70);
	lBound = (17, 170, 255);
	The obtained skin mask will be applied using a 10,10 kernel which with the use of
	morphologyex (opening), clearing out false positives outside the face

*/
Mat skinDetection(Mat frameC, Rect originalFaceRect,Rect actualRegion) {

	Mat frameFace = frameC(originalFaceRect).clone();
	//shrink the region of interest to a face centric region
	Point2i tempTL, tempBR;
	int tlX, tlY, brX, brY;
	tlX = 0;
	tlY = 0;
	brX = frameFace.cols;
	brY = frameFace.rows;

	tempTL.x = tlX + (brX - tlX) * 0.1;
	if (tempTL.x <= 0) {
		tempTL.x = originalFaceRect.tl().x;
	}
	tempTL.y = tlY + (brY - tlY) * 0.1;
	if (tempTL.y <= 0) {
		tempTL.y = originalFaceRect.tl().y;
	}
	tempBR.x = brX - (brX - tlX) * 0.1;
	if (tempBR.x >= frameFace.cols) {
		tempBR.x = originalFaceRect.br().x;
		std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
	}
	tempBR.y = brY - (brY - tlY) * 0.35;
	if (tempBR.y >= frameFace.rows) {
		tempBR.y = originalFaceRect.br().y;
		std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
	}
	Rect tempRect(tempTL, tempBR);

	tlX = actualRegion.tl().x;
	tlY = actualRegion.tl().y;
	brX = actualRegion.br().x;
	brY = actualRegion.br().y;

	tempTL.x = tlX;
	if (tempTL.x <= 0) {
		tempTL.x = actualRegion.tl().x;
	}
	tempTL.y = tlY;
	if (tempTL.y <= 0) {
		tempTL.y = actualRegion.tl().y;
	}
	tempBR.x = brX;
	if (tempBR.x >= brX) {
		tempBR.x = actualRegion.br().x;
		std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
	}
	tempBR.y = brY - (brY - tlY) * 0.35;
	if (tempBR.y >= brY) {
		tempBR.y = actualRegion.br().y;
		std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
	}
	Rect tempRect2(tempTL, tempBR);
	
	frameFace = frameFace(tempRect);

	Mat yccFace, imgFilter;
	cv::cvtColor(frameFace, yccFace, COLOR_BGR2YCrCb, CV_8U);

	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	//low-pass spatial filtering to remove high frequency content
	blur(yccFace, yccFace, Size(5, 5));
	std::vector<Mat> YCrCb_planes;
	split(yccFace, YCrCb_planes);
	Mat y_hist, cr_hist, cb_hist;
	calcHist(&YCrCb_planes[0], 1, 0, Mat(), y_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&YCrCb_planes[1], 1, 0, Mat(), cr_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&YCrCb_planes[2], 1, 0, Mat(), cb_hist, 1, &histSize, &histRange, uniform, accumulate);
	
	//unfortunately i need to define all the following parameters for this to work..
	double y_min, cr_min, cb_min;
	double y_max, cr_max, cb_max;
	Point y_min_loc, cr_min_loc, cb_min_loc;
	Point y_max_loc, cr_max_loc, cb_max_loc;
	cv::minMaxLoc(y_hist, &y_min, &y_max, &y_min_loc, &y_max_loc);
	cv::minMaxLoc(cr_hist, &cr_min, &cr_max, &cr_min_loc, &cr_max_loc);
	cv::minMaxLoc(cb_hist, &cb_min, &cb_max, &cb_min_loc, &cb_max_loc);
	
	/*std::cout << "y_max location: " << y_max_loc.y
		<< " cr_max location: " << cr_max_loc.y
		<< " cb_max location: " << cb_max_loc.y << std::endl;*/

	int min_cr , min_cb , max_cr , max_cb ;
	min_cr = cr_max_loc.y - 10;
	max_cr = cr_max_loc.y + 10;
	min_cb = cb_max_loc.y - 10;
	max_cb = cb_max_loc.y + 10;
  
	
	//colour segmentation
	cv::inRange(yccFace, Scalar(0, min_cr, min_cb), Scalar(255, max_cr, max_cb), imgFilter);

	//Morphology on the imgFilter to remove noise introduced by colour segmentation
	//Mat kernel = Mat::ones(Size(7, 7), CV_8U);
	Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
	cv::morphologyEx(imgFilter, imgFilter, cv::MORPH_OPEN, kernel, Point(-1, -1),3);
	cv::morphologyEx(imgFilter, imgFilter, cv::MORPH_CLOSE, kernel, Point(-1, -1),3);
		
	Mat skin;
	//return our detected skin valuessc
	frameFace.copyTo(skin, imgFilter);
	//imshow("SKIN", skin);
	imshow("imgFilter raw", imgFilter);
	Mat frameCC;
	frameCC = frameC.clone();
	rectangle(frameCC, originalFaceRect, Scalar(0, 255, 255));
	//rectangle(frameCC, actualRegion, Scalar(255, 0, 255));
	rectangle(frameCC, tempRect2, Scalar(255, 0, 255));
	resize(frameCC, frameCC, cv::Size(), 0.5, 0.5);
	imshow("ROI Comparison", frameCC);
	cv::waitKey(1);

	return skin;
}


