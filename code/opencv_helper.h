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

using namespace cv;
namespace plt = matplotlibcpp;
cv::CascadeClassifier face_cascade;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

const bool DEBUG_MODE = false;
const bool DEBUG_MODE2 = false;

std::deque<double> topLeftX;
std::deque<double> topLeftY;
std::deque<double> botRightX;
std::deque<double> botRightY;
std::vector<double> timeVec;
std::vector<double> timeVecOutput;
std::vector<double> rPPGSignal;
std::vector<double> rPPGSignalFiltered;
std::vector<double> rBCGSignal;
std::vector<double> rBCGSignalFiltered;

int strideLength = 45; // size of the temporal stride 
					//(45 frames as this should capture the heart rate information)
bool firstStride = true;


const int FILTER_SECTIONS = 12;

//the following filter uses Cheby II bandpass
/*
'IIR Digital Filter (real)                              '
	'-------------------------                              '
	'Number of Sections  : 12                               '
	'Stable              : Yes                              '
	'Linear Phase        : No                               '
	'                                                       '
	'Design Method Information                              '
	'Design Algorithm : Chebyshev type II                   '
	'                                                       '
	'Design Options                                         '
	'Match Exactly : stopband                               '
	'                                                       '
	'Design Specifications                                  '
	'Sample Rate            : 30 Hz                         '
	'Response               : Bandpass                      '
	'Specification          : Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2'
	'Second Passband Edge   : 2.6 Hz                        '
	'First Stopband Edge    : 600 mHz                       '
	'Passband Ripple        : 1 dB                          '
	'First Stopband Atten.  : 45 dB                         '
	'Second Stopband Atten. : 60 dB                         '
	'First Passband Edge    : 800 mHz                       '
	'Second Stopband Edge   : 3 Hz                          '
	*/
const double sos_matrix[12][6] = {
	{0.821904631289823, -1.62696489036367, 0.821904631289823, 1, -1.65115775247009, 0.956239445176575},
	{ 0.821904631289823, -1.32674174567684, 0.821904631289823, 1, -1.96106396889903, 0.986723611465211 },
	{ 0.780764081275640, -1.54692952632525, 0.780764081275640, 1, -1.56427767500315, 0.864961197373822 },
	{ 0.780764081275640, -1.23429225313410, 0.780764081275640, 1, -1.93459287711451, 0.959002322765484 },
	{ 0.714686410007531, -1.41854618499363, 0.714686410007531, 1, -1.45294320387794, 0.754222796360954 },
	{ 0.714686410007531, -1.06823178848402, 0.714686410007531, 1, -1.90430891262403, 0.926726397665631 },
	{ 0.611402310616563, -1.21661813952701, 0.611402310616563, 1, -1.86538074368294, 0.885520925318713 },
	{ 0.611402310616563, -0.787144826085887, 0.611402310616563, 1, -1.31144754653070, 0.610589268539194 },
	{ 0.461489552066884, -0.920876339017339, 0.461489552066884, 1, -1.81057593401990, 0.829235052618707 },
	{ 0.461489552066884, -0.322599196669853, 0.461489552066884, 1, -1.16218727318019, 0.441359420631550 },
	{ 0.299969123764612, -0.599764553627771, 0.299969123764612, 1, -1.73181537720060, 0.752138831145407 },
	{ 0.299969123764612, 0.349792836547195, 0.299969123764612, 1, -1.08728905725033, 0.313519738807378 } };

/*	'IIR Digital Filter (real)                              '
	'-------------------------                              '
	'Number of Sections  : 17                               '
	'Stable              : Yes                              '
	'Linear Phase        : No                               '
	'                                                       '
	'Design Method Information                              '
	'Design Algorithm : Chebyshev type II                   '
	'                                                       '
	'Design Options                                         '
	'Match Exactly : stopband                               '
	'                                                       '
	'Design Specifications                                  '
	'Sample Rate            : 30 Hz                         '
	'Response               : Bandpass                      '
	'Specification          : Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2'
	'Second Passband Edge   : 4 Hz                          '
	'First Stopband Edge    : 400 mHz                       '
	'First Stopband Atten.  : 45 dB                         '
	'Second Stopband Edge   : 4.2 Hz                        '
	'Passband Ripple        : 1 dB                          '
	'Second Stopband Atten. : 45 dB                         '
	'First Passband Edge    : 700 mHz                       '*/
const int FILTER_SECTIONS_2 = 17;
const double sos_matrix2[17][6] =
{ {0.940366323, -1.862774655, 0.940366323, 1, -1.299740364, 0.967078909},
{ 0.940366323, -1.195268478	,0.940366323,1, -1.972845695,0.99360238},
{ 0.918599724, -1.820541289	,0.918599724,1, -1.232519133,0.899884024},
{ 0.918599724, -1.138679916	,0.918599724,1, -1.960890165,0.980667121},
{ 0.891360127, -1.768235451	,0.891360127,1, -1.135220512,0.82340388},
{ 0.891360127, -1.042363165	,0.891360127,1, -1.949143543,0.967142085},
{ 0.85524642 ,-1.698907786,	0.85524642	,1, -0.997765752,0.728069166},
{ 0.85524642, -0.891810875,	0.85524642	,1, -1.937124601,0.952644804},
{ 0.805827276, -1.603417843	,0.805827276,1, -0.808305984,0.603484961},
{ 0.805827276 ,-0.66265824	,0.805827276,1, -1.924522282,0.937042602},
{ 0.738392147, -1.471891246	,0.738392147,1, -1.911450164,0.920745389},
{ 0.738392147, -0.319565142	,0.738392147,1, -0.559761346,0.441968437},
{ 0.652480987, -1.302813134	,0.652480987,1, -1.898898218,0.905191294},
{0.652480987,0.16888325,	0.652480987	,1 ,-0.26891347	,0.251218407},
{0.565179381,-1.129858484	,0.565179381,1 ,-1.889175876,0.893274654},
{0.565179381,0.742562256,	0.565179381	,1 ,-0.007319884,0.076933969},
{0.524257303, 0				,-0.524257303,1 ,-0.891236583 ,-0.048514606} };

// End Filter Design Section

/*C++ findpeaks in a noisy data sample implemented by a github user: claydergc
* https://github.com/claydergc/find-peaks
*/
namespace Peaks {
	const double EPS = 2.2204e-16f;
	void findPeaks(std::vector<double> x0, std::vector<int>& peakInds);
}

void diff(std::vector<double> in, std::vector<double>& out) {
	out = std::vector<double>(in.size() - 1);

	for (int i = 1; i < in.size(); ++i)
		out[i - 1] = in[i] - in[i - 1];
}

void vectorProduct(std::vector<double> a, std::vector<double> b, std::vector<double>& out) {
	out = std::vector<double>(a.size());

	for (int i = 0; i < a.size(); ++i)
		out[i] = a[i] * b[i];
}

void findIndicesLessThan(std::vector<double> in, double threshold, std::vector<int>& indices) {
	for (int i = 0; i < in.size(); ++i)
		if (in[i] < threshold)
			indices.push_back(i + 1);
}

void selectElements(std::vector<double> in, std::vector<int> indices, std::vector<double>& out) {
	for (int i = 0; i < indices.size(); ++i)
		out.push_back(in[indices[i]]);
}

void selectElements(std::vector<int> in, std::vector<int> indices, std::vector<int>& out) {
	for (int i = 0; i < indices.size(); ++i)
		out.push_back(in[indices[i]]);
}

void signVector(std::vector<double> in, std::vector<int>& out) {
	out = std::vector<int>(in.size());

	for (int i = 0; i < in.size(); ++i) {
		if (in[i] > 0)
			out[i] = 1;
		else if (in[i] < 0)
			out[i] = -1;
		else
			out[i] = 0;
	}
}

void Peaks::findPeaks(std::vector<double> x0, std::vector<int>& peakInds) {
	int minIdx = distance(x0.begin(), min_element(x0.begin(), x0.end()));
	int maxIdx = distance(x0.begin(), max_element(x0.begin(), x0.end()));

	double sel = (x0[maxIdx] - x0[minIdx]) / 4.0;

	int len0 = x0.size();

	std::vector<double> dx;
	diff(x0, dx);
	std::replace(dx.begin(), dx.end(), 0.0, -Peaks::EPS);
	std::vector<double> dx0(dx.begin(), dx.end() - 1);
	std::vector<double> dx1(dx.begin() + 1, dx.end());
	std::vector<double> dx2;

	vectorProduct(dx0, dx1, dx2);

	std::vector<int> ind;
	findIndicesLessThan(dx2, 0, ind); // Find where the derivative changes sign

	std::vector<double> x;

	std::vector<int> indAux(ind.begin(), ind.end());
	selectElements(x0, indAux, x);
	x.insert(x.begin(), x0[0]);
	x.insert(x.end(), x0[x0.size() - 1]);;


	ind.insert(ind.begin(), 0);
	ind.insert(ind.end(), len0);

	int minMagIdx = distance(x.begin(), min_element(x.begin(), x.end()));
	double minMag = x[minMagIdx];
	double leftMin = minMag;
	int len = x.size();

	if (len > 2) {
		double tempMag = minMag;
		bool foundPeak = false;
		int ii;

		// Deal with first point a little differently since tacked it on
		// Calculate the sign of the derivative since we tacked the first
		//  point on it does not neccessarily alternate like the rest.
		std::vector<double> xSub0(x.begin(), x.begin() + 3);//tener cuidado subvector
		std::vector<double> xDiff;//tener cuidado subvector
		diff(xSub0, xDiff);

		std::vector<int> signDx;
		signVector(xDiff, signDx);

		if (signDx[0] <= 0) // The first point is larger or equal to the second
		{
			if (signDx[0] == signDx[1]) // Want alternating signs
			{
				x.erase(x.begin() + 1);
				ind.erase(ind.begin() + 1);
				len = len - 1;
			}
		}
		else // First point is smaller than the second
		{
			if (signDx[0] == signDx[1]) // Want alternating signs
			{
				x.erase(x.begin());
				ind.erase(ind.begin());
				len = len - 1;
			}
		}

		if (x[0] >= x[1])
			ii = 0;
		else
			ii = 1;

		double maxPeaks = ceil((double)len / 2.0);
		std::vector<int> peakLoc(maxPeaks, 0);
		std::vector<double> peakMag(maxPeaks, 0.0);
		int cInd = 1;
		int tempLoc;

		while (ii < len) {
			ii = ii + 1;//This is a peak
			//Reset peak finding if we had a peak and the next peak is bigger
			//than the last or the left min was small enough to reset.
			if (foundPeak) {
				tempMag = minMag;
				foundPeak = false;
			}

			//Found new peak that was lager than temp mag and selectivity larger
			//than the minimum to its left.

			if (x[ii - 1] > tempMag && x[ii - 1] > leftMin + sel) {
				tempLoc = ii - 1;
				tempMag = x[ii - 1];
			}

			//Make sure we don't iterate past the length of our vector
			if (ii == len)
				break; //We assign the last point differently out of the loop

			ii = ii + 1; // Move onto the valley

			//Come down at least sel from peak
			if (!foundPeak && tempMag > sel + x[ii - 1]) {
				foundPeak = true; //We have found a peak
				leftMin = x[ii - 1];
				peakLoc[cInd - 1] = tempLoc; // Add peak to index
				peakMag[cInd - 1] = tempMag;
				cInd = cInd + 1;
			}
			else if (x[ii - 1] < leftMin) // New left minima
				leftMin = x[ii - 1];

		}

		// Check end point
		if (x[x.size() - 1] > tempMag && x[x.size() - 1] > leftMin + sel) {
			peakLoc[cInd - 1] = len - 1;
			peakMag[cInd - 1] = x[x.size() - 1];
			cInd = cInd + 1;
		}
		else if (!foundPeak && tempMag > minMag)// Check if we still need to add the last point
		{
			peakLoc[cInd - 1] = tempLoc;
			peakMag[cInd - 1] = tempMag;
			cInd = cInd + 1;
		}

		//Create output
		if (cInd > 0) {
			std::vector<int> peakLocTmp(peakLoc.begin(), peakLoc.begin() + cInd - 1);
			selectElements(ind, peakLocTmp, peakInds);
			//peakMags = vector<double>(peakLoc.begin(), peakLoc.begin()+cInd-1);
		}

	}

}


// Common Function Design Section
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
	Output: Skin image

	The obtained skin mask will be applied using a 10,10 kernel which with the use of
	morphologyex (opening), clearing out false positives outside the face

*/
Mat skinDetection(Mat frameC, Rect originalFaceRect) {

	//shrink the region of interest to a face centric region
	Point2i tempTL, tempBR;
	int tlX, tlY, brX, brY;
	tlX = originalFaceRect.tl().x;
	tlY = originalFaceRect.tl().y;
	brX = originalFaceRect.br().x;
	brY = originalFaceRect.br().y;

	tempTL.x = tlX + (brX - tlX) * 0.05;
	if (tempTL.x <= 0) {
		tempTL.x = originalFaceRect.tl().x;
	}
	tempTL.y = tlY + (brY - tlY) * 0.1;
	if (tempTL.y <= 0) {
		tempTL.y = originalFaceRect.tl().y;
	}
	tempBR.x = brX - (brX - tlX) * 0.05;
	if (tempBR.x >= frameC.cols) {
		tempBR.x = originalFaceRect.br().x;
		std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
	}
	tempBR.y = brY - (brY - tlY) * 0.05;
	if (tempBR.y >= frameC.rows) {
		tempBR.y = originalFaceRect.br().y;
		std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
	}
	Rect tempRect(tempTL, tempBR);
	Mat frameFace = frameC(tempRect);

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

	//finds the minmax locations for the chrominance channels in the YCrCB image (local min and max)
	double cr_min, cb_min;
	double cr_max, cb_max;
	Point cr_min_loc, cb_min_loc;
	Point cr_max_loc, cb_max_loc;
	cv::minMaxLoc(cr_hist, &cr_min, &cr_max, &cr_min_loc, &cr_max_loc);
	cv::minMaxLoc(cb_hist, &cb_min, &cb_max, &cb_min_loc, &cb_max_loc);
	//define arbitrary range of values which we can accept for the skin as +/- 10 chrominance (R-G, B-G);
	int min_cr, min_cb, max_cr, max_cb;
	min_cr = cr_max_loc.y - 10;
	max_cr = cr_max_loc.y + 10;
	min_cb = cb_max_loc.y - 10;
	max_cb = cb_max_loc.y + 10;

	//Median Colour Filtering (Ensures minimal noise for the skin detection)
	cv::inRange(yccFace, Scalar(0, min_cr, min_cb), Scalar(255, max_cr, max_cb), imgFilter);

	//Morphology on the imgFilter to remove noise introduced by colour segmentation
	Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
	// Ellipse was found to work the best, the rectangle made the skin mask really noise for unknown reasons
	cv::morphologyEx(imgFilter, imgFilter, cv::MORPH_OPEN, kernel, Point(-1, -1), 3);
	cv::morphologyEx(imgFilter, imgFilter, cv::MORPH_CLOSE, kernel, Point(-1, -1), 3);

	//Possible implementation which can be useful would be medianblur to further smoothen the skin mask
	Mat skin;
	
	//return our detected skin values
	frameFace.copyTo(skin, imgFilter);
	if (DEBUG_MODE2) {
		Mat frameCC;
		frameC.copyTo(frameCC);
		double sF = 1.0 / 2.0;
		imshow("skinFrame", skin);
		rectangle(frameCC, originalFaceRect, Scalar(255, 0, 255));
		rectangle(frameCC, tempRect, Scalar(255, 0, 0));
		resize(frameCC, frameCC, cv::Size(), sF, sF);
		imshow("frame ROI", frameCC);
		cv::waitKey(1);
	}


	return skin;
}

/** Function detectAndDisplay
	Detects a face in a video feed from camera and
	return the frame of information which contains the color information in it.
	Input: current frame;
	Output: current skin pixel values, current region of interest location for rBCG;
*/
std::tuple<std::deque<Mat>, std::vector<Rect>> detectAndDisplay(std::deque<Mat> frameQ, int numFrames, int choice) {

	std::deque<Mat> skin_FrameQ;
	std::vector<Rect> rBCG_ROI_Q;

	for (int i = 0; i < numFrames; i++) {
		Mat frameClone = frameQ[i];
		Mat procFrame;
		Mat frameGray;
		std::vector<Rect> faces;
		std::vector<int> numDetections;

		//downsizing image before processing
		double scaleFactor;
		if (choice == 1) {
			scaleFactor = 1.0 / 6.0;
		}
		else {
			scaleFactor = 1.0 / 3.0;
		}
		resize(frameClone, procFrame, cv::Size(), scaleFactor, scaleFactor);
		//convert the image into a grayscale image which will be equalised for face detection
		cvtColor(procFrame, frameGray, COLOR_BGR2GRAY); // convert the current frame into grayscale
		equalizeHist(frameGray, frameGray); // equalise the grayscale img
		//use the Haar cascade classifier to process the image with training files.
		face_cascade.detectMultiScale(frameGray.clone(), faces, numDetections, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		//finds the best face possible on the current frame
		int bestIndex = std::distance(
			numDetections.begin(),
			std::max_element(numDetections.begin(), numDetections.end()));

		Rect rBCGFocus;
		Rect faceROI;
		Mat rPPGROI;

		if (!faces.empty()) {
			//find the faceROI using a moving average to ease the noise out	
			faceROI = findPoints(faces, bestIndex, scaleFactor);

			if (!faceROI.empty()) {

				// finding suitable region for rBCG feature extraction
				Point2i tempTL, tempBR;
				int tlX, tlY, brX, brY;
				tlX = faceROI.tl().x;
				tlY = faceROI.tl().y;
				brX = faceROI.br().x;
				brY = faceROI.br().y;

				tempTL.x = tlX + (brX - tlX) * 0.15;
				if (tempTL.x <= 0) {
					tempTL.x = faceROI.tl().x;
				}
				tempTL.y = tlY + (brY - tlY) * 0.45;
				if (tempTL.y <= 0) {
					tempTL.y = faceROI.tl().y;
				}
				tempBR.x = brX - (brX - tlX) * 0.15;
				if (tempBR.x >= frameClone.cols) {
					tempBR.x = faceROI.br().x;
					std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
				}
				tempBR.y = brY - (brY - tlY) * 0.001;
				if (tempBR.y >= frameClone.rows) {
					tempBR.y = faceROI.br().y;
					std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
				}

				rBCGFocus = Rect(tempTL, tempBR);

				// finding appropriate section for skin analysis;
				Point2i skinTL, skinBR;
				skinTL.x = tlX + (brX - tlX) * 0.1;
				if (skinTL.x <= 0) {
					skinTL.x = faceROI.tl().x;
				}
				skinTL.y = tlY + (brY - tlY) * 0.05; //basically not changed
				if (skinTL.y <= 0) {
					skinTL.y = faceROI.tl().y;
				}
				skinBR.x = brX - (brX - tlX) * 0.1;
				if (skinBR.x >= frameClone.cols) {
					skinBR.x = faceROI.br().x;
					std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
				}
				skinBR.y = brY - (brY - tlY) * 0.35;
				if (skinBR.y >= frameClone.rows) {
					skinBR.y = faceROI.br().y;
					std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
				}

				Rect tempRect(skinTL, skinBR);
				//rectangle(frameClone, tempRect, Scalar(0, 0, 255), 1, LINE_4, 0);
				rPPGROI = skinDetection(frameClone, tempRect); //skin detection conducted here

				rBCG_ROI_Q.push_back(rBCGFocus);
				skin_FrameQ.push_back(rPPGROI);
			}
		}
	}

	return std::make_tuple(skin_FrameQ, rBCG_ROI_Q);
}

/* Function: sosFilter
* Uses a predefined IIR Bandpass filter (Cheby-II) to filter the raw signal results obtained from both methods.
* Input: signal vector raw (common to both methods)
* Output: filtered signal
*/
std::vector<double> sosFilter(std::vector<double> signal) {

	std::vector<double> output;
	output.resize(signal.size());

	double** tempOutput = new double* [FILTER_SECTIONS];
	for (int i = 0; i < FILTER_SECTIONS; i++)
		tempOutput[i] = new double[signal.size()];
	for (int i = 0; i < FILTER_SECTIONS; i++) {
		for (int j = 0; j < signal.size(); j++) {
			tempOutput[i][j] = 0;
		}
	}

	for (int i = 0; i < signal.size(); i++) {

		if (i - 2 < 0) {
			//std::cout << "skipping some stuff" << std::endl;
			continue;
		}

		double b0, b1, b2, a1, a2;
		double result;
		//for each section
		for (int j = 0; j < FILTER_SECTIONS; j++) {

			b0 = sos_matrix[j][0];
			b1 = sos_matrix[j][1];
			b2 = sos_matrix[j][2];
			a1 = sos_matrix[j][4];
			a2 = sos_matrix[j][5];

			if (j == 0) {
				result = b0 * signal[i] + b1 * signal[i - 1] + b2 * signal[i - 2]
					- a1 * tempOutput[j][i - 1] - a2 * tempOutput[j][i - 2];
				tempOutput[j][i] = result;
			}
			else {
				result = b0 * tempOutput[j - 1][i] + b1 * tempOutput[j - 1][i - 1] + b2 * tempOutput[j - 1][i - 2]
					- a1 * tempOutput[j][i - 1] - a2 * tempOutput[j][i - 2];
				tempOutput[j][i] = result;
			}

		}

	}
	for (int x = 0; x < signal.size(); x++) {
		output[x] = tempOutput[FILTER_SECTIONS - 1][x];
		//std::cout << "output: " << output[x] << std::endl;
	}

	return output;
}
/* Function: sosFilter2
* Uses a predefined IIR Bandpass filter (Cheby-II) to filter the raw signal results obtained from both methods.
* Input: signal vector raw (common to both methods)
* Output: filtered signal
*/
std::vector<double> sosFilter2(std::vector<double> signal) {

	std::vector<double> output;
	output.resize(signal.size());

	double** tempOutput = new double* [FILTER_SECTIONS_2];
	for (int i = 0; i < FILTER_SECTIONS_2; i++)
		tempOutput[i] = new double[signal.size()];
	for (int i = 0; i < FILTER_SECTIONS_2; i++) {
		for (int j = 0; j < signal.size(); j++) {
			tempOutput[i][j] = 0;
		}
	}

	for (int i = 0; i < signal.size(); i++) {

		if (i - 2 < 0) {
			//std::cout << "skipping some stuff" << std::endl;
			continue;
		}

		double b0, b1, b2, a1, a2;
		double result;
		//for each section
		for (int j = 0; j < FILTER_SECTIONS_2; j++) {

			b0 = sos_matrix2[j][0];
			b1 = sos_matrix2[j][1];
			b2 = sos_matrix2[j][2];
			a1 = sos_matrix2[j][4];
			a2 = sos_matrix2[j][5];

			if (j == 0) {
				result = b0 * signal[i] + b1 * signal[i - 1] + b2 * signal[i - 2]
					- a1 * tempOutput[j][i - 1] - a2 * tempOutput[j][i - 2];
				tempOutput[j][i] = result;
			}
			else {
				result = b0 * tempOutput[j - 1][i] + b1 * tempOutput[j - 1][i - 1] + b2 * tempOutput[j - 1][i - 2]
					- a1 * tempOutput[j][i - 1] - a2 * tempOutput[j][i - 2];
				tempOutput[j][i] = result;
			}

		}

	}
	for (int x = 0; x < signal.size(); x++) {
		output[x] = tempOutput[FILTER_SECTIONS_2 - 1][x];
		//std::cout << "output: " << output[x] << std::endl;
	}

	return output;
}

/** Function calcPSD
* Function used to generate the frequency power spectrum used to plot
* Input: Signal and the number of Frames used
* Output: the frequency spectrum and relevant power values in dB
*/
std::tuple<std::vector<double>, std::vector<double>>
calcPSD(std::vector<double> sig, int numFrames) {
	//Pre-filtering FFT and PLOTTING ~~~~~~~~~~~~~~~~~~~~~
	//preallocate memory
	double* in = (double*)fftw_malloc(sizeof(double) * numFrames);
	fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numFrames);

	for (int i = 0; i < numFrames; i++) {
		double sig_ = sig[i];
		in[i] = sig_;
	}

	fftw_plan fftPlan = fftw_plan_dft_r2c_1d(numFrames, in, out, FFTW_ESTIMATE);
	fftw_execute(fftPlan);

	std::vector<double> v, ff;
	double sampRate = 30;
	for (int i = 0; i <= ((numFrames / 2) - 1); i++) {
		double a, b;
		a = sampRate * i / numFrames;
		//Here I have calculated the y axis of the spectrum in dB
		b = (20 * log(sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]))) / numFrames;

		/*if (i == 0) {
			b = b * -1.0;
		}*/

		v.push_back((double)b);
		ff.push_back((double)a);
	}

	fftw_destroy_plan(fftPlan);
	fftw_free(in);
	fftw_free(out);

	return std::make_tuple(v, ff);
}


// rPPG Function Design Section
/* Function: spatialRotation
* Inputs:
*	- a vector of skinFrames
*	- the current long term pulse analysis/estimation
*	- number of frames in the current capture
* Outputs:
*	- the current long term pulse analysis/estimation (the same length as the number of frames in the current capture)
*/
Mat spatialRotation(std::deque<Mat> skinFrameQ, Mat longTermPulseVector, int capLength) {
	std::deque<Mat> eigValArray; // This will contain the eigenvalues
	std::deque<Mat> eigVecArray; // This will contain the eigenvectors
	std::vector<Mat> SRDash;
	// for each frame in the queue 
	for (int i = 0; i < capLength; i++) {
		//our skinFrames are queued in a deque, which can be accessed using a for loop
		//this retrieves a single skin frame from the queue, but this will contain many zeros
		Mat temp = skinFrameQ.front().clone();
		skinFrameQ.pop_front();
		if (DEBUG_MODE2) {
			//cout << "The skinFrame is :" << endl << temp << endl;
		}
		//for each skinFrame, split the skin pixles into 3 channels, blue, green and red.
		std::vector<Mat> colorVector;
		split(temp, colorVector);
		// colorVector has size 3xN, 
		// colorVector[0] contains all blue pixels
		// colorVector[1] contains all green pixels 
		// colorVector[2] contains all red pixels 

		/**code used to get rid of the unnecessary zeros**/
		// note: we only need to use one mask here since skin pixel values cannot be (0,0,0)
		std::vector<Point> mask;
		findNonZero(colorVector[0], mask);
		Mat bVal, gVal, rVal;
		for (Point p : mask) {
			bVal.push_back(colorVector[0].at<uchar>(p)); // collect blue values
			gVal.push_back(colorVector[1].at<uchar>(p)); // collect green values
			rVal.push_back(colorVector[2].at<uchar>(p)); // collect red values
		}

		Mat colorValues; //colorValues is a Nx3 matrix
		std::vector<Mat> matrices = { rVal, gVal, bVal };
		hconcat(matrices, colorValues);
		if (DEBUG_MODE2) {
			//cout << "The concatenanted skinFrame (ignoring zeros) is : " << endl << colorValues << endl;
		}
		// identity of the colorValues matirx
		int rows = colorValues.rows; // this will be the number of skin-pixels N
		int cols = colorValues.cols; // this will be the 3 color channels (r,g,b)

		Mat vTv; //Transposed colorValues * colorValues, normalised matrix
		mulTransposed(colorValues, vTv, true);
		// Divide the multiplied vT*v vector by the total number of skin-pixels
		double N = rows; // number of skin-pixels
		Mat C = vTv / N;
		Mat eigVal, eigVec;
		cv::eigen(C, eigVal, eigVec);	//computes the eigenvectors and eigenvalues used 
										//for our spatial subspace rotation calculations

		Mat sortEigVal;
		cv::sort(eigVal, sortEigVal, cv::SortFlags::SORT_EVERY_COLUMN + cv::SortFlags::SORT_DESCENDING);

		/* ~~~~temporal stride analysis~~~~ */
		//Within the stride, we will analyze the temporal changes in the eigenvector subspace.
		eigValArray.push_back(sortEigVal.clone()); // This will contain the first frame's content for eigenvalues
		eigVecArray.push_back(eigVec.clone()); // This will contain the first frame's content for eigenvectors
		Mat uTau, lambdaTau;
		Mat uTime, lambdaTime;
		Mat RDash, S; //R' and S

		int tempTau = (i + 1) - strideLength + 1;
		if (tempTau > 0) {
			uTau = eigVecArray[tempTau - 1].clone();
			lambdaTau = eigValArray[tempTau - 1].clone();

			int t; //this will need to be re-used for t 
			for (t = tempTau; t < (i + 2); t++) { // note this needs to be i+2
				//current values of the eigenvector and eigenvalues
				uTime = eigVecArray[t - 1].clone();
				lambdaTime = eigValArray[t - 1].clone();

				// calculation for R'
				Mat t1, t_2, t_3, r1, r2;
				t1 = uTime.col(0).clone(); //current frame's U (or u1 vector)
				cv::transpose(t1, t1);
				t_2 = uTau.col(1).clone(); //first frame's u2
				r1 = t1 * t_2;
				t_3 = uTau.col(2).clone(); //first frame's u3
				r2 = t1 * t_3;
				double result[2] = { sum(r1)[0],sum(r2)[0] };
				RDash = (Mat_<double>(1, 2) << result[0], result[1]); // obtains the R' values required to calculate the SR'

				// calculation for S
				Mat temp, tau2, tau3;
				temp = lambdaTime.row(0).clone(); //lambda t1
				tau2 = lambdaTau.row(1).clone();  //lambda tau2
				tau3 = lambdaTau.row(2).clone();  //lambda tau3

				double result2[3] = { sum(temp)[0],sum(tau2)[0],sum(tau3)[0] };
				double lambda1, lambda2;
				lambda1 = sqrt(result2[0] / result2[1]);
				lambda2 = sqrt(result2[0] / result2[2]);
				S = (Mat_<double>(1, 2) << lambda1, lambda2);
				Mat SR = S.mul(RDash);; // Obtain SR matrix, and adjust with rotation vectors (step (10) of 2SR paper)

				SR.convertTo(SR, 5); // adjust to 32F rather than double --- needs fixing later

				//back-projection into the original RGB space
				Mat backProjectTau;
				vconcat(t_2.t(), t_3.t(), backProjectTau);
				Mat SRBackProjected = SR * backProjectTau;

				// Obtain the SR' vector
				SRDash.push_back(SRBackProjected);
			}
			// Calculate SR'1 and SR'2
			Mat SRDashConcat;
			vconcat(SRDash, SRDashConcat);
			Mat SR_1, SR_2;
			SR_1 = SRDashConcat.col(0);
			SR_2 = SRDashConcat.col(1);
			Scalar SR_1Mean, SR_2Mean, SR_1Stdev, SR_2Stdev;
			cv::meanStdDev(SR_1, SR_1Mean, SR_1Stdev);
			cv::meanStdDev(SR_2, SR_2Mean, SR_2Stdev);
			float SR_1std, SR_2std;
			SR_1std = sum(SR_1Stdev)[0];
			SR_2std = sum(SR_2Stdev)[0];

			//Calculate pulse vector 
			Mat pulseVector;
			pulseVector = SR_1 - (SR_1std / SR_2std) * SR_2;

			//Calculate long-term pulse vector over successive strides using overlap-adding
			Mat tempPulse = pulseVector - mean(pulseVector);

			if (firstStride) {
				longTermPulseVector = tempPulse;
				firstStride = false;
			}
			else {
				longTermPulseVector.push_back(float(0));
				int y = 0;
				for (size_t x = t - strideLength; x < (i + 1); x++) {
					longTermPulseVector.at<float>(x) = longTermPulseVector.at<float>(x) + tempPulse.at<float>(y);
					y++;
				}
			}

			SRDash.clear();
		}
	}
	//clean memory registers
	eigVecArray.clear();
	eigValArray.clear();
	SRDash.clear();

	return longTermPulseVector;
}

/* Function rPPGEstimate
* This function does filtering of the signal acquired from 2SR
* Input:
* - LongTermPulseVector obtained from 2SR analysis;
* - Number of frames in this recording
* Output:
* - Time Vector Unique to rPPG (vector<double>)
* - Raw Signal (vector<double>)
* - Filtered Signal (vector<double>)
*/
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, double>
rPPGEstimate(Mat SRResult, int numFrames) {

	std::vector <double> x, raw_rPPG_signal;
	for (int i = 0; i < numFrames; i++) {
		double t = timeVec[i] / 1000;
		double sig = (double)SRResult.at<float>(i);

		x.push_back(t);
		raw_rPPG_signal.push_back(sig);
		//public memeory allocation (consider optimising for local memory)
		rPPGSignal.push_back(sig);
		timeVecOutput.push_back(t);
	}

	std::vector<double> fOut = sosFilter(raw_rPPG_signal);

	std::vector <double> filtered_rPPG_signal;
	for (int m = 0; m < numFrames; m++) {
		double sig = fOut[m];
		filtered_rPPG_signal.push_back(sig);
		rPPGSignalFiltered.push_back(sig);
	}

	//peak detection 
	std::vector<int> peakLocs;
	Peaks::findPeaks(filtered_rPPG_signal, peakLocs);
	//peak distance calculation
	std::vector<double> diffTime;
	for (size_t i = 0; i < peakLocs.size(); i++) {
		if (peakLocs[i] != numFrames && x[peakLocs[i]] > 1) {
			diffTime.push_back(x[peakLocs[i]]);
		}
		else {

		}
		//std::cout << "time: " << x[peakLocs[i]] << "peak location at: " << peakLocs[i] << std::endl;
	}

	//HR estimation via peak detection 
	//(not the best way, but FFT will result in higher inaccuracies due to poor temporal frequency resolution)
	diff(diffTime, diffTime);
	double mean_diffTime = std::accumulate(diffTime.begin(), diffTime.end(), 0.0) / diffTime.size();
	std::cout << "rPPG - Average time between peaks: " << mean_diffTime << std::endl;
	std::cout << "rPPG - Estimated heart rate: " << 60.0 / mean_diffTime << std::endl;
	double heartRate = 60.0 / mean_diffTime; //update the current heart rate estimate

	if (mean_diffTime < 0) {
		for (size_t i = 0; i < peakLocs.size(); i++) {
			std::cout << "time: " << x[peakLocs[i]] << "peak location at: " << peakLocs[i] << std::endl;
		}
	}

	return std::make_tuple(x, raw_rPPG_signal, filtered_rPPG_signal, heartRate);

}


// rBCG estimation functions
/* Function KLTEstimate
* Input:
* - Recorded frames in the past window of frames
* - Analysed region of interests for the face
* - The current number of frames
* Output:
* - The filtered signal vector
* - The raw signal vector (mean-subtracted version)
* - The unique time vector
* - Estimated heart rate (double)
*/
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
KLTEstimate(std::deque<Mat> frameQ, std::vector<Rect> ROI_queue, int numFrames) {

	const int POINTS_UPPER_LIM = 20;
	double** BCGResult = new double* [numFrames];
	for (int i = 0; i < numFrames; i++) {
		BCGResult[i] = new double[POINTS_UPPER_LIM];
	}

	//Define a criteria for opticalFlow
	TermCriteria termCrit(TermCriteria::COUNT + TermCriteria::EPS, 35, 0.05);
	Size subPixWinSize(10, 10), winSize(31, 31);
	int MAX_COUNT = 1000;
	std::vector<Point2f> points[2];
	bool initialising = true;

	Mat temp;
	double scaleFactor = 1.0 / 2.0;
	Rect ROI, ROI_postInit;
	Mat greyFrame, prevGrey;
	bool reinit = false;

	//run after the frame has finished capturing
	for (int id = 0; id < frameQ.size(); id++) {

		if (!frameQ[id].empty()) {
			//keep a golden frame
			if (id < 10) {
				ROI = ROI_queue[id];
			}
			else {
				ROI_postInit = ROI_queue[id];

				// finding suitable region for rBCG feature extraction
				Point2i tempTL, tempBR;
				int tlX, tlY, brX, brY;
				tlX = ROI_postInit.tl().x;
				tlY = ROI_postInit.tl().y;
				brX = ROI_postInit.br().x;
				brY = ROI_postInit.br().y;

				tempTL.x = tlX - 100;
				if (tempTL.x <= 0) {
					tempTL.x = ROI_postInit.tl().x;
				}
				tempTL.y = tlY - 100;
				if (tempTL.y <= 0) {
					tempTL.y = ROI_postInit.tl().y;
				}
				tempBR.x = brX + 100;
				if (tempBR.x >= frameQ[id].cols) {
					tempBR.x = ROI_postInit.br().x;
					std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
				}
				tempBR.y = brY + 100;
				if (tempBR.y >= frameQ[id].rows) {
					tempBR.y = ROI_postInit.br().y;
					std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
				}

				ROI = Rect(tempTL, tempBR);
			}

			//convert to greyscale
			cv::cvtColor(frameQ[id], greyFrame, COLOR_BGR2GRAY);

			if (initialising || reinit) {
				goodFeaturesToTrack(greyFrame, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
				cornerSubPix(greyFrame, points[1], subPixWinSize, Size(-1, -1), termCrit);
				reinit = false;
			}
			else if (!points[0].empty()) {
				std::vector<uchar> status;
				std::vector<float> err;

				if (prevGrey.empty())
					greyFrame.copyTo(prevGrey);

				calcOpticalFlowPyrLK(prevGrey, greyFrame, points[0], points[1], status, err, winSize,
					3, termCrit, 0, 0.001);
				size_t i, k;
				for (i = k = 0; i < points[1].size(); i++) {
					if (!status[i])
						continue;
					//checks to see if the points are in the ROI of the face, if it is not then we reject it by not keeping it.
					if (ROI.contains(points[1][i])) {
						points[1][k++] = points[1][i];
						circle(frameQ[id], points[1][i], 3, Scalar(0, 255, 0), -1, 8);
					}
					else {
					}
				}
				points[1].resize(k);
				if (points[1].size() > POINTS_UPPER_LIM) {
					for (int i = 0; i < (points[1].size() - POINTS_UPPER_LIM); i++) {
						points[1].pop_back();
					}
				}
				else if (points[1].empty()) {
					reinit = true;
				}
				else {
					for (size_t j = 0; j < points[1].size(); j++) {
						BCGResult[id][j] = static_cast<double>(points[1][j].y);
					}
				}
			}
			if (reinit) {
				initialising = true;
			}
			else {
				initialising = false;
			}

			if (DEBUG_MODE2) {
				cv::rectangle(frameQ[id], ROI, Scalar(0, 255, 0));
				cv::resize(frameQ[id], temp, cv::Size(), scaleFactor, scaleFactor);

				cv::imshow("Detected Frame", temp);
				if (cv::waitKey(1) > 0) break;
			}
			//frameQ.push_back(newFrame.clone());
			std::swap(points[1], points[0]);

			cv::swap(prevGrey, greyFrame);

		}
	}

	std::vector<std::vector<double>> resultVector;
	// for the first few frames, the points may disappear, which may cause large discrepancies in the results
	// so if any vector appears to have 0 value, then we ignore these vectors all together
	// this tells me how much I need to take away from the timeVector
	int startingNoise = 15;
	int p = 0, q = 0;
	int tempUpperLim = POINTS_UPPER_LIM;
	bool foundNoise = false;
	for (p = 0; p < tempUpperLim; ++p) {
		//for each point we make a vector in time 
		for (int _x = 0; _x < 100; ++_x) {
			if (round(BCGResult[_x][p]) == 0) {
				startingNoise = _x;
			}
		}
	}
	
	// Determine the location where points are allocated to
	// Potential issue may arise if we don't have a good signal of detected points
	for (p = 0; p < tempUpperLim; ++p) {
		std::vector<double> tempRow;
		for (q = startingNoise + 1; q < numFrames; ++q) {
			//add to the point in the vector
			tempRow.push_back(BCGResult[q][p]);
			//check if there's any discontinuity in the frame before the end
			if (round(BCGResult[q][p]) == 0.0) {
				
				tempUpperLim = p;
				tempRow.clear(); // dump the row entirely
				break;
			}
		}
		if (!tempRow.empty()) {
			resultVector.push_back(tempRow);
		}

	}

	//mean for each point across the entire section is:
	std::vector<double> pointAvgs;
	double pointAvg_temp = 0;
	for (int x = 0; x < tempUpperLim; x++) {
		pointAvg_temp = std::accumulate(resultVector[x].begin(), resultVector[x].end(), 0.0) / resultVector[x].size();
		pointAvgs.push_back(pointAvg_temp);
	}

	std::vector<double> t;
	for (int x = startingNoise + 1; x < numFrames; x++) {
		double t_ = timeVec[x] / 1000.0;
		t.push_back(t_);//get timevector for plotting rBCG
	}

	//mean subtraction for each point average
	std::vector<std::vector<double>> sig_lessMean;
	sig_lessMean.resize(resultVector.size());
	for (int y = 0; y < tempUpperLim; y++) {
		double res = 0.0;
		for (int x = 0; x < resultVector[y].size(); x++) {
			res = resultVector[y][x] - pointAvgs[y]; //mean subtraction
			sig_lessMean[y].push_back(res);
		}

	}
	std::vector<double> sig_avg;
	for (int x = 0; x < resultVector[0].size(); x++) {
		double temp = 0.0;

		for (int y = 0; y < tempUpperLim; y++) {
			temp += sig_lessMean[y][x];
		}
		temp = temp / tempUpperLim;
		sig_avg.push_back(temp);
	}

	//Filter the signal with a IIR Butterworth Bandpass Filter
	std::vector<double> filtSig = sosFilter(sig_avg);

	for (int i = 0; i < sig_lessMean.size(); i++) {
		double temp = filtSig[i];
		rBCGSignalFiltered.push_back(temp);
	}

	//peak detection 
	std::vector<int> peakLocs;
	Peaks::findPeaks(filtSig, peakLocs);
	
	//peak-to-peak distance calculation
	std::vector<double> diffTime;
	for (size_t i = 0; i < peakLocs.size(); i++) {
		if (peakLocs[i] != numFrames && peakLocs[i] != 0 && (t[peakLocs[i]] > 0.0)) {
			diffTime.push_back(t[peakLocs[i]]);
		}
		else {
		}
		//std::cout << "time: " << t[peakLocs[i]] << "peak location at: " << peakLocs[i] << std::endl;
	}
	
	diff(diffTime, diffTime);
	double mean_diffTime = std::accumulate(diffTime.begin(), diffTime.end(), 0.0) / diffTime.size();
	std::cout << "rBCG - Average time between peaks: " << mean_diffTime << std::endl;
	double heartRate = 60.0 / mean_diffTime;
	//this estimation will almost always be poor
	std::cout << "rBCG - Estimated from Peak Detection heart rate: " << heartRate << std::endl;

	return std::make_tuple(filtSig, sig_avg, t);
}