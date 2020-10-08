#include "opencv_helper.h"

using namespace cv;
namespace plt = matplotlibcpp;

constexpr auto M_PI = 3.141592653589793;

CascadeClassifier face_cascade;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;
const bool DEBUG_MODE = false;
const bool DEBUG_MODE2 = false;

std::deque<double> topLeftX;
std::deque<double> topLeftY;
std::deque<double> botRightX;
std::deque<double> botRightY;
std::vector<double> timeVec;


int strideLength = 45; // size of the temporal stride 
					//(45 frames as this should capture the heart rate information)
bool firstStride = true;

Rect detectAndDisplay(
	Mat frame);

Rect findPoints(
	std::vector<Rect> faces,
	int bestIndex,
	double scaleFactor);

void opticalFlowCalc(
	std::deque<Mat> frameQ);

std::deque<double> filterDesign(
	double f1,
	double f2,
	double fs,
	double N);

std::deque<double> convFunc(
	std::deque<double> a,
	std::deque<double> b,
	int aLen,
	int bLen);

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
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	//Get video input/output setup
	cv::VideoCapture capture(1); // 1 for front facing webcam
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(CAP_PROP_FRAME_HEIGHT, 1080);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(CAP_PROP_AUTOFOCUS, 1);
	capture.set(CAP_PROP_AUTO_WB, 0);

	std::cout << "CAPTURE FOMRAT IS: " << capture.get(CAP_PROP_FORMAT) << std::endl;
	if (!capture.isOpened()) {
		return -1;
	}
	//capture.set(CAP_PROP_FOURCC, ('D', 'I', 'V', 'X'));

	Mat frame;
	std::deque<Mat> frameQ;


	int numFrames; //15 seconds worth of frames
	const int FPS = capture.get(CAP_PROP_FPS);
	const int msPerFrame = 33;

	std::cout << "Frames per second according to capture.get(): " << FPS << std::endl;
	bool recording = true;
	bool processing = false;
	bool firstTime = true;


	while (true) {

		if (recording && !processing) {
			if (firstTime) {
				numFrames = 185;
			}
			else {
				numFrames = 450;
			}
			for (size_t i = 0; i < numFrames; i++) {

				Clock::time_point start = Clock::now();

				if (capture.read(frame)) {
					if (!frame.empty()) {
						frameQ.push_back(frame.clone());
					}
				}
				else {
					return -10;
				}

				Clock::time_point end = Clock::now();
				auto ms = std::chrono::duration_cast<milliseconds>(end - start);
				double captureTime = ms.count() / 1000;

				if (!frame.empty()) {
					imshow("Live-Camera Footage", frame);
				}
				if (cv::waitKey(1) > 0) break;

				if (DEBUG_MODE) {
					std::cout << "time taken to record frame " << i << "is : "
						<< captureTime << "ms " << std::endl;
				}

				if (!timeVec.empty()) {
					auto cTime = timeVec.back() + ms.count();
					timeVec.push_back(cTime);
				}
				else {
					timeVec.push_back(ms.count());
				}
			}
			cv::destroyWindow("Live-Camera Footage");

			//handover to processing
			recording = false;
			processing = true;
		}

		else if (processing && !recording) {
			//initialising 
			if (firstTime) {
				frameQ.clear();
				timeVec.clear();
				firstTime = false;
			}
			else {
				const int POINTS_UPPER_LIM = 10;
				double** BCGResult = new double* [numFrames];
				for (int i = 0; i < numFrames; i++) {
					BCGResult[i] = new double[POINTS_UPPER_LIM];
				}

				//Define a criteria for opticalFlow
				TermCriteria termCrit(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03);
				Size subPixWinSize(10, 10), winSize(31, 31);
				int MAX_COUNT = 1000;
				std::vector<Point2f> points[2];
				bool initialising = true;

				Rect ROI;
				Mat greyFrame, prevGrey, goldFrame;
				bool reinit = false;

				//run after the frame has finished capturing
				for (size_t id = 0; id < frameQ.size(); id++) {
					Mat temp = frameQ[id];
					if (!temp.empty()) {
						//keep a golden frame
						temp.copyTo(goldFrame);
						ROI = detectAndDisplay(temp);
						//convert to greyscale
						cv::cvtColor(goldFrame, greyFrame, COLOR_BGR2GRAY);

						if (initialising || reinit) {
							goodFeaturesToTrack(greyFrame, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
							cornerSubPix(greyFrame, points[1], subPixWinSize, Size(-1, -1), termCrit);

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
									circle(temp, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
								}
								else {

								}
							}
							points[1].resize(k);
							if (points[1].size() > POINTS_UPPER_LIM) {
								points[1].pop_back();
							}
							else if (points[1].size() == 0) {
								reinit = true;
							}
							else {
								for (size_t j = 0; j < points[1].size(); j++) {
									BCGResult[id][j] = static_cast<double>(points[1][j].y);
									//std::cout << static_cast<double>(points[1][i].y);
								}
							}
						}
						if (reinit) {
							initialising = true;
						}
						else {
							initialising = false;
						}

						cv::rectangle(temp, ROI, Scalar(0, 255, 0));
						cv::imshow("Detected Frame", temp);
						//frameQ.push_back(newFrame.clone());
						std::swap(points[1], points[0]);

						cv::swap(prevGrey, greyFrame);
						if (cv::waitKey(1) > 0) break;

						char c = (char)waitKey(10);
						if (c == ' ') break;
						switch (c) {
						case 'r': // re-initialise all points.
							initialising = true;
							break;
						case 'c': // clear all points
							points[0].clear();
							points[1].clear();
							break;

						}
					}
				}
				if (DEBUG_MODE) {
					for (int x = 0; x < numFrames; x++) {
						for (int y = 0; y < POINTS_UPPER_LIM; y++) {
							std::cout << "Row: " << x <<
								" Col: " << y << " Val: " <<
								BCGResult[x][y] << std::endl;
						}
						std::cout << std::endl;
					}
				}
				std::vector<std::vector<double>> resultVector;
				//for the first few frames, the points may disappear
				//so if any vector appears to have 0 value, then we ignore these vectors all together
				int ripItems = 0; // this tells me how much I need to take away from the timeVector
				int startingNoise = 20;
				int p = 0, q = 0;
				int tempUpperLim = POINTS_UPPER_LIM;

				for (p = 0; p < tempUpperLim; ++p) {
					//for each point we make a vector in time 
					std::vector<double> tempRow;
					for (int _x = 0; _x < 80; ++_x) {
						if (round(BCGResult[_x][p]) == 0) {
							startingNoise = _x;
						}
					}
					std::cout << "starting position: " << startingNoise;
					for (q = startingNoise + 1; q < numFrames; ++q) {
						//add to the point in the vector
						tempRow.push_back(BCGResult[q][p]);

						//check if there's any discontinuity in the frame
						if (round(BCGResult[q][p]) == (double)0) {
							std::cout << "hit a 0 at row: " << p
								<< "time: " << q << std::endl;
							tempUpperLim = p;
							tempRow.clear();
							break;
						}
					}

					if (!tempRow.empty()) {
						resultVector.push_back(tempRow);
					}

				}

				std::cout << "resultVector size is:" << resultVector.size() << std::endl;

			}

			cv::destroyWindow("Detected Frame");

			frameQ.clear();
			timeVec.clear();
			//handover to recording
			recording = true;
			processing = false;

		}


		if (DEBUG_MODE) {
			std::cout << "finished recording and processing N frames" << std::endl;
			std::cout << "going to repeat" << std::endl;
		}
	}

	frameQ.clear();
	capture.release();
	return 0;
}



/* Algorithm for FIR bandpass filter
Input:
f1, the lowest frequency to be included, in Hz
f2, the highest frequency to be included, in Hz
f_samp, sampling frequency of the audio signal to be filtered, in Hz
N, the order of the filter; assume N is odd
Output:
a bandpass FIR filter in the form of an N-element array
*/
std::deque<double> filterDesign(double f1, double f2, double fs, double N) {
	std::deque<double> output(N);
	std::cout << "entered filter design" << std::endl;
	double f1_c, f2_c, w1_c, w2_c;
	f1_c = f1 / fs;
	f2_c = f2 / fs;
	w1_c = 2 * M_PI * f1_c;
	w2_c = 2 * M_PI * f2_c;
	std::cout << "f1_c: " << f1_c << " "
		<< "f2_c" << f2_c << std::endl
		<< "w1_c" << w1_c << " "
		<< "w2_c" << w2_c << std::endl;

	int mid = N / 2; //integer division, drops the remainder.

	for (int i = 0; i < N; i++) {
		if (i == mid) {
			output[i] = (double)2 * f2_c - 2 * f1_c;
		}
		else {
			if (i < mid) {
				int i_low = -N / 2 + i;
				output[i] = (double)(std::sin(w2_c * i_low) / (M_PI * i_low))
					- (std::sin(w1_c * i_low) / (M_PI * i_low));
			}
			else if (i > mid) {
				int i_high = N / 2 + i;
				output[i] = (double)(std::sin(w2_c * i_high) / (M_PI * i_high))
					- (std::sin(w1_c * i_high) / (M_PI * i_high));
			}

		}
	}


	return output;
}

/* Convolution function
* input:
* vectors a and b
* length of vectors a and b
* output:
* vector of convolution result
*/
std::deque<double> convFunc(std::deque<double> a, std::deque<double> b, int aLen, int bLen) {

	std::cout << "entered convolution " << std::endl;
	int abMax = std::max(aLen, bLen);
	int convLen = aLen + bLen - 1;
	std::deque<double> output(convLen);

	for (int n = 0; n < convLen; n++) {
		float prod = 0;
		int kMax = std::min(abMax, n);
		for (int k = 0; k <= kMax; k++) {
			//make sure we're in bounds for both arrays
			if (k < aLen && (n - k) < bLen) {
				prod += a[k] * b[n - k];
			}
		}
		output[n] = prod;
	}
	return output;
}




/** Function detectAndDisplay
	Detects a face in a video feed from camera and
	return the frame of information which contains the color information in it.
	Input: current frame (raw)
	Output: current skin (raw);
*/
/** Function detectAndDisplay
	Detects a face in a video feed from camera and
	return the frame of information which contains the color information in it.
	Input: current frame (raw)
	Output: current skin (raw);
*/
Rect detectAndDisplay(Mat frame) {

	Mat frameClone = frame.clone();
	Mat procFrame;
	Mat frameGray;
	std::vector<Rect> faces;
	std::vector<int> numDetections;

	//downsizing image before processing
	const double scaleFactor = 1.0 / 8.0;
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

	Rect faceROI;
	Rect tempRect;

	if (!faces.empty()) {
		//find the faceROI using a moving average to filter the noise of the face detection method out 
		faceROI = findPoints(faces, bestIndex, scaleFactor);

		if (!faceROI.empty()) {
			// expands the detected ROI by fixed integer
			Point2i tempTL, tempBR;
			int tlX,
				tlY,
				brX,
				brY;
			tlX = faceROI.tl().x;
			tlY = faceROI.tl().y;
			brX = faceROI.br().x;
			brY = faceROI.br().y;


			tempTL.x = tlX + (brX - tlX) * 0.2;
			if (tempTL.x <= 0) {
				tempTL.x = faceROI.tl().x;
			}
			tempTL.y = tlY + (brY - tlY) * 0.55;
			if (tempTL.y <= 0) {
				tempTL.y = faceROI.tl().y;
			}
			tempBR.x = brX - (brX - tlX) * 0.2;
			if (tempBR.x >= frameClone.cols) {
				tempBR.x = faceROI.br().x;
				std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
			}
			tempBR.y = brY - (brY - tlY) * 0.05;
			if (tempBR.y >= frameClone.rows) {
				tempBR.y = faceROI.br().y;
				std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
			}

			tempRect = Rect(tempTL, tempBR);

		}
	}
	return tempRect;
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


void opticalFlowCalc(std::deque<Mat> frameQ) {


}