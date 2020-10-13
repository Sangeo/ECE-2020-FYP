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

std::vector<double> sosFilter(
	std::vector<double> signal);

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
				// this tells me how much I need to take away from the timeVector
				int startingNoise = 20;
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
				//std::cout << "starting position: " << startingNoise;
				for (p = 0; p < tempUpperLim; ++p) {
					std::vector<double> tempRow;
					for (q = startingNoise + 1; q < numFrames; ++q) {
						//add to the point in the vector
						tempRow.push_back(BCGResult[q][p]);
						//check if there's any discontinuity in the frame before the end
						if (round(BCGResult[q][p]) == 0.0) {
							std::cout << "hit a 0 at row: " << p
								<< "time: " << q << std::endl;
							tempUpperLim = p;
							tempRow.clear(); // dump the row entirely
							break;
						}
					}

					if (!tempRow.empty()) {
						resultVector.push_back(tempRow);
						//std::cout << "resultVector size is:" << resultVector.size() << std::endl;
					}

				}
				std::vector<double> t, sig;
				for (int x = startingNoise + 1; x < numFrames; x++) {
					double t_ = timeVec[x] / 1000.0;
					double sig_ = 0;
					for (int y = 0; y < tempUpperLim; y++) {
						double temp = resultVector[y][x - (startingNoise + 1)];
						sig_ += temp;
					}
					sig_ /= tempUpperLim;
					sig.push_back(sig_);
					t.push_back(t_);
				}

				double sig_mean = std::accumulate(sig.begin(), sig.end(), 0.0) / sig.size();

				std::vector<double> sig_lessMean;
				for (int i = 0; i < sig.size(); i++) {
					double temp = sig[i] - sig_mean;
					sig_lessMean.push_back(temp);
					//std::cout << "i: " << i << "signal value: " << temp << std::endl;
				}

				if (DEBUG_MODE) {
					//std::cout << "sig_mean: " << sig_mean << std::endl;
					//std::cout << "t size: " << t.size() << "sig vec size: " << sig_lessMean.size() << std::endl;
				}
				std::vector<double> filtSig = sosFilter(sig_lessMean);
				
				//peak detection 
				std::vector<int> peakLocs;
				Peaks::findPeaks(filtSig, peakLocs);
				std::cout << "peakLocs size: " << peakLocs.size() << std::endl;
				//peak distance calculation
				std::vector<double> diffTime;
				for (size_t i = 0; i < peakLocs.size(); i++) {
					diffTime.push_back(t[peakLocs[i]]);
					std::cout << "time: " << t[peakLocs[i]] << "peak location at: " << peakLocs[i] << std::endl;
				}
				diff(diffTime, diffTime);
				double mean_diffTime = std::accumulate(diffTime.begin(), diffTime.end(), 0.0) / diffTime.size();
				std::cout << "Average time between peaks: " << mean_diffTime << std::endl;
				std::cout << "Estimated heart rate: " << 60.0 / mean_diffTime << std::endl;

				plt::figure(0);
				if (!sig.empty()) {
					plt::subplot(2, 1, 1);
					plt::plot(t, sig_lessMean);
					plt::xlabel("time (seconds)");
					plt::ylabel("nominal displacement (pixels)");
					plt::subplot(2, 1, 2);
					plt::plot(t, filtSig);
					plt::xlabel("time (seconds)");
					plt::ylabel("nominal displacement (pixels)");
				}
				plt::show();

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
			tempTL.y = tlY + (brY - tlY) * 0.48;
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

