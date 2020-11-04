#include "opencv_helper.h"

using namespace cv;
using namespace std;

typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

Rect detectAndDisplay(
	Mat frame);

Rect findPoints(
	vector<Rect> faces,
	int bestIndex,
	double scaleFactor);

void write_CSV(string filename, vector<float> arr1, vector<float> arr2, vector<float> time);

CascadeClassifier face_cascade;

deque<double> topLeftX;
deque<double> topLeftY;
deque<double> botRightX;
deque<double> botRightY;
vector<float> pointsX;
vector<float> pointsY;

VideoCapture capture(1);

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
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	
	int fs = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int nFrames = capture.get(CAP_PROP_FRAME_COUNT);

	cout << "Sampling rate: " << fs << endl
		<< "width: " << width << endl
		<< "height :" << height << endl;

	capture.set(CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(CAP_PROP_FRAME_HEIGHT, 1080);

	Mat frame;
	deque<Mat> frameQ;
	deque<float> timeVector;

	if (!capture.isOpened()) {
		return 0;
	}
	bool initialRun = true;
	bool capturing = true;
	//Define a criteria for opticalFlow
	TermCriteria termCrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	int MAX_COUNT = 1000;
	vector<Point2f> points[2];
	bool initialising = true;
	// Creating the CSV file to write location data into
	ofstream myFile;
	myFile.open("rBCG_trials_live.csv");
	Rect ROI;
	Mat greyFrame, prevGrey, goldFrame;

	size_t capLen = 500;

	while (true) {

		Clock::time_point start = Clock::now();

		if (capture.read(frame) && capturing) {
			if (!frame.empty()) {
				imshow("frame", frame);
				frameQ.push_back(frame.clone());
			}
			if (frameQ.size() > capLen) {
				capturing = false;
				cout << "frame Q has reached 300 and processing now" << endl;
			}
		}
		Clock::time_point end = Clock::now();
		auto ms = chrono::duration_cast<milliseconds>(end - start);
		timeVector.push_back(ms.count());


		if (cv::waitKey(1) > 0) {
			break;
		}

		if (frameQ.size() > 100 && initialRun) {
			frameQ.clear();
			timeVector.clear();
			initialRun = false;
			cout << "frame queue has been cleared" << endl;
		}
		
		if (frameQ.size() > capLen) {
			cout << "entered into processing" << endl;
			for (size_t i = 0; i < frameQ.size(); i++) {
				Mat temp = frameQ[i];
				if (!temp.empty()) {
					//keep a golden frame
					temp.copyTo(goldFrame);
					ROI = detectAndDisplay(temp);
					//convert to greyscale
					cv::cvtColor(goldFrame, greyFrame, COLOR_BGR2GRAY);

					if (initialising) {
						goodFeaturesToTrack(greyFrame, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
						cornerSubPix(greyFrame, points[1], subPixWinSize, Size(-1, -1), termCrit);
					}
					else if (!points[0].empty()) {
						vector<uchar> status;
						vector<float> err;
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
						if (points[1].size() > 11) {
							points[1].pop_back();
						}
						else if (points[1].size() == 0) {
							initialising = true;
						}
						else {
							for (size_t i = 0; i < points[1].size(); i++) {
								myFile << i << ";" << points[1][i].y << ";";
							}
							myFile << timeVector.front() << endl;
							timeVector.pop_front();
						}
					}

					initialising = false;

					cv::rectangle(temp, ROI, Scalar(0, 255, 0));
					cv::imshow("detected frame", temp);
					//frameQ.push_back(newFrame.clone());
					std::swap(points[1], points[0]);
					cv::swap(prevGrey, greyFrame);
					
					char c = (char)waitKey(5);
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
				timeVector.pop_front();
			}
			myFile << endl;
			capturing = true;
			frameQ.clear();
			break; // otherwise this will go on indefinitely
		}
	}

	capture.release();

	cout << "function successfully terminated" << endl;
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
			tempTL.y = tlY + (brY - tlY) * 0.5;
			if (tempTL.y <= 0) {
				tempTL.y = faceROI.tl().y;
			}
			tempBR.x = brX - (brX - tlX) * 0.2;
			if (tempBR.x >= frameClone.cols) {
				tempBR.x = faceROI.br().x;
				std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
			}
			tempBR.y = brY - (brY - tlY) * 0.1;
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
Rect findPoints(vector<Rect> faces, int bestIndex, double scaleFactor) {

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



/** Function: write_CSV
	Input: filename, vector<double> of numbers
	Output: prints to csv file the results
*/
void write_CSV(string filename, vector<float> arr1, vector<float> arr2, vector<float> time) {
	ofstream myfile;
	myfile.open(filename.c_str());
	int vsize = arr1.size();
	for (int n = 0; n < vsize; n++) {
		myfile << n << ";" << arr1[n] << ";" << arr2[n] << ";" << time[n] << endl;
	}
}


