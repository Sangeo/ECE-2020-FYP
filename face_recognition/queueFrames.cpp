#include "opencv_helper.h"
#include <cmath>


using namespace cv;
using namespace std;

deque<double> getColorValues(
	Mat frame);

Mat detectAndDisplay(
	Mat frame);

void write_CSV(
	string filename,
	deque<double> arr,
	double fps);

Rect findPoints(
	vector<Rect> faces,
	int bestIndex,
	double scaleFactor);

CascadeClassifier face_cascade;
deque<double> OUTPUT_CSV_VAR;
deque<double> topLeftX;
deque<double> topLeftY;
deque<double> botRightX;
deque<double> botRightY;

constexpr int maxColors = 3;

typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

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
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	//Get video input/output setup
	cv::VideoCapture capture(0);
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(CAP_PROP_AUTOFOCUS, 0);
	capture.set(CAP_PROP_ISO_SPEED, 400);

	// introduce queues
	deque<double> blue, green, red;

	// current frame
	Mat newFrame;
	int maxBufferSize = 300;
	int windowSize = 10; //minimum 6 frames per window to detect heart rate pattern
	bool initializing = true;

	for (;;) {
		// if a frame is captured, decoded and stored into the newFrame matrix
		if (capture.read(newFrame)) {
			Mat temp;
			//find the face and take the faceROI and output as a frame of just forehead region pixels
			temp = detectAndDisplay(newFrame);
			//average of current colors, output with a rawData vector containing red, green, and blue values separately
			deque<double> rawData = getColorValues(temp);
			if (!rawData.empty()) {
				blue.push_back(rawData[0]);
				green.push_back(rawData[1]);
				red.push_back(rawData[2]);
			}
		}

		// if it is the first 5 seconds of the video recording, get rid of it so we don't get any spikes in readings
		if (initializing) {
			if (blue.size() > 150) {
				while (!blue.empty()) {
					blue.pop_front();
					green.pop_front();
					red.pop_front();
				}
				initializing = false;
			}
		}
		//if there is sufficient information to process, create a sliding window with length l (enough to contain one heart beat)
		//sliding window implementation
		if ((green.size() > windowSize) && (initializing == false)) {

			Mat bgrIntensityz(windowSize, maxColors, CV_64F);
			for (size_t i = 0; i < windowSize; i++) {
				for (size_t j = 0; j < maxColors; j++) {
					if (j == 0) {
						bgrIntensityz.at<double>(i,j) = blue[i];
					}
					else if (j == 1) {
						bgrIntensityz.at<double>(i,j) = green[i];
					}
					else if (j == 2) {
						bgrIntensityz.at<double>(i, j) = red[i];
					}
				}
			}
			//cout << "M = " << endl << " " << bgrIntensityz << endl << endl;
			green.pop_front();
			blue.pop_front();
			red.pop_front();

		}

		//get rid of the buffer if it exceeds a certain number of frames (contingency planning)
		if (blue.size() > maxBufferSize) {
			cout << "exceeded buffer size!! deleting information now!!" << endl;
			while (!blue.empty()) {
				blue.pop_front();
				green.pop_front();
				red.pop_front();
			}
		}
		if (waitKey(1) > 0) {
			break;
		}
	}
	//write_CSV("output_file2.csv", OUTPUT_CSV_VAR, 30);

	return 1;
}

/** Function: getColorValues
	takes the current frameWindow of frames alongside the current ROI
	and calculates the mean color values in all three color channels
	and passes these values to the output csv
*/
deque<double> getColorValues(Mat frame) {
	double blue, green, red;
	deque<double> output;
	if (!frame.empty()) {
		Mat procFrame = frame;
		
		vector<Mat> temp;
		split(procFrame, temp);	//resplits the channels (extracting the color green for default/testing cases)

		Scalar averageColorb = mean(temp[0]); //takes the average of the color along a selected spectrum B/R/G
		double blue = sum(averageColorb)[0];
		Scalar averageColorg = mean(temp[1]); //takes the average of the color along a selected spectrum B/R/G
		double green = sum(averageColorg)[0];
		Scalar averageColorr = mean(temp[2]); //takes the average of the color along a selected spectrum B/R/G
		double red = sum(averageColorr)[0];
		output = { blue, green, red };
		/*OUTPUT_CSV_VAR.push_back(blue);
		OUTPUT_CSV_VAR.push_back(green);
		OUTPUT_CSV_VAR.push_back(red); */
		
		//cout << "b: " << blue << " g: " << green << " r: " << red << endl;
	}

	return output;
}

/** to detect a face in a video feed from camera and
	return the frame of information which contains the color information in it.

*/
Mat detectAndDisplay(Mat frame) {

	Mat frameClone = frame.clone();
	Mat procFrame;
	Mat frameGray;
	std::vector<Rect> faces;
	std::vector<int> numDetections;

	//resizing image before processing
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
	Rect myROI;
	Rect faceROI;
	Mat calcFrame;
	if (!faces.empty()) {
		//find the faceROI using a moving average to ease the noise out	
		faceROI = findPoints(faces, bestIndex, scaleFactor);

		if (!faceROI.empty()) {
			// draws on the current face region of interest
			rectangle(frameClone, faceROI, Scalar(0, 0, 255), 1, LINE_4, 0);
			// locate region of interest (ROI) for the forehead
			Point innerTopLC(faceROI.x + (faceROI.width * 3 / 10), faceROI.y + (faceROI.height / 20));
			Point innerBotRC(faceROI.x + (faceROI.width * 7 / 10), faceROI.y + (faceROI.height * 1 / 5));
			Rect myROI(innerTopLC, innerBotRC);
			// draws on the current region of interest for forehead
			rectangle(frameClone, myROI, Scalar(0, 0, 255), 1, LINE_8, 0);
			calcFrame = frameClone(myROI);
		}
	}
	imshow("frame", frameClone);

	return calcFrame;
}

/** Function findPoints
	locates the best face ROI coordinates using a simple moving average to eliminate noise
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
	int frameWindow = 8;
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

void write_CSV(string filename, deque<double> arr, double fps) {
	ofstream myfile;
	myfile.open(filename.c_str());
	int vsize = arr.size();
	for (int n = 0; n < vsize; n++)
	{
		myfile << fps << ";" << n << ";" << arr[n] << endl;
	}
}
