/** Motive: to capture color signal information from a selected ROI from the frame
	Jerry Li
*/
#include "opencv_helper.h"

using namespace cv;
using namespace std;

Mat skinDetection(
	Mat frame,
	Rect originalFaceRect,
	Scalar lBound,
	Scalar uBound);

void detectAndDisplay(
	Mat frame,
	int c);

void write_CSV(
	string filename,
	vector<double> arr,
	double fps);

Rect findPoints(
	vector<Rect> faces,
	int bestIndex,
	double scaleFactor);

vector<double> OUTPUT_CSV_VAR;
deque<double> topLeftX;
deque<double> topLeftY;
deque<double> botRightX;
deque<double> botRightY;

CascadeClassifier face_cascade;
typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

int main(int argc, const char** argv) {
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

	//-- 2. Read the video stream
	cv::VideoCapture capture(0); //inputs into the Mat frame as CV_8UC3 format (unsigned integer)
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(CAP_PROP_AUTOFOCUS, 0);
	capture.set(CAP_PROP_ISO_SPEED, 400);
	// check that the camera is open
	if (!capture.isOpened())
	{
		std::cout << "--(!)Error opening video capture\n";
		return -1;
	}
	//prompt to select color
	cout << "Please select the color for signal decomposition\n";
	cout << "1 for blue, 2 for green, 3 for red : \n";
	int color_sel;
	cin >> color_sel;


	Mat frame;
	vector<int> topLeftX;
	vector<int> topLeftY;
	vector<int> botRightX;
	vector<int> botRightY;


	for (;;)
	{
		if (capture.read(frame))
		{
			detectAndDisplay(frame, color_sel);
		}

		if (cv::waitKey(1) >= 0)
		{
			break; // if escape is pressed at any time
		}
	}
	long long fps = 30; // gives frames per second
	write_CSV("output_file2.csv", OUTPUT_CSV_VAR, fps);

}

/** to detect a specified region of interest in a video feed from camera
*/
void detectAndDisplay(Mat frame, int cSel) {

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

	if (!faces.empty()) {
		//find the faceROI using a moving average to ease the noise out	
		Rect faceROI = findPoints(faces, bestIndex, scaleFactor);

		if (!faceROI.empty()) {
			rectangle(frameClone, faceROI, Scalar(0, 0, 255), 1, LINE_4, 0);

			// locate region of interest (ROI) for the forehead
			Point innerTopLC(faceROI.x + (faceROI.width * 3 / 10), faceROI.y + (faceROI.height / 20));
			Point innerBotRC(faceROI.x + (faceROI.width * 7 / 10), faceROI.y + (faceROI.height * 1 / 5));
			Rect myROI(innerTopLC, innerBotRC);
			rectangle(frameClone, myROI, Scalar(0, 0, 255), 1, LINE_8, 0);

			Mat resultFrame = frameClone(myROI);
			Mat colorImg;
			vector<Mat> temp;
			split(resultFrame, temp);	//resplits the channels (extracting the color green for default/testing cases)
			colorImg = temp[cSel - 1];

			Scalar averageColor = mean(temp[cSel - 1]); //takes the average of the color along a selected spectrum B/R/G
			double s = sum(averageColor)[0];
			OUTPUT_CSV_VAR.push_back(s);
		}
	}
	imshow("frame", frameClone);
}

Rect findPoints(vector<Rect> faces, int bestIndex, double scaleFactor) {
	//locates the best face ROI coordinates
	double tlX = floor(faces[bestIndex].tl().x * (1 / scaleFactor));
	double tlY = floor(faces[bestIndex].tl().y * (1 / scaleFactor));
	double brX = floor(faces[bestIndex].br().x * (1 / scaleFactor));
	double brY = floor(faces[bestIndex].br().y * (1 / scaleFactor));

	//temporary variables
	double avgtlX;
	double avgtlY;
	double avgbrX;
	double avgbrY;
	int sumTX = 0;
	int sumTY = 0;
	int sumBX = 0;
	int sumBY = 0;
	//if the queue size is above a certain number we start to take the average of the frames
	int frameWindow = 30;
	if (topLeftX.size() >= frameWindow) {

		// Take the sum of all elements in the current frame window
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
	}

	return Rect();
}

void write_CSV(string filename, vector<double> arr, double fps) {
	ofstream myfile;
	myfile.open(filename.c_str());
	int vsize = arr.size();
	for (int n = 0; n < vsize; n++)
	{
		myfile << fps << ";" << n << ";" << arr[n] << endl;
	}
}


Mat skinDetection(Mat frameC, Rect originalFaceRect, Scalar lBound, Scalar uBound) {

	cv::Mat normalFace;
	cv::Mat faceRegion;
	cv::Mat YCrCbFrame;
	cv::cvtColor(frameC, YCrCbFrame, cv::COLOR_BGR2YCrCb);
	cv::Mat skinMask;
	cv::Mat skin;

	//zone out faceRegion
	faceRegion = YCrCbFrame(originalFaceRect).clone();
	cv::inRange(faceRegion, lBound, uBound, skinMask);

	//for the real frame output we use the camera feed
	normalFace = frameC(originalFaceRect).clone();

	//apply morphology to image to remove noise and ensure cleaner frame
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
	cv::erode(skinMask, skinMask, kernel, cv::Point(-1, -1), 1);
	cv::dilate(skinMask, skinMask, kernel, cv::Point(-1, -1), 1);

	cv::GaussianBlur(skinMask, skinMask, cv::Size(7, 7), 1, 1);

	//bitwise and to get the actual skin color back;
	cv::bitwise_and(normalFace, normalFace, skin, skinMask);

	return skin;
}
