#include "opencv_helper.h"

using namespace cv;
using namespace std;

void getColorValues(
	deque<Mat> fQueue);

Mat detectAndDisplay(
	deque<Mat> fQueue);

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
	int colorSelect = 2;
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

	//introduce queues
	deque<Mat> frameQueue;
	deque<Mat> ROIQueue;
	
	Mat newFrame;
	
	int maxBufferSize = 200;
	int minBufferSize = 60;
	int windowSize = 12; //minimum 6 frames per window to detect heart rate pattern
	for (;;) {

		//get new frame
		capture.read(newFrame);
		if (!newFrame.empty()){
			//frame producer
			frameQueue.push_back(newFrame);
			
			//find the face ROI and add to co-ordinates face ROI queue
			Clock::time_point start = Clock::now();
			Mat tempFrame = detectAndDisplay(frameQueue);
			Clock::time_point end = Clock::now();
			milliseconds ms = chrono::duration_cast<milliseconds>(end-start);
			long long duration = ms.count();
			//cout << "duration for detection: " << duration << "ms" << endl;
			//faceROI producer
			
			if (!tempFrame.empty()) {
				ROIQueue.push_back(tempFrame);
				getColorValues(ROIQueue);
			}
			
			cout << "frameQueue is at: " << frameQueue.size() << endl;
			cout << "ROIQueue is at: " << ROIQueue.size() << endl;
						
		}


		
		if (frameQueue.size() > maxBufferSize) {
			cout << "Too many frames being queued, deleting down to min size..." << endl;
			while (frameQueue.size() > minBufferSize) {
				frameQueue.pop_front();
			}
		}
		if (ROIQueue.size() > maxBufferSize) {
			cout << "Too many ROIs being queued, deleting down to min size..." << endl;
			while (ROIQueue.size() > minBufferSize) {
				ROIQueue.pop_front();
			}
		}
		if (waitKey(1) > 0) {
			break;
		}
	}
	return 1;
}

/** Function: getColorValues
	takes the current frameWindow of frames alongside the current ROI 
	and calculates the mean color values in all three color channels 
	and passes these values to the output csv
*/
void getColorValues(deque<Mat> fQueue) {
	while (!fQueue.empty()) {
		Mat procFrame = fQueue.front();
		vector<Mat> temp;
		split(procFrame, temp);	//resplits the channels (extracting the color green for default/testing cases)
		Scalar averageColorb = mean(temp[0]); //takes the average of the color along a selected spectrum B/R/G
		double blue = sum(averageColorb)[0];
		Scalar averageColorg = mean(temp[1]); //takes the average of the color along a selected spectrum B/R/G
		double green = sum(averageColorg)[0];
		Scalar averageColorr = mean(temp[2]); //takes the average of the color along a selected spectrum B/R/G
		double red = sum(averageColorr)[0];

		/*OUTPUT_CSV_VAR.push_back(blue);
		OUTPUT_CSV_VAR.push_back(green);
		OUTPUT_CSV_VAR.push_back(red); */
		fQueue.pop_front();

	}
}

/** to detect a face in a video feed from camera and
	return the frame of information which contains the color information in it.
	
*/
Mat detectAndDisplay(deque<Mat> fQueue) {

	Mat frameClone = fQueue.front().clone();
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

	fQueue.pop_front();
	return calcFrame;
}

/** locates the best face ROI coordinates
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
	int frameWindow = 18;
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
