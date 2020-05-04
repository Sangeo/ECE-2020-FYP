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


#include "opencv_helper.h"

using namespace std;
using namespace cv;

void write_CSV(string filename, vector<double> arr, double fps);
void detectAndDisplay(Mat frame, int cSel, Scalar lower, Scalar upper);
Mat skinDetection(Mat frame, Rect originalFaceRect, Scalar lBound, Scalar uBound);
typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

CascadeClassifier face_cascade;
vector<double> OUTPUT_CSV_VAR;
int faceCascadeSize = 30;

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
	//set camera settings to capture at 720p 30fps
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(CAP_PROP_AUTOFOCUS, 0);
	capture.set(CAP_PROP_ISO_SPEED, 400);
	// check that the camera is open
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}
	// this frame will store all information about the video captured by the camera
	Mat frame;

	// these are the colors which will determine the skin section on my face
	// these values are in YCrCb format, which helps with determining skin color from luminance
	cv::Scalar lower, upper;
	lower = cv::Scalar(90, 133, 70);
	upper = cv::Scalar(255, 173, 127);

	Clock::time_point start = Clock::now();
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
	Clock::time_point end = Clock::now();
	milliseconds ms = chrono::duration_cast<milliseconds>(end - start);
	cout << "ran for duration: " << ms.count() << "ms\n" << endl;
	long long duration = ms.count();
	long long n = OUTPUT_CSV_VAR.size();
	cout << n << "\n" << endl;
	long long fps = n / (duration / 1000); // gives frames per second
	cout << "Frames per second: " << fps << "\n" << endl;
	write_CSV("output_file2.csv", OUTPUT_CSV_VAR, fps);

	capture.release();
	return 0;
}

void detectAndDisplay(Mat frame, int cSel, Scalar lBound, Scalar uBound) {
	Mat frameClone = frame.clone();
	Mat procFrame; //frame used for face recognition
	// resize the frame upon entry
	const double scaleFactor = 1.0 / 7;
	cv::resize(frameClone, procFrame, cv::Size(), scaleFactor, scaleFactor);

	// clone the frame for processing
	Mat frame_gray;
	cvtColor(procFrame, frame_gray, COLOR_BGR2GRAY); // convert the current frame into grayscale
	equalizeHist(frame_gray, frame_gray); // equalise the grayscale img
	//-- Detect faces
	std::vector<int> numDetections;
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray.clone(), faces, numDetections, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(faceCascadeSize, faceCascadeSize));
	// finds the best face possible on the current frame
	int bestIndex = std::distance(
		numDetections.begin(),
		std::max_element(numDetections.begin(), numDetections.end()));

	Mat skin;
	Mat colorImg;
	Mat zeroMatrix = Mat::zeros(Size(frameClone.cols, frameClone.rows), CV_8UC1);
	// ensure that the frame is only processed when a face is obtained)
	if (!faces.empty()) {

		Rect bestFaceRect = faces[bestIndex]; //this contains the rectangle for the best face detected
		Rect originalFaceRect = Rect(bestFaceRect.tl() * ((1 / scaleFactor) + 0.1), bestFaceRect.br() * ((1 / scaleFactor) - 0.1)); //rescaling back to normal size
		// draw on the frame clone
		rectangle(frameClone, originalFaceRect, Scalar(0, 0, 255), 1, LINE_4, 0);
		//-- This is used for color separation for later

		//skin detection code
		skin = skinDetection(frameClone, originalFaceRect, lBound, uBound);
		//split skin into color spectrums
		vector<Mat> temp;
		split(skin, temp);//resplits the channels (extracting the color green for default/testing cases)
		colorImg = temp[cSel - 1];
		Scalar averageColor = mean(temp[cSel - 1]); //takes the average of the color along a selected spectrum B/R/G
		double s = sum(averageColor)[0];
		OUTPUT_CSV_VAR.push_back(s);
	}
	// show what is obtained
	cv::imshow("Capture - Face detection", frameClone);
	if (!faces.empty()) cv::imshow("face ROI", skin);

}


/** Function to get skin color values on the face region of interest
	This function was written to obtain a larger sample size when doing calculations
	for signal averaging

*/
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
	
	cv::GaussianBlur(skinMask, skinMask, cv::Size(7,7), 1, 1);

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