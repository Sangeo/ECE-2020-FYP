/** A tool to detect skin from a region of interest, might be useful to get a nice average of skin pixels
	that is not affected by ROI changing due to tracking differences between frames.

*/
#include "opencv_helper.h"

cv::CascadeClassifier faceCascade;
void detectAndDisplay(cv::Mat f,cv::Scalar vec, cv::Scalar vec2,int useHSV);

int main(int argc, const char** argv) {
	//Introducing the module
	cv::CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nPress Esc to quit program\nThis program demonstrates signal decomposition by color in openCV with facial recognition in a video stream");
	parser.printMessage();
	//-- 1. Load the cascades
	cv::String faceCascadeName = cv::samples::findFile(parser.get<cv::String>("face_cascade"));
	if (!faceCascade.load(faceCascadeName))
	{
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	//-- 2. Read the video stream
	cv::VideoCapture capture(0); //inputs into the Mat frame as CV_8UC3 format (unsigned integer)
	capture.set(cv::CAP_PROP_FPS, 30);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	// check that the camera is open
	if (!capture.isOpened())
	{
		std::cout << "--(!)Error opening video capture\n";
		return -1;
	}
	// this frame will store all information about the video captured by the camera
	cv::Mat frame;
	cv::Scalar lower;
	cv::Scalar upper;
	//define lower and upper bounds for skin detection
	//these values will be in HSV format NOT BGR
	int useHSV = 0;
	if (useHSV == 1) {
		//HSV values - Hue, Saturation and Value 
		lower = cv::Scalar(0, 48, 80);
		upper = cv::Scalar(20, 255, 255);
	}
	else {
		// YCrCb values - 
	//The Y' channel (luma) is basically the grayscale version of the original image. 
	// The Cr and Cb channels contain the colour information. They can be highly compressed.
		lower = cv::Scalar(0, 133, 70);
		upper = cv::Scalar(255, 173, 127);
	}

	
	for (;;)
	{
		if (capture.read(frame))
		{
			detectAndDisplay(frame,lower,upper,useHSV);
		}

		if (cv::waitKey(1) == 27)
		{
			break; // if escape is pressed at any time
		}
	}
}

void detectAndDisplay(cv::Mat frame, cv::Scalar lBound, cv::Scalar uBound, int useHSV) {
	cv::Mat frameClone = frame.clone();
	cv::Mat procFrame; //frame used for face recognition

	// resize the frame upon entry
	const double scaleFactor = 1.0 / 9;
	cv::resize(frameClone, procFrame, cv::Size(), scaleFactor, scaleFactor);

	// detectMultiscale requires a gray format image to work best
	cv::Mat frame_gray;
	cvtColor(procFrame, frame_gray, cv::COLOR_BGR2GRAY); // convert the current frame into grayscale
	equalizeHist(frame_gray, frame_gray); // equalise the grayscale img

	//-- Detect faces using the detectMultiscale function of the openCV library
	std::vector<int> numDetections;
	std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(frame_gray.clone(), faces, numDetections, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20));
	// finds the best face possible on the current frame
	int bestIndex = std::distance(
		numDetections.begin(),
		std::max_element(numDetections.begin(), numDetections.end()));
	cv::Rect bestFaceRect;
	cv::Rect originalFaceRect;
	cv::Mat faceRegion;
	
	cv::Mat normalFace;
	cv::Mat cvtColorFrame;

	cv::Mat skinMask;
	cv::Mat skin;	
	
	if (useHSV == 1) {
		cv::cvtColor(frame, cvtColorFrame, cv::COLOR_BGR2HSV);
	}
	else {
		cv::cvtColor(frame, cvtColorFrame, cv::COLOR_BGR2YCrCb);
	}
	
	

	if (!faces.empty()) {
		bestFaceRect = faces[bestIndex];
		originalFaceRect = cv::Rect(bestFaceRect.tl() * (1 / scaleFactor), bestFaceRect.br() * (1 / scaleFactor));
		rectangle(frameClone, originalFaceRect, cv::Scalar(0, 0, 255), 1, cv::LINE_4, 0);

		//zone out faceRegion
		faceRegion = cvtColorFrame(originalFaceRect).clone();
		cv::inRange(faceRegion, lBound, uBound, skinMask);

		normalFace = frame(originalFaceRect).clone();

		//apply morphology to image to remove noise and ensure cleaner frame
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
		cv::erode(skinMask, skinMask, kernel, cv::Point(-1,-1), 1);
		cv::dilate(skinMask, skinMask, kernel, cv::Point(-1, -1), 1);
		cv::GaussianBlur(skinMask, skinMask, cv::Size(3,3),0,0);
		//bitwise and to get the actual skin color back;
		cv::bitwise_and(normalFace, normalFace, skin, skinMask);
		

		//show what you've got
		imshow("FaceRegion", skin);
		imshow("skinMask", skinMask);
		
	}
	//imshow("Camera", frameClone);
	//cv:print(skinMask);
	
}