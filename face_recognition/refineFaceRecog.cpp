/** Motivation: To make a better face recognition code to enable more stable face ROI determination using DND
*/
#include "opencv_helper.h"
#include "opencv2/dnn/dnn.hpp"

void faceRecogHAAR(cv::Mat Frame);
void faceRecogCaffe(cv::Mat frame, cv::dnn::dnn4_v20200310::Net net);


cv::CascadeClassifier face_cascade;
constexpr auto faceCascadeSize = 30;

int main(int argc, const char** argv) {
	cv::CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nPress Esc to quit program\nThis program demonstrates signal decomposition by color in openCV with facial recognition in a video stream");
	parser.printMessage();

	//prompt to select color
	std::cout << "Please select the color for signal decomposition\n";
	std::cout << "1 for blue, 2 for green, 3 for red : \n";
	int color_sel;
	std::cin >> color_sel;

	//-- 1. Load the cascades
	cv::String face_cascade_name = cv::samples::findFile(parser.get<cv::String>("face_cascade"));
	if (!face_cascade.load(face_cascade_name))
	{
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	//code to import caffe neural network settings
	const std::string caffeConfigFile = "C:/Users/jerry/Documents/OpenCV/sources/samples/dnn/face_detector/deploy.prototxt";
	const std::string caffeWeightFile = "C:/Users/jerry/Documents/OpenCV/sources/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	cv::dnn::dnn4_v20200310::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

	//-- 2. Read the video stream
	cv::VideoCapture capture(0); //inputs into the Mat frame as CV_8UC3 format (unsigned integer)
	//set camera settings to capture at 720p 30fps
	capture.set(cv::CAP_PROP_FPS, 30);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(cv::CAP_PROP_AUTOFOCUS, 0);
	capture.set(cv::CAP_PROP_ISO_SPEED, 400);
	// check that the camera is open
	if (!capture.isOpened())
	{
		std::cout << "--(!)Error opening video capture\n";
		return -1;
	}
	// this frame will store all information about the video captured by the camera
	cv::Mat frame;


	for (;;)
	{
		if (capture.read(frame))
		{
			faceRecogHAAR(frame);
		}
		
		if (cv::waitKey(1) >= 0)
		{
			break; // if escape is pressed at any time
		}
	}
}

void faceRecogHAAR(cv::Mat frame) {
	cv::Mat frameClone = frame.clone();
	cv::Mat procFrame; //frame used for face recognition
	// resize the frame upon entry
	const double scaleFactor = 1.0 / 7;
	cv::resize(frameClone, procFrame, cv::Size(), scaleFactor, scaleFactor);
	cv::GaussianBlur(procFrame, procFrame, cv::Size(7, 7), 1, 1);

	// clone the frame for processing
	cv::Mat frame_gray;
	cvtColor(procFrame, frame_gray, cv::COLOR_BGR2GRAY); // convert the current frame into grayscale
	equalizeHist(frame_gray, frame_gray); // equalise the grayscale img
	//-- Detect faces
	std::vector<int> numDetections;
	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(frame_gray.clone(), faces, numDetections, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(faceCascadeSize, faceCascadeSize));
	// finds the best face possible on the current frame
	int bestIndex = std::distance(
		numDetections.begin(),
		std::max_element(numDetections.begin(), numDetections.end()));

	if (!faces.empty()) {

		cv::Rect bestFaceRect = faces[bestIndex]; //this contains the rectangle for the best face detected
		cv::Rect originalFaceRect = cv::Rect(bestFaceRect.tl() * ((1 / scaleFactor)), bestFaceRect.br() * (1 / scaleFactor)); //rescaling back to normal size
		// draw on the frame clone
		rectangle(frameClone, originalFaceRect, cv::Scalar(0, 0, 255), 1, cv::LINE_4, 0);
		//-- This is used for color separation for later
	}
	cv::imshow("Frame", frameClone);

}

void faceRecogCaffe(cv::Mat frame, cv::dnn::dnn4_v20200310::Net net) {

	cv::Mat frameClone = frame.clone();
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameClone);
	net.setInput(inputBlob, "Data");
	cv::Mat detection = net.forward("Detection_Out");
	cv::Mat detectionVec(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	for (int i = 0; i < detectionVec.rows; i++) {
		float confidence = detectionVec.at<float>(i, 2);
		if (confidence > 0.95) {
			int x1 = static_cast <int>(detectionVec.at<float>(i, 3) * frameClone.size().width);
			int y1 = static_cast <int>(detectionVec.at<float>(i, 4) * frameClone.size().height);
			int x2 = static_cast <int>(detectionVec.at<float>(i, 5) * frameClone.size().width);
			int y2 = static_cast <int>(detectionVec.at<float>(i, 6) * frameClone.size().height);
			cv::rectangle(frameClone, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
		}
	}

	cv::imshow("frame", frameClone);
}