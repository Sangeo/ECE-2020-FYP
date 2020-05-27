/**
	Spatial Subspace Rotation C++ Implementation
	Translated from Matlab file
	https://github.com/partofthestars/Spatial-Subspace-Rotation/blob/master/SpatialSubspaceRotation.m
	Original idea from:
	"A Novel Algorithm for Remote Photoplethysmography: Spatial Subspace Rotation"
	Authors: Wenjin Wang, Sander Stuijk and Gerard de Haan
	Available: IEEE Transactions on Biomedical Engineering, vol. 63, no. 9, pp. 1974–1984, Sept. 2016.
*/

#include "opencv_helper.h"

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

Mat skinDetection(
	Mat frameC,
	Rect originalFaceRect);

CascadeClassifier face_cascade;
deque<double> OUTPUT_CSV_VAR;
deque<double> topLeftX;
deque<double> topLeftY;
deque<double> botRightX;
deque<double> botRightY;

constexpr int maxColors = 3; //B,G,R

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

	Mat newFrame;
	Mat skinFrame;
	deque<Mat> skinFrameQ;

	int capLength = capture.get(CAP_PROP_FPS) * 10; //get 10 seconds worth of information
	bool firstTime = true;

	while (true) {

		//-----Get buffer of skin pixels with a decent length (500 frames)
		//if the current capture is available, add it to the newFrame matrix
		if (capture.read(newFrame)) {
			//send newFrame into detectAndDisplay to obtain skin mask
			if (!newFrame.empty()) {
				skinFrame = detectAndDisplay(newFrame);

				//queue if the skinFrame is ready to be pushed into the queue
				if (!skinFrame.empty()) skinFrameQ.push_back(skinFrame);
			}
		}

		//-----Do the processing required on the skin pixels using spatial subspace rotation (2SR) algorithm 
		//dump the initial frames as they are usually not stable
		if ((skinFrameQ.size() > capLength / 2) && firstTime) {

			while (!skinFrameQ.empty()) {
				skinFrameQ.pop_front();
			}
			firstTime = false;
		}
		//grab skin pixel information from the queue and process the data
		else if (skinFrameQ.size() > capLength) {
			
			// with each frame in the queue
			for (size_t i = 0; i < capLength; i++) {
				//our skinFrames are queued in a deque, which can be accessed using a for loop
				//this retrieves a single skin frame from the queue, but this will contain many zeros
				Mat temp = skinFrameQ.front().clone();
				skinFrameQ.pop_front();
				//for each skinFrame, split the skin pixles into 3 channels, blue, green and red.
				vector<Mat> colorVector;
				split(temp, colorVector);
				// colorVector has size 3xN, 
				// colorVector[0] contains all blue pixels, this includes 0s
				// colorVector[1] contains all green pixels 
				// colorVector[2] contains all red pixels 

				// note: we only need to use one mask here since skin pixel values cannot be (0,0,0)
				vector<Point> mask;
				/**code used to get rid of the unnecessary zeros**/
				findNonZero(colorVector[0], mask);
				Mat bVal, gVal, rVal;
				for (Point p : mask) {
					bVal.push_back(colorVector[0].at<uchar>(p)); // collect blue values
					gVal.push_back(colorVector[1].at<uchar>(p)); // collect green values
					rVal.push_back(colorVector[2].at<uchar>(p)); // collect red values
				}

				Mat colorValues; //colorValues is a Nx3 matrix
				vector<Mat> matrices = { bVal, gVal, rVal };
				hconcat(matrices, colorValues);
				// identity of the colorValues matirx
				int rows = colorValues.rows; // this will be the number of skin-pixels N
				int cols = colorValues.cols; // this will be the 3 color channels (b,g,r)

				Mat vTv; //Transposed colorValues * colorValues normalising 
				cv::mulTransposed(colorValues, vTv, true); //This will convert our N X 3 matrix into a 3 X 3 matrix.

				// Divide the multiplied vT*v vector by the total number of skin-pixels
				double N = rows; // number of skin-pixels
				Mat C;
				cv::divide(N, vTv, C); // Correlation vector C, consitutes the matrix used for spatial subspace
				
				Mat eigVal, eigVec;
				cv::eigen(C, eigVal, eigVec);	//computes the eigenvectors and eigenvalues used 
												//for our spatial subspace rotation calculations
				Mat sortEigVal;
				cv::sort(eigVal, sortEigVal, cv::SortFlags::SORT_EVERY_COLUMN + cv::SortFlags::SORT_DESCENDING);

				/* ~~~~temporal stride analysis~~~~ */
				int strideLength = 20; // size of the temporal stride
				//Within the stride, we will analyze the temporal changes in the eigenvector subspace.

				// NOTE MOVE THIS OUT SIDE OF THE LOOP
				vector<Mat> eigValVec; // This will contain the eigenvalues
				vector<Mat> eigVecVec; // This will contain the eigenvectors
				
				eigValVec.push_back(sortEigVal.clone()); // This will contain the first frame's content for eigenvalues
				eigVecVec.push_back(eigVec.clone()); // This will contain the first frame's content for eigenvectors

				if (i > strideLength) {
					cout << "current eigVecvec size: " << eigVecVec.size() << endl;
					for (size_t j = 0; j < strideLength; j++) {

						cout << "eigVecVec at position: " << j << endl << eigVecVec[j].col(0) << endl;
						
						/*transpose(strideUT1, strideUT1);
						Mat t1, t2;
						multiply(strideUT1, strideUTau2, t1);
						multiply(strideUT1, strideUTau3, t2);

						cout << "t1: " << endl << t1 << endl;
						cout << "t2: " << endl << t2 << endl;*/

					}
				}

				/*cout << "C: " << endl << C << endl;
				cout << "eigenvalues: " << endl << eigVal << endl;
				cout << "eigenvectors: " << endl << eigVec << endl;*/


			}

		}

		//return the pulse 

		//exit condition is when the keyboard is pressed anywhere.
		if (waitKey(1) > 0) break;
	}

}


/** Function detectAndDisplay
	Detects a face in a video feed from camera and
	return the frame of information which contains the color information in it.
	Input: current frame (raw)
	Output: current skin (raw);
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

	Rect faceROI;
	Mat trueROI;
	if (!faces.empty()) {
		//find the faceROI using a moving average to ease the noise out	
		faceROI = findPoints(faces, bestIndex, scaleFactor);

		if (!faceROI.empty()) {
			// draws on the current face region of interest
			rectangle(frameClone, faceROI, Scalar(0, 0, 255), 1, LINE_4, 0);

			trueROI = skinDetection(frameClone, faceROI);
		}
	}
	imshow("frame", frameClone);

	return trueROI;
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


/** Function: skinDetection
	Input: Current frame, face ROI
	Output: Skin information

	Obtains skin in the detected face ROI, two thresholds are applied to the frame
	These following values work best under a bright white lamp
	1. YCrCb values
	lBound = (0, 133, 70);
	uBound = (255, 173, 127);

	2. HSV values
	lBound = (0, 30, 70);
	lBound = (17, 170, 255);
	The obtained skin mask will be applied using a 10,10 kernel which with the use of
	morphologyex (opening), clearing out false positives outside the face

*/
Mat skinDetection(Mat frameC, Rect originalFaceRect) {

	cv::Mat normalFace;
	cv::Mat faceRegion;
	cv::Mat faceRegion2;
	cv::Mat YCrCbFrame;
	cv::Mat HSVFrame;
	// YCrCb values used for skin detection
	Scalar lBound = cv::Scalar(0, 133, 70);
	Scalar uBound = cv::Scalar(255, 173, 127);
	// HSV values used for skin detection
	Scalar HSVUpper, HSVLower;
	HSVLower = Scalar(0, 30, 70);
	HSVUpper = Scalar(17, 170, 255);


	cv::cvtColor(frameC, YCrCbFrame, cv::COLOR_BGR2YCrCb);
	cv::cvtColor(frameC, HSVFrame, cv::COLOR_BGR2HSV);
	//use two masks, one to detect for skin, one to detect for valid skin values
	cv::Mat skinMask;
	cv::Mat skinMask2;
	cv::Mat skin;
	Mat trialMask;
	//medianBlur(YCrCbFrame, YCrCbFrame, 5);
	//medianBlur(HSVFrame, HSVFrame, 5);
	//zone out faceRegion
	faceRegion = YCrCbFrame(originalFaceRect).clone();
	cv::inRange(faceRegion, lBound, uBound, skinMask);
	faceRegion2 = HSVFrame(originalFaceRect).clone();
	cv::inRange(faceRegion2, HSVLower, HSVUpper, skinMask2);

	Mat kernel = Mat::ones(Size(10, 10), CV_8U);

	bitwise_and(skinMask, skinMask2, trialMask, noArray());

	//apply morphology to image to remove noise and ensure cleaner frame
	cv::morphologyEx(trialMask, trialMask, MORPH_OPEN, kernel);

	medianBlur(trialMask, trialMask, 5);

	//for the real frame output we use the camera feed
	normalFace = frameC(originalFaceRect).clone();

	//bitwise and to get the actual skin color back;
	cv::bitwise_and(normalFace, normalFace, skin, trialMask);

	if (!trialMask.empty()) imshow("trialMask", trialMask);

	return skin;
}
