#include "opencv_helper.h"

namespace plt = matplotlibcpp;
using namespace cv;
cv::CascadeClassifier face_cascade;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

const bool DEBUG_MODE = false;
const bool DEBUG_MODE2 = false;


std::deque<double> topLeftX;
std::deque<double> topLeftY;
std::deque<double> botRightX;
std::deque<double> botRightY;
std::vector<double> timeVec;
std::vector<double> rPPGSignal;
std::vector<double> rBCGSignal;
const int MAX_BUFFER_SIZE = 900;

int strideLength = 45; // size of the temporal stride 
					//(45 frames as this should capture the heart rate information)
bool firstStride = true;

Mat detectAndDisplay(
	Mat frame);

Rect findPoints(
	std::vector<Rect> faces,
	int bestIndex,
	double scaleFactor);

Mat skinDetection(
	Mat frameC,
	Rect originalFaceRect);


Mat spatialRotation(
	std::deque<Mat> skinFrameQ,
	Mat longTermPulseVector,
	int capLength);

int main(int argc, const char** argv) {

	//introduce the facial recog
	cv::CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("ECSE FYP 2020");
	parser.printMessage();

	//-- 1. Load the cascades
	cv::String face_cascade_name = cv::samples::findFile(parser.get<cv::String>("face_cascade"));
	if (!face_cascade.load(face_cascade_name)) {
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	cv::VideoCapture capture(1);
	cv::VideoCapture capture2(0);

	if (!capture.isOpened()) {
		std::cout << "First camera not detected, trying for second one... " << std::endl;
		if (!capture2.isOpened()) {
			std::cout << "Second camera not detected as well, exiting program..." << std::endl;
			return -1;
		}
		return -1;
	}
	capture.set(cv::CAP_PROP_FPS, 30);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(cv::CAP_PROP_AUTOFOCUS, 1);
	capture.set(cv::CAP_PROP_AUTO_WB, 0);
	std::cout << "CAPTURE FOMRAT IS: " << capture.get(cv::CAP_PROP_FORMAT) << std::endl;

	std::deque<cv::Mat> frameQ;
	Mat SRResult;
	int numFrames = 0;
	bool initialising = true;
	bool processing = false;
	bool recording = true;

	while (true) {
		//loop until user presses any key to exit
		if (cv::waitKey(1) > 0) break;
		//Record frames into queue
		cv::Mat frame;
		cv::Mat text(100, 500, CV_8UC1);;
		if (recording && !processing) {
			if (initialising) {
				numFrames = 30 * 5;
			}
			else {
				numFrames = 30 * 15;
			}
			for (int i = 0; i < numFrames; i++) {
				Clock::time_point start = Clock::now();
				if (capture.read(frame)) {
					frameQ.push_back(frame.clone());
				}
				Clock::time_point end = Clock::now();
				auto ms = std::chrono::duration_cast<milliseconds>(end - start);
				double captureTime = ms.count() / 1000;
				if (!frame.empty()) {

					if (initialising) {
						text.setTo(0);
						cv::putText(frame, "Initialising", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(128));
						cv::imshow("Live-Camera Footage", frame);
						if (cv::waitKey(1) > 0) break;
					}
					else {
						cv::imshow("Live-Camera Footage", frame);
						if (cv::waitKey(1) > 0) break;

					}
				}
				if (!timeVec.empty()) {
					auto cTime = timeVec.back() + ms.count();
					timeVec.push_back(cTime);
				}
				else {
					timeVec.push_back(ms.count());
				}

				if (timeVec.size() > MAX_BUFFER_SIZE) {
					assert(!timeVec.empty());
					timeVec.erase(timeVec.begin());
				}
			}
			cv::destroyWindow("Live-camera Footage");

			recording = false;
			processing = true;
		}
		//Process frames in the queue, store result in vector 
		if (!recording && processing) {
			if (initialising) {
				frameQ.clear();
				timeVec.clear();
				initialising = false;
			}
			else {
				Mat skinFrame;
				std::deque<Mat> skinFrameQ;
				for (size_t j = 0; j < numFrames; j++) {
					skinFrame = detectAndDisplay(frameQ[j].clone());
					
					if (!skinFrame.empty()) {
						skinFrameQ.push_back(skinFrame);
					}
					else {
						skinFrame = skinFrameQ[j - 1];
						skinFrameQ.push_back(skinFrame);
					}
				}
				SRResult = spatialRotation(skinFrameQ, SRResult, numFrames);
				std::vector <double> x, y;
				for (int i = 0; i < (numFrames - 1); i++) {
					double t = timeVec.at(i) / 1000;
					double sig = (double)SRResult.at<float>(i);

					x.push_back(t);
					y.push_back(sig);
					rPPGSignal.push_back(sig);
					if (rPPGSignal.size() > MAX_BUFFER_SIZE) {
						assert(!rPPGSignal.empty());
						rPPGSignal.erase(rPPGSignal.begin());
					}
				}

				plt::figure(0);
				plt::plot(timeVec, rPPGSignal);
				plt::title("Estimated heart rate signal (time-domain)");
				plt::xlabel("Time (s)");
				plt::ylabel("Measured Rotation");
				plt::pause(0.05);

			}
			plt::show();
			recording = true;
			processing = false;
		}

		//Display results on matplotlib and return to record frames w/o waiting for user interaction
	}




	std::cout << "the program has stopped" << std::endl;
	return 99;
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
	Mat trueROI;
	if (!faces.empty()) {
		//find the faceROI using a moving average to ease the noise out	
		faceROI = findPoints(faces, bestIndex, scaleFactor);

		if (!faceROI.empty()) {
			// draws on the current face region of interest
			// draws on the current face region of interest
			Point2i tempTL, tempBR;
			tempTL.x = faceROI.tl().x - 40;
			if (tempTL.x <= 0) {
				tempTL.x = faceROI.tl().x;
			}
			tempTL.y = faceROI.tl().y - 40;
			if (tempTL.y <= 0) {
				tempTL.y = faceROI.tl().y;
			}
			tempBR.x = faceROI.br().x + 40;
			if (tempBR.x >= frameClone.cols) {
				tempBR.x = faceROI.br().x;
				std::cout << "tempBR.x is over the frame's allowable limit" << std::endl;
			}
			tempBR.y = faceROI.br().y + 40;
			if (tempBR.y >= frameClone.rows) {
				tempBR.y = faceROI.br().y;
				std::cout << "tempBR.y is over the frame's allowable limit" << std::endl;
			}

			Rect tempRect(tempTL, tempBR);
			rectangle(frameClone, tempRect, Scalar(0, 0, 255), 1, LINE_4, 0);


			trueROI = skinDetection(frameClone, tempRect);
		}
	}
	/*imshow("Detected Face", frameClone);
	cv::waitKey(1);*/
	if (!trueROI.empty()) {
		return trueROI;
	}
	else {
		Mat zeros;
		return zeros;
	}
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

	Mat frameFace = frameC(originalFaceRect).clone();
	//shrink the region of interest to a face centric region
	Point2i tempTL, tempBR;
	tempTL.x = 60;
	tempTL.y = 60;
	tempBR.x = frameFace.rows - 60;
	tempBR.y = frameFace.cols - 140;
	Rect tempRect(tempTL, tempBR);
	frameFace = frameFace(tempRect);

	Mat yccFace, imgFilter;
	cv::cvtColor(frameFace, yccFace, COLOR_BGR2YCrCb, CV_8U);
	int min_cr, min_cb, max_cr, max_cb;
	min_cr = 133;
	max_cr = 173;
	min_cb = 77;
	max_cb = 127;

	//low-pass spatial filtering to remove high frequency content
	blur(yccFace, yccFace, Size(5, 5));

	//colour segmentation
	cv::inRange(yccFace, Scalar(0, min_cr, min_cb), Scalar(255, max_cr, max_cb), imgFilter);

	//Morphology on the imgFilter to remove noise introduced by colour segmentation
	Mat kernel = Mat::ones(Size(7, 7), CV_8U);
	cv::morphologyEx(imgFilter, imgFilter, MORPH_OPEN, kernel, Point(-1, -1), 2);
	cv::morphologyEx(imgFilter, imgFilter, MORPH_CLOSE, kernel, Point(-1, -1), 2);
	Mat skin;

	//return our detected skin values
	frameFace.copyTo(skin, imgFilter);
	//imshow("SKIN", skin);

	return skin;
}


Mat spatialRotation(std::deque<Mat> skinFrameQ, Mat longTermPulseVector, int capLength) {
	std::deque<Mat> eigValArray; // This will contain the eigenvalues
	std::deque<Mat> eigVecArray; // This will contain the eigenvectors
	std::vector<Mat> SRDash;
	// for each frame in the queue 
	for (size_t i = 1; i <= capLength; i++) {
		//our skinFrames are queued in a deque, which can be accessed using a for loop
		//this retrieves a single skin frame from the queue, but this will contain many zeros
		Mat temp = skinFrameQ.front().clone();
		skinFrameQ.pop_front();
		if (DEBUG_MODE2) {
			//cout << "The skinFrame is :" << endl << temp << endl;
		}
		//for each skinFrame, split the skin pixles into 3 channels, blue, green and red.
		std::vector<Mat> colorVector;
		split(temp, colorVector);
		// colorVector has size 3xN, 
		// colorVector[0] contains all blue pixels
		// colorVector[1] contains all green pixels 
		// colorVector[2] contains all red pixels 

		/**code used to get rid of the unnecessary zeros**/
		// note: we only need to use one mask here since skin pixel values cannot be (0,0,0)
		std::vector<Point> mask;
		findNonZero(colorVector[0], mask);
		Mat bVal, gVal, rVal;
		for (Point p : mask) {
			bVal.push_back(colorVector[0].at<uchar>(p)); // collect blue values
			gVal.push_back(colorVector[1].at<uchar>(p)); // collect green values
			rVal.push_back(colorVector[2].at<uchar>(p)); // collect red values
		}

		Mat colorValues; //colorValues is a Nx3 matrix
		std::vector<Mat> matrices = { rVal, gVal, bVal };
		hconcat(matrices, colorValues);
		if (DEBUG_MODE2) {
			//cout << "The concatenanted skinFrame (ignoring zeros) is : " << endl << colorValues << endl;
		}
		// identity of the colorValues matirx
		int rows = colorValues.rows; // this will be the number of skin-pixels N
		int cols = colorValues.cols; // this will be the 3 color channels (r,g,b)

		Mat vTv; //Transposed colorValues * colorValues, normalised matrix
		mulTransposed(colorValues, vTv, true);
		// Divide the multiplied vT*v vector by the total number of skin-pixels
		double N = rows; // number of skin-pixels
		Mat C = vTv / N;
		Mat eigVal, eigVec;
		cv::eigen(C, eigVal, eigVec);	//computes the eigenvectors and eigenvalues used 
										//for our spatial subspace rotation calculations

		Mat sortEigVal;
		cv::sort(eigVal, sortEigVal, cv::SortFlags::SORT_EVERY_COLUMN + cv::SortFlags::SORT_DESCENDING);

		if (DEBUG_MODE2) {
			std::cout << "C" << std::endl << C << std::endl;
			/*cout << "eigVal" << endl << eigVal << endl;
			cout << "sortEigVal" << endl << sortEigVal << endl;
			cout << "eigVec" << endl << eigVec << endl;*/
		}

		/* ~~~~temporal stride analysis~~~~ */
		//Within the stride, we will analyze the temporal changes in the eigenvector subspace.
		eigValArray.push_back(sortEigVal.clone()); // This will contain the first frame's content for eigenvalues
		eigVecArray.push_back(eigVec.clone()); // This will contain the first frame's content for eigenvectors
		Mat uTau, lambdaTau;
		Mat uTime, lambdaTime;
		Mat RDash, S; //R' and S

		int tempTau = i - strideLength + 1;
		if (tempTau > 0) {
			uTau = eigVecArray[tempTau - 1].clone();
			lambdaTau = eigValArray[tempTau - 1].clone();

			int t; //this will need to be re-used for t 
			for (t = tempTau; t < i; t++) {
				//current values of the eigenvector and eigenvalues
				uTime = eigVecArray[t - 1].clone();
				lambdaTime = eigValArray[t - 1].clone();

				if (DEBUG_MODE2) {
					std::cout << "Current values for tau and time are -----------------------------------" << std::endl;
					std::cout << "uTau and lambdaTau" << uTau << "," << std::endl << lambdaTau << "," << std::endl;
					std::cout << "uTime and lambdaTime" << uTime << "," << std::endl << lambdaTime << "," << std::endl;
					std::cout << std::endl;
				}

				// calculation for R'
				Mat t1, t_2, t_3, r1, r2;
				t1 = uTime.col(0).clone(); //current frame's U (or u1 vector)
				cv::transpose(t1, t1);
				t_2 = uTau.col(1).clone(); //first frame's u2
				r1 = t1 * t_2;
				t_3 = uTau.col(2).clone(); //first frame's u3
				r2 = t1 * t_3;
				double result[2] = { sum(r1)[0],sum(r2)[0] };
				RDash = (Mat_<double>(1, 2) << result[0], result[1]); // obtains the R' values required to calculate the SR'
				if (DEBUG_MODE2) {
					/*cout << endl << "R'(1) = " << r1 << "and R'(2) =" << r2 << endl;
					cout << "RDash is equals: ......................................." << endl << RDash << endl;*/
				}
				// calculation for S
				Mat temp, tau2, tau3;
				temp = lambdaTime.row(0).clone(); //lambda t1
				tau2 = lambdaTau.row(1).clone();  //lambda tau2
				tau3 = lambdaTau.row(2).clone();  //lambda tau3

				double result2[3] = { sum(temp)[0],sum(tau2)[0],sum(tau3)[0] };
				double lambda1, lambda2;
				lambda1 = sqrt(result2[0] / result2[1]);
				lambda2 = sqrt(result2[0] / result2[2]);
				S = (Mat_<double>(1, 2) << lambda1, lambda2);
				Mat SR = S.mul(RDash);; // Obtain SR matrix, and adjust with rotation vectors (step (10) of 2SR paper)

				if (DEBUG_MODE2) {
					std::cout << "result matrix S': " << std::endl << S << std::endl;
					std::cout << "result matrix SR': " << std::endl << SR << std::endl;
					std::cout << "Type of matrix SR': " << SR.type() << std::endl;
					std::cout << "Rows and columns of SR' are: " << SR.rows << " and " << SR.cols << std::endl;
				}

				SR.convertTo(SR, 5); // adjust to 32F rather than double --- needs fixing later

				//back-projection into the original RGB space
				Mat backProjectTau;
				vconcat(t_2.t(), t_3.t(), backProjectTau);
				Mat SRBackProjected = SR * backProjectTau;
				if (DEBUG_MODE2) {
					std::cout << "the back project vector used for multiplication is : " << std::endl << backProjectTau << std::endl;
					std::cout << "Backprojected SR' is then equals: " << std::endl << SRBackProjected << std::endl;
				}
				// Obtain the SR'
				SRDash.push_back(SRBackProjected);
			}
			// Calculate SR'1 and SR'2
			Mat SRDashConcat;
			vconcat(SRDash, SRDashConcat);
			Mat SR_1, SR_2;
			SR_1 = SRDashConcat.col(0);
			SR_2 = SRDashConcat.col(1);
			Scalar SR_1Mean, SR_2Mean, SR_1Stdev, SR_2Stdev;
			cv::meanStdDev(SR_1, SR_1Mean, SR_1Stdev);
			cv::meanStdDev(SR_2, SR_2Mean, SR_2Stdev);
			float SR_1std, SR_2std;
			SR_1std = sum(SR_1Stdev)[0];
			SR_2std = sum(SR_2Stdev)[0];

			if (DEBUG_MODE2) {
				std::cout << "Current value of t is: " << t << std::endl;
				std::cout << "Current value of i is: " << i << std::endl;
				std::cout << "therefore the current value of t - stridelength + 1 is : " << t - strideLength << std::endl << std::endl;
			}
			//Calculate pulse vector 
			Mat pulseVector;
			pulseVector = SR_1 - (SR_1std / SR_2std) * SR_2;

			//Calculate long-term pulse vector over successive strides using overlap-adding
			Mat tempPulse = pulseVector - mean(pulseVector);

			if (firstStride) {
				longTermPulseVector = tempPulse;
				firstStride = false;
			}
			else {
				longTermPulseVector.push_back(float(0));
				int y = 0;
				for (size_t x = t - strideLength; x < i; x++) {

					longTermPulseVector.at<float>(x) = longTermPulseVector.at<float>(x) + tempPulse.at<float>(y);
					y++;
				}
			}

			if (DEBUG_MODE2) {
				std::cout << "Current PulseVector before overlap-adding: " << std::endl << pulseVector << std::endl;
				std::cout << "Temp: " << std::endl << tempPulse << std::endl;
			}
			SRDash.clear();



		}
	}
	//clean memory registers
	eigVecArray.clear();
	eigValArray.clear();
	SRDash.clear();

	//std::cout << "Current Length of Long-term Pulse Vector :" << longTermPulseVector.rows << std::endl;
	return longTermPulseVector;
}
