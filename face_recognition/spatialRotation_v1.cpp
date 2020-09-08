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

void write_CSV(string filename, vector<float> arr);


CascadeClassifier face_cascade;
vector<float> OUTPUT_CSV_VAR;
deque<double> topLeftX;
deque<double> topLeftY;
deque<double> botRightX;
deque<double> botRightY;

constexpr int maxColors = 3; //B,G,R

typedef chrono::high_resolution_clock Clock;
typedef chrono::milliseconds milliseconds;

static bool DEBUGGING_MODE = false;


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
	//Get video input/output setup
	cv::VideoCapture capture(0);
	capture.set(CAP_PROP_FPS, 25);
	capture.set(CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 1);
	capture.set(CAP_PROP_AUTOFOCUS, 1);
	capture.set(CAP_PROP_AUTO_WB, 1);

	cout << "CAPTURE FOMRAT IS: " << capture.get(CAP_PROP_FORMAT) << endl;

	int capLength = capture.get(CAP_PROP_FPS) * 12; //get 8 seconds worth of information
	int strideLength = 45; // size of the temporal stride 
										//(10 frames as this should capture the heart rate information)
	bool firstTime = true;
	bool firstStride = true;

	Mat newFrame;
	Mat skinFrame;
	deque<Mat> skinFrameQ;
	Mat longTermPulseVector;
	deque<Mat> tempPulseVector;

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
		//	dump the initial frames as they are usually not stable
		if ((skinFrameQ.size() >= capLength) && firstTime) {

			skinFrameQ.clear();
			firstTime = false;
		}

		//grab skin pixel information from the queue and process the data
		else if (skinFrameQ.size() > capLength) {
			if (DEBUGGING_MODE) {
				cout << "There are: " << skinFrameQ.size() << " frames stored in the queue." << endl;
				cout << ".................................." << endl;
			}
			deque<Mat> eigValArray; // This will contain the eigenvalues
			deque<Mat> eigVecArray; // This will contain the eigenvectors
			vector<Mat> SRDash;
			// for each frame in the queue 
			for (size_t i = 1; i <= capLength; i++) {
				//our skinFrames are queued in a deque, which can be accessed using a for loop
				//this retrieves a single skin frame from the queue, but this will contain many zeros
				Mat temp = skinFrameQ.front().clone();
				skinFrameQ.pop_front();
				if (DEBUGGING_MODE) {
					//cout << "The skinFrame is :" << endl << temp << endl;
				}
				//for each skinFrame, split the skin pixles into 3 channels, blue, green and red.
				vector<Mat> colorVector;
				split(temp, colorVector);
				// colorVector has size 3xN, 
				// colorVector[0] contains all blue pixels
				// colorVector[1] contains all green pixels 
				// colorVector[2] contains all red pixels 

				/**code used to get rid of the unnecessary zeros**/
				// note: we only need to use one mask here since skin pixel values cannot be (0,0,0)
				vector<Point> mask;
				findNonZero(colorVector[0], mask);
				Mat bVal, gVal, rVal;
				for (Point p : mask) {
					bVal.push_back(colorVector[0].at<uchar>(p)); // collect blue values
					gVal.push_back(colorVector[1].at<uchar>(p)); // collect green values
					rVal.push_back(colorVector[2].at<uchar>(p)); // collect red values
				}

				Mat colorValues; //colorValues is a Nx3 matrix
				vector<Mat> matrices = { rVal, gVal, bVal };
				hconcat(matrices, colorValues);
				if (DEBUGGING_MODE) {
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


				////sort the eigen values from largest to lowest (so as to get the diagonal elements)
				//Mat CPtr = C.clone();
				//CPtr.convertTo(CPtr, 6);
				//arma::mat arma_C(reinterpret_cast<double*>(CPtr.data), CPtr.rows, CPtr.cols);
				//arma::mat arma_U, arma_lambda;
				//arma::qr(arma_U, arma_lambda, arma_C);

				////convert the QR decomposed algorithms back to normal
				////use the same dimensions as they are going to be 3x3 full rank matrices.
				//cv::Mat cv_C(CPtr.rows, CPtr.cols, CV_64FC1, arma_C.memptr());
				//cv::Mat cv_U(CPtr.rows, CPtr.cols, CV_64FC1, arma_U.memptr());
				//transpose(cv_U, cv_U);
				//cv::Mat cv_lambda(CPtr.rows, CPtr.cols, CV_64FC1, arma_lambda.memptr());
				////transpose(cv_lambda, cv_lambda);

				//if (DEBUGGING_MODE) {
				//	cout << "Original C: " << C << endl;
				//	cout << "Original C type: " << C.type() << endl;
				//	cout << "CPtr type :" << CPtr.type() << endl;
				//	cout << "Armadillo's version of C: " << endl << arma_C << endl;
				//	cout << "QR Decomposition of U: " << endl << arma_U << endl;
				//	cout << "QR Decomposition of lambda: " << endl << arma_lambda << endl;
				//	cout << "Converted back C: " << endl << cv_C << endl;
				//	cout << "Converted back C: " << endl << cv_U << endl;
				//	cout << "Converted back C: " << endl << cv_lambda << endl;

				//}

				Mat sortEigVal;
				cv::sort(eigVal, sortEigVal, cv::SortFlags::SORT_EVERY_COLUMN + cv::SortFlags::SORT_DESCENDING);

				if (DEBUGGING_MODE) {
					cout << "C" << endl << C << endl;
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

						if (DEBUGGING_MODE) {
							cout << "Current values for tau and time are -----------------------------------" << endl;
							cout << "uTau and lambdaTau" << uTau << "," << endl << lambdaTau << "," << endl;
							cout << "uTime and lambdaTime" << uTime << "," << endl << lambdaTime << "," << endl;
							cout << endl;
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
						if (DEBUGGING_MODE) {
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

						if (DEBUGGING_MODE) {
							cout << "result matrix S': " << endl << S << endl;
							cout << "result matrix SR': " << endl << SR << endl;
							cout << "Type of matrix SR': " << SR.type() << endl;
							cout << "Rows and columns of SR' are: " << SR.rows << " and " << SR.cols << endl;
						}

						SR.convertTo(SR, 5); // adjust to 32F rather than double --- needs fixing later

						//back-projection into the original RGB space
						Mat backProjectTau;
						vconcat(t_2.t(), t_3.t(), backProjectTau);
						Mat SRBackProjected = SR * backProjectTau;
						if (DEBUGGING_MODE) {
							cout << "the back project vector used for multiplication is : " << endl << backProjectTau << endl;
							cout << "Backprojected SR' is then equals: " << endl << SRBackProjected << endl;

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

					if (DEBUGGING_MODE) {
						/*cout << "Size of SRDashConcat is: " << SRDashConcat.size() << endl;
						cout << "SR'1 is currently:" << endl << SR_1 << endl;
						cout << "SR'1 standard deviation is: " << SR_1std << endl;
						cout << "SR'2 is currently:" << endl << SR_2 << endl;
						cout << "SR'2 standard deviation is: " << SR_2std << endl;*/

						cout << "Current value of t is: " << t << endl;
						cout << "Current value of i is: " << i << endl;
						cout << "therefore the current value of t - stridelength + 1 is : " << t - strideLength << endl << endl;
					}
					//Calculate pulse vector 
					Mat pulseVector;
					pulseVector = SR_1 - (SR_1std / SR_2std) * SR_2;

					//Calculate long-term pulse vector over successive strides using overlap-adding
					//longTermPulseVector.push_back(pulseVector - mean(pulseVector));
					//cout << "Current Pulse Vector Length " << pulseVector.rows << endl;
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

					if (DEBUGGING_MODE) {
						cout << "Current PulseVector before overlap-adding: " << endl << pulseVector << endl;
						cout << "Temp: " << endl << tempPulse << endl;
					}

					SRDash.clear();

					if (DEBUGGING_MODE) {
						cout << "Current Long-term Pulse Vector :" << endl << longTermPulseVector << endl;

					}

				}

			}
			//clean memory registers
			eigVecArray.clear();
			eigValArray.clear();
			SRDash.clear();
		}

		if (!longTermPulseVector.empty()) {
			for (int i = 0; i < longTermPulseVector.rows; i++) {
				OUTPUT_CSV_VAR.push_back(longTermPulseVector.at<float>(i));
			}

			write_CSV("output_file3.csv", OUTPUT_CSV_VAR);

			cout << "the program has finished running for the capture of one stroll of frames" << endl;

			break;
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

	cv::Mat normalFace;
	cv::Mat faceRegion;
	cv::Mat faceRegion2;
	cv::Mat YCrCbFrame;
	cv::Mat HSVFrame;
	// YCrCb values used for skin detection
	Scalar lBound = cv::Scalar(0, 100, 60);
	Scalar uBound = cv::Scalar(255, 180, 160);
	// HSV values used for skin detection
	Scalar HSVUpper, HSVLower;
	HSVLower = Scalar(0, 30, 80);
	HSVUpper = Scalar(20, 180, 255);


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

	/*GaussianBlur(skinMask,skinMask,Size(5,5),0,0,4);
	GaussianBlur(skinMask2,skinMask2,Size(5,5),0,0,4);*/

	Mat kernel = Mat::ones(Size(10, 10), CV_8U);

	bitwise_and(skinMask, skinMask2, trialMask, noArray());

	//apply morphology to image to remove noise and ensure cleaner frame
	cv::morphologyEx(trialMask, trialMask, MORPH_OPEN, kernel);

	medianBlur(trialMask, trialMask, 7);

	//for the real frame output we use the camera feed
	normalFace = frameC(originalFaceRect).clone();

	//bitwise and to get the actual skin color back;
	cv::bitwise_and(normalFace, normalFace, skin, trialMask);

	if (!trialMask.empty()) imshow("trialMask", trialMask);

	return skin;
}

/** Function: write_CSV
	Input: filename, vector<double> of numbers
	Output: prints to csv file the results
*/
void write_CSV(string filename, vector<float> arr) {
	ofstream myfile;
	myfile.open(filename.c_str());
	int vsize = arr.size();
	for (int n = 0; n < vsize; n++) {
		myfile << n << ";" << arr[n] << endl;
	}
}
