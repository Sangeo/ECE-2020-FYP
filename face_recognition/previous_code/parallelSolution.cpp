#include "opencv_helper.h"

int main(int argc, const char** argv) {

	//introduce the module
	CommandLineParser parser(argc, argv,
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nPress Esc to quit program\nThis program demonstrates signal decomposition by color and motion in openCV with facial recognition in a video stream");
	parser.printMessage();

	//-- 1. Load the cascades
	String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
	if (!face_cascade.load(face_cascade_name)) {
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	std::cout << "Please select the correct configuration for your camera: " << std::endl;
	std::cout << "Enter '1' for 1920*1080, or '2' for 1280*720" << std::endl;
	int choice;
	std::cin >> choice;

	cv::VideoCapture capture(1); // 1 for front facing webcam
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_AUTOFOCUS, 1);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(CAP_PROP_AUTO_WB, 0);
	if (choice == 1) {
		//Get video input/output setup
		capture.set(CAP_PROP_FRAME_WIDTH, 1920);
		capture.set(CAP_PROP_FRAME_HEIGHT, 1080);
	}
	else {
		capture.set(CAP_PROP_FRAME_WIDTH, 1280);
		capture.set(CAP_PROP_FRAME_HEIGHT, 720);
	}

	std::cout << "CAPTURE FOMRAT IS: " << capture.get(CAP_PROP_FORMAT) << std::endl;

	//checks to see if the camera works for both methods
	if (!capture.isOpened()) {
		capture.release();
		capture.open(0);
	}
	//capture.set(CAP_PROP_FOURCC, ('D', 'I', 'V', 'X'));

	Mat frame;
	std::deque<Mat> frameQ;
	Mat SRResult;

	int numFrames; //15 seconds worth of frames
	const int FPS = capture.get(CAP_PROP_FPS);
	const int msPerFrame = 33;

	std::cout << "Frames per second according to capture.get(): " << FPS << std::endl;
	bool recording = true;
	bool processing = false;
	bool initialising = true;
	bool plotLegends = true;
	bool firstRecording = true;
	bool run = true; //state of the system

	//Live-plotting (non-blocking operation of plotting)
	plt::figure_size(1000, 800);
	plt::show(false);

	double rPPG_HREst = 0.0;
	double rBCG_HREst = 0.0;

	while (run) {
		//loop until user presses any key to exit
		if (cv::waitKey(1) > 0) {
			std::cout << "User terminated the program" << std::endl;
			run = false;
			processing = false;
			recording = false;
			break;
		}
		else {

		}
		//Record frames into queue
		cv::Mat frame;
		if (recording && !processing) {
			if (initialising) {
				numFrames = 33 * 5;
			}
			else {
				numFrames = 33 * 15;
			}
			for (int i = 0; i < numFrames; i++) {
				Clock::time_point start = Clock::now();
				if (capture.read(frame)) {
					frameQ.push_back(frame.clone());
				}
				Clock::time_point end = Clock::now();
				auto ms = std::chrono::duration_cast<milliseconds>(end - start);
				double captureTime = ms.count() / 1000.0;
				if (!frame.empty()) {
					double scaleFactor;
					if (choice == 2) {
						scaleFactor = 1.0;
					}
					else {
						scaleFactor = 1.0 / 2.0;
					}

					if (initialising) {
						cv::putText(frame, "Initialising", cv::Point(15, 70),
							cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
						Mat resizedFrame;
						cv::resize(frame, resizedFrame, cv::Size(), scaleFactor, scaleFactor);
						if (!resizedFrame.empty())
							imshow("Live-Camera Footage", resizedFrame);
						if (cv::waitKey(1) > 0) {
							run = false;
							processing = false;
							recording = false;
							break;
						}
					}
					else {
						cv::putText(frame, "Recording! Try to keep still...", cv::Point(15, 70),
							cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
						if (firstRecording) {
						}
						else {
							if (rPPG_HREst >= 0) {
								char buffer[50];
								sprintf_s(buffer, "rPPG Estimate of Heart-Rate is: %.2f", rPPG_HREst);
								cv::putText(frame, buffer, cv::Point(15, 140),
									cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
							}
							else {
								char buffer[50];
								sprintf_s(buffer, "Please try again - bad rPPG estimation");
								cv::putText(frame, buffer, cv::Point(15, 140),
									cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
							}
							if (rBCG_HREst >= 0) {
								char buffer2[50];
								sprintf_s(buffer2, "rBCG Estimate of Heart-Rate is: %.2f", rBCG_HREst);
								cv::putText(frame, buffer2, cv::Point(15, 210),
									cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
							}
							else {
								char buffer[50];
								sprintf_s(buffer, "Please try again - bad rBCG estimation");
								cv::putText(frame, buffer, cv::Point(15, 210),
									cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
							}
						}

						Mat resizedFrame;
						cv::resize(frame, resizedFrame, cv::Size(), scaleFactor, scaleFactor);
						if (!resizedFrame.empty())
							imshow("Live-Camera Footage", resizedFrame);

						if (cv::waitKey(1) > 0) {
							run = false;
							processing = false;
							recording = false;
							break;
						}
					}
				}
				if (!timeVec.empty()) {
					auto cTime = timeVec.back() + ms.count();
					timeVec.push_back(cTime);
				}
				else {
					timeVec.push_back(ms.count());
				}

				if (timeVec.size() >= numFrames) {
					//timeVec will continuously be replaced with newer values 
					assert(!timeVec.empty());
					timeVec.erase(timeVec.begin());
				}
			}
			cv::destroyWindow("Live-Camera Footage");

			if (run == false) {
				processing = false;
				recording = false;
				firstRecording = false;
			}
			else {
				recording = false;
				processing = true;
			}
		}
		//Process frames in the queue, store result in vector 

		if (processing && !recording) {
			//initialising 
			if (initialising) {
				frameQ.clear();
				timeVec.clear();
				initialising = false;
			}
			else {
				firstRecording = false;

				Rect rBCG_ROI;
				Mat skinFrame;
				std::vector<Rect> rBCG_ROI_Queue;
				std::deque<Mat> skinFrameQ;
				Clock::time_point start = Clock::now();
				std::tie(skinFrameQ, rBCG_ROI_Queue) = detectAndDisplay(frameQ, numFrames, choice);
				Clock::time_point end = Clock::now();
				auto ms = std::chrono::duration_cast<milliseconds>(end - start);
				double captureTime = ms.count() / 1000.0;
				std::cout << "Skin/ROI Detection Took: " << captureTime << std::endl;

				start = Clock::now();
				// Spatial Rotation Method
				std::vector<double> filSig_rPPG, rawSig_rPPG, rPPG_time;
				SRResult = spatialRotation(skinFrameQ, SRResult, numFrames);
				std::tie(rPPG_time, rawSig_rPPG, filSig_rPPG, rPPG_HREst) = rPPGEstimate(SRResult, numFrames);
				end = Clock::now();
				ms = std::chrono::duration_cast<milliseconds>(end - start);
				captureTime = ms.count() / 1000.0;
				std::cout << "Skin/ROI Detection Took: " << captureTime << " seconds " << std::endl;

				Clock::time_point start1 = Clock::now();
				// KLT Optical Flow Method
				std::vector<double> filSig_rBCG, rawSig_rBCG, rBCG_time;
				std::tie(filSig_rBCG, rawSig_rBCG, rBCG_time, rBCG_HREst) = KLTEstimate(frameQ, rBCG_ROI_Queue, numFrames);

				Clock::time_point end1 = Clock::now();
				auto ms1 = std::chrono::duration_cast<milliseconds>(end1 - start1);
				double captureTime2 = ms1.count() / 1000.0;
				std::cout << "KLT Estimation took around: " << captureTime2 << " seconds " << std::endl;

				// Plotting relevant figures
				if (!filSig_rPPG.empty()) {
					plt::subplot(2, 1, 1);
					plt::plot(rPPG_time, rawSig_rPPG, { {"color","blue"},{"label","Raw rPPG Signal"} });
					plt::plot(rPPG_time, filSig_rPPG, { {"color","darkorchid"},{"label","Filtered rPPG Signal"} });
					plt::title("Estimated heart rate signal (time-domain)");
					plt::xlabel("Time (seconds)");
					plt::ylabel("Measured Rotation");

				}
				if (plotLegends) {
					plt::legend();
				}

				if (!filSig_rBCG.empty()) {
					plt::subplot(2, 1, 2);
					plt::plot(rBCG_time, rawSig_rBCG, { {"color", "red"}, {"label","Raw rBCG Signal"} });
					plt::plot(rBCG_time, filSig_rBCG, { {"color", "black"}, {"label","Filtered rBCG Signal"} });
					plt::xlabel("Time (seconds)");
					plt::ylabel("nominal displacement (pixels)");

				}
				if (plotLegends) {
					plt::legend();
					plotLegends = false;
				}

				skinFrameQ.clear(); // clear out the skinFrameQ to save RAM
				firstStride = true; // reset the stride starting point for rPPG 
			}

			if (DEBUG_MODE2) {
				cv::destroyWindow("Detected Frame");
				cv::destroyWindow("skinFrame");
			}
			plt::draw();
			plt::pause(0.001);
			plt::save("plot.pdf");

			frameQ.clear();
			//handover to recording
			recording = true;
			processing = false;
		}


		if (DEBUG_MODE) {
			std::cout << "finished recording and processing N frames" << std::endl;
			std::cout << "going to repeat" << std::endl;
		}
	}

	plt::draw();
	plt::pause(0.001);
	plt::save("plot.pdf");

	std::cout << "the program has stopped" << std::endl;

	frameQ.clear();
	capture.release();
	return 0;
}