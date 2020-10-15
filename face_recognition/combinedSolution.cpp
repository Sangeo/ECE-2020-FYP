#include "opencv_helper.h"

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
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	//Get video input/output setup
	cv::VideoCapture capture(1); // 1 for front facing webcam
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(CAP_PROP_FRAME_HEIGHT, 1080);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(CAP_PROP_AUTOFOCUS, 1);
	capture.set(CAP_PROP_AUTO_WB, 0);

	std::cout << "CAPTURE FOMRAT IS: " << capture.get(CAP_PROP_FORMAT) << std::endl;
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

	//Live-plotting (non-blocking operation of plotting)
	plt::figure_size(1000, 800);
	plt::show(false);
	bool run = true;

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
					double scaleFactor = 1.0 / 2.0;
					if (initialising) {
						cv::putText(frame, "Initialising", cv::Point(15, 70),
							cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);
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
							cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);

						char buffer[50];
						sprintf_s(buffer, "rPPG Estimate of Heart-Rate is: %.2f", rPPG_HREst);
						cv::putText(frame, buffer, cv::Point(15, 140),
							cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
						
						char buffer2[50];
						sprintf_s(buffer2, "rBCG Estimate of Heart-Rate is: %.2f", rBCG_HREst);
						cv::putText(frame, buffer2, cv::Point(15, 210),
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
				}
				if (!timeVec.empty()) {
					auto cTime = timeVec.back() + ms.count();
					timeVec.push_back(cTime);
				}
				else {
					timeVec.push_back(ms.count());
				}

				if (timeVec.size() > numFrames) {
					//timeVec will continuously be replaced with newer values 
					assert(!timeVec.empty());
					timeVec.erase(timeVec.begin());
				}
			}
			cv::destroyWindow("Live-Camera Footage");
			if (run == false) {
				processing = false;
				recording = false;
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
				Rect rBCG_ROI;
				Mat skinFrame;
				std::vector<Rect> rBCG_ROI_Queue;
				std::deque<Mat> skinFrameQ;
				for (size_t j = 0; j < numFrames; j++) {
					std::tie(skinFrame, rBCG_ROI) = detectAndDisplay(frameQ[j].clone());
					
					if (!rBCG_ROI.empty()) {
						rBCG_ROI_Queue.push_back(rBCG_ROI);
					}
					else {
						std::cout << "rBCG_ROI is empty " << std::endl;
						//if(j!=0) implementation in the future
						rBCG_ROI = rBCG_ROI_Queue[j - 1];
						rBCG_ROI_Queue.push_back(rBCG_ROI);
					}

					if (!skinFrame.empty()) {
						skinFrameQ.push_back(skinFrame);
					}
					else { 
						//to avoid crashing if a face is not detected
						std::cout << "skinframe is empty " << std::endl;
						skinFrame = skinFrameQ[j - 1];
						skinFrameQ.push_back(skinFrame);
					}
				}
				
				// Spatial Rotation Method
				std::vector<double> filSig_rPPG, rawSig_rPPG, rPPG_time;
				SRResult = spatialRotation(skinFrameQ, SRResult, numFrames);
				std::tie(rPPG_time, rawSig_rPPG, filSig_rPPG, rPPG_HREst) = rPPGEstimate(SRResult, numFrames);
				
				if (DEBUG_MODE) {
					for (int i = 0; i < rPPG_time.size(); i++) {
						std::cout << "time: " << rPPG_time[i] <<
							" rawSig: " << rawSig_rPPG[i] <<
							" filSig: " << filSig_rPPG[i] << std::endl;
					}
				}

				// KLT Optical Flow Method
				std::vector<double> filSig_rBCG, rawSig_rBCG, rBCG_time;
				std::tie(filSig_rBCG, rawSig_rBCG, rBCG_time, rBCG_HREst) = KLTEstimate(frameQ, rBCG_ROI_Queue, numFrames);

				if (DEBUG_MODE) {
					for (int i = 0; i < rBCG_time.size(); i++) {
						std::cout << "time: " << rBCG_time[i] <<
							" rawSig: " << rawSig_rBCG[i] <<
							" filSig: " << filSig_rBCG[i] << std::endl;
					}

				}
				
				if (!filSig_rPPG.empty()) {
					plt::subplot(2, 1, 1);
					plt::plot(rPPG_time, rawSig_rPPG, { {"color","blue"},{"label","Raw rPPG Signal"} });
					plt::plot(rPPG_time, filSig_rPPG, { {"color","darkorchid"},{"label","Filtered rPPG Signal"} });
					plt::title("Estimated heart rate signal (time-domain)");
					plt::xlabel("Time (seconds)");
					plt::ylabel("Measured Rotation");

				}
				
				if (!filSig_rBCG.empty()) {
					plt::subplot(2, 1, 2);
					plt::plot(rBCG_time, rawSig_rBCG, { {"color", "red"}, {"label","Raw rBCG Signal"} });
					plt::plot(rBCG_time, filSig_rBCG, { {"color", "black"}, {"label","Filtered rBCG Signal"} });
					plt::xlabel("Time (seconds)");
					plt::ylabel("nominal displacement (pixels)");

				}

				skinFrameQ.clear(); // clear out the skinFrameQ to save RAM
				firstStride = true; // reset the stride starting point for rPPG 
			}

			cv::destroyWindow("Detected Frame");

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
	plt::legend();
	plt::save("plot.pdf");

	std::cout << "the program has stopped" << std::endl;

	frameQ.clear();
	capture.release();
	return 0;

}