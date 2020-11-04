#include "opencv_helper.h"

// RPPG Live implementation 
// current issues:
// 1. Takes roughly a bit more than 4GB of RAM to run this efficiently... 
// This is due to the high resolution video input we are currently using for the sake of rBCG
// rPPG does not need such high resolution video inputs, so we can significantly cut down
// processing time and the RAM used.

// Tasks left to do:
// 1. Filtering the signal and plotting the signal (DONE)
// 2. Outputting the heart rate estimation to the user in the form of puttext to frame. (DONE)
// 3. Test for robustness to obstruction to face detection method...


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

	plt::figure_size(1000, 500);
	plt::show(false);
	bool run = true;
	
	double rPPG_WindowEstimate = 0.0;

	while (run) {
		//loop until user presses any key to exit
		if (cv::waitKey(1) > 0) {
			std::cout << "User terminated" << std::endl;
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
							cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);
						
						char buffer[50];
						sprintf_s(buffer, "Previous Estimate of Heart-Rate is: %.2f", rPPG_WindowEstimate);
						cv::putText(frame, buffer, cv::Point(15, 140), 
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
		if (!recording && processing) {
			if (initialising) {
				frameQ.clear();
				timeVec.clear();
				initialising = false;
			}
			else {
				Mat skinFrame;
				Rect waste;
				std::deque<Mat> skinFrameQ;
				for (size_t j = 0; j < numFrames; j++) {
					std::tie(skinFrame, waste) = detectAndDisplay(frameQ[j].clone());

					if (!skinFrame.empty()) {
						skinFrameQ.push_back(skinFrame);
					}
					else { //to avoid crashing if a face is not detected
						skinFrame = skinFrameQ[j - 1];
						skinFrameQ.push_back(skinFrame);
					}
				}
				SRResult = spatialRotation(skinFrameQ, SRResult, numFrames);

				std::vector <double> x, Raw_rPPG_signal;
				for (int i = 0; i < numFrames; i++) {
					double t = timeVec.at(i) / 1000;
					double sig = (double)SRResult.at<float>(i);

					x.push_back(t);
					Raw_rPPG_signal.push_back(sig);
					rPPGSignal.push_back(sig);
					timeVecOutput.push_back(t);

				}

				


				std::vector<double> fOut = sosFilter(Raw_rPPG_signal);

				std::vector <double> Filtered_rPPG_signal;
				for (int m = 0; m < numFrames; m++) {
					double sig = fOut[m];
					Filtered_rPPG_signal.push_back(sig);
					rPPGSignalFiltered.push_back(sig);
				}
				plt::plot(x, Raw_rPPG_signal, { {"color","blue"},{"label","Raw Signal"} });
				plt::title("Estimated heart rate signal (time-domain)");
				plt::xlabel("Time (s)");
				plt::ylabel("Measured Rotation");

				plt::plot(x, Filtered_rPPG_signal, { {"color","darkorchid"},{"label","Filtered Signal"} });

				//peak detection 
				std::vector<int> peakLocs;
				Peaks::findPeaks(Filtered_rPPG_signal, peakLocs);
				//peak distance calculation
				std::vector<double> diffTime;
				for (size_t i = 0; i < peakLocs.size(); i++) {
					diffTime.push_back(x[peakLocs[i]]);
					std::cout << "time: " << x[peakLocs[i]] << "peak location at: " << peakLocs[i] << std::endl;
				}
				diff(diffTime, diffTime);
				double mean_diffTime = std::accumulate(diffTime.begin(), diffTime.end(), 0.0) / diffTime.size();
				std::cout << "Average time between peaks: " << mean_diffTime << std::endl;
				std::cout << "Estimated heart rate: " << 60.0 / mean_diffTime << std::endl;
				rPPG_WindowEstimate = 60.0 / mean_diffTime; //update the current heart rate estimate

				frameQ.clear();
				skinFrameQ.clear();
				firstStride = true;
			}

			plt::draw();
			plt::pause(0.001);
			plt::save("plot.pdf");

			recording = true;
			processing = false;
		}

		//Display results on matplotlib and return to record frames w/o waiting for user interaction
	}
	plt::draw();
	plt::pause(0.001);
	plt::legend();
	plt::save("plot.pdf");

	std::cout << "the program has stopped" << std::endl;
	return 99;
}


