#include "opencv_helper.h"

int main(int argc, const char** argv) {

	// Introduction to the system
	CommandLineParser parser(argc, argv,
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nPress any key during recording phase to quit program\n This program was made as a part of the ECSE 2020 Final Year Project at Monash.");
	parser.printMessage();

	//-- 1. Load the cascades
	String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
	if (!face_cascade.load(face_cascade_name)) {
		std::cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	int choice = 1;

	//Initiliase the video capture
	cv::VideoCapture capture(0, CAP_DSHOW); 
	//checks to see if the camera works for both methods
	if (!capture.isOpened()) {
		capture.release();
		capture.open(1);
		if (!capture.isOpened()) {
			std::cout << "Unable to open the camera device" << std::endl;
			return -100;
		}
	}
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_AUTOFOCUS, 1);
	capture.set(CAP_PROP_AUTO_EXPOSURE, 0);
	capture.set(CAP_PROP_AUTO_WB, 0);

	std::cout << "CAPTURE BACKEND: " << capture.getBackendName() << std::endl;
	std::cout << "CAPTURE FOMRAT IS: " << capture.get(CAP_PROP_FORMAT) << std::endl;
	capture.set(CAP_PROP_FOCUS, cv::CAP_MSMF);
	std::cout << "CAPTURE FOMRAT IS: " << capture.get(CAP_PROP_FORMAT) << std::endl;
	// try for the highest resolution available, this forces the camera to be on its maximum resolution. 
	capture.set(CAP_PROP_FRAME_WIDTH, 19200); 
	capture.set(CAP_PROP_FRAME_HEIGHT, 10800);
	//Get video input/output setup
	std::cout << "capture get width " << capture.get(CAP_PROP_FRAME_WIDTH) << std::endl;
	std::cout << "capture get HEIGHT" << capture.get(CAP_PROP_FRAME_HEIGHT) << std::endl;
	// reset the camera so that we don't actually interpolate/upscale the image to the preset resolution
	capture.set(CAP_PROP_FRAME_WIDTH, capture.get(CAP_PROP_FRAME_WIDTH));
	capture.set(CAP_PROP_FRAME_HEIGHT, capture.get(CAP_PROP_FRAME_HEIGHT));



	const int FPS = capture.get(CAP_PROP_FPS);
	std::cout << "Frames per second according to capture.get(): " << FPS << std::endl;

	std::ofstream myfile;
	myfile.open("CombiResults.csv");
	myfile << "trial number" << ";" << "rPPG_HREst" << ";" << "rBCG_HREst" << std::endl;
	std::ofstream myfile2;
	myfile2.open("rBCG_RawSignals.csv");
	
	Mat frame;
	std::deque<Mat> frameQ;
	Mat SRResult;
	Rect rBCG_ROI;
	Mat skinFrame;
	std::vector<Rect> rBCG_ROI_Queue;
	std::deque<Mat> skinFrameQ;

	bool recording = true;
	bool processing = false;
	bool initialising = true;
	bool plotLegends = true;
	bool firstRecording = true;
	//Live-plotting (non-blocking operation of plotting)
	bool plottingMode = true;
	if (plottingMode) {
		plt::figure_size(1000, 800);
		plt::show(false);
	}
	bool run = true;
	int trialNumber = 0;
	double ground_HREst = 0.0;
	double rPPG_HREst = 0.0;
	double rBCG_HREst = 0.0;
	int numFrames; //15 seconds worth of frames

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
				numFrames = 30 * 4;
			}
			else {
				numFrames = 30 * 15;
			}

			Clock::time_point start = Clock::now();
			std::chrono::duration<double, std::milli> ms;

			for (int i = 0; i < numFrames; i++) {

				if (capture.read(frame)) {
					Clock::time_point frameTime = Clock::now();
					ms = frameTime - start;
					start = frameTime;
					frameQ.push_back(frame.clone());
				}
				else throw 10;
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
							cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
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
				trialNumber++;

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
				std::cout << "rPPG Detection Took: " << captureTime << " seconds " << std::endl;

				Clock::time_point start1 = Clock::now();
				// KLT Optical Flow Method
				std::vector<double> filSig_rBCG, rawSig_rBCG, rBCG_time;
				std::tie(filSig_rBCG, rawSig_rBCG, rBCG_time) = KLTEstimate(frameQ, rBCG_ROI_Queue, numFrames);

				Clock::time_point end1 = Clock::now();
				auto ms1 = std::chrono::duration_cast<milliseconds>(end1 - start1);
				double captureTime2 = ms1.count() / 1000.0;
				std::cout << "KLT Estimation took around: " << captureTime2 << " seconds " << std::endl;

				std::vector<double> v_, ff_;
				//std::tie(v, ff) = calcPSD(rawSig_rBCG, numFrames);
				int rBCGSize = filSig_rBCG.size();
				std::tie(v_, ff_) = calcPSD(filSig_rBCG, rBCGSize);

				int temp_max_loc = 0;
				double temp_max = 0.0;
				for (int i = 0; i < v_.size(); i++) {
					//Ignoring high energy coefficients at the start of the bandpass filter
					//Ask James if there's a way to do this.
					if ((v_[i] > temp_max) && (ff_[i] > 1.1)) {
						temp_max_loc = i;
						temp_max = v_[i];
						std::cout << "max freq:" << ff_[i] << "max val: " << temp_max << std::endl;
					}
				}
				rBCG_HREst = ff_[temp_max_loc] * 60; //bpm estimation for rBCG
				std::cout << "rBCG - Estimated heart rate using FFT: " << rBCG_HREst << std::endl;


				if (DEBUG_MODE2) {
					cv::destroyWindow("Detected Frame");
					cv::destroyWindow("skinFrame");
				}

				//code used for testing purposes (uncomment to use)
				/*std::cout << "Please enter the average reading for this trial:  ... " << std::endl;
				std::cin >> ground_HREst;
				myfile << trialNumber << ";" << rPPG_HREst << ";" << rBCG_HREst << ";" << ground_HREst << std::endl;*/

				for (int q = 0; q < rBCG_time.size(); q++) {
					myfile2 << rBCG_time[q] << ";" << rawSig_rBCG[q] << ";" << std::endl;
				}


				// Plotting relevant figures
				if (!filSig_rPPG.empty() && plottingMode) {
					plt::subplot(2, 1, 1);
					plt::plot(rPPG_time, rawSig_rPPG, { {"color","blue"},{"label","Raw rPPG Signal"} });
					plt::plot(rPPG_time, filSig_rPPG, { {"color","darkorchid"},{"label","Filtered rPPG Signal"} });
					plt::title("Estimated heart rate signal (time-domain)");
					plt::xlabel("Time (seconds)");
					plt::ylabel("Measured Rotation");

				}
				if (plotLegends && plottingMode) {
					plt::legend();
				}

				if (!filSig_rBCG.empty() && plottingMode) {
					plt::subplot(2, 1, 2);
					plt::plot(rBCG_time, rawSig_rBCG, { {"color", "red"}, {"label","Raw rBCG Signal"} });
					plt::plot(rBCG_time, filSig_rBCG, { {"color", "black"}, {"label","Filtered rBCG Signal"} });
					plt::xlabel("Time (seconds)");
					plt::ylabel("Signal estimation");

					//code to plot frequency power spectrum

					//plt::subplot(2, 1, 2);
					//plt::plot(ff, v, { {"color", "red"}, {"label","Raw rBCG Spectrum"} });
					//plt::plot(ff_, v_, { {"color", "purple"}, {"label","Filtered rBCG Spectrum"} });
					//plt::xlabel("Frequency (Hz)");
					//plt::ylabel("Signal Power");
				}
				if (plotLegends && plottingMode) {
					plt::legend();
					plotLegends = false;
				}

				skinFrameQ.clear(); // clear out the skinFrameQ to save RAM
				rBCG_ROI_Queue.clear();
				firstStride = true; // reset the stride starting point for rPPG 
			}

			if (plottingMode) {
				plt::draw();
				plt::pause(0.001);
				plt::save("plot.pdf");
			}

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

	if (plottingMode) {
		plt::draw();
		plt::pause(0.001);
		plt::save("plot.pdf");
	}

	std::cout << "the program has stopped" << std::endl;

	frameQ.clear();
	capture.release();
	return 0;

}