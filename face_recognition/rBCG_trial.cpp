//	THIS VERSION FIRST OUTPUTS A RECTANGLE FACE ROI DISPLAYED BY A BLUE RECTANGLE
//		BEFORE CROPPING THE BOUNDED BLUE REGION
//		AND THEN APPLYING THE MOST BASIC FACE DETECTION

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdint.h>
#include <fstream>

using namespace dlib;
using namespace std;
using namespace cv;


int main(int argc, const char** argv) {
	// Load face detection and pose estimation models.
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor predictor;
	deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

	//	Read the video stream
	cout << "Opening video capture ..." << endl << endl;
	cv::VideoCapture capture(0);

	//	Set camera settings to capture at 720p 30fps
	capture.set(CAP_PROP_FPS, 30);
	capture.set(CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CAP_PROP_FRAME_HEIGHT, 720);

	//	Output video settings
	cout << "\nFrame dimensions: " << capture.get(CAP_PROP_FRAME_WIDTH) << " x " << capture.get(CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "Video FPS: " << capture.get(CAP_PROP_FPS) << endl;

	cout << endl << "Press 'ESC' to exit video capture" << endl;

	// Check that the camera is open
	if (!capture.isOpened()) {
		cerr << "ERROR: Can't initialize camera capture!" << endl;
		return -1;
	}

	//	Store a video frame
	cv::Mat frame;

	image_window window;	//output dlib window

	int numFrames = 0;

	// Creating the CSV file to write location data into
	ofstream myFile;
	myFile.open("locationData.csv");

	for (;;) {
		if (capture.read(frame)) {
			Mat frameClone = frame.clone();
			Mat procFrame;	//frame used for face recognition

			numFrames++;	//counting the number of frames

			//	Resize the frame for speed of processing
			const double scaleFactor = 1;	//1 = no resizing
			resize(frameClone, procFrame, Size(), scaleFactor, scaleFactor);

			//	Defining dlib image
			cv_image<bgr_pixel> dlibImage(procFrame);

			//	face ROI frame size
			//const double frameScaleFactor = 0.4;
			//const double frameWidthScaled = 1280 * frameScaleFactor;
			//const double frameHeightScaled = 720 * frameScaleFactor;

			//	Disregard the first 150 frames (5 seconds)
			if (numFrames <= 150) {
				//	Draw blue rectangle for face ROI - centre of screen horizontally
				dlib::rectangle faceBorder;
				faceBorder.left() = 1280 / 2 - 400 / 2;
				faceBorder.top() = 50;	//arbitrary
				faceBorder.right() = faceBorder.left() + 400;
				faceBorder.bottom() = faceBorder.top() + 400;
				//dlib::draw_rectangle(dlibImage, faceBorder, bgr_pixel(255, 0, 0));

				// Show what is obtained
				window.clear_overlay();
				window.add_overlay(dlib::image_window::overlay_rect(faceBorder, bgr_pixel(255, 0, 0), "POSITION HEAD IN BLUE BOX"));
				window.set_image(dlibImage);	//outputs dlib frames (ie. displays video feed)
			}

			else if (numFrames > 150) {
				//	CROPPING WINDOW (in OpenCV)
				cv::Point2i croppedTL, croppedBR;
				croppedTL.x = 1280 / 2 - 400 / 2;
				croppedTL.y = 50;
				croppedBR.x = croppedTL.x + 400;
				croppedBR.y = croppedTL.y + 400;
				cv::Rect croppedRect(croppedTL, croppedBR);

				procFrame = procFrame(croppedRect);
				cv_image<bgr_pixel> dlibImage(procFrame);	//converting backt to dlib image

				//	Detecting faces
				std::vector<dlib::rectangle> faces = detector(dlibImage);

				// Ensure that the frame is only processed when a face is obtained
				if (!faces.empty()) {
					std::vector<full_object_detection> shapes;
					shapes.push_back(predictor(dlibImage, faces[0]));

					//cout << "	# of Frames: " << numFrames << endl;	//Printing out the frame number
					myFile << numFrames << ",";	//Writing frame number to file

					int numLandmarks = shapes[0].num_parts();

					for (int l = 0; l < numLandmarks; l++) {
						//	If an eye (36-47) or mouth (48-67) landmark is encountered, it is skipped/ignored
						if ((l >= 36) && (l <= 67)) {
							continue;
						}

						//	Extracting the landmark coordinates and drawing a green circle
						dlib::point coords = shapes[0].part(l);
						//cout << "Point " << l << ": " << coords << endl << endl;
						dlib::draw_solid_circle(dlibImage, coords, 4, bgr_pixel(0, 255, 0));

						//	Writing the landmark coordinates to a CSV file
						//		Data in file in form of:	frame number, x1, y1, x2, y2, ...
						myFile << coords.x() << "," << coords.y() << ",";

					}

					myFile << endl;

					//----- USING DLIB TO DISPLAY FRAME -----//
					// Placing a red rectangle around the detected face
					dlib::draw_rectangle(dlibImage, faces[0], bgr_pixel(0, 0, 255));

					// Show what is obtained
					window.clear_overlay();
					window.set_image(dlibImage);	//outputs dlib frames (ie. displays video feed)
				
				}
			}

		}

		if (waitKey(0) > 0 )	//need to check this works properly***
		{
			break;	//break out of loop if ESC is pressed at any time
		}
	}

	capture.release();
	return 0;
}

