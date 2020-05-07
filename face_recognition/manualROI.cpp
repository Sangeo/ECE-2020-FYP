/** Motive: to capture color signal information from a selected ROI from the frame
	Jerry Li
*/
#include "opencv_helper.h"

using namespace cv;
using namespace std;


void detectAndDisplay(Mat frame, int c);
void write_CSV(string filename, vector<double> arr, double fps);
vector<double> OUTPUT_CSV_VAR;


int main() {

	//-- 2. Read the video stream
	cv::VideoCapture capture(0); //inputs into the Mat frame as CV_8UC3 format (unsigned integer)
	capture.set(cv::CAP_PROP_FPS, 30);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	// check that the camera is open
	if (!capture.isOpened())
	{
		std::cout << "--(!)Error opening video capture\n";
		return -1;
	}
	//prompt to select color
	cout << "Please select the color for signal decomposition\n";
	cout << "1 for blue, 2 for green, 3 for red : \n";
	int color_sel;
	cin >> color_sel;

	Mat frame;
	for (;;)
	{
		if (capture.read(frame))
		{
			detectAndDisplay(frame, color_sel);
		}

		if (cv::waitKey(1) >= 0)
		{
			break; // if escape is pressed at any time
		}
	}
	long long fps = 30; // gives frames per second
	write_CSV("output_file2.csv", OUTPUT_CSV_VAR, fps);

}

/** to detect a specified region of interest in a video feed from camera
*/
void detectAndDisplay(Mat frame, int cSel) {

	Mat frameClone = frame.clone();
	////set a region of interest in the center of the frame
	//const double scaleFactor = 1.0 / 7;
	//resize(frameClone, frameClone, cv::Size(), scaleFactor, scaleFactor);
	Size frameSize = frameClone.size();

	Rect faceROI = Rect(1280 / 3, 720 / 3, 200, 100);

	rectangle(frameClone, faceROI, Scalar(0, 0, 255), 1, LINE_4, 0);

	imshow("frame", frameClone);
	Mat procFrame = frame(faceROI);
	Mat colorImg;
	vector<Mat> temp;
	split(procFrame, temp);//resplits the channels (extracting the color green for default/testing cases)
	colorImg = temp[cSel - 1];
	
	Scalar averageColor = mean(temp[cSel - 1]); //takes the average of the color along a selected spectrum B/R/G
	double s = sum(averageColor)[0];
	OUTPUT_CSV_VAR.push_back(s);

}


void write_CSV(string filename, vector<double> arr, double fps)
{
	ofstream myfile;
	myfile.open(filename.c_str());
	int vsize = arr.size();
	for (int n = 0; n < vsize; n++)
	{
		myfile << fps << ";" << n << ";" << arr[n] << endl;
	}
}
