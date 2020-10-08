#include "opencv_helper.h"

namespace plt = matplotlibcpp;
using namespace cv;
using namespace std;

constexpr auto M_PI = 3.141592653589793;

deque<double> filterDesign(
	double f1,
	double f2,
	double fs,
	double N);

deque<double> convFunc(
	deque<double> signal,
	deque<double> filter,
	int aLen,
	int bLen);

deque<double> fftSpect(
	deque<double> signal,
	double fs,
	double N);

int main() {

	vector<double> time, signal;
	deque<double> signal_deque;

	//signal generation
	for (double i = 0; i < 1 * M_PI; i = i + 0.01) {
		time.push_back(i);
		//signal has peaks at 1.5hz, 7.5hz, 15hz 
		signal.push_back(0.5*std::sin(2 * M_PI * i) + std::sin(12 * M_PI * i) + 0.5*std::sin(30 * M_PI * i));
		signal_deque.push_back(0.5*std::sin(2 * M_PI * i) + std::sin(12 * M_PI * i) + 0.5*std::sin(30 * M_PI * i));

	}


	double f1, f2, fs, N;
	f1 = 3;
	f2 = 10;
	fs = 100;
	N = 128;

	//filter design
	deque<double> filter = filterDesign(f1, f2, fs, N);
	for (size_t j = 0; j < filter.size(); j++) {
		cout << "Coefficient: " << j << " " << filter[j] << endl;
	}

	//convolution
	int len = signal_deque.size() + filter.size() - 1;
	deque<double> output = convFunc(filter, signal_deque, filter.size(), signal_deque.size());
	cout << "CONVOLUTION COMPLETE, convolved signal size: " << output.size() << endl;
	cout << "Original vector length: " << signal_deque.size() << endl;

	vector<double> bp_signal, bp_time;
	deque<double> bp_signal_deque;
	for (int i = 0; i < output.size(); i++) {
		bp_signal.push_back(output[i]);
		bp_signal_deque.push_back(output[i]);
		
		bp_time.push_back((double)i / 100);
	}

	plt::figure(0);
	plt::subplot(2, 1, 1);
	plt::plot(time, signal);
	plt::subplot(2, 1, 2);
	plt::plot(bp_time, bp_signal);
	plt::show();
	
	//redefine N
	double N1 = signal.size();
	double N2 = bp_signal.size();

	deque<double> preFilterPS, postFilterPS;
	preFilterPS = fftSpect(signal_deque, fs, N1);
	postFilterPS = fftSpect(bp_signal_deque, fs, N2);
	
	vector<double> ff1,ff2;
	vector<double> rawPS, filtPS;
	for (int i = 0; i <= ((N1 / 2) - 1); i++) {
		ff1.push_back(fs * i / N1);
		rawPS.push_back(preFilterPS[i]);
	}
	for (int i = 0; i <= ((N2 / 2) - 1); i++) {
		ff2.push_back(fs * i / N2);
		filtPS.push_back(postFilterPS[i]);
	}



	plt::figure(1);
	plt::subplot(2, 1, 1);
	plt::plot(ff1, rawPS);
	plt::subplot(2, 1, 2);
	plt::plot(ff2, filtPS);
	plt::show();
	

	return 100;
}

/* Algorithm for FIR bandpass filter
Input:
f1, the lowest frequency to be included, in Hz
f2, the highest frequency to be included, in Hz
f_samp, sampling frequency of the audio signal to be filtered, in Hz
N, the order of the filter; assume N is odd
Output:
a bandpass FIR filter in the form of an N-element array
*/
deque<double> filterDesign(double f1, double f2, double fs, double N) {
	deque<double> output(N);

	double f1_c, f2_c, w1_c, w2_c;
	f1_c = f1 / fs;
	f2_c = f2 / fs;
	w1_c = 2 * M_PI * f1_c;
	w2_c = 2 * M_PI * f2_c;
	cout << "f1_c: " << f1_c << " "
		<< "f2_c" << f2_c << endl
		<< "w1_c" << w1_c << " "
		<< "w2_c" << w2_c << endl;

	int mid = N / 2; //integer division, drops the remainder.

	for (int i = 0; i < N; i++) {
		if (i == mid) {
			output[i] = (double)2 * f2_c - 2 * f1_c;
		}
		else {
			if (i < mid) {
				int i_low = -N / 2 + i;
				output[i] = (double)(std::sin(w2_c * i_low) / (M_PI * i_low))
					- (std::sin(w1_c * i_low) / (M_PI * i_low));
			}
			else if (i > mid) {
				int i_high = N / 2 + i;
				output[i] = (double)(std::sin(w2_c * i_high) / (M_PI * i_high))
					- (std::sin(w1_c * i_high) / (M_PI * i_high));
			}

		}
	}


	return output;
}

/* Convolution function
* input: 
* vectors a and b
* length of vectors a and b
* output:
* vector of convolution result
*/
deque<double> convFunc(deque<double> a, deque<double> b, int aLen, int bLen) {

	int abMax = std::max(aLen, bLen);
	int convLen = aLen + bLen - 1;
	deque<double> output(convLen);

	for (int n = 0; n < convLen; n++) {
		float prod = 0;
		int kMax = std::min(abMax, n);
		for (int k = 0; k <= kMax; k++) {
			//make sure we're in bounds for both arrays
			if (k < aLen && (n - k) < bLen) {
				prod += a[k] * b[n - k];
			}
		}
		output[n] = prod;
	}
	return output;
}


/* this is a function used to generate the power spectrum 

*/
deque<double> fftSpect(deque<double> signal, double fs, double N) {
	
	cout << "entered into fftspect" << endl;
	cout << N << endl;
	double* in;
	fftw_complex* out;
	fftw_plan plan_forward;

	in = (double*)fftw_malloc(sizeof(double) * N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

	for (int i = 0; i < N; i++) {
		in[i] = signal[i];
	}
	plan_forward = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

	fftw_execute(plan_forward);

	deque<double> v;

	for (int i = 0; i <= ((N / 2) - 1); i++) {

		//Here I have calculated the y axis of the spectrum in dB
		v.push_back((20 * log(sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]))) / N);

	}

	fftw_destroy_plan(plan_forward);
	fftw_free(in);
	fftw_free(out);
	return v;
}
