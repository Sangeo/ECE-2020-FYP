/**
	This file generates a sample 50 Hz sinusoidal signal and passes it into a Hanning window (Smoothing) 
	and then the signal is passed through a FFT using the fftw3 library
	The output will be stored in example2.csv in the project's repository
	This is essentially what I want to do for the 2SR method.
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fftw3.h>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;
constexpr auto M_PI = 3.141592653589793;

int main() {
	int i;
	double y;
	const int N = 550;//Number of points acquired inside the window
	double Fs = 200;//sampling frequency
	double dF = Fs / N;
	double  T = 1 / Fs;//sample time 
	double f = 50;//frequency
	double* in;
	fftw_complex* out;
	double t[N];//time vector 
	double ff[N];
	fftw_plan plan_forward;

	in = (double*)fftw_malloc(sizeof(double) * N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

	for (int i = 0; i <= N; i++) {
		t[i] = i * T;

		in[i] = 0.7 * sin(2 * M_PI * f * t[i]);// generate sine waveform
		double multiplier = 0.5 * (1 - cos(2 * M_PI * i / (N - 1)));//Hanning Window
		in[i] = multiplier * in[i];
	}

	for (int i = 0; i <= ((N / 2) - 1); i++) {
		ff[i] = Fs * i / N;
	}
	plan_forward = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

	fftw_execute(plan_forward);

	double v[N];

	for (int i = 0; i <= ((N / 2) - 1); i++) {
				
		//Here I have calculated the y axis of the spectrum in dB
		v[i] = (20 * log(sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]))) / N;  
		
	}

	fstream myfile;

	myfile.open("example2.csv", fstream::out);


	for (i = 0; i < ((N / 2) - 1); i++)

	{

		myfile << ff[i] << ";" << v[i] << std::endl;

	}

	myfile.close();

	fftw_destroy_plan(plan_forward);
	fftw_free(in);
	fftw_free(out);
	return 0;
}