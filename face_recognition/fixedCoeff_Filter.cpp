#include "opencv_helper.h"

namespace plt = matplotlibcpp;


const int FILTER_SECTIONS = 12;

//the following filter uses butterworth bandpass
const double sos_matrix[12][6] = {
	{0.821904631289823, -1.62696489036367, 0.821904631289823, 1, -1.65115775247009, 0.956239445176575},
	{ 0.821904631289823, -1.32674174567684, 0.821904631289823, 1, -1.96106396889903, 0.986723611465211 },
	{ 0.780764081275640, -1.54692952632525, 0.780764081275640, 1, -1.56427767500315, 0.864961197373822 },
	{ 0.780764081275640, -1.23429225313410, 0.780764081275640, 1, -1.93459287711451, 0.959002322765484 },
	{ 0.714686410007531, -1.41854618499363, 0.714686410007531, 1, -1.45294320387794, 0.754222796360954 },
	{ 0.714686410007531, -1.06823178848402, 0.714686410007531, 1, -1.90430891262403, 0.926726397665631 },
	{ 0.611402310616563, -1.21661813952701, 0.611402310616563, 1, -1.86538074368294, 0.885520925318713 },
	{ 0.611402310616563, -0.787144826085887, 0.611402310616563, 1, -1.31144754653070, 0.610589268539194 },
	{ 0.461489552066884, -0.920876339017339, 0.461489552066884, 1, -1.81057593401990, 0.829235052618707 },
	{ 0.461489552066884, -0.322599196669853, 0.461489552066884, 1, -1.16218727318019, 0.441359420631550 },
	{ 0.299969123764612, -0.599764553627771, 0.299969123764612, 1, -1.73181537720060, 0.752138831145407 },
	{ 0.299969123764612, 0.349792836547195, 0.299969123764612, 1, -1.08728905725033, 0.313519738807378 } };


//the following filter uses cheby II bandpass
//const double sos_matrix[30][6] = {
//	{ 0.220923216439658, 0, -0.220923216439658, 1.000000000000000, -1.581173319170414, 0.965821622922831},
//	{ 0.220923216439658 ,   0 ,-0.220923216439658  , 1.000000000000000 ,-1.954925718475871 ,  0.989235891934891 },
//	{ 0.216183427007106,   0 ,-0.216183427007106  , 1.000000000000000 ,-1.532101119224441 ,  0.901296130268447 },
//	{ 0.216183427007106,   0 ,-0.216183427007106  , 1.000000000000000 ,-1.933732949675144 ,  0.967961853468526 },
//	{ 0.211807189018726,   0 ,-0.211807189018726  , 1.000000000000000 ,-1.489878648986974 ,  0.842120811331373 },
//	{ 0.211807189018726,   0 ,-0.211807189018726  , 1.000000000000000 ,-1.912432920118631 ,  0.946871408044003 },
//	{ 0.207797303248617,   0 ,-0.207797303248617  , 1.000000000000000 ,-1.454413918882560 ,  0.788488550562242 },
//	{ 0.207797303248617,   0 ,-0.207797303248617  , 1.000000000000000 ,-1.890833345266653 ,  0.925782794499707 },
//	{ 0.204152454314275,   0 ,-0.204152454314275  , 1.000000000000000 ,-1.425595409123324 ,  0.740522333184677 },
//	{ 0.204152454314275,   0 ,-0.204152454314275  , 1.000000000000000 ,-1.868728596427614 ,  0.904513795668882 },
//	{ 0.200868547871834,   0 ,-0.200868547871834  , 1.000000000000000 ,-1.403324117347747 ,  0.698314623510787 },
//	{ 0.200868547871834,   0 ,-0.200868547871834  , 1.000000000000000 ,-1.845891145592909 ,  0.882875654900996 },
//	{ 0.197939759431276,   0 ,-0.197939759431276  , 1.000000000000000 ,-1.822061546986437 ,  0.860667280604746 },
//	{ 0.197939759431276,   0 ,-0.197939759431276  , 1.000000000000000 ,-1.387540089696096 ,  0.661961686288664 },
//	{ 0.195359344457641,   0 ,-0.195359344457641  , 1.000000000000000 ,-1.796936706998081 ,  0.837670231954310 },
//	{ 0.195359344457641,   0 ,-0.195359344457641  , 1.000000000000000 ,-1.378245401559255 ,  0.631595107517786 },
//	{ 0.193120255704142,   0 ,-0.193120255704142  , 1.000000000000000 ,-1.770157003484578 ,  0.813646206890749 },
//	{ 0.193120255704142,   0 ,-0.193120255704142  , 1.000000000000000 ,-1.375524431746631 ,  0.607412404088223 },
//	{ 0.191215607800265,   0 ,-0.191215607800265  , 1.000000000000000 ,-1.741295295224693 ,  0.788341680618151 },
//	{ 0.191215607800265,   0 ,-0.191215607800265  , 1.000000000000000 ,-1.379559496017317 ,  0.589707096211115 },
//	{ 0.189639022566721,   0 ,-0.189639022566721  , 1.000000000000000 ,-1.709857749818007 ,  0.761511185077859 },
//	{ 0.189639022566721,   0 ,-0.189639022566721  , 1.000000000000000 ,-1.390632759703335 ,  0.578893701384191 },
//	{ 0.188384882202515,   0 ,-0.188384882202515  , 1.000000000000000 ,-1.675323834725396 ,  0.732985548109764 },
//	{ 0.188384882202515,   0 ,-0.188384882202515  , 1.000000000000000 ,-1.409087720106584 ,  0.575508890913461 },
//	{ 0.187448511784658,   0 ,-0.187448511784658  , 1.000000000000000 ,-1.637289663227242 ,  0.702836916873098 },
//	{ 0.187448511784658,   0 ,-0.187448511784658  , 1.000000000000000 ,-1.435186533537827 ,  0.580134962905609 },
//	{ 0.186826307564141,   0 ,-0.186826307564141  , 1.000000000000000 ,-1.595823119795508 ,  0.671704816109547 },
//	{ 0.186826307564141,   0 ,-0.186826307564141  , 1.000000000000000 ,-1.468755098164943 ,  0.593132949607912 },
//	{ 0.186515823306459,   0 ,-0.186515823306459  , 1.000000000000000 ,-1.552059405951876 ,  0.641224221569519 },
//	{ 0.186515823306459,   0 ,-0.186515823306459  , 1.000000000000000 ,-1.508587486638711 ,  0.614084913637172 } };


constexpr auto M_PI = 3.141592653589793;
std::vector<double> sosFilter(
	std::vector<double> signal);

int main() {

	std::vector<double> time, signal;


	//signal generation
	for (double i = 0; i < 4 * M_PI; i = i + (double)1 / 30) {
		time.push_back(i);
		//signal has peaks at 1.5hz, 7.5hz, 15hz 
		signal.push_back(1 * std::sin(2.666 * M_PI * i) + 0.4 * std::sin(5 * 2 * M_PI * i) + 0.2 * std::sin(8 * 2 * M_PI * i));
	}

	//filter signal using the existing values


	std::vector<double> filSignal;
	filSignal = sosFilter(signal);

	plt::figure(0);
	plt::subplot(2, 1, 1);
	plt::plot(time, signal);
	plt::title("prefiltered signal");
	plt::subplot(2, 1, 2);
	plt::plot(time, filSignal);
	plt::title("filtered signal");
	plt::show();


	int	numFrames = signal.size();
	double* in = (double*)fftw_malloc(sizeof(double) * signal.size());
	fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numFrames);
	std::vector <double> x, y;
	std::deque<double> rPPGRawValues;
	for (int i = 0; i < (numFrames - 1); i++) {
		double t = time.at(i);
		double sig = (double)signal.at(i);

		x.push_back(t);
		y.push_back(sig);
		rPPGRawValues.push_back(sig);
		in[i] = sig;

	}

	fftw_plan fftPlan = fftw_plan_dft_r2c_1d(numFrames, in, out, FFTW_ESTIMATE);
	fftw_execute(fftPlan);

	std::vector<double> v, ff;
	double sampRate = 30;
	for (int i = 0; i <= ((numFrames / 2) - 1); i++) {
		double a, b;
		a = sampRate * i / numFrames;
		//Here I have calculated the y axis of the spectrum in dB
		b = (20 * log(sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]))) / numFrames;
		if (i == 0) {
			b = b * -1;
		}

		v.push_back((double)b);
		ff.push_back((double)a);

	}

	plt::figure(1);
	plt::subplot(2, 1, 1);
	plt::plot(x, y);
	plt::title("Normal signal (tisme-domain)");
	plt::xlabel("Time (s)");
	plt::ylabel("Measured Rotation");

	plt::subplot(2, 1, 2);
	plt::plot(ff, v);
	plt::title("Normal signal (frequency-domain)");
	plt::xlabel("Frequency (Hz)");
	plt::ylabel("Power");
	plt::show();

	fftw_destroy_plan(fftPlan);
	fftw_free(in);
	fftw_free(out);


	numFrames = filSignal.size();
	double* in1 = (double*)fftw_malloc(sizeof(double) * filSignal.size());
	fftw_complex* out1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numFrames);
	std::vector <double> x1, y1;
	for (int i = 0; i < (numFrames - 1); i++) {
		double t = time.at(i);
		double sig = (double)filSignal.at(i);

		x1.push_back(t);
		y1.push_back(sig);
		in1[i] = sig;

	}

	fftw_plan fftPlan1 = fftw_plan_dft_r2c_1d(numFrames, in1, out1, FFTW_ESTIMATE);
	fftw_execute(fftPlan1);

	std::vector<double> v1, ff1;

	for (int i = 0; i <= ((numFrames / 2) - 1); i++) {
		double a, b;
		a = sampRate * i / numFrames;
		//Here I have calculated the y axis of the spectrum in dB
		b = (20 * log(sqrt(out1[i][0] * out1[i][0] + out1[i][1] * out1[i][1]))) / numFrames;
		if (i == 0) {
			b = b * -1;
		}

		v1.push_back((double)b);
		ff1.push_back((double)a);

	}

	plt::figure(2);
	plt::subplot(2, 1, 1);
	plt::plot(x1, y1);
	plt::title("Filtered signal (tisme-domain)");
	plt::xlabel("Time (s)");
	plt::ylabel("Measured Rotation");

	plt::subplot(2, 1, 2);
	plt::plot(ff1, v1);
	plt::title("Filtered signal (frequency-domain)");
	plt::xlabel("Frequency (Hz)");
	plt::ylabel("Power");
	plt::show();

	fftw_destroy_plan(fftPlan1);
	fftw_free(in1);
	fftw_free(out1);

	return 100;
}


std::vector<double> sosFilter(std::vector<double> signal) {

	std::vector<double> output;
	output.resize(signal.size());

	double** tempOutput = new double* [FILTER_SECTIONS];
	for (int i = 0; i < FILTER_SECTIONS; i++)
		tempOutput[i] = new double[signal.size()];
	for (int i = 0; i < FILTER_SECTIONS; i++) {
		for (int j = 0; j < signal.size(); j++) {
			tempOutput[i][j] = 0;
		}
	}

	for (int i = 0; i < signal.size(); i++) {

		if (i - 2 < 0) {
			std::cout << "skipping some stuff" << std::endl;
			continue;
		}

		double b0, b1, b2, a1, a2;
		double result;
		//for each section
		for (int j = 0; j < FILTER_SECTIONS; j++) {

			b0 = sos_matrix[j][0];
			b1 = sos_matrix[j][1];
			b2 = sos_matrix[j][2];
			a1 = sos_matrix[j][4];
			a2 = sos_matrix[j][5];

			if (j == 0) {
				result = b0 * signal[i] + b1 * signal[i - 1] + b2 * signal[i - 2]
					- a1 * tempOutput[j][i - 1] - a2 * tempOutput[j][i - 2];
				tempOutput[j][i] = result;
			}
			else {
				result = b0 * tempOutput[j - 1][i] + b1 * tempOutput[j - 1][i - 1] + b2 * tempOutput[j - 1][i - 2]
					- a1 * tempOutput[j][i - 1] - a2 * tempOutput[j][i - 2];
				tempOutput[j][i] = result;
			}

		}

		
	}
	for (int x = 0; x < signal.size(); x++) {
		output[x] = tempOutput[FILTER_SECTIONS-1][x];
		std::cout << "output: " << output[x] << std::endl;
	}
	
	return output;
}

/*
std::vector<double> sosFilter(std::vector<double> signal) {

	//initialise vectors for signal processing
	std::vector<double> output;
	output.resize(signal.size());

	std::vector<double>
		_output1, _output2,
		_output3, _output4,
		_output5, _output6,
		_output7, _output8,
		_output9, _output10,
		_output11;
	_output1.resize(signal.size());
	_output2.resize(signal.size());
	_output3.resize(signal.size());
	_output4.resize(signal.size());
	_output5.resize(signal.size());
	_output6.resize(signal.size());
	_output7.resize(signal.size());
	_output8.resize(signal.size());
	_output9.resize(signal.size());
	_output10.resize(signal.size());
	_output11.resize(signal.size());



	for (int i = 0; i < signal.size(); i++) {

		if (i - 2 < 0) continue;

		//hardcoding IIR filter with 12 second order sections
		int j = 0;
		double b0, b1, b2, a1, a2;
		double num0, num1, num2;
		double den1, den2;

		//section 1
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		//a0 is not used as it is 1;
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];

		num0 = b0 * (signal[i]);
		num1 = b1 * (signal[i - 1]);
		num2 = b2 * (signal[i - 2]);
		den1 = a1 * _output1[i - 1];
		den2 = a2 * _output1[i - 2];
		_output1[i] = (num0 + num1 + num2 - den1 - den2);

		//section 2
		j = 1;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output1[i]);
		num1 = b1 * (_output1[i - 1]);
		num2 = b2 * (_output1[i - 2]);
		den1 = a1 * _output2[i - 1];
		den2 = a2 * _output2[i - 2];
		_output2[i] = (num0 + num1 + num2 - den1 - den2);

		//section 3
		j = 2;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output2[i]);
		num1 = b1 * (_output2[i - 1]);
		num2 = b2 * (_output2[i - 2]);
		den1 = a1 * _output3[i - 1];
		den2 = a2 * _output3[i - 2];
		_output3[i] = (num0 + num1 + num2 - den1 - den2);

		//section 4
		j = 3;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output3[i]);
		num1 = b1 * (_output3[i - 1]);
		num2 = b2 * (_output3[i - 2]);
		den1 = a1 * _output4[i - 1];
		den2 = a2 * _output4[i - 2];
		_output4[i] = (num0 + num1 + num2 - den1 - den2);

		//section 5
		j = 4;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output4[i]);
		num1 = b1 * (_output4[i - 1]);
		num2 = b2 * (_output4[i - 2]);
		den1 = a1 * _output5[i - 1];
		den2 = a2 * _output5[i - 2];
		_output5[i] = (num0 + num1 + num2 - den1 - den2);

		//section 6
		j = 5;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output5[i]);
		num1 = b1 * (_output5[i - 1]);
		num2 = b2 * (_output5[i - 2]);
		den1 = a1 * _output6[i - 1];
		den2 = a2 * _output6[i - 2];
		_output6[i] = (num0 + num1 + num2 - den1 - den2);

		//section 7
		j = 6;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output6[i]);
		num1 = b1 * (_output6[i - 1]);
		num2 = b2 * (_output6[i - 2]);
		den1 = a1 * _output7[i - 1];
		den2 = a2 * _output7[i - 2];
		_output7[i] = (num0 + num1 + num2 - den1 - den2);


		//section 8
		j = 7;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output7[i]);
		num1 = b1 * (_output7[i - 1]);
		num2 = b2 * (_output7[i - 2]);
		den1 = a1 * _output8[i - 1];
		den2 = a2 * _output8[i - 2];
		_output8[i] = (num0 + num1 + num2 - den1 - den2);


		//section 9
		j = 8;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output8[i]);
		num1 = b1 * (_output8[i - 1]);
		num2 = b2 * (_output8[i - 2]);
		den1 = a1 * _output9[i - 1];
		den2 = a2 * _output9[i - 2];
		_output9[i] = (num0 + num1 + num2 - den1 - den2);

		//section 10
		j = 9;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output9[i]);
		num1 = b1 * (_output9[i - 1]);
		num2 = b2 * (_output9[i - 2]);
		den1 = a1 * _output10[i - 1];
		den2 = a2 * _output10[i - 2];
		_output10[i] = (num0 + num1 + num2 - den1 - den2);

		//section 11
		j = 10;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output10[i]);
		num1 = b1 * (_output10[i - 1]);
		num2 = b2 * (_output10[i - 2]);
		den1 = a1 * _output11[i - 1];
		den2 = a2 * _output11[i - 2];
		_output11[i] = (num0 + num1 + num2 - den1 - den2);

		//section 12
		j = 11;
		b0 = sos_matrix[j][0];
		b1 = sos_matrix[j][1];
		b2 = sos_matrix[j][2];
		a1 = sos_matrix[j][4];
		a2 = sos_matrix[j][5];
		num0 = b0 * (_output11[i]);
		num1 = b1 * (_output11[i - 1]);
		num2 = b2 * (_output11[i - 2]);
		den1 = a1 * output[i - 1];
		den2 = a2 * output[i - 2];
		output[i] = (num0 + num1 + num2 - den1 - den2);
		std::cout << "output at i: " << i << " value: " << output[i] << std::endl;

	}

	return output;
}
*/