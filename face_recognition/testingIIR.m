load('filterWorkspace.mat');

t = 0:1/30:4*pi;
x=sin(2.666*pi*t) + 0.4*sin(5*2*pi*t) + 0.2*sin(8*2*pi*t);
plot(t,x);

D = designfilt('bandpassiir', 'StopbandFrequency1', .6, 'PassbandFrequency1', .9, ...
    'PassbandFrequency2', 3, 'StopbandFrequency2', 3.2, 'StopbandAttenuation1', 25,...
    'PassbandRipple', 1, 'StopbandAttenuation2', 25, 'SampleRate', 30);
bpSig=filtfilt(D,x);
hold on;
plot(t,bpSig);

D2 = designfilt('bandpassiir', 'StopbandFrequency1', 0.6, 'PassbandFrequency1', 0.8,...
    'PassbandFrequency2', 2.6, 'StopbandFrequency2', 3, 'StopbandAttenuation1', 45, ...
    'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', 30, 'DesignMethod', 'cheby2');
bpSig=filtfilt(D2,x);
plot(t,bpSig);
legend('raw','filtered 1','filtered 2');

D2Coeff = D2.Coefficients;