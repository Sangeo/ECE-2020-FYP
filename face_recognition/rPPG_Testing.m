fileToRead = 'rPPG_TestRes.csv';
M = readmatrix(fileToRead);
t = M(:,1);
x = M(:,2);

plot(t,x);
title('raw vs filtered signal');
hold on;

% D = designfilt('bandpassiir', 'StopbandFrequency1', .6, 'PassbandFrequency1', .9, ...
%     'PassbandFrequency2', 3, 'StopbandFrequency2', 3.2, 'StopbandAttenuation1', 25,...
%     'PassbandRipple', 1, 'StopbandAttenuation2', 25, 'SampleRate', 29);
% bpSig=filtfilt(D,x);
% plot(t,bpSig);

[y,d] = bandpass(x,[0.8 3],29);
plot(t,y);

D2 = designfilt('bandpassiir', 'StopbandFrequency1', 0.6, 'PassbandFrequency1', 0.8,...
    'PassbandFrequency2', 3, 'StopbandFrequency2', 3.2, 'StopbandAttenuation1', 40, ...
    'PassbandRipple', 1, 'StopbandAttenuation2', 40, 'SampleRate', 30);
bpSig=filtfilt(D2,x);
% plot(t,bpSig/100);
legend('raw','filtered 1','filtered 2');


