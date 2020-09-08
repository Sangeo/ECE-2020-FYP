%run this script after placing the answer result in x
%this file wll plot two plots, one of the x raw values against each frame
%number, and then the next is the one sided power spectral density function
clear all;
fileToRead = 'rBCG_analysis.xlsm';
M = readmatrix(fileToRead,'Sheet',2);
x = M(:,3);
t = M(:,1);
t1 = t;
for i = 1:length(t)-1
    t1(i+1) = t1(i) + t(i+1);
end
t1 = t1 - t1(1);

clc; close all;
figure(1);
frameNumber = 1:length(x);
time = t1;
% fs = floor(1/mean(diff(t1/1000)));
% x = x - mean(x);
% x = lowpass(x,3,fs);

fs = 29;

[rows, cols] = size(M);
xMat = zeros(rows,cols);
sumSig = zeros(rows,cols);
hold on;
for i = 1:cols-2
    xMat(:,i) = M(:,i+2);
    plot(time,xMat(:,i));
    sigMat(:,i) = xMat(:,i)-mean(xMat(:,i));
    sumSig = sumSig + sigMat(:,i);
    
end

xlabel('time (s)');
ylabel('y displacement');
figure(2)
hold on;

avgSig = sumSig(:,end)./(cols-2);
time = time(300:end-100);
avgSig = avgSig(300:end-100);
plot(time,avgSig);

xlabel('time (s)');
ylabel('y displacement');

x = avgSig;
figure(3);
L=length(x);        
NFFT=1024;       
X=fft(x,NFFT);       
Px=X.*conj(X)/(NFFT*L); %Power of each freq components     
Px=Px(1:NFFT/2);
fVals=fs*(0:NFFT/2-1)/NFFT;      
plot(fVals,Px,'b','LineSmoothing','on','LineWidth',1);         
title('One Sided Power Spectral Density');       
xlabel('Frequency (Hz)')         
ylabel('PSD');

[val, loc] = max(Px);

fprintf('the average heart rate measured was: %.2f \n',(60*fVals(loc)))
