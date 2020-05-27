%code to decode the signal
clear all; clc; close all;
fileToRead = 'output_file2.csv';
M = readmatrix(fileToRead);
fps = M(1,1);
frameNumber = M(:,2);
ROI_ColorIntensity = M(:,3);
time = frameNumber./fps;

%discard first 10 seconds of sampled data
n = fps*10;
time(1:n) = [];
ROI_ColorIntensity(1:n)=[];

figure(1)
plot(time,ROI_ColorIntensity);
xlabel('Time (Seconds)');
ylabel('Color Intensity');
if(0)
    fromX = 10;
    toX = 20;
    axis([fromX toX min(simple(fromX:toX))-3 max(simple(fromX:toX))+5]) 
end


% windowSize = 10; 
% b = (1/windowSize)*ones(1,windowSize);
% a = 1;
% simple = filter(b,a,ROI_ColorIntensity);
simple = bandpass(ROI_ColorIntensity,[0.8,2.55],fps);
n = (3*fps)/fps;
time(1:n) = [];
simple(1:n)=[];

figure(2)
findpeaks(simple,time,'MinPeakProminence', 0.3);
xlabel('Time (Seconds)')
ylabel('Color Intensity')
title('Find Prominent Peaks')


%figure(3)
[pks,locs] = findpeaks(simple,time,'MinPeakProminence', 0.25);
peakInterval = diff(locs);
%histogram(peakInterval,30)
xlabel('time')
ylabel('number of signal peaks received')

fprintf('the average heart rate measured was: %.2f \n',(60/mean(peakInterval)))

% figure(4)
% %sampling freq = fps
% Fs = fps;
% T = 1/Fs;
% L = time(end);
% Y = fft(simple);
% P2 = abs(Y/L);
% P1 = P2(1:(floor(L/2+1)));
% P1(2:end-1) = 2*P1(2:end-1);
% f = Fs*(0:(L/2))/L;
% plot(f,P1) 
    %filtering the content
    %lowpass(ROI_ColorIntensity,1,1/30);