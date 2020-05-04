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
simple = bandpass(ROI_ColorIntensity,[0.8,3],fps);
n = fps/fps;
time(1:n) = [];
simple(1:n)=[];

figure(2)
findpeaks(simple,time,'MinPeakProminence', 0.3);
xlabel('Time (Seconds)')
ylabel('Color Intensity')
title('Find Prominent Peaks')


figure(3)
[pks,locs] = findpeaks(simple,time,'MinPeakProminence', 0.3);
peakInterval = diff(locs);
hist(peakInterval)
xlabel('time')
ylabel('frequency of signal peaks')

fprintf('the average heart rate measured was: %.2f \n',(60/mode(peakInterval)))

fft(simple,10);
    %filtering the content
    %lowpass(ROI_ColorIntensity,1,1/30);