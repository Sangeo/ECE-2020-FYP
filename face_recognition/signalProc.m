%code to decode the signal
clear all; clc; close all;
fileToRead = 'output_file2.csv';
M = readmatrix(fileToRead);
fps = M(1,1);
frameNumber = M(:,2);
ROI_ColorIntensity = M(:,3);
time = frameNumber./fps;

lag = 10;
simple = movmean(ROI_ColorIntensity,lag);

figure(1)
plot(time,simple);
xlabel('Time (Seconds)');
ylabel('Color Intensity');
if(0)
    fromX = 5;
    toX = 15;
    axis([fromX toX min(simple(fromX:toX))-3 max(simple(fromX:toX))+5]) 
end

figure(2)
findpeaks(simple,time,'MinPeakProminence',1.2);
xlabel('Time (Seconds)')
ylabel('Color Intensity')
title('Find Prominent Peaks')


    %filtering the content
    %lowpass(ROI_ColorIntensity,1,1/30);