%code to decode the signal
clear all; clc;
fileToRead = 'output_file2.csv';
M = readmatrix(fileToRead);
fps = M(1,1);
frameNumber = M(:,2);
ROI_ColorIntensity = M(:,3);
time = frameNumber./fps;

figure(1)
plot(time,ROI_ColorIntensity);
xlabel('Seconds');
ylabel('Color Intensity');