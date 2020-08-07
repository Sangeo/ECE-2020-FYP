%run this script after placing the answer result in x
%this file wll plot two plots, one of the x raw values against each frame
%number, and then the next is the one sided power spectral density function
clear all;
fileToRead = 'output_file3.csv';
M = readmatrix(fileToRead);
x = M(:,2);

clc; close all;
figure(1);
plot(x);

figure(2);
fs = 30; %30 frames per second
L=length(x);        
NFFT=1024;       
X=fft(x,NFFT);       
Px=X.*conj(X)/(NFFT*L); %Power of each freq components       
fVals=fs*(0:NFFT/2-1)/NFFT;      
plot(fVals,Px(1:NFFT/2),'b','LineSmoothing','on','LineWidth',1);         
title('One Sided Power Spectral Density');       
xlabel('Frequency (Hz)')         
ylabel('PSD');