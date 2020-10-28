clc; close all; clear all;
%Best case conditions
rPPG_results = [79.6217, 69.674, 68.5042, 84.4639, 80.0129, 72.1477, 75.1186, 75.7337, 77.9712,...
 74.7228, 72.6059, 74.7296, 78.9453, 75.7608, 76.4472, 74.3711, 77.4079, 74.1805, 71.4149, ...
 75.4954, 80.0699, 77.5925, 82.5168, 78.1897, 81.7476, 74.3613, 77.2469 , 74.2024, 71.186, ...
 68.1478 ,72.4187 ,74.1299];

% rBCG_results = [77.3333,73.4667,81.2,73.4667,69.6,77.3333,73.4667,81.2,77.3333,69.6,69.6,81.2,...
% 82.2857,128.571,92.5714 ,66.6667,66.8571,68.764,77.1429,113.77,80.5369,66.8571,68.9189,...
% 72,77.1429,72,82.2857,72,85.5204,69.0745,69.2308,72];
% 
rBCG_results = [77.3333,73.4667,81.2,73.4667,69.6,77.3333,73.4667,81.2,77.3333,69.6,69.6,81.2,...
82.2857,80,92.5714 ,66.6667,66.8571,68.764,77.1429,80.77,80.5369,66.8571,68.9189,...
72,77.1429,72,82.2857,72,77.5204,69.0745,69.2308,72];

measured_results = [78, 77,81,84,79,79,75,76,78,78,80,90,78,77,79,77,78,77,78,78 ,81,76 ,...
79,76,80,75,82,78,76,70,73,72];

% combined_results = [rPPG_results;rBCG_results;measured_results]';
combined_results = [rBCG_results;measured_results]';
% group_name = {'rPPG','rBCG','PPG'};
group_name = {'rBCG','PPG'};
p = anova1(combined_results,group_name)
ylabel('Group Mean Estimations');
xlabel('Tested Method');
title('Resting Condition - Best Condition Test');
figure
[R,PVal,H]= corrplot(combined_results,'type','Pearson','varNames',group_name,'testR','on','alpha',0.05,'tail','right');