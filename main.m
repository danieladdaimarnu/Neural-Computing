% Main entry point.

% Clear workspace and Command window
close all; clc;

% add libs folder to path
addpath(genpath("scripts"));

% add data
% data source https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+
rawdataPath = sprintf('%s/data/new_HTRU.csv', pwd); % raw data path
elecdataPath = sprintf('%s/data/smote_new_HTRU.csv', pwd); % smote data path
rawData = readtable(rawdataPath); % load raw data
elecData = readtable(elecdataPath); % load smote data

% define input and target columns
rng('default');
% Cross varidation (train: 70%, test: 30%)
cv = cvpartition(size(elecData,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
elecDataTrain = elecData(~idx,:);
elecDataTest  = elecData(idx,:);

trainInput = elecDataTrain(:, 1:end-1); % input training data 
trainTarget = elecDataTrain(:, end); %target training data
testInput = elecDataTest(:, 1:end-1); % input testing data 
testTarget = elecDataTest(:, end); %target testing data

%% Main entry points to run scripts

section = 1;
while section ~= 0
    %close all; clear all;
    fprintf('\nPlease, type a number between 1 and 9 to run the related script, or 0 to exit the program\n\n')
    fprintf('>> 1 : Exploratory Data Analysis\n')
    fprintf('>> 2 : Feature Selection\n')
    fprintf('>> 3 : Decision Boundaries: SVM vs MLP\n')
    fprintf('>> 4 : SVM Hyperparameter Tuning\n')
    fprintf('>> 5 : MLP Hyperparameter Tuning\n')
    fprintf('>> 6 : Learning Curve\n')
    fprintf('>> 7 : Best Models Performance\n')
    fprintf('>> 8 : SVM Time Space Complexity\n')
    fprintf('>> 9 : MLP Time Space Complexity\n')
    fprintf('Type 0 to exit the program ...\n\n')
    
    section = input('Enter a number: ');
    
    switch section
        case 1
            % Exploratory Data Analysis
            ExploratoryDataAnalysis(rawData, elecData);
            pause(3)
        case 2
            % Feature Selection using minimum redundancy maximum
            % and decision trees
            FeatureSelection(elecDataTrain, trainTarget);
            pause(3)
        case 3
            % Decision Boundaries: Naive Bayes vs Random Forest
            DecisionBoundary(elecDataTrain);
            pause(3)
        case 4
            % SVM Hyperparameter Tuning
            SVMHyperParameterTuning(elecDataTrain);
            pause(3)
        case 5
            % MLP Hyperparameter Tuning
            MLPHyperParameterTuning(elecDataTrain);
            pause(3)
        case 6
            % Learning Curve
            LearningCurve(elecDataTrain)
            pause(3)
        case 7
            % Performance Evaluation
            FinalModels(elecDataTrain, elecDataTest)
            pause(3)
        case 8
            % Time Space Complexity
            SVMSpaceTimeComplexity(elecDataTrain)
            pause(3)
        case 9
            % Time Space Complexity
            MLPSpaceTimeComplexity(elecDataTrain)
            pause(3)
        case 0
            continue
        otherwise
            % if number is invalid
            fprintf('\nPlease pick a viable number between 1 and 6.\n')
            fprintf('Type 0 to exit the program ...\n\n')
            pause(1)
    end
end

% Clear workspace, console
close all; clc;
fprintf('Exit.\n\n')